"""
Created on 20 Sep, 2020

@author: Xiaokun Zhang
"""

import os
import time
import random
import argparse
import pickle
import numpy as np
from tqdm import tqdm
from os.path import join

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from torch.backends import cudnn

import metric
from utils import collate_fn
from didn import DIDN, DIDN_retrieval_enhanced, negative_sampling_loss
from dataset import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', default='datasets/yoochoose1_64/', help='dataset directory path: datasets/diginetica/yoochoose1_4/yoochoose1_64')
parser.add_argument('--batch_size', type=int, default=512, help='input batch size 512')
parser.add_argument('--hidden_size', type=int, default=64, help='hidden state size of gru module')
parser.add_argument('--embed_dim', type=int, default=64, help='the dimension of item embedding')
parser.add_argument('--epoch', type=int, default=150, help='the number of epochs to train for 100')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=80,
                    help='the number of steps after which the learning rate decay')
parser.add_argument('--test', default=False, help='test')
parser.add_argument('--topk', type=int, default=20, help='number of top score items selected for calculating recall and mrr metrics')
parser.add_argument('--valid_portion', type=float, default=0.1,
                    help='split the portion of training set as validation set')

parser.add_argument('--position_embed_dim', type=int, default=64, help='the dimension of position embedding')
parser.add_argument('--max_len', type=float, default=19, help='max length of input session')
parser.add_argument('--alpha1', type=float, default=0.1, help='degree of alpha pooling')
parser.add_argument('--alpha2', type=float, default=0.1, help='degree of alpha pooling')
parser.add_argument('--alpha3', type=float, default=0.1, help='degree of alpha pooling')
parser.add_argument('--pos_num', type=int, default=2000, help='the number of position encoding')
parser.add_argument('--neighbor_num', type=int, default=5, help='the number of neighboring sessions')
parser.add_argument('--k', type=int, default=3, help='the number of retrieved sessions, if k=0, then it is the same as the original DIDN')

args = parser.parse_args()
print(args)

here = os.path.dirname(os.path.abspath(__file__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# if torch.cuda.is_available():
#     torch.cuda.set_device(1)

# debug mode
debug = False
if debug:
    torch.autograd.set_detect_anomaly(True)


def main():
    print('Loading data...')
    train, valid, test = load_data(args.dataset_path, valid_portion=args.valid_portion)

    train_data = RecSysDataset(train)
    valid_data = RecSysDataset(valid)
    test_data = RecSysDataset(test)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    if args.dataset_path.split('/')[-2] in ['diginetica', 'yoochoose1_64']:
        with open(args.dataset_path + 'num_items.txt', 'r') as f:
            n_items = int(f.readline().strip())
    else:
        raise Exception('Unknown Dataset!')
    model = DIDN(n_items, args.hidden_size, args.embed_dim, args.batch_size, args.max_len, args.position_embed_dim, args.alpha1, args.alpha2, args.alpha3,args.pos_num, args.neighbor_num).to(device)

    if args.test:
        ckpt = torch.load('latest_checkpoint.pth.tar')
        model.load_state_dict(ckpt['state_dict'])
        recall, mrr = validate(test_loader, model)
        # print("Test: Recall@{}: {:.4f}, MRR@{}: {:.4f}".format(args.topk, recall, args.topk, mrr))
        print("Test: Recall@{}\t MRR@{}\t".format(args.topk, args.topk))
        print("Test: {:.4f}\t {:.4f}".format(recall * 100, mrr * 100))
        return

    optimizer = optim.Adam(model.parameters(), args.lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size=args.lr_dc_step, gamma=args.lr_dc)

    for epoch in tqdm(range(args.epoch)):
        # train for one epoch
        scheduler.step(epoch=epoch)
        trainForEpoch(train_loader, model, optimizer, epoch, args.epoch, criterion, log_aggr=200)

        recall, mrr = validate(valid_loader, model)
        print('Epoch {} validation: Recall@{}: {:.4f}, MRR@{}: {:.4f} \n'.format(epoch, args.topk, recall, args.topk,
                                                                                 mrr))

        # store best loss and save a model checkpoint
        ckpt_dict = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        torch.save(ckpt_dict, 'latest_checkpoint.pth.tar')

def main_retrieval_enhanced():
    print("Loading data...")
    train, valid, test = load_retrieved_data(args.dataset_path, valid_portion=args.valid_portion, k = args.k)
    train_data = RecSysRetrievedDataset(train)
    valid_data = RecSysRetrievedDataset(valid)
    test_data = RecSysRetrievedDataset(test)
    train_loader = DataLoader(train_data, 
                              batch_size=args.batch_size, 
                              shuffle=True, 
                              collate_fn=collate_fn_for_retrieved_dataset)
    valid_loader = DataLoader(valid_data, 
                              batch_size=args.batch_size, 
                              shuffle=False, 
                              collate_fn=collate_fn_for_retrieved_dataset)
    test_loader = DataLoader(test_data, 
                             batch_size=args.batch_size, 
                             shuffle=False, 
                             collate_fn=collate_fn_for_retrieved_dataset)
    
    
    if args.dataset_path.split('/')[-2] in ['diginetica', 'yoochoose1_64']:
        with open(args.dataset_path + 'num_items.txt', 'r') as f:
            n_items = int(f.readline().strip())
    else:
        raise Exception('Unknown Dataset!')
    
    model = DIDN_retrieval_enhanced(n_items=n_items,
                                    hidden_size=args.hidden_size,
                                    embedding_dim=args.embed_dim,
                                    batch_size=args.batch_size,
                                    position_embed_dim=args.position_embed_dim,
                                    max_len=args.max_len,
                                    alpha1=args.alpha1,
                                    alpha2=args.alpha2,
                                    alpha3=args.alpha3,
                                    pos_num=args.pos_num,
                                    neighbor_num=args.neighbor_num,
                                    k=args.k).to(device)
    
    print(model.parameters)
    
    # if args.test:
    #     ckpt = torch.load('latest_checkpoint.pth.tar')
    #     model.load_state_dict(ckpt['state_dict'])
    #     recall, mrr = validate_retrieval_enhanced(test_loader, model)
    #     # print("Test: Recall@{}: {:.4f}, MRR@{}: {:.4f}".format(args.topk, recall, args.topk, mrr))
    #     print("Test: Recall@{}\t MRR@{}\t".format(args.topk, args.topk))
    #     print("Test: {:.4f}\t {:.4f}".format(recall * 100, mrr * 100))
    #     return

    optimizer = optim.Adam(model.parameters(), args.lr)
    criterion = nn.CrossEntropyLoss()
    # criterion = negative_sampling_loss
    scheduler = StepLR(optimizer, step_size=args.lr_dc_step, gamma=args.lr_dc)

    best_recall = 0
    best_mrr = 0
    patient = 10
    patient_count = patient
    timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    best_model_path = f'best_model_weight/best_model_{timestamp}.pth'
    print(f"Best model path: {best_model_path}")

    for epoch in tqdm(range(args.epoch)):
        # train for one epoch
        trainForEpoch_retrieval_enhanced(train_loader, model, optimizer, epoch, args.epoch, criterion, log_aggr=200)
        scheduler.step(epoch=epoch)

        recall, mrr = validate_retrieval_enhanced(valid_loader, model)
        print('Epoch {} validation: Recall@{}: {:.4f}, MRR@{}: {:.4f} \n'.format(epoch, args.topk, recall, args.topk,
                                                                                 mrr))
        
        # recall_test, mrr_test = validate_retrieval_enhanced(test_loader, model)
        # print('Epoch {} test: Recall@{}: {:.4f}, MRR@{}: {:.4f} \n'.format(epoch, args.topk, recall_test, args.topk,
        #                                                                          mrr_test))
        
        # save the best model
        if epoch == 0:
            best_recall = recall
            best_mrr = mrr
        else:
            if recall > best_recall:
                best_recall = recall
                best_mrr = mrr
                
                torch.save(model.state_dict(), best_model_path)
                print("Best model saved.")
                patient_count = patient
            else:
                patient_count -= 1
                print(f"Patient count: {patient_count}")
                if patient_count == 0:
                    print("Early stopping...")
                    break


        # store best loss and save a model checkpoint
        # ckpt_dict = {
        #     'epoch': epoch + 1,
        #     'state_dict': model.state_dict(),
        #     'optimizer': optimizer.state_dict()
        # }

        # torch.save(ckpt_dict, 'latest_checkpoint.pth.tar')

    print("Best model on validation set: Recall@{}\t MRR@{}\t".format(args.topk, args.topk))
    print("Best model on validation set: {:.4f}\t {:.4f}".format(best_recall * 100, best_mrr * 100))

    # load the best model and evaluate on the test set
    model.load_state_dict(torch.load(best_model_path))
    recall, mrr = validate_retrieval_enhanced(test_loader, model)
    # print("Test: Recall@{}: {:.4f}, MRR@{}: {:.4f}".format(args.topk, recall, args.topk, mrr))
    print("Test: Recall@{}\t MRR@{}\t".format(args.topk, args.topk))
    print("Test: {:.4f}\t {:.4f}".format(recall * 100, mrr * 100))



def trainForEpoch(train_loader, model, optimizer, epoch, num_epochs, criterion, log_aggr=1):
    model.train()

    sum_epoch_loss = 0

    start = time.time()
    for i, (seq, target, lens) in tqdm(enumerate(train_loader), total=len(train_loader)):
        seq = seq.to(device)
        target = target.to(device)
        # neighbors = neighbors.to(device)

        optimizer.zero_grad()
        outputs = model(seq, lens)

        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        sum_epoch_loss += loss_val

        iter_num = epoch * len(train_loader) + i + 1

        if i % log_aggr == 0:
            print('[TRAIN] epoch %d/%d batch loss: %.4f (avg %.4f) (%.2f im/s)'
                  % (epoch + 1, num_epochs, loss_val, sum_epoch_loss / (i + 1),
                     len(seq) / (time.time() - start)))

        start = time.time()


def validate(valid_loader, model):
    model.eval()
    recalls = []
    mrrs = []
    with torch.no_grad():
        for seq, target, lens in tqdm(valid_loader):
            seq = seq.to(device)
            target = target.to(device)
            # neighbors = neighbors.to(device)
            # case study
            # case_seq = seq.permute(1,0)
            # np.save('seq.npy', case_seq.cpu())
            # np.save('label.npy', target.cpu())
            outputs = model(seq, lens)
            logits = F.softmax(outputs, dim=1)
            recall, mrr = metric.evaluate(logits, target, k=args.topk)
            recalls.append(recall)
            mrrs.append(mrr)

    mean_recall = np.mean(recalls)
    mean_mrr = np.mean(mrrs)
    return mean_recall, mean_mrr

def trainForEpoch_retrieval_enhanced(train_loader, model, optimizer, epoch, num_epochs, criterion, log_aggr=1):
    model.train()

    sum_epoch_loss = 0


    start = time.time()
    for i, (given_session, given_session_label, given_session_len, retrieved_sessions, retrieved_sessions_labels, retrieved_sessions_lens) in tqdm(enumerate(train_loader), total=len(train_loader)):
        given_session = given_session.to(device)
        given_session_label = given_session_label.to(device)
        retrieved_sessions = retrieved_sessions.to(device)
        retrieved_sessions_labels = retrieved_sessions_labels.to(device)

        # train all
        optimizer.zero_grad()
        # outputs, negative_sample_scores, classification_weight = model.forward_retrieval_enhanced(given_session, given_session_len, retrieved_sessions, retrieved_sessions_lens, retrieved_sessions_labels, 0)
        # loss = criterion(outputs, negative_sample_scores, classification_weight, given_session_label)
        outputs = model.forward_retrieval_enhanced(given_session, given_session_len, retrieved_sessions, retrieved_sessions_lens, retrieved_sessions_labels)
        loss = criterion(outputs, given_session_label)
        if debug:
            with torch.autograd.detect_anomaly():
                loss.backward()
        else:
            loss.backward()
        optimizer.step()


        # # two-stage training
        # with torch.no_grad():
        #     outputs, negative_sample_scores, classification_weight = model.forward_retrieval_enhanced(given_session, given_session_len, retrieved_sessions, retrieved_sessions_lens, retrieved_sessions_labels)
        #     loss = criterion(outputs, negative_sample_scores, classification_weight, given_session_label)
        #     loss_val = loss.item()

        #     if i % log_aggr == 0:
        #         print('[TRAIN] epoch %d/%d batch loss 0: %.4f ' % (epoch + 1, num_epochs, loss_val))

        # # training step 1
        # optimizer.zero_grad()
        # scores, negative_sample_scores, classification_weight = model.forward_retrieval_enhanced(given_session, given_session_len, retrieved_sessions, retrieved_sessions_lens, retrieved_sessions_labels, 1)
        # loss = criterion(scores, negative_sample_scores, classification_weight, given_session_label)

        # if debug:
        #     with torch.autograd.detect_anomaly():
        #         loss.backward()
        # else:
        #     loss.backward()
        # optimizer.step()

        # loss_val = loss.item()

        # if i % log_aggr == 0:
        #     print('[TRAIN] epoch %d/%d batch loss 1: %.4f ' % (epoch + 1, num_epochs, loss_val))

        # # training step 2
        # optimizer.zero_grad()
        # scores, negative_sample_scores, classification_weight = model.forward_retrieval_enhanced(given_session, given_session_len, retrieved_sessions, retrieved_sessions_lens, retrieved_sessions_labels, 2)
        # loss = criterion(scores, negative_sample_scores, classification_weight, given_session_label)

        # if debug:
        #     with torch.autograd.detect_anomaly():
        #         loss.backward()
        # else:
        #     loss.backward()
        # optimizer.step()
        

        loss_val = loss.item()
        sum_epoch_loss += loss_val

        iter_num = epoch * len(train_loader) + i + 1

        if i % log_aggr == 0:
            print('[TRAIN] epoch %d/%d batch loss 2: %.4f (avg %.4f) (%.2f im/s)'
                  % (epoch + 1, num_epochs, loss_val, sum_epoch_loss / (i + 1),
                     len(given_session) / (time.time() - start)))
            # print(f"Loss 0: {loss_0_val:.4f}, Loss 1: {loss_1_val:.4f}, Loss 2: {loss_2_val:.4f}")

        start = time.time()


def validate_retrieval_enhanced(valid_loader, model):
    model.eval()
    recalls = []
    mrrs = []
    with torch.no_grad():
        for given_session, given_session_label, given_session_len, retrieved_sessions, retrieved_sessions_labels, retrieved_sessions_lens in tqdm(valid_loader):
            given_session = given_session.to(device)
            given_session_label = given_session_label.to(device)
            retrieved_sessions = retrieved_sessions.to(device)
            retrieved_sessions_labels = retrieved_sessions_labels.to(device)

            outputs = model.forward_retrieval_enhanced(given_session, given_session_len, retrieved_sessions, retrieved_sessions_lens, retrieved_sessions_labels)
            # outputs, output_merged_session, output_merged_future_item = model.forward_retrieval_enhanced(given_session, given_session_len, retrieved_sessions, retrieved_sessions_lens, retrieved_sessions_labels)
            logits = F.softmax(outputs, dim=1)
            recall, mrr = metric.evaluate(logits, given_session_label, k=args.topk)
            recalls.append(recall)
            mrrs.append(mrr)

    mean_recall = np.mean(recalls)
    mean_mrr = np.mean(mrrs)
    return mean_recall, mean_mrr


class Set_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        cross_entropy = F.nll_loss(input, target)

        return cross_entropy


if __name__ == '__main__':
    if args.k == 0:
        main()
    else:
        main_retrieval_enhanced()

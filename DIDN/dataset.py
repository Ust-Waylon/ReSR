# -*- coding: utf-8 -*-
"""
Created on 20 Sep, 2020

@author: Xiaokun Zhang

Reference: https://github.com/lijingsdu/sessionRec_NARM/blob/master/data_process.py
"""

import pickle
import torch
from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset
import numpy as np


def load_data(root, valid_portion=0.1, maxlen=19, sort_by_len=False):
    '''Loads the dataset
    
    root: The path to the dataset
    maxlen: the max sequence length we use in the train/valid set.
    sort_by_len: Sort by the sequence lenght for the train,
        valid and test set. This allow faster execution as it cause
        less padding per minibatch. Another mechanism must be used to
        shuffle the train set at each epoch.

    '''

    # Load the dataset
    path_train_data = root + 'train.txt'
    path_valid_data = root + 'valid.txt'
    path_test_data = root + 'test.txt'
    # with open(path_train_data, 'rb') as f1:
    #     train_set = pickle.load(f1)

    # with open(path_test_data, 'rb') as f2:
    #     test_set = pickle.load(f2)

    train_set_sessions = []
    train_set_labels = []
    with open(path_train_data, 'r') as f:
        for line in f:
            line = line.strip().split(',')
            for i in range(1, len(line)):
                train_set_sessions.append(list(map(int, line[:i])))
                train_set_labels.append(int(line[i]))
    train_set = (train_set_sessions, train_set_labels)

    valid_set_sessions = []
    valid_set_labels = []
    with open(path_valid_data, 'r') as f:
        for line in f:
            line = line.strip().split(',')
            for i in range(1, len(line)):
                valid_set_sessions.append(list(map(int, line[:i])))
                valid_set_labels.append(int(line[i]))
    valid_set = (valid_set_sessions, valid_set_labels)

    test_set_sessions = []
    test_set_labels = []
    with open(path_test_data, 'r') as f:
        for line in f:
            line = line.strip().split(',')
            for i in range(1, len(line)):
                test_set_sessions.append(list(map(int, line[:i])))
                test_set_labels.append(int(line[i]))
    test_set = (test_set_sessions, test_set_labels)
    

    if maxlen:
        new_train_set_x = []
        new_train_set_y = []
        for x, y in zip(train_set[0], train_set[1]):
            if len(x) < maxlen:
                new_train_set_x.append(x)
                new_train_set_y.append(y)
            else:
                # choose first 19 clicks in the sequence
                new_train_set_x.append(x[:maxlen])
                new_train_set_y.append(y)
        train_set = (new_train_set_x, new_train_set_y)
        del new_train_set_x, new_train_set_y

        new_valid_set_x = []
        new_valid_set_y = []
        for xx, yy in zip(valid_set[0], valid_set[1]):
            if len(xx) <= maxlen:
                new_valid_set_x.append(xx)
                new_valid_set_y.append(yy)
            else:
                new_valid_set_x.append(xx[:maxlen])
                new_valid_set_y.append(xx[maxlen])
        valid_set = (new_valid_set_x, new_valid_set_y)
        del new_valid_set_x, new_valid_set_y

        new_test_set_x = []
        new_test_set_y = []
        for xx, yy in zip(test_set[0], test_set[1]):
            if len(xx) <= maxlen:
                new_test_set_x.append(xx)
                new_test_set_y.append(yy)
            else:
                new_test_set_x.append(xx[:maxlen])
                new_test_set_y.append(xx[maxlen])
        test_set = (new_test_set_x, new_test_set_y)
        del new_test_set_x, new_test_set_y

    # # split training set into validation set
    # train_set_x, train_set_y = train_set
    # n_samples = len(train_set_x)
    # sidx = np.arange(n_samples, dtype='int32')
    # np.random.shuffle(sidx)
    # n_train = int(np.round(n_samples * (1. - valid_portion)))
    # valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    # valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    # train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    # train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    (train_set_x, train_set_y) = train_set
    (valid_set_x, valid_set_y) = valid_set
    (test_set_x, test_set_y) = test_set

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    if sort_by_len:
        sorted_index = len_argsort(test_set_x)
        test_set_x = [test_set_x[i] for i in sorted_index]
        test_set_y = [test_set_y[i] for i in sorted_index]
        sorted_index = len_argsort(valid_set_x)
        valid_set_x = [valid_set_x[i] for i in sorted_index]
        valid_set_y = [valid_set_y[i] for i in sorted_index]

    train = (train_set_x, train_set_y)
    valid = (valid_set_x, valid_set_y)
    test = (test_set_x, test_set_y)

    return train, valid, test


class RecSysDataset(Dataset):
    """define the pytorch Dataset class for yoochoose and diginetica datasets.
    """

    def __init__(self, data):
        self.data = data
        print('-' * 50)
        print('Dataset info:')
        print('Number of sessions: {}'.format(len(data[0])))
        print('-' * 50)

    def __getitem__(self, index):
        session_items = self.data[0][index]
        target_item = self.data[1][index]
        return session_items, target_item

    def __len__(self):
        return len(self.data[0])
    
def read_retrieved_sessions(file_path):
    sessions = pd.read_csv(file_path, sep='\t', header=None)

    given_sessions = sessions[0].apply(lambda x: list(map(int, x.split(',')))).values
    retrieved_sessions = []
    for i in range(1, sessions.shape[1]):
        retrieved_sessions.append(sessions[i].apply(lambda x: list(map(int, x.split(',')))).values)
    return given_sessions, retrieved_sessions

def load_retrieved_data(folder_path, valid_portion=0.1, maxlen=19, k=3):
    retrieve_folder_name = "retrieved_sessions" + f"_{k}"
    # check if the folder exists
    if not (Path(folder_path) / retrieve_folder_name).exists():
        raise FileNotFoundError(f"Folder {retrieve_folder_name} does not exist in {folder_path}")
    
    train_data_path = Path(folder_path) / retrieve_folder_name / "train.txt"
    valid_data_path = Path(folder_path) / retrieve_folder_name / "valid.txt"
    if not valid_data_path.exists():
        valid_data_path = Path(folder_path) / retrieve_folder_name / "test.txt"
    test_data_path = Path(folder_path) / retrieve_folder_name / "test.txt"

    train_given_sessions, train_retrieved_sessions = read_retrieved_sessions(train_data_path)
    valid_given_sessions, valid_retrieved_sessions = read_retrieved_sessions(valid_data_path)
    test_given_sessions, test_retrieved_sessions = read_retrieved_sessions(test_data_path)

    train = (train_given_sessions, train_retrieved_sessions)
    valid = (valid_given_sessions, valid_retrieved_sessions)
    test = (test_given_sessions, test_retrieved_sessions)

    return train, valid, test

class RecSysRetrievedDataset(Dataset):
    def __init__(self, data):
        self.data = data
        print('-' * 50)
        print('Dataset info:')
        print('Number of sessions: {}'.format(len(data[0])))
        print('-' * 50)

    def __getitem__(self, index):
        given_session = self.data[0][index]
        retrieved_sessions = [self.data[1][i][index] for i in range(len(self.data[1]))]
        return given_session, retrieved_sessions

    def __len__(self):
        return len(self.data[0])
    
    
def collate_fn_for_retrieved_dataset(data):
    given_session, retrieved_sessions = zip(*data)
    
    # get the last item in the sessions as the label
    given_session_label = [sess[-1] for sess in given_session]
    given_session_without_label = [sess[:-1] for sess in given_session]
    retrieved_sessions_labels = []
    retrieved_sessions_without_label = []
    for i in range(len(retrieved_sessions)):
        retrieved_sessions_labels.append([sess[-1] for sess in retrieved_sessions[i]])
        retrieved_sessions_without_label.append([sess[:-1] for sess in retrieved_sessions[i]])
    given_session_label = torch.tensor(given_session_label).long()
    retrieved_sessions_labels = [torch.tensor(sess).long() for sess in retrieved_sessions_labels]
    retrieved_sessions_labels = torch.stack(retrieved_sessions_labels, dim=1)

    # get the length of each session
    given_session_len = [len(sess) for sess in given_session_without_label]
    retrieved_sessions_lens = []
    error_i, error_j = -1, -1
    for i in range(len(retrieved_sessions_without_label)):
        # retrieved_sessions_lens.append([len(sess) for sess in retrieved_sessions_without_label[i]])
        len_list = []
        for j in range(len(retrieved_sessions_without_label[i])):
            sess = retrieved_sessions_without_label[i][j]
            if len(sess) == 0:
                error_i, error_j = i, j
                print(f"Error: {error_i}, {error_j}")
                print(given_session[error_i])
                print(retrieved_sessions[error_i][0], retrieved_sessions[error_i][1], retrieved_sessions[error_i][2])
                print(retrieved_sessions_without_label[error_i][0], retrieved_sessions_without_label[error_i][1], retrieved_sessions_without_label[error_i][2])
                raise ValueError("Error in collate_fn_for_retrieved_dataset")
            len_list.append(len(sess))
        retrieved_sessions_lens.append(len_list)        
        

    # pad the sessions to the same length
    max_session_len = 19
    padded_given_session = torch.zeros(len(given_session_without_label), max_session_len).long()
    for i, sess in enumerate(given_session_without_label):
        padded_given_session[i, :len(sess)] = torch.LongTensor(sess)
    padded_retrieved_sessions = []
    for i in range(len(retrieved_sessions_without_label)):
        padded_sess = torch.zeros(len(retrieved_sessions_without_label[i]), max_session_len).long()
        for j, sess in enumerate(retrieved_sessions_without_label[i]):
            padded_sess[j, :len(sess)] = torch.LongTensor(sess)
        padded_retrieved_sessions.append(padded_sess)
    padded_retrieved_sessions = torch.stack(padded_retrieved_sessions, dim=0)

    return padded_given_session, given_session_label, given_session_len, padded_retrieved_sessions, retrieved_sessions_labels, retrieved_sessions_lens

    
    
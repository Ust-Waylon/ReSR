import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--dataset-dir', 
    default='../datasets/sample', 
    help='the dataset directory'
)
parser.add_argument(
    '--embedding-dim', 
    type=int, 
    default=64, 
    help='the embedding size'
)
parser.add_argument(
    '--num-layers', 
    type=int, 
    default=3, 
    help='the number of layers'
)
parser.add_argument(
    '--feat-drop', 
    type=float, 
    default=0.2, 
    help='the dropout ratio for features'
)
parser.add_argument(
    '--lr', 
    type=float, 
    default=1e-3, 
    help='the learning rate'
)
parser.add_argument(
    '--batch-size', 
    type=int, 
    default=512, 
    help='the batch size for training'
)
parser.add_argument(
    '--epochs', 
    type=int, 
    default=50, 
    help='the number of training epochs'
)
parser.add_argument(
    '--weight-decay',
    type=float,
    default=1e-4,
    help='the parameter for L2 regularization',
)
parser.add_argument(
    '--Ks',
    default='10,20',
    help='the values of K in evaluation metrics, separated by commas',
)
parser.add_argument(
    '--patience',
    type=int,
    default=2,
    help='the number of epochs that the performance does not improves after which the training stops',
)
parser.add_argument(
    '--num-workers',
    type=int,
    default=4,
    help='the number of processes to load the input graphs',
)
parser.add_argument(
    '--valid-split',
    type=float,
    default=None,
    help='the fraction for the validation set',
)
parser.add_argument(
    '--log-interval',
    type=int,
    default=100,
    help='print the loss after this number of iterations',
)
parser.add_argument(
    '--max-session-len',
    type=int,
    default=19,
    help='maximum length of a session',
)
args = parser.parse_args()
print(args)

import sys
sys.path.append('..')
from pathlib import Path
import torch as th
# th.autograd.set_detect_anomaly(True)
from torch.utils.data import DataLoader
from data.dataset import read_dataset, AugmentedDataset
from pretrain.pretrain_runner import PretrainRunner, evaluate, print_results
from pretrain.model import SelfAttentiveSessionEncoder

dataset_dir = Path(args.dataset_dir)
args.Ks = [int(K) for K in args.Ks.split(',')]
print(f'reading dataset from {dataset_dir}')
train_sessions, valid_sessions, test_sessions, num_items = read_dataset(dataset_dir)

if args.valid_split is not None:
    num_valid = int(len(train_sessions) * args.valid_split)
    test_sessions = train_sessions[-num_valid:]
    train_sessions = train_sessions[:-num_valid]

train_set = AugmentedDataset(train_sessions)
valid_set = AugmentedDataset(valid_sessions)
test_set = AugmentedDataset(test_sessions)

def collate_fn(samples):
    sessions, labels = zip(*samples)
    sessions = th.LongTensor(sessions)
    labels = th.LongTensor(labels)
    return sessions, labels

train_loader = DataLoader(
    train_set,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=args.num_workers,
    collate_fn=collate_fn,
    pin_memory=True,
)

valid_loader = DataLoader(
    valid_set,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    collate_fn=collate_fn,
    pin_memory=True,
)

test_loader = DataLoader(
    test_set,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    collate_fn=collate_fn,
    pin_memory=True,
)

model = SelfAttentiveSessionEncoder(num_items=num_items, 
                                    n_layers=args.num_layers,
                                    hidden_size=args.embedding_dim,
                                    hidden_dropout_prob=args.feat_drop,
                                    max_session_length=args.max_session_len)
device = th.device('cuda' if th.cuda.is_available() else 'cpu')
model.to(device)

runner = PretrainRunner(
    model,
    train_loader,
    valid_loader,
    device,
    lr=args.lr,
    weight_decay=args.weight_decay,
    patience=args.patience,
    Ks=args.Ks,
)

print('start training')
runner.train(args.epochs, args.log_interval)

print('test results')
model.load_state_dict(th.load("best_model.pth"))
test_results = evaluate(model, test_loader, device, Ks=args.Ks)
print_results(test_results)
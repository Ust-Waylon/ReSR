import itertools
import numpy as np
import pandas as pd


def create_index(sessions):
    lens = np.fromiter(map(len, sessions), dtype=int)
    session_idx = np.repeat(np.arange(len(sessions)), lens - 1)
    label_idx = map(lambda l: range(l - 1, 0, -1), lens)
    label_idx = itertools.chain.from_iterable(label_idx)
    label_idx = np.fromiter(label_idx, dtype=int)
    idx = np.column_stack((session_idx, label_idx))
    return idx


def read_sessions(filepath):
    sessions = pd.read_csv(filepath, sep='\t', header=None).squeeze('columns')
    sessions = sessions.apply(lambda x: list(map(int, x.split(',')))).values
    return sessions


def read_dataset(dataset_dir):
    train_sessions = read_sessions(dataset_dir / 'train.txt')
    if (dataset_dir / 'valid.txt').exists():
        valid_sessions = read_sessions(dataset_dir / 'valid.txt')
    else:
        valid_sessions = read_sessions(dataset_dir / 'test.txt')
    test_sessions = read_sessions(dataset_dir / 'test.txt')
    
    with open(dataset_dir / 'num_items.txt', 'r') as f:
        num_items = int(f.readline())
    return train_sessions, valid_sessions, test_sessions, num_items


class AugmentedDataset:
    def __init__(self, sessions, sort_by_length=False):
        self.sessions = sessions
        index = create_index(sessions)  # columns: sessionId, labelIndex
        if sort_by_length:
            # sort by labelIndex in descending order
            ind = np.argsort(index[:, 1])[::-1]
            index = index[ind]
        self.index = index

    def __getitem__(self, idx):
        max_session_len = 19
        sid, lidx = self.index[idx]
        seq = self.sessions[sid][:lidx]
        # if seq is longer than max_session_len, take the last max_session_len items
        if len(seq) > max_session_len:
            seq = seq[-max_session_len:]
        # if seq is shorter than max_session_len, pad with zeros
        elif len(seq) < max_session_len:
            seq = [0] * (max_session_len - len(seq)) + seq
        label = self.sessions[sid][lidx]
        return seq, label

    def __len__(self):
        return len(self.index)    

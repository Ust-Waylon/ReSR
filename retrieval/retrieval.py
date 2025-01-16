import torch
import numpy as np
import sys
import tqdm
sys.path.append('..')
from pretrain.model import SelfAttentiveSessionEncoder
from pathlib import Path
from torch.utils.data import DataLoader
from data.dataset import read_dataset, AugmentedDataset
from annoy import AnnoyIndex
import random

def retrieve_top_k_similar_sessions(train_embeddings, idx_session_map, dataloader, k, loader_type, folder):
    with open(f"{folder}/{loader_type}.txt", "w") as f2:
        session_idx = 0
        for batch in tqdm.tqdm(dataloader):
            inputs, labels = batch
            inputs_gpu = torch.stack([torch.LongTensor(x) for x in inputs]).to(device)
            # labels_gpu = labels.to(device)
            with torch.no_grad():
                outputs = model(inputs_gpu)
                session_embeddings = outputs[:, -1, :]

                normalized_session_embeddings = session_embeddings / torch.norm(session_embeddings, dim=1, keepdim=True)
                cosine_similarity = torch.matmul(normalized_session_embeddings, train_embeddings.transpose(0, 1))

                topk_indices = (torch.topk(cosine_similarity, k+1, dim=1).indices).cpu().numpy().tolist()
            
            # get the top-k most similar session
            topk_sessions = []
            for i in range(len(topk_indices)):

                topk_session = []
                input = inputs[i].cpu().numpy()
                label = labels[i].cpu().numpy()

                # remove zero padding starting from the beginning
                first_index = 0
                while  first_index < len(input) and input[first_index] == 0:
                    first_index += 1
                input = input[first_index:]

                input_complete_session = np.concatenate((input, [label]))
                topk_session.append(input_complete_session)
                
                for j in range(len(topk_indices[i])):
                    if len(topk_session) == k+1:
                        break
                    elif loader_type == "train" and topk_indices[i][j] == session_idx:
                        continue
                        
                    input_session, label = idx_session_map[topk_indices[i][j]]

                    # remove zero padding starting from the beginning
                    first_index = 0
                    while  first_index < len(input_session) and input_session[first_index] == 0:
                        first_index += 1
                    input_session = input_session[first_index:]

                    complete_session = np.concatenate((input_session, [label]))

                    topk_session.append(complete_session)
                        
                if len(topk_session) != k+1:
                    continue
                topk_sessions.append(topk_session)

                session_idx += 1
            
                    
            # save the given session and the top-k most similar sessions pairs as
            # <given_session> \t <session1> \t <session2> \t <session3>
            for topk_session in topk_sessions:
                f2.write("\t".join([",".join(map(str, session)) for session in topk_session]) + "\n")

if __name__ == '__main__':
    # load model
    dataset_name = "diginetica"
    dataset_path = "/rwproject/kdd-db/students/wtanae/research/retrieval_based/ReSR/datasets/diginetica"
    model_path = "/rwproject/kdd-db/students/wtanae/research/retrieval_based/ReSR/pretrain/best_model_diginetica.pth"
    # dataset_name = "yoochoose1_64"
    # dataset_path = "/rwproject/kdd-db/students/wtanae/research/retrieval_based/ReSR/datasets/yoochoose1_64"
    # model_path = "/rwproject/kdd-db/students/wtanae/research/retrieval_based/ReSR/pretrain/best_model_yoochoose1_64.pth"

    with open(dataset_path + "/num_items.txt", "r") as f:
        n_items = int(f.readline().strip())

    n_layers = 3
    hidden_size = 64
    hidden_dropout_prob = 0.2
    max_session_len = 19

    ks = [8]

    model = SelfAttentiveSessionEncoder(num_items=n_items, n_layers=n_layers, hidden_size=hidden_size, hidden_dropout_prob=hidden_dropout_prob, max_session_length=max_session_len)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)


    # dataloader
    train_sessions, valid_sessions, test_sessions, num_items = read_dataset(Path(dataset_path))
    print(f"dataset name: {dataset_name}, #items: {n_items}")

    train_set = AugmentedDataset(train_sessions)
    valid_set = AugmentedDataset(valid_sessions)
    test_set = AugmentedDataset(test_sessions)

    def collate_fn(samples):
        sessions, labels = zip(*samples)
        sessions = torch.LongTensor(sessions)
        labels = torch.LongTensor(labels)
        return sessions, labels

    train_loader = DataLoader(
        train_set,
        batch_size=1024,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    valid_loader = DataLoader(
        valid_set,
        batch_size=1024,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=1024,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )


    # build index
    session_idx = 0
    idx_session_map = {}
    train_embeddings = []
    print("collecting session embeddings from train set")
    for batch in tqdm.tqdm(train_loader):
        sessions, labels = batch
        sessions = sessions.to(device)
        with torch.no_grad():
            session_embedding = model(sessions)
            session_embedding = session_embedding[:, -1, :]
        for i, embedding in enumerate(session_embedding):
            idx_session_map[session_idx] = (sessions[i].cpu().numpy(), labels[i].cpu().numpy())
            train_embeddings.append(embedding.cpu().numpy())
            session_idx += 1

    train_embeddings = torch.stack([torch.tensor(embedding) for embedding in train_embeddings])
    train_embeddings = train_embeddings.to(device)
    # normalize embeddings
    train_embeddings = train_embeddings / torch.norm(train_embeddings, dim=1, keepdim=True)


    for k in ks:
        folder = dataset_path + "/retrieved_sessions" + "_" + str(k)
        Path(folder).mkdir(parents=True, exist_ok=True)
        # retrieve top-k similar sessions for train, valid, and test sets
        print(f"retrieving top-{k} similar sessions on train set")
        retrieve_top_k_similar_sessions(train_embeddings, idx_session_map, train_loader, k, "train", folder)
        print(f"retrieving top-{k} similar sessions on valid set")
        retrieve_top_k_similar_sessions(train_embeddings, idx_session_map, valid_loader, k, "valid", folder)
        print(f"retrieving top-{k} similar sessions on test set")
        retrieve_top_k_similar_sessions(train_embeddings, idx_session_map, test_loader, k, "test", folder)
    
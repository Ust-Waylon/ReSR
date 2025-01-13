import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math
import numpy as np
import pandas as pd
from scipy import stats
import sys
sys.path.append('..')
from module import AttentionBasedMerger, FutureItemMerger, HistoryFutureFusionModule, RetrievalClassifier

def negative_sampling_loss(scores, negative_sample_scores, classification_weight, labels):
    # scores: [batch_size, n_items]
    # negative_sample_scores: [batch_size, k]
    # classification_weight: [batch_size, k]
    # labels: [batch_size]
    # output: loss

    cross_entropy_loss = nn.CrossEntropyLoss()
    cross_entropy_loss = cross_entropy_loss(scores, labels)


    # version 3
    # # sigmoid negative sample scores
    # sigmoid_negative_sample_scores = torch.sigmoid(negative_sample_scores)
    # negative_part = torch.ones_like(sigmoid_negative_sample_scores) - sigmoid_negative_sample_scores
    # negative_part = negative_part
    # negative_part = -torch.log(negative_part + 1e-20)
    # negative_sample_loss = torch.mean(torch.sum(negative_part, dim=1))

    # version 2
    # # get the positive sample scores
    # positive_sample_scores = scores[torch.arange(scores.size(0)), labels] # [batch_size]
    # exp_positive_sample_scores = torch.exp(positive_sample_scores)
    # # softmax with weight
    # exp_negative_sample_scores = torch.exp(negative_sample_scores) * (1 - classification_weight) # [batch_size, k]
    # exp_all_scores = torch.cat([exp_positive_sample_scores.unsqueeze(1), exp_negative_sample_scores], dim=1) # [batch_size, k+1]
    # softmax_all_scores = exp_all_scores / (torch.sum(exp_all_scores, dim=1, keepdim=True) + 1e-20)
    # negative_sample_loss = -torch.log(softmax_all_scores[:, 0] + 1e-20)
    # # negative_sample_loss = mean of negative_sample_loss * k / n_items
    # negative_sample_loss = torch.mean(negative_sample_loss) * negative_sample_scores.size(1) / scores.size(1)

    # version 1
    # # softmax
    # all_scores = F.softmax(all_scores, dim=1)
    # positive_part = all_scores[:, 0] # [batch_size]
    # negative_part = all_scores[:, 1:] # [batch_size, k]
    # # negative_part <- 1 - negative_part
    # negative_part = torch.ones_like(negative_part) - negative_part # [batch_size, k]
    # all_scores = torch.cat([positive_part.unsqueeze(1), negative_part], dim=1) # [batch_size, k+1]
    # all_scores = torch.log(all_scores) # [batch_size, k+1]
    # # calculate the loss
    # positive_sample_loss = -all_scores[:, 0] # [batch_size]
    # # negative_sample_weight = 1 - classification_weight
    # negative_sample_weight = torch.ones_like(classification_weight) - classification_weight
    # negative_sample_loss = torch.sum(-all_scores[:, 1:] * negative_sample_weight, dim=1) # [batch_size]
    # negative_sample_loss = torch.mean(positive_sample_loss + negative_sample_loss)

    # output_loss = cross_entropy_loss + negative_sample_loss
    # print("cross_entropy_loss: ", cross_entropy_loss)
    # print("negative_sample_loss: ", negative_sample_loss)

    output_loss = cross_entropy_loss
    return output_loss



class DIDN_retrieval_enhanced(nn.Module):
    """Dynamic intent-aware Iterative Denoising Network Class

    Args:
        n_items(int): the number of items
        hidden_size(int): the hidden size of gru
        position_embed_dim/embedding_dim(int): the dimension of item embedding
        batch_size(int):
        n_layers(int): the number of gru layers

    """

    def __init__(self, n_items, hidden_size, embedding_dim, batch_size, max_len, position_embed_dim, alpha1,  alpha2, alpha3, pos_num, neighbor_num, k, n_layers=1):
        super(DIDN_retrieval_enhanced, self).__init__()
        self.n_items = n_items
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        self.position_embed_dim = position_embed_dim
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3
        self.pos_num = pos_num
        self.neighbor_num = neighbor_num
        self.k = k
        self.emb = nn.Embedding(self.n_items, self.embedding_dim, padding_idx=0)

        self.position_emb = nn.Embedding(self.pos_num, self.position_embed_dim, padding_idx=0)
        self.emb_dropout = nn.Dropout(0.3)
        self.position_dropout = nn.Dropout(0.3)
        self.dropout15 = nn.Dropout(0.15)
        self.dropout30 = nn.Dropout(0.30)
        self.dropout40 = nn.Dropout(0.40)
        self.dropout70 = nn.Dropout(0.70)

        # batchnormalization
        self.bn = torch.nn.BatchNorm1d(max_len, affine=False)
        self.bn1 = torch.nn.BatchNorm1d(embedding_dim, affine=False)
        self.final2std_cur = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)
        self.final2std_last = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)
        self.v_f2std = nn.Linear(self.embedding_dim, 1, bias=True)

        self.final2std2_std = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)
        self.final2std2_cur = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)
        self.v_f2std2 = nn.Linear(self.embedding_dim, 1, bias=True)

        self.final2std3_std = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)
        self.final2std3_cur = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)
        self.v_f2std3 = nn.Linear(self.embedding_dim, 1, bias=True)

        self.gru = nn.GRU(self.embedding_dim, self.hidden_size, self.n_layers)

        self.user2item_dim = nn.Linear(self.hidden_size, self.embedding_dim, bias=True)
        self.pos2item_dim = nn.Linear(self.position_embed_dim, self.embedding_dim, bias=True)
        # Dual gating mechanism
        self.w_u_z = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)
        self.w_u_r = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)
        self.w_u = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)

        self.u_u_z = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)
        self.u_u_r = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)
        self.u_u = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)

        self.w_p_f = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)
        self.w_p_i = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)
        self.w_p_c = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)

        self.u_p_f = nn.Linear(self.position_embed_dim, self.embedding_dim, bias=True)
        self.u_p_i = nn.Linear(self.position_embed_dim, self.embedding_dim, bias=True)
        self.u_p_c = nn.Linear(self.position_embed_dim, self.embedding_dim, bias=True)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.v_n2se = nn.Linear(self.embedding_dim, 1, bias=True)


        self.neighbor_v_t = nn.Linear(self.embedding_dim, 1, bias=True)


        self.merge_n_c = nn.Linear(self.embedding_dim*4, self.embedding_dim, bias=True)
        # attention to get stand
        self.v1_w = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)
        self.v1 = nn.Linear(self.embedding_dim, 1, bias=True)

        self.b = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.sf = nn.Softmax()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Retrival enhancing modules
        self.attention_based_merger = AttentionBasedMerger(n_head=1, hidden_size=hidden_size, hidden_dropout_prob=0.3, layer_norm_eps=1e-12)
        self.history_future_fusion_module = HistoryFutureFusionModule(n_layers=5, n_head=1, hidden_size=hidden_size, hidden_dropout_prob=0.3)
        # self.retrieval_classifier = RetrievalClassifier(hidden_size=hidden_size)
        # self.future_item_merger = FutureItemMerger(hidden_size=hidden_size, hidden_dropout_prob=0.3)

    def get_session_embedding(self, seq, lengths):
        # get original item embedding
        embs_origin = self.emb_dropout(self.emb(seq))
        lengths_ori = lengths
        embs_origin = embs_origin.permute(1, 0, 2)

        # get position embedding  pos+length
        item_position = torch.tensor(range(1, seq.size()[0]+1), device=self.device)
        item_position = item_position.unsqueeze(1).expand_as(seq).permute(1, 0)
        len_d = torch.Tensor(lengths_ori).unsqueeze(1).expand_as(seq.permute(1, 0))*5 - 1
        len_d = len_d.type(torch.cuda.LongTensor)
        mask_position = torch.where(seq.permute(1, 0) > 0, torch.tensor([1], device=self.device),
                           torch.tensor([0], device=self.device))
        item_position = item_position*mask_position*len_d

        mask = torch.where(seq.permute(1, 0) > 0, torch.tensor([1.], device=self.device),
                           torch.tensor([0.], device=self.device))

        hidden = self.init_hidden(seq.size(1))
        embs_padded = pack_padded_sequence(embs_origin.permute(1, 0, 2), lengths, enforce_sorted=False)
        gru_out, hidden = self.gru(embs_padded, hidden)
        # get user embeding gru_out (19,512,100)
        gru_out, lengths = pad_packed_sequence(gru_out)
        ht = hidden[-1]

        pos_embs = self.position_dropout(self.position_emb(item_position))
        pos_embs = self.dropout30(self.pos2item_dim(pos_embs))
        user_embs = self.dropout30(self.user2item_dim(ht))
        user_emb_expand = user_embs.unsqueeze(1).expand_as(embs_origin)

        user_z = torch.sigmoid(self.w_u_z(user_emb_expand)+self.u_u_z(embs_origin))
        user_r = torch.sigmoid(self.w_u_r(user_emb_expand) + self.u_u_r(embs_origin))
        uw_emb_h = self.tanh(self.w_u(user_emb_expand) + self.u_u(user_r*embs_origin))
        uw_emb = (1-user_z)*embs_origin + user_z * uw_emb_h

        pos_f = torch.sigmoid(self.w_p_f(uw_emb) + self.u_p_f(pos_embs))
        pos_i = torch.sigmoid(self.w_p_i(uw_emb) + self.u_p_i(pos_embs))
        pw_emb_h = self.tanh(self.w_p_c(uw_emb) + self.u_p_c(pos_embs))

        embs_final = pos_f*embs_origin + pos_i*pw_emb_h

        # batchnormalization
        embs_final = self.bn(embs_final)

        qq_current = self.final2std_cur(embs_final.contiguous().view(-1, self.embedding_dim)).view(embs_final.size())
        v_atten_q = self.v1_w(embs_final.contiguous().view(-1, self.embedding_dim)).view(embs_final.size())
        v_atten_q = self.tanh(v_atten_q)
        v_atten_w = self.v1(v_atten_q).expand_as(qq_current) * qq_current
        qq_avg_masked = mask.unsqueeze(2).expand_as(qq_current) * v_atten_w
        qq_avg_masked = torch.sum(qq_avg_masked, 1)
        qq_avg_masked = mask.unsqueeze(2).expand_as(qq_current) * qq_avg_masked.unsqueeze(1).expand_as(qq_current)

        beta1 = self.v_f2std(torch.sigmoid(qq_current + qq_avg_masked ).view(-1, self.embedding_dim)).view(
            mask.size())

        beta1 = self.sf(beta1)
        beta1_v = torch.mean(beta1, 1, True)[0].expand_as(beta1)
        beta1_mask = beta1 - self.alpha1 * beta1_v

        beta1 = torch.where(beta1_mask > 0, beta1,
                           torch.tensor([0.], device=self.device))

        sess_std = torch.sum(beta1.unsqueeze(2).expand_as(embs_final) * embs_final, 1)

        sess_std = self.dropout70(sess_std)

        q22_current = self.final2std2_cur(embs_final.contiguous().view(-1, self.embedding_dim)).view(embs_final.size())

        q22_std = self.final2std2_std(sess_std)
        q22_std_expand = q22_std.unsqueeze(1).expand_as(q22_current)
        q22_std_masked = mask.unsqueeze(2).expand_as(q22_current) * q22_std_expand

        beta2 = self.v_f2std2(torch.sigmoid(q22_current + q22_std_masked ).view(-1, self.embedding_dim)).view(mask.size())

        beta2 = self.sf(beta2)
        beta2_v = torch.mean(beta2, 1, True)[0].expand_as(beta2)
        beta2_mask = beta2 - self.alpha2 * beta2_v

        beta2 = torch.where(beta2_mask > 0, beta2,
                            torch.tensor([0.], device=self.device))
        sess_std2 = torch.sum(beta2.unsqueeze(2).expand_as(embs_final) * embs_final, 1)
        sess_std2 = self.dropout30(sess_std2)

        q3_current = self.final2std3_cur(embs_final.contiguous().view(-1, self.embedding_dim)).view(embs_final.size())

        q3_std = self.final2std3_std(sess_std2)
        q3_std_expand = q3_std.unsqueeze(1).expand_as(q3_current)
        q3_std_masked = mask.unsqueeze(2).expand_as(q3_current) * q3_std_expand

        beta3 = self.v_f2std3(torch.sigmoid(q3_current + q3_std_masked ).view(-1, self.embedding_dim)).view(mask.size())

        beta3 = self.sf(beta3)
        beta3_v = torch.mean(beta3, 1, True)[0].expand_as(beta3)
        beta3_mask = beta3 - self.alpha3 * beta3_v

        beta3 = torch.where(beta3_mask > 0, beta3,
                            torch.tensor([0.], device=self.device))
        sess_std3 = torch.sum(beta3.unsqueeze(2).expand_as(embs_final) * embs_final, 1)

        sess_current = sess_std3

        # cosine similarity
        fenzi = torch.matmul(sess_current, sess_current.permute(1, 0)) #512*512
        fenmu_l = torch.sum(sess_current * sess_current + 0.000001, 1)
        fenmu_l = torch.sqrt(fenmu_l).unsqueeze(1)
        fenmu = torch.matmul(fenmu_l, fenmu_l.permute(1, 0))
        cos_sim = fenzi / fenmu #512*512
        cos_sim = nn.Softmax(dim=-1)(cos_sim)

        k_v = self.neighbor_num
        if cos_sim.size()[0] < k_v:
            k_v = cos_sim.size()[0]
        cos_topk, topk_indice = torch.topk(cos_sim, k=k_v, dim=1)
        cos_topk = nn.Softmax(dim=-1)(cos_topk)
        sess_topk = sess_current[topk_indice]

        cos_sim = cos_topk.unsqueeze(2).expand(cos_topk.size()[0], cos_topk.size()[1], self.embedding_dim)

        neighbor_sess = torch.sum(cos_sim * sess_topk, 1)
        neighbor_sess = self.dropout40(neighbor_sess)
        sess_final = torch.cat(
            [sess_current, neighbor_sess, sess_current + neighbor_sess, sess_current * neighbor_sess], 1)

        sess_final = self.dropout30(sess_final)
        sess_final = self.merge_n_c(sess_final)

        return sess_final

    def forward_retrieval_enhanced(self, seq, lengths, retrieved_seqs, retrieved_lengths, retrieved_labels, training_type=0):

        if training_type == 1:
            # only train the retrieval_classifier
            for param in self.parameters():
                param.requires_grad = False
            for param in self.retrieval_classifier.parameters():
                param.requires_grad = True
            self.attention_based_merger.dense.requires_grad = True

        elif training_type == 2:
            # train the whole model except the retrieval_classifier
            for param in self.parameters():
                param.requires_grad = True
            for param in self.retrieval_classifier.parameters():
                param.requires_grad = False

        elif training_type == 0:
            for param in self.parameters():
                param.requires_grad = True

        else:
            raise ValueError("Invalid training type")



        # get the session embedding of the given session
        seq = seq.transpose(0, 1)
        session_embedding = self.get_session_embedding(seq, lengths)
        final_embedding = session_embedding

        # # original DIDN
        # item_embs = self.emb(torch.arange(self.n_items).to(self.device))
        # item_embs = self.dropout15(item_embs)
        # item_embs = self.bn1(item_embs)
        # item_embs = self.b(item_embs).permute(1, 0)
        # scores = torch.matmul(final_embedding, item_embs)
        # return scores


        # get the session embeddings of the top-k retrieved sessions
        input_retrieved_seqs = retrieved_seqs.view(-1, retrieved_seqs.size(2)) # [batch_size*k, max_len]
        input_retrieved_seqs = input_retrieved_seqs.transpose(0, 1)
        retrieved_lengths = [length for sublist in retrieved_lengths for length in sublist]
        retrieved_session_embeddings = self.get_session_embedding(input_retrieved_seqs, retrieved_lengths) # [batch_size*k, hidden_size]
        # retrieved_session_embeddings = retrieved_session_embeddings.view(retrieved_seqs.size(0), retrieved_seqs.size(1), retrieved_session_embeddings.size(1)) # [batch_size, k, hidden_size]

        # history future fusion module
        future_item_embedding = self.emb(retrieved_labels)
        future_item_embedding = future_item_embedding.permute(1, 0, 2) # [batch_size, k, hidden_size]
        future_item_embedding = future_item_embedding.reshape(retrieved_seqs.size(0)*retrieved_seqs.size(1), future_item_embedding.size(2)) # [batch_size*k, hidden_size]
        fused_item_embedding = self.history_future_fusion_module(retrieved_session_embeddings, future_item_embedding)
        future_item_embedding = future_item_embedding.reshape(retrieved_seqs.size(0), retrieved_seqs.size(1), future_item_embedding.size(1)) # [batch_size, k, hidden_size]
        fused_item_embedding = fused_item_embedding.reshape(retrieved_seqs.size(0), retrieved_seqs.size(1), fused_item_embedding.size(1)) # [batch_size, k, hidden_size]

        # retrieval classifier
        # classification_weight = self.retrieval_classifier(session_embedding, fused_item_embedding) # [batch_size, k]
        # classification_weight = torch.ones(retrieved_seqs.size(0), retrieved_seqs.size(1)).to(self.device) # [batch_size, k]

        # attention based merger
        session_embedding = session_embedding.unsqueeze(1) # [batch_size, 1, hidden_size]
        final_embedding = self.attention_based_merger(session_embedding, fused_item_embedding)

        final_embedding = final_embedding + session_embedding.squeeze(1) # [batch_size, hidden_size]

        # two training types
        # if training_type == 'o':
        #     # train original DIDN
        #     final_embedding = final_embedding.detach() + session_embedding.squeeze(1)
        # elif training_type == 'r':
        #     # train retrieval enhanced modules
        #     final_embedding = final_embedding + session_embedding.squeeze(1).detach()

        item_embs = self.emb(torch.arange(self.n_items).to(self.device))
        item_embs = self.dropout15(item_embs)
        item_embs = self.bn1(item_embs)
        item_embs = self.b(item_embs).permute(1, 0)

        scores = torch.matmul(final_embedding, item_embs) # [batch_size, n_items]

        # negative sampling scores
        # negative_sample_scores = torch.matmul(future_item_embedding, final_embedding.unsqueeze(2)).squeeze(2) # [batch_size, k]

        return scores

    def forward(self, seq, lengths):
        # get original item embedding
        embs_origin = self.emb_dropout(self.emb(seq))
        lengths_ori = lengths
        embs_origin = embs_origin.permute(1, 0, 2)

        # get position embedding  pos+length
        item_position = torch.tensor(range(1, seq.size()[0]+1), device=self.device)
        item_position = item_position.unsqueeze(1).expand_as(seq).permute(1, 0)
        len_d = torch.Tensor(lengths_ori).unsqueeze(1).expand_as(seq.permute(1, 0))*5 - 1
        len_d = len_d.type(torch.cuda.LongTensor)
        mask_position = torch.where(seq.permute(1, 0) > 0, torch.tensor([1], device=self.device),
                           torch.tensor([0], device=self.device))
        item_position = item_position*mask_position*len_d

        mask = torch.where(seq.permute(1, 0) > 0, torch.tensor([1.], device=self.device),
                           torch.tensor([0.], device=self.device))

        hidden = self.init_hidden(seq.size(1))
        embs_padded = pack_padded_sequence(embs_origin.permute(1, 0, 2), lengths)
        gru_out, hidden = self.gru(embs_padded, hidden)
        # get user embeding gru_out (19,512,100)
        gru_out, lengths = pad_packed_sequence(gru_out)
        ht = hidden[-1]

        pos_embs = self.position_dropout(self.position_emb(item_position))
        pos_embs = self.dropout30(self.pos2item_dim(pos_embs))
        user_embs = self.dropout30(self.user2item_dim(ht))
        user_emb_expand = user_embs.unsqueeze(1).expand_as(embs_origin)

        user_z = torch.sigmoid(self.w_u_z(user_emb_expand)+self.u_u_z(embs_origin))
        user_r = torch.sigmoid(self.w_u_r(user_emb_expand) + self.u_u_r(embs_origin))
        uw_emb_h = self.tanh(self.w_u(user_emb_expand) + self.u_u(user_r*embs_origin))
        uw_emb = (1-user_z)*embs_origin + user_z * uw_emb_h

        pos_f = torch.sigmoid(self.w_p_f(uw_emb) + self.u_p_f(pos_embs))
        pos_i = torch.sigmoid(self.w_p_i(uw_emb) + self.u_p_i(pos_embs))
        pw_emb_h = self.tanh(self.w_p_c(uw_emb) + self.u_p_c(pos_embs))

        embs_final = pos_f*embs_origin + pos_i*pw_emb_h

        # batchnormalization
        embs_final = self.bn(embs_final)

        qq_current = self.final2std_cur(embs_final.contiguous().view(-1, self.embedding_dim)).view(embs_final.size())
        v_atten_q = self.v1_w(embs_final.contiguous().view(-1, self.embedding_dim)).view(embs_final.size())
        v_atten_q = self.tanh(v_atten_q)
        v_atten_w = self.v1(v_atten_q).expand_as(qq_current) * qq_current
        qq_avg_masked = mask.unsqueeze(2).expand_as(qq_current) * v_atten_w
        qq_avg_masked = torch.sum(qq_avg_masked, 1)
        qq_avg_masked = mask.unsqueeze(2).expand_as(qq_current) * qq_avg_masked.unsqueeze(1).expand_as(qq_current)

        beta1 = self.v_f2std(torch.sigmoid(qq_current + qq_avg_masked ).view(-1, self.embedding_dim)).view(
            mask.size())

        beta1 = self.sf(beta1)
        beta1_v = torch.mean(beta1, 1, True)[0].expand_as(beta1)
        beta1_mask = beta1 - self.alpha1 * beta1_v

        beta1 = torch.where(beta1_mask > 0, beta1,
                           torch.tensor([0.], device=self.device))

        sess_std = torch.sum(beta1.unsqueeze(2).expand_as(embs_final) * embs_final, 1)

        sess_std = self.dropout70(sess_std)

        q22_current = self.final2std2_cur(embs_final.contiguous().view(-1, self.embedding_dim)).view(embs_final.size())

        q22_std = self.final2std2_std(sess_std)
        q22_std_expand = q22_std.unsqueeze(1).expand_as(q22_current)
        q22_std_masked = mask.unsqueeze(2).expand_as(q22_current) * q22_std_expand

        beta2 = self.v_f2std2(torch.sigmoid(q22_current + q22_std_masked ).view(-1, self.embedding_dim)).view(mask.size())

        beta2 = self.sf(beta2)
        beta2_v = torch.mean(beta2, 1, True)[0].expand_as(beta2)
        beta2_mask = beta2 - self.alpha2 * beta2_v

        beta2 = torch.where(beta2_mask > 0, beta2,
                            torch.tensor([0.], device=self.device))
        sess_std2 = torch.sum(beta2.unsqueeze(2).expand_as(embs_final) * embs_final, 1)
        sess_std2 = self.dropout30(sess_std2)

        q3_current = self.final2std3_cur(embs_final.contiguous().view(-1, self.embedding_dim)).view(embs_final.size())

        q3_std = self.final2std3_std(sess_std2)
        q3_std_expand = q3_std.unsqueeze(1).expand_as(q3_current)
        q3_std_masked = mask.unsqueeze(2).expand_as(q3_current) * q3_std_expand

        beta3 = self.v_f2std3(torch.sigmoid(q3_current + q3_std_masked ).view(-1, self.embedding_dim)).view(mask.size())

        beta3 = self.sf(beta3)
        beta3_v = torch.mean(beta3, 1, True)[0].expand_as(beta3)
        beta3_mask = beta3 - self.alpha3 * beta3_v

        beta3 = torch.where(beta3_mask > 0, beta3,
                            torch.tensor([0.], device=self.device))
        sess_std3 = torch.sum(beta3.unsqueeze(2).expand_as(embs_final) * embs_final, 1)

        sess_current = sess_std3

        # cosine similarity
        fenzi = torch.matmul(sess_current, sess_current.permute(1, 0)) #512*512
        fenmu_l = torch.sum(sess_current * sess_current + 0.000001, 1)
        fenmu_l = torch.sqrt(fenmu_l).unsqueeze(1)
        fenmu = torch.matmul(fenmu_l, fenmu_l.permute(1, 0))
        cos_sim = fenzi / fenmu #512*512
        cos_sim = nn.Softmax(dim=-1)(cos_sim)

        k_v = self.neighbor_num
        if cos_sim.size()[0] < k_v:
            k_v = cos_sim.size()[0]
        cos_topk, topk_indice = torch.topk(cos_sim, k=k_v, dim=1)
        cos_topk = nn.Softmax(dim=-1)(cos_topk)
        sess_topk = sess_current[topk_indice]

        cos_sim = cos_topk.unsqueeze(2).expand(cos_topk.size()[0], cos_topk.size()[1], self.embedding_dim)

        neighbor_sess = torch.sum(cos_sim * sess_topk, 1)
        neighbor_sess = self.dropout40(neighbor_sess)
        sess_final = torch.cat(
            [sess_current, neighbor_sess, sess_current + neighbor_sess, sess_current * neighbor_sess], 1)

        sess_final = self.dropout30(sess_final)
        sess_final = self.merge_n_c(sess_final)

        item_embs = self.emb(torch.arange(self.n_items).to(self.device))
        item_embs = self.dropout15(item_embs)

        item_embs = self.bn1(item_embs)
        scores = torch.matmul(sess_final, self.b(item_embs).permute(1, 0))

        return scores

    def init_hidden(self, batch_size):
        return torch.zeros((self.n_layers, batch_size, self.hidden_size), requires_grad=True).to(self.device)

    def transpose_for_scores(self, x, attention_head_size):
        # INPUT:  x'shape = [bs, seqlen, hid_size] hid_size=128
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, attention_head_size)  # [bs, seqlen, 8, 16]
        x = x.view(*new_x_shape)  #
        return x.permute(0, 2, 1, 3)  # [bs, 8, seqlen, 16]


class DIDN(nn.Module):
    """Dynamic intent-aware Iterative Denoising Network Class

    Args:
        n_items(int): the number of items
        hidden_size(int): the hidden size of gru
        position_embed_dim/embedding_dim(int): the dimension of item embedding
        batch_size(int):
        n_layers(int): the number of gru layers

    """

    def __init__(self, n_items, hidden_size, embedding_dim, batch_size, max_len, position_embed_dim, alpha1,  alpha2, alpha3, pos_num,neighbor_num, n_layers=1):
        super(DIDN, self).__init__()
        self.n_items = n_items
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        self.position_embed_dim = position_embed_dim
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3
        self.pos_num = pos_num
        self.neighbor_num = neighbor_num
        self.emb = nn.Embedding(self.n_items, self.embedding_dim, padding_idx=0)

        self.position_emb = nn.Embedding(self.pos_num, self.position_embed_dim, padding_idx=0)
        self.emb_dropout = nn.Dropout(0.3)
        self.position_dropout = nn.Dropout(0.3)
        self.dropout15 = nn.Dropout(0.15)
        self.dropout30 = nn.Dropout(0.30)
        self.dropout40 = nn.Dropout(0.40)
        self.dropout70 = nn.Dropout(0.70)

        # batchnormalization
        self.bn = torch.nn.BatchNorm1d(max_len, affine=False)
        self.bn1 = torch.nn.BatchNorm1d(embedding_dim, affine=False)
        self.final2std_cur = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)
        self.final2std_last = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)
        self.v_f2std = nn.Linear(self.embedding_dim, 1, bias=True)

        self.final2std2_std = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)
        self.final2std2_cur = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)
        self.v_f2std2 = nn.Linear(self.embedding_dim, 1, bias=True)

        self.final2std3_std = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)
        self.final2std3_cur = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)
        self.v_f2std3 = nn.Linear(self.embedding_dim, 1, bias=True)

        self.gru = nn.GRU(self.embedding_dim, self.hidden_size, self.n_layers)

        self.user2item_dim = nn.Linear(self.hidden_size, self.embedding_dim, bias=True)
        self.pos2item_dim = nn.Linear(self.position_embed_dim, self.embedding_dim, bias=True)
        # Dual gating mechanism
        self.w_u_z = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)
        self.w_u_r = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)
        self.w_u = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)

        self.u_u_z = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)
        self.u_u_r = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)
        self.u_u = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)

        self.w_p_f = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)
        self.w_p_i = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)
        self.w_p_c = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)

        self.u_p_f = nn.Linear(self.position_embed_dim, self.embedding_dim, bias=True)
        self.u_p_i = nn.Linear(self.position_embed_dim, self.embedding_dim, bias=True)
        self.u_p_c = nn.Linear(self.position_embed_dim, self.embedding_dim, bias=True)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.v_n2se = nn.Linear(self.embedding_dim, 1, bias=True)


        self.neighbor_v_t = nn.Linear(self.embedding_dim, 1, bias=True)


        self.merge_n_c = nn.Linear(self.embedding_dim*4, self.embedding_dim, bias=True)
        # attention to get stand
        self.v1_w = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)
        self.v1 = nn.Linear(self.embedding_dim, 1, bias=True)

        self.b = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.sf = nn.Softmax()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, seq, lengths):
        # get original item embedding
        embs_origin = self.emb_dropout(self.emb(seq))
        lengths_ori = lengths
        embs_origin = embs_origin.permute(1, 0, 2)

        # get position embedding  pos+length
        item_position = torch.tensor(range(1, seq.size()[0]+1), device=self.device)
        item_position = item_position.unsqueeze(1).expand_as(seq).permute(1, 0)
        len_d = torch.Tensor(lengths_ori).unsqueeze(1).expand_as(seq.permute(1, 0))*5 - 1
        len_d = len_d.type(torch.cuda.LongTensor)
        mask_position = torch.where(seq.permute(1, 0) > 0, torch.tensor([1], device=self.device),
                           torch.tensor([0], device=self.device))
        item_position = item_position*mask_position*len_d

        mask = torch.where(seq.permute(1, 0) > 0, torch.tensor([1.], device=self.device),
                           torch.tensor([0.], device=self.device))

        hidden = self.init_hidden(seq.size(1))
        embs_padded = pack_padded_sequence(embs_origin.permute(1, 0, 2), lengths)
        gru_out, hidden = self.gru(embs_padded, hidden)
        # get user embeding gru_out (19,512,100)
        gru_out, lengths = pad_packed_sequence(gru_out)
        ht = hidden[-1]

        pos_embs = self.position_dropout(self.position_emb(item_position))
        pos_embs = self.dropout30(self.pos2item_dim(pos_embs))
        user_embs = self.dropout30(self.user2item_dim(ht))
        user_emb_expand = user_embs.unsqueeze(1).expand_as(embs_origin)

        user_z = torch.sigmoid(self.w_u_z(user_emb_expand)+self.u_u_z(embs_origin))
        user_r = torch.sigmoid(self.w_u_r(user_emb_expand) + self.u_u_r(embs_origin))
        uw_emb_h = self.tanh(self.w_u(user_emb_expand) + self.u_u(user_r*embs_origin))
        uw_emb = (1-user_z)*embs_origin + user_z * uw_emb_h

        pos_f = torch.sigmoid(self.w_p_f(uw_emb) + self.u_p_f(pos_embs))
        pos_i = torch.sigmoid(self.w_p_i(uw_emb) + self.u_p_i(pos_embs))
        pw_emb_h = self.tanh(self.w_p_c(uw_emb) + self.u_p_c(pos_embs))

        embs_final = pos_f*embs_origin + pos_i*pw_emb_h

        # batchnormalization
        embs_final = self.bn(embs_final)

        qq_current = self.final2std_cur(embs_final.contiguous().view(-1, self.embedding_dim)).view(embs_final.size())
        v_atten_q = self.v1_w(embs_final.contiguous().view(-1, self.embedding_dim)).view(embs_final.size())
        v_atten_q = self.tanh(v_atten_q)
        v_atten_w = self.v1(v_atten_q).expand_as(qq_current) * qq_current
        qq_avg_masked = mask.unsqueeze(2).expand_as(qq_current) * v_atten_w
        qq_avg_masked = torch.sum(qq_avg_masked, 1)
        qq_avg_masked = mask.unsqueeze(2).expand_as(qq_current) * qq_avg_masked.unsqueeze(1).expand_as(qq_current)

        beta1 = self.v_f2std(torch.sigmoid(qq_current + qq_avg_masked ).view(-1, self.embedding_dim)).view(
            mask.size())

        beta1 = self.sf(beta1)
        beta1_v = torch.mean(beta1, 1, True)[0].expand_as(beta1)
        beta1_mask = beta1 - self.alpha1 * beta1_v

        beta1 = torch.where(beta1_mask > 0, beta1,
                           torch.tensor([0.], device=self.device))

        sess_std = torch.sum(beta1.unsqueeze(2).expand_as(embs_final) * embs_final, 1)

        sess_std = self.dropout70(sess_std)

        q22_current = self.final2std2_cur(embs_final.contiguous().view(-1, self.embedding_dim)).view(embs_final.size())

        q22_std = self.final2std2_std(sess_std)
        q22_std_expand = q22_std.unsqueeze(1).expand_as(q22_current)
        q22_std_masked = mask.unsqueeze(2).expand_as(q22_current) * q22_std_expand

        beta2 = self.v_f2std2(torch.sigmoid(q22_current + q22_std_masked ).view(-1, self.embedding_dim)).view(mask.size())

        beta2 = self.sf(beta2)
        beta2_v = torch.mean(beta2, 1, True)[0].expand_as(beta2)
        beta2_mask = beta2 - self.alpha2 * beta2_v

        beta2 = torch.where(beta2_mask > 0, beta2,
                            torch.tensor([0.], device=self.device))
        sess_std2 = torch.sum(beta2.unsqueeze(2).expand_as(embs_final) * embs_final, 1)
        sess_std2 = self.dropout30(sess_std2)

        q3_current = self.final2std3_cur(embs_final.contiguous().view(-1, self.embedding_dim)).view(embs_final.size())

        q3_std = self.final2std3_std(sess_std2)
        q3_std_expand = q3_std.unsqueeze(1).expand_as(q3_current)
        q3_std_masked = mask.unsqueeze(2).expand_as(q3_current) * q3_std_expand

        beta3 = self.v_f2std3(torch.sigmoid(q3_current + q3_std_masked ).view(-1, self.embedding_dim)).view(mask.size())

        beta3 = self.sf(beta3)
        beta3_v = torch.mean(beta3, 1, True)[0].expand_as(beta3)
        beta3_mask = beta3 - self.alpha3 * beta3_v

        beta3 = torch.where(beta3_mask > 0, beta3,
                            torch.tensor([0.], device=self.device))
        sess_std3 = torch.sum(beta3.unsqueeze(2).expand_as(embs_final) * embs_final, 1)

        sess_current = sess_std3

        # cosine similarity
        fenzi = torch.matmul(sess_current, sess_current.permute(1, 0)) #512*512
        fenmu_l = torch.sum(sess_current * sess_current + 0.000001, 1)
        fenmu_l = torch.sqrt(fenmu_l).unsqueeze(1)
        fenmu = torch.matmul(fenmu_l, fenmu_l.permute(1, 0))
        cos_sim = fenzi / fenmu #512*512
        cos_sim = nn.Softmax(dim=-1)(cos_sim)

        k_v = self.neighbor_num
        if cos_sim.size()[0] < k_v:
            k_v = cos_sim.size()[0]
        cos_topk, topk_indice = torch.topk(cos_sim, k=k_v, dim=1)
        cos_topk = nn.Softmax(dim=-1)(cos_topk)
        sess_topk = sess_current[topk_indice]

        cos_sim = cos_topk.unsqueeze(2).expand(cos_topk.size()[0], cos_topk.size()[1], self.embedding_dim)

        neighbor_sess = torch.sum(cos_sim * sess_topk, 1)
        neighbor_sess = self.dropout40(neighbor_sess)
        sess_final = torch.cat(
            [sess_current, neighbor_sess, sess_current + neighbor_sess, sess_current * neighbor_sess], 1)

        sess_final = self.dropout30(sess_final)
        sess_final = self.merge_n_c(sess_final)

        item_embs = self.emb(torch.arange(self.n_items).to(self.device))
        item_embs = self.dropout15(item_embs)

        item_embs = self.bn1(item_embs)
        scores = torch.matmul(sess_final, self.b(item_embs).permute(1, 0))

        return scores

    def init_hidden(self, batch_size):
        return torch.zeros((self.n_layers, batch_size, self.hidden_size), requires_grad=True).to(self.device)

    def transpose_for_scores(self, x, attention_head_size):
        # INPUT:  x'shape = [bs, seqlen, hid_size] hid_size=128
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, attention_head_size)  # [bs, seqlen, 8, 16]
        x = x.view(*new_x_shape)  #
        return x.permute(0, 2, 1, 3)  # [bs, 8, seqlen, 16]

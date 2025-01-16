import torch
import torch.nn as nn
import numpy as np

class AttentionBasedMerger(nn.Module):
    def __init__(
        self,
        n_head,
        hidden_size,
        hidden_dropout_prob,
        layer_norm_eps
    ):
        super(AttentionBasedMerger, self).__init__()
        if hidden_size % n_head != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_head)
            )
        self.num_attention_heads = n_head
        self.attention_head_size = int(hidden_size / n_head)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = np.sqrt(self.attention_head_size)
        self.hidden_size = hidden_size

        self.softmax = nn.Softmax(dim=-1)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        # self.out_dropout = nn.Dropout(hidden_dropout_prob)

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dense = nn.Linear(hidden_size, hidden_size)

        self.tau = 1

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x
    
    def forward(self, input_tensor, retrieval_tensor):
        # input_tensor is the session embedding of the given session, [batch_size, 1, hidden_size]
        # retrieval_tensor is the session embeddings of the top-k retrieved sessions, [batch_size, k, hidden_size]

        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(retrieval_tensor)
        mixed_value_layer = self.value(retrieval_tensor)

        # [batch_size, 1, hidden_size] -> [batch_size, num_attention_heads, 1, attention_head_size]
        query_layer = self.transpose_for_scores(mixed_query_layer).permute(0, 2, 1, 3)
        # [batch_size, k, hidden_size] -> [batch_size, num_attention_heads, attention_head_size, k]
        key_layer = self.transpose_for_scores(mixed_key_layer).permute(0, 2, 3, 1)
        value_layer = self.transpose_for_scores(mixed_value_layer).permute(0, 2, 3, 1)

        # # Take the dot product between "query" and "key" to get the raw attention scores.
        # attention_scores = torch.matmul(query_layer, key_layer) # [batch_size, num_attention_heads, 1, k]
        # attention_scores = attention_scores / self.sqrt_attention_head_size
        # attention_probs = self.softmax(attention_scores)
        # attention_probs = attention_probs * classification_weight.unsqueeze(1).unsqueeze(1)
        # attention_probs = self.out_dropout(attention_probs)

        norm_query_layer = query_layer / torch.norm(query_layer, dim=-1, keepdim=True)
        norm_key_layer = key_layer / torch.norm(key_layer, dim=-2, keepdim=True)
        attention_scores = torch.matmul(norm_query_layer, norm_key_layer) # [batch_size, num_attention_heads, 1, k]
        attention_probs = (attention_scores + 1) / 2

        # Gumbel-softmax
        attention_probs_pn = torch.stack([attention_probs, 1-attention_probs], dim=-1) # [batch_size, num_attention_heads, 1, k, 2]
        logit = torch.log(attention_probs_pn + 1e-20)
        logit = torch.nn.functional.gumbel_softmax(logit, tau=self.tau, hard=False, dim=-1)
        self.tau = max(self.tau * 0.995, 0.1)
        attention_probs = logit[:, :, :, :, 0]

        # print(attention_probs)


        # weighted sum of the values
        context_layer = torch.matmul(attention_probs, value_layer.transpose(-1, -2)) # [batch_size, num_attention_heads, 1, attention_head_size]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous() # [batch_size, 1, num_attention_heads, attention_head_size]
        context_layer = context_layer.view(context_layer.size(0), 1, self.all_head_size) # [batch_size, 1, hidden_size]
        context_layer = context_layer.squeeze(1) # [batch_size, hidden_size]      
        # context_layer = self.out_dropout(context_layer)
        
        output = self.dense(context_layer) # [batch_size, hidden_size]

        # add residual connection and layer normalization
        # output = self.LayerNorm(output + input_tensor.squeeze(1))

        return output
    
class HistoryFutureFusionModule(nn.Module):
    def __init__(
        self,
        n_layers,
        n_head,
        hidden_size,
        hidden_dropout_prob
    ):
        super(HistoryFutureFusionModule, self).__init__()
        self.n_layers = n_layers
        self.n_head = n_head
        self.hidden_size = hidden_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.layer_norm = nn.LayerNorm(hidden_size)

        self.position_embedding = nn.Embedding(2, hidden_size) # 0 for history, 1 for future
        for i in range(n_layers):
            setattr(self, f"attention_{i}", torch.nn.MultiheadAttention(hidden_size, n_head, hidden_dropout_prob, batch_first=True))
        
    def forward(self, session_embbeding, item_embedding):
        # sequence: session_embbeding, item_embedding
        # session_embbeding: [batch_size, hidden_size]
        # item_embedding: [batch_size, hidden_size]
        # output: [batch_size, hidden_size]
        position_ids = torch.tensor([0, 1], device=session_embbeding.device)
        position_embedding = self.position_embedding(position_ids) # [2, hidden_size]
        input_embedding = torch.stack([session_embbeding, item_embedding], dim=1)
        input_embedding = input_embedding + position_embedding # [batch_size, 2, hidden_size]
        for i in range(self.n_layers):
            attention_layer = getattr(self, f"attention_{i}")

            input_embedding, _ = attention_layer(input_embedding, input_embedding, input_embedding)

            # # residual connection
            # attention_output, _ = attention_layer(input_embedding, input_embedding, input_embedding)
            # input_embedding = input_embedding + attention_output
            # # layer normalization
            # input_embedding = self.layer_norm(input_embedding)
        output = input_embedding[:, 1, :]
        return output    

class RetrievalClassifier(nn.Module):
    def __init__(
        self,
        hidden_size
    ):
        super(RetrievalClassifier, self).__init__()
        self.dense_given = nn.Linear(hidden_size, hidden_size)
        self.dense_fused = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=-1)

        self.tau = 1

    def forward(self, input_tensor, retrieval_tensor):
        # input_tensor is the session embedding of the given session
        # retrieval_tensor is the session embeddings of the top-k retrieved sessions

        output_given = self.dense_given(input_tensor) # [batch_size, hidden_size]
        output_given = output_given / torch.norm(output_given, dim=-1, keepdim=True)
        output_fused = self.dense_fused(retrieval_tensor) # [batch_size, k, hidden_size]
        output_fused = output_fused / torch.norm(output_fused, dim=-1, keepdim=True)
        similarity = torch.matmul(output_given.unsqueeze(1), output_fused.transpose(-1, -2)) # [batch_size, 1, k]
        similarity = similarity.squeeze(1) # [batch_size, k]
        # cosine similarity to probability
        similarity = (similarity + 1) / 2

        # classification probability
        probability = torch.stack([similarity, 1-similarity], dim=-1) # [batch_size, k, 2]
        logit = torch.log(probability + 1e-20)

        # Gumple-softmax
        # # for each class, sample from Gumbel(0, 1)
        # # Gumbel(0, 1) = -log(-log(U)), U ~ Uniform(0, 1)
        # U = torch.rand_like(probability)
        # gumbel = -torch.log(-torch.log(U + 1e-20) + 1e-20)
        # # add the gumbel noise to the logit
        # logit = logit + gumbel
        # # softmax
        # temperature = 1
        # logit = self.softmax(logit / temperature) # [batch_size, k, 2]

        # logit = torch.nn.functional.gumbel_softmax(logit, tau=1, hard=True, dim=-1)
        logit = torch.nn.functional.gumbel_softmax(logit, tau=self.tau, hard=False, dim=-1)
        self.tau = max(self.tau * 0.995, 0.1)

        output = logit[:, :, 0] # [batch_size, k]

        # output_mean = torch.mean(torch.mean(output, dim=-1), dim=-1)
        # print(output_mean)

        return output
    

class FutureItemMerger(nn.Module):
    def __init__(
        self,
        hidden_size,
        hidden_dropout_prob
    ):
        super(FutureItemMerger, self).__init__()
        self.hidden_size = hidden_size
        self.out_dropout = nn.Dropout(hidden_dropout_prob)
        self.dense = nn.Linear(hidden_size, hidden_size)

    def forward(self, input_tensor, retrieval_tensor, future_item_embedding):
        # input_tensor is the session embedding of the given session
        # retrieval_tensor is the session embeddings of the top-k retrieved sessions
        # future_item_embedding is the embeddings of the "future" items in the top-k retrieved sessions

        # calculate tht similarity between the input session and the retrieved sessions
        normalized_input = input_tensor / torch.norm(input_tensor, dim=-1, keepdim=True)
        normalized_retrieval = retrieval_tensor / torch.norm(retrieval_tensor, dim=-1, keepdim=True)
        similarity = torch.matmul(normalized_input, normalized_retrieval.transpose(-1, -2))

        # calculate the weighted sum of the future item embeddings
        weighted_sum = torch.matmul(similarity, future_item_embedding)
        weighted_sum = weighted_sum.view(weighted_sum.size(0), weighted_sum.size(-1))
        weighted_sum = self.out_dropout(weighted_sum)

        output = self.dense(weighted_sum)

        return output
        
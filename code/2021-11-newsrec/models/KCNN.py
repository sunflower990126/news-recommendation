import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class KCNN(torch.nn.Module):

    def __init__(self, data_config, model_config):
        super(KCNN, self).__init__()
        self.data_config = data_config
        self.model_config = model_config

        # 是否初始化词向量
        if "pretrained_word_embedding" not in self.data_config:
            self.word_embedding = nn.Embedding(self.model_config["word_size"],
                                               self.model_config["word_embedding_dim"],
                                               padding_idx=0)
        else:
            pretrained_word_embedding_file = os.path.join(
                self.data_config["data_root"], self.data_config["pretrained_word_embedding"])
            pretrained_word_embedding = torch.Tensor(np.load(pretrained_word_embedding_file))
            self.word_embedding = nn.Embedding.from_pretrained(
                pretrained_word_embedding, freeze=False, padding_idx=0)

        # 是否初始化实体向量
        if "pretrained_entity_embedding" not in self.data_config:
            self.entity_embedding = nn.Embedding(self.model_config["entity_size"],
                                                 self.model_config["entity_embedding_dim"],
                                                 padding_idx=0)
        else:
            pretrained_entity_embedding_file = os.path.join(
                self.data_config["data_root"], self.data_config["pretrained_entity_embedding"])
            pretrained_entity_embedding = torch.Tensor(np.load(pretrained_entity_embedding_file))
            self.entity_embedding = nn.Embedding.from_pretrained(
                pretrained_entity_embedding, freeze=False, padding_idx=0)

        # 是否初始化实体关联向量
        if self.model_config["use_context"]:
            if "pretrained_context_embedding" not in self.data_config:
                self.context_embedding = nn.Embedding(self.model_config["entity_size"],
                                                      self.model_config["entity_embedding_dim"],
                                                      padding_idx=0)
            else:
                pretrained_context_embedding_file = os.path.join(
                    self.data_config["data_root"], self.data_config["pretrained_context_embedding"])
                pretrained_context_embedding = torch.Tensor(np.load(pretrained_context_embedding_file))
                self.context_embedding = nn.Embedding.from_pretrained(
                    pretrained_context_embedding, freeze=False, padding_idx=0)

        # 实体和实体关联嵌入的空间转化层
        self.entity_transform = nn.Linear(
            self.model_config["entity_embedding_dim"], self.model_config["word_embedding_dim"])
        if self.model_config["use_context"]:
            self.context_transform = nn.Linear(
                self.model_config["entity_embedding_dim"], self.model_config["word_embedding_dim"])

        self.conv_filters = nn.ModuleDict({
            str(window): nn.Conv2d(3 if self.model_config["use_context"] else 2,
                                   self.model_config["num_filters"],
                                   (window, self.model_config["word_embedding_dim"]))
            for window in self.model_config["filter_sizes"]
        })

    # batch_size: N
    # history_length: L_h
    # word_seq_len: L_w
    # word_embedding_dim: E_w
    # entity_embedding_dim: E_e

    # news_indexs: (N, L_h, L_w) 或 (N, L_w)
    # news_entity_indexs: (N, L_h, L_w) 或 (N, L_w)
    def forward(
        self,
        news_indexs,
        news_entity_indexs
    ):
        word_vector = self.word_embedding(news_indexs)   # (N, L_h, L_w, E_w)
        entity_vector = self.entity_embedding(news_entity_indexs)   # (N, L_h, L_w, E_e)
        if self.model_config["use_context"]:
            context_vector = self.context_embedding(news_entity_indexs)   # (N, L_h, L_w, E_e)

        transformed_entity_vector = self.entity_transform(entity_vector)   # (N, L_h, L_w, E_w)
        transformed_entity_vector = torch.tanh(transformed_entity_vector)   # (N, L_h, L_w, E_w)

        if self.model_config["use_context"]:
            transformed_context_vector = self.entity_transform(context_vector)   # (N, L_h, L_w, E_w)
            transformed_context_vector = torch.tanh(transformed_context_vector)   # (N, L_h, L_w, E_w)

            multi_channel_vector = torch.stack([
                word_vector, transformed_entity_vector,
                transformed_context_vector
            ],
                dim=-3)      # (N, L_h, 3, L_w, E_w)
        else:
            multi_channel_vector = torch.stack(
                [word_vector, transformed_entity_vector], dim=-3)   # (N, L_h, 2, L_w, E_w)

        if len(multi_channel_vector.size()) == 5:   # 输入是click history还是candidate
            ifhistory = True
        else:
            ifhistory = False

        if ifhistory:
            N, L_h, input_channels, L_w, E_w = multi_channel_vector.size()
            multi_channel_vector = multi_channel_vector.view(
                N * L_h, input_channels, L_w, E_w)   # (N * L_h, 3, L_w, E_w)

        # window: w
        # num_filters: nf
        # len(filter_sizes): fs
        pooled_vectors = []
        for window in self.model_config["filter_sizes"]:
            convoluted = self.conv_filters[str(window)](
                multi_channel_vector).squeeze(dim=-1)   # (N * L_h, nf, L_w + 1 - w)
            activated = F.relu(convoluted)   # (N * L_h, nf, L_w + 1 - w)
            pooled = F.max_pool1d(
                activated, activated.size(-1)).squeeze(dim=-1)   # (N * L_h, nf)
            pooled_vectors.append(pooled)
        
        final_vector = torch.cat(pooled_vectors, dim=-1)   # (N * L_h, fs * nf)
        if ifhistory:
            final_vector = final_vector.view(
                N, L_h, len(self.model_config["filter_sizes"]) * self.model_config["num_filters"])   # (N, L_h, fs * nf)
        return final_vector

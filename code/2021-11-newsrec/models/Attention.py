import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(torch.nn.Module):

    def __init__(self, config):
        super(Attention, self).__init__()
        self.config = config
        self.dnn = nn.Sequential(
            nn.Linear(len(self.config["filter_sizes"]) * 2 * self.config["num_filters"], 
                      self.config["attention_hidden_size"]),
            nn.ReLU(),
            nn.Linear(self.config["attention_hidden_size"], 1)
        )

    # batch_size: N
    # history_length: L_h
    # num_filters: nf
    # len(filter_sizes): fs

    # candidate_news_vector: (N, fs * nf)
    # clicked_news_vector: (N, L_h, fs * nf)
    def forward(self, candidate_news_vector, clicked_news_vector):
        candidate_news_vector = candidate_news_vector.unsqueeze(1)   # (N, 1, fs * nf)
        candidate_news_vector = candidate_news_vector.expand(
            clicked_news_vector.size())   # (N, L_h, fs * nf)
        concat_news_vector = torch.cat(
            [candidate_news_vector, clicked_news_vector], dim=-1)   # (N, L_h, fs * nf * 2)
        clicked_news_weights = self.dnn(concat_news_vector).squeeze(dim=-1)   # (N, L_h)
        clicked_news_weights = F.softmax(clicked_news_weights, dim=-1)   # (N, L_h)

        user_vector = torch.bmm(clicked_news_weights.unsqueeze(dim=1),
                                clicked_news_vector).squeeze(dim=1)   # (N, fs * nf)
        return user_vector


class DNNClickPredictor(torch.nn.Module):
    def __init__(self, config):
        super(DNNClickPredictor, self).__init__()

        hidden_size = config["predictor_hidden_sizes"]
        layer_list = []
        last_layer_size=len(config["filter_sizes"]) * 2 * config["num_filters"]
        for index in range(len(hidden_size)):
            layer_list.append(nn.Linear(last_layer_size, hidden_size[index]))
            last_layer_size = hidden_size[index]
            layer_list.append(nn.ReLU())
            if config["dropout"]:
                layer_list.append(nn.Dropout(config["dropout"]))

        layer_list.append(nn.Linear(last_layer_size, 1))
        
        self.dnn = nn.Sequential(*layer_list)

    # batch_size: N
    # num_filters: nf
    # len(filter_sizes): fs

    # candidate_news_vector: (N, fs * nf)
    # user_vector: (N, fs * nf)
    def forward(self, candidate_news_vector, user_vector):
        concat_vector = torch.cat((candidate_news_vector, user_vector), dim=1)   # (N, fs * nf * 2)
        return self.dnn(concat_vector).squeeze(dim=1)   # (N, )

import torch
from models.KCNN import KCNN
from models.Attention import Attention, DNNClickPredictor

class DKN(torch.nn.Module):
    def __init__(self, data_config, model_config):
        super(DKN, self).__init__()
        self.data_config = data_config
        self.model_config = model_config
        self.kcnn = KCNN(self.data_config, self.model_config)
        self.attention = Attention(self.model_config)
        self.click_predictor = DNNClickPredictor(self.model_config)

    # batch_size: N
    # history_length: L_h
    # word_seq_len: L_w
    # word_embedding_dim: E_w
    # entity_embedding_dim: E_e
    # num_filters: nf
    # len(window_sizes): ws

    # candidate_news_indexs: (N, L_w)
    # candidate_news_entity_indexs: (N, L_w)
    # click_news_indexs: (N, L_h, L_w)
    # click_news_entity_indexs: (N, L_h, L_w)
    def forward(
        self, 
        candidate_news_indexs, 
        candidate_news_entity_indexs, 
        click_news_indexs, 
        click_news_entity_indexs
    ):
        candidate_news_vector = self.kcnn(
            candidate_news_indexs, candidate_news_entity_indexs)   # (N, ws * nf)
        clicked_news_vector = self.kcnn(
            click_news_indexs, click_news_entity_indexs)   # (N, L_h, ws * nf)

        user_vector = self.attention(candidate_news_vector, clicked_news_vector)   # (N, ws * nf)

        click_probability = self.click_predictor(candidate_news_vector, user_vector)   # (N, )
        return click_probability

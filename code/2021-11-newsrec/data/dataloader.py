import os
import numpy as np
from torch.utils.data import Dataset, DataLoader

class MindDataset(Dataset):

    def __init__(self, config, mode):

        print("加载新闻词和实体")
        """
        self.news_word_index为字典，字典的键为所有的新闻id，值为列表，列表内为处理好的该新闻的词的索引
        例：{'N39563': [29138, 393, 470, 1121, 118, 1579, 612, 3170, 0, 0]}
        self.news_entity_index与news_word_index形式相同，但存的是新闻的实体的索引
        例：{'N36658': [0, 0, 0, 0, 2846, 0, 0, 0, 0, 0]}
        """
        self.news_word_index = {}
        self.news_entity_index = {}
        news_feature_file = os.path.join(
            config["data_root"], config["news_feature_file"])
        with open(news_feature_file) as rd:
            for line in rd:
                newsid, word_index, entity_index = line.strip().split(" ")
                self.news_word_index[newsid] = [
                    int(item) for item in word_index.split(",")
                ]
                self.news_entity_index[newsid] = [
                    int(item) for item in entity_index.split(",")
                ]

        print("加载用户历史记录")
        """
        user_history为每个用户浏览的历史新闻id列表
        例：['N55189', 'N42782', 'N34694', 'N45794', 'N18445', 'N63302', 'N10414', 'N19347', 'N31801']
        click_news_index为根据user_history中的新闻id从上面的self.news_word_index中查找到的对应的值的列表，列表做了过长的截断，并对缺失的部分进行补0
        例：[[4837, 17, 4646, 950, 4838, 995, 753, 1, 10992, 5051], [56, 2249, 122, 1667, 1668, 584, 66, 1669, 589, 1]]
        click_news_entity_index与click_news_index类似
        self.user_history存储了所有训练、验证和测试用户的历史新闻的上述数据
        """
        self.user_history = {}
        user_history_file = os.path.join(
            config["data_root"], config["user_history_file"])
        with open(user_history_file) as rd:
            for line in rd:
                if len(line.strip().split(" ")) == 1:
                    userid = line.strip()
                    user_history = []
                else:
                    userid, user_history_string = line.strip().split(" ")
                    user_history = user_history_string.split(",")
                click_news_index = []
                click_news_entity_index = []
                if len(user_history) > config["history_size"]:
                    user_history = user_history[-config["history_size"]:]
                for newsid in user_history:
                    click_news_index.append(self.news_word_index[newsid])
                    click_news_entity_index.append(
                        self.news_entity_index[newsid])
                for i in range(config["history_size"] - len(user_history)):
                    click_news_index.append([0] * config["doc_size"])
                    click_news_entity_index.append([0] * config["doc_size"])
                self.user_history[userid] = (
                    click_news_index, click_news_entity_index)

        print("加载对应的数据情况")
        """
        self.labels存储的是所有的拆分后记录的标签
        例：[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        self.users_id存储的是所有的记录对应的用户id
        例：['train_U64800', 'train_U64800', 'train_U64800', 'train_U64800']
        self.users_candidate_news存储的是记录对应的待选新闻id
        例：['N19661', 'N41934', 'N61233', 'N61233']
        self.users_impression_id存储的是记录对应的impression（即原始数据集中的一条记录，仅验证集和测试集有有意义的值）
        例：[0, 0, 0, 0, 0, 0, 0, 0]
        """
        self.labels = []
        self.users_id = []
        self.users_candidate_news = []
        self.users_impression_id = []
        file_key = "{}_file".format(mode)
        file_path = os.path.join(config["data_root"], config[file_key])
        with open(file_path) as rd:
            for line in rd:
                impression_id = 0
                words = line.strip().split("%")
                if len(words) == 2:
                    impression_id = words[1].strip()

                cols = words[0].strip().split(" ")
                label = float(cols[0])

                userid = cols[1]
                candidate_news = cols[2]

                self.labels.append(label)
                self.users_id.append(userid)
                self.users_candidate_news.append(candidate_news)
                self.users_impression_id.append(impression_id)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        impression_id = self.users_impression_id[index]
        label = self.labels[index]
        user_id = self.users_id[index]

        candidate_news = self.users_candidate_news[index]
        candidate_news_index = np.array(self.news_word_index[candidate_news]).astype(np.int64)
        candidate_news_entity_index = np.array(self.news_entity_index[candidate_news]).astype(np.int64)
        click_news = self.user_history[user_id]
        click_news_index = np.array(click_news[0]).astype(np.int64)
        click_news_entity_index = np.array(click_news[1]).astype(np.int64)

        return impression_id, label, user_id, candidate_news_index, candidate_news_entity_index, click_news_index, click_news_entity_index


# Load datasets
def load_data(data_config, training_config, train=True):
    batch_size = training_config["batch_size"]
    num_workers = training_config["num_workers"]

    if train:
        train_dataset = MindDataset(data_config, "train")
        train_dataloader = DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        valid_dataset = MindDataset(data_config, "valid")
        valid_dataloader = DataLoader(
            dataset=valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    else:
        train_dataloader, valid_dataloader = None, None

    if "test_file" in data_config:
        test_dataset = MindDataset(data_config, "test")
        test_dataloader = DataLoader(
            dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    else:
        test_dataloader = None
        
    return train_dataloader, valid_dataloader, test_dataloader

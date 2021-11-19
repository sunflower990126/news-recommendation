import os
import argparse
import pprint
from data import dataloader
import warnings
import yaml
import torch
from run_networks import model

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default=None, type=str)

args = parser.parse_args()

# 训练参数解析
with open(args.cfg) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

torch.cuda.set_device(config["setting"]["gpu"])

if not config["train"]["evaluate"]:

    # 训练函数入口
    # 定义采样器
    print("加载数据")
    train_dataloader, valid_dataloader, test_dataloader = dataloader.load_data(
        config["data"], config["train"], train=True)
    data = {
        "train": train_dataloader, 
        "valid": valid_dataloader,
        "test": test_dataloader
    }
    print("数据加载完毕")

    training_model = model(config, data)
    training_model.train()

else:
    # 测试函数入口
    _, _, test_dataloader = dataloader.load_data(
        config["data"], config["train"], train=False)
    data = {
        "test": test_dataloader
    }

    training_model = model(config, data)
    training_model.eval(phase="test")

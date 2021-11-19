"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Portions of the source code are from the OLTR project which
notice below and in LICENSE in the root directory of
this source tree.

Copyright (c) 2019, Zhongqi Miao
All rights reserved.
"""

import os
import copy
import pickle
import datetime
from scipy.sparse.dia import dia_matrix
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import dataloader
from tensorboardX import SummaryWriter
from tqdm import tqdm
from logger import Logger
import time
import numpy as np

from utils import *
from models.DKN import DKN


class model():

    def __init__(self, config, data):

        self.data_config = config['data']
        self.model_config = config['model']
        self.training_config = config['train']
        self.setting_config = config['setting']
        self.data = data

        # 训练文件夹创建
        self.log_dir = os.path.join(
            self.training_config['log_root'], self.data_config['dataset'], self.training_config['model_name'])
        if not os.path.isdir(self.log_dir):
            os.makedirs(self.log_dir)

        # 日志对象生成
        self.logger = Logger(self.log_dir)
        self.tf_writer = SummaryWriter(log_dir=self.log_dir)

        # 初始化模型架构，生成模型对象
        self.model = DKN(self.data_config, self.model_config)
        self.model = self.model.cuda()

        # 加载预训练模型或训练好的模型
        if self.training_config['trained_model_dir'] is not None:
            self.load_model(self.training_config['trained_model_dir'])

        if not self.training_config["evaluate"]:
            # 针对一个epoch的数据，定义为一个完整的数据集样本的数量，利用step来固定此数据（即只学习固定个step）
            self.training_data_num = len(self.data['train'].dataset)
            self.epoch_steps = int(self.training_data_num
                                   / self.training_config['batch_size'])

            # 初始化优化器和损失函数
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=self.training_config["learning_rate"])
            self.criterions = nn.BCEWithLogitsLoss()

            # 创建新的日志文件并将当前的更新后的参数重新保存一份至目录文件夹下
            self.log_file = os.path.join(self.log_dir, 'log.txt')
            if os.path.isfile(self.log_file):
                os.remove(self.log_file)
            self.logger.log_cfg(config)
        else:
            self.log_file = None

    # 模型训练
    def train(self):
        print_write(['Phase: train'], self.log_file)

        # 初始化最好的模型参数与精度
        best_model_weights = copy.deepcopy(self.model.state_dict())
        best_auc = 0.0
        best_step = 0
        global_step = 0

        end_epoch = self.training_config['num_epochs']

        # 开始逐轮训练
        for epoch in range(1, end_epoch + 1):
            # 使模型处于训练模式
            self.model.train()

            # 缓存的显存清空
            torch.cuda.empty_cache()

            # 记录所有的预测标签
            total_preds = []
            total_labels = []

            # 在单轮次内实际按照step执行训练
            for step, batch in enumerate(self.data['train']):
                # 检查是否达到一个epoch限制的step数量，如果是的话停止该epoch
                if step == self.epoch_steps:
                    break

                _, labels, _, candidate_news_indexs, candidate_news_entity_indexs, click_news_indexs, click_news_entity_indexs = batch
                labels = labels.cuda()
                candidate_news_indexs = candidate_news_indexs.cuda()
                candidate_news_entity_indexs = candidate_news_entity_indexs.cuda()
                click_news_indexs =click_news_indexs.cuda()
                click_news_entity_indexs = click_news_entity_indexs.cuda()

                with torch.set_grad_enabled(True):

                    # 执行前馈、计算损失与反传的操作
                    logits = self.model(candidate_news_indexs, candidate_news_entity_indexs,
                               click_news_indexs, click_news_entity_indexs)
                    loss = self.criterions(logits, labels)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # 得到预测的结果
                    preds = torch2numpy(torch.sigmoid(logits))
                    labels = torch2numpy(labels)
                    total_preds.append(preds)
                    total_labels.append(labels)

                    # 间隔一定step打印结果
                    if step % self.training_config['display_step'] == 0:

                        minibatch_loss_total = loss.item()
                        minibatch_metric = cal_metric(
                            labels, preds, imp_indexs=None)

                        # 打印记录batch内的损失和精度
                        print_str = [datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                     'Epoch: [%d/%d]'
                                     % (epoch, self.training_config['num_epochs']),
                                     'Step: [%d/%d]'
                                     % (step, self.epoch_steps),
                                     'Minibatch_loss: %.4f'
                                     % (minibatch_loss_total),
                                     'Minibatch_auc: %.4f'
                                     % (minibatch_metric["auc"])]
                        print_write(print_str, self.log_file)

                        # 记录损失信息
                        loss_info = {
                            'Epoch': epoch,
                            'Step': step,
                            'Total': minibatch_loss_total
                        }

                        self.logger.log_loss(loss_info)

                global_step += 1

                if global_step % self.training_config['save_step'] == 0:
                    # 每轮结束后进行验证
                    rsls = {'step': global_step}
                    rsls_train = self.eval_with_preds(total_preds, total_labels)
                    rsls_eval = self.eval(phase='valid')
                    rsls.update(rsls_train)
                    rsls.update(rsls_eval)

                    # 记录验证结果
                    self.logger.log_auc(rsls)
                    self.tf_writer.add_scalar('train/auc', rsls["train_auc"], global_step)
                    self.tf_writer.add_scalar('valid/auc', rsls["valid_auc"], global_step)
                    self.tf_writer.add_scalar('valid/mean_mrr', rsls["valid_mean_mrr"], global_step)
                    self.tf_writer.add_scalar('valid/ndcg@5', rsls["valid_ndcg@5"], global_step)
                    self.tf_writer.add_scalar('valid/ndcg@10', rsls["valid_ndcg@10"], global_step)
                    self.tf_writer.add_scalar('valid/group_auc', rsls["valid_group_auc"], global_step)

                    # 更新最优的模型
                    if rsls["valid_auc"] > best_auc:
                        best_step = global_step
                        best_auc = rsls["valid_auc"]
                        best_model_weights = copy.deepcopy(self.model.state_dict())

                    self.save_latest(epoch)

        print_str = ['Best validation auc is %.4f at step %d' %
                     (best_auc, best_step)]
        print_write(print_str, self.log_file)
        # 保存最优的模型
        self.save_model(epoch, best_step,
                        best_model_weights, best_auc)

        # 在测试集上进行测试
        if "test" in self.data:
            self.reset_model(best_model_weights)
            self.eval('test' if 'test' in self.data else 'val')
        print('Done')

    # 用于在训练中对于训练集进行验证
    def eval_with_preds(self, preds, labels):

        rsl = {}
        preds, labels = list(map(np.concatenate, [preds, labels]))
        train_auc = cal_metric(labels, preds)
        rsl['train_auc'] = train_auc["auc"]

        # 打印结果
        print_str = ['\n Training auc: %.4f \n' % (rsl['train_auc']), '\n']
        print_write(print_str, self.log_file)

        return rsl

    # 验证阶段
    def eval(self, phase='valid'):

        print_str = ['Phase: %s' % (phase)]
        print_write(print_str, self.log_file)

        torch.cuda.empty_cache()

        # 调整为验证模式
        self.model.eval()

        total_logits = []
        total_labels = []
        total_impression_ids = []

        for batch in tqdm(self.data[phase]):
            impression_ids, labels, _, candidate_news_indexs, candidate_news_entity_indexs, click_news_indexs, click_news_entity_indexs = batch
            candidate_news_indexs = candidate_news_indexs.cuda()
            candidate_news_entity_indexs = candidate_news_entity_indexs.cuda()
            click_news_indexs = click_news_indexs.cuda()
            click_news_entity_indexs = click_news_entity_indexs.cuda()

            with torch.set_grad_enabled(False):
                logits = self.model(candidate_news_indexs, candidate_news_entity_indexs,
                                    click_news_indexs, click_news_entity_indexs)
                total_logits.append(logits.detach())
                total_labels.append(torch2numpy(labels))
                total_impression_ids.append(torch2numpy(impression_ids))

        total_logits = torch.cat(total_logits)
        preds = torch.sigmoid(total_logits).cpu().numpy()
        labels, impression_ids = list(
            map(np.concatenate, [total_labels, total_impression_ids]))

        # 计算精度
        eval_metric = cal_metric(labels, preds, impression_ids)

        # 打印结果
        print_str = ['\n\n',
                     'Phase: %s'
                     % (phase),
                     '\n\n',
                     'auc: %.4f'
                     % (eval_metric["auc"]),
                     '\n',
                     'mean_mrr: %.4f'
                     % (eval_metric["mean_mrr"]),
                     '\n',
                     'ndcg@5: %.4f'
                     % (eval_metric["ndcg@5"]),
                     'ndcg@10: %.4f'
                     % (eval_metric["ndcg@10"]), 
                     '\n',
                     'group_auc: %.4f'
                     % (eval_metric["group_auc"]),
                     '\n']

        rsl = {phase + '_auc': eval_metric["auc"],
               phase + '_mean_mrr': eval_metric["mean_mrr"],
               phase + '_ndcg@5': eval_metric["ndcg@5"],
               phase + '_ndcg@10': eval_metric["ndcg@10"],
               phase + '_group_auc': eval_metric["group_auc"]}

        print_write(print_str, self.log_file)

        return rsl

    # 从内存中加载最优的模型参数
    def reset_model(self, model_state):
        weights = {k: model_state[k]
                       for k in model_state if k in self.model.state_dict()}
        self.model.load_state_dict(weights)

    # 模型参数加载
    def load_model(self, model_dir):
        # 可以指定模型文件夹或模型文件自身
        if not model_dir.endswith('.pth'):
            model_dir = os.path.join(model_dir, 'final_model_checkpoint.pth')

        # 模型加载与最优参数读取
        checkpoint = torch.load(model_dir)
        model_state = checkpoint['state_dict_best']

        # 加载网络参数
        weights = {k: model_state[k]
                       for k in model_state if k in self.model.state_dict()}
        self.model.load_state_dict(weights)

    # 保存最新的模型
    def save_latest(self, epoch):
        model_weights= copy.deepcopy(self.model.state_dict())

        model_states = {
            'epoch': epoch,
            'state_dict': model_weights
        }

        model_dir = os.path.join(self.log_dir, 'latest_model_checkpoint.pth')
        torch.save(model_states, model_dir)

    # 保存最优的模型
    def save_model(self, epoch, best_step, best_model_weights, best_auc):

        model_states = {'epoch': epoch,
                        'best_step': best_step,
                        'state_dict_best': best_model_weights,
                        'best_auc': best_auc}

        model_dir = os.path.join(self.log_dir, 'final_model_checkpoint.pth')

        torch.save(model_states, model_dir)

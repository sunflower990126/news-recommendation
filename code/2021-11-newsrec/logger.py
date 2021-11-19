import os
import yaml
import csv


class Logger(object):
    def __init__(self, logdir):
        self.logdir = logdir
        if not os.path.isdir(logdir):
            os.makedirs(logdir)
        self.cfg_file = os.path.join(self.logdir, 'cfg.yaml')
        self.auc_file = os.path.join(self.logdir, 'auc.csv')
        self.loss_file = os.path.join(self.logdir, 'loss.csv')
        self.auc_keys = None
        self.loss_keys = None
        self.logging_ws = False

    def log_cfg(self, cfg):
        # 更新后的参数复制一份到日志文件夹下
        with open(self.cfg_file, 'w') as f:
            yaml.dump(cfg, f)

    def log_auc(self, aucs):
        if self.auc_keys is None:
            self.auc_keys = [k for k in aucs.keys()]
            with open(self.auc_file, 'w') as f:
                writer = csv.DictWriter(f, fieldnames=self.auc_keys)
                writer.writeheader()
                writer.writerow(aucs)
        else:
            with open(self.auc_file, 'a') as f:
                writer = csv.DictWriter(f, fieldnames=self.auc_keys)
                writer.writerow(aucs)

    def log_loss(self, losses):
        # valid_losses = {k: v for k, v in losses.items() if v is not None}
        valid_losses = losses
        if self.loss_keys is None:
            self.loss_keys = [k for k in valid_losses.keys()]
            with open(self.loss_file, 'w') as f:
                writer = csv.DictWriter(f, fieldnames=self.loss_keys)
                writer.writeheader()
                writer.writerow(valid_losses)
        else:
            with open(self.loss_file, 'a') as f:
                writer = csv.DictWriter(f, fieldnames=self.loss_keys)
                writer.writerow(valid_losses)
        
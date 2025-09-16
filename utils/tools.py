import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import math
import seaborn as sns

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')



# 注意力可视化

# 定义注意力可视化函数
def visualize_attention(attentions, layer_idx=0, head_idx=8, time_step=0, epoch=0, past_step=0, pred_step=0, save_path='./attns_visual/'):
    """
    可视化注意力权重热力图
    :param attentions: 注意力权重列表 [LAYER x (batch_size, n_heads, L, L)]
    :param layer_idx: 指定编码器层
    :param head_idx: 指定注意力头
    :param time_step: 指定 batch 中的样本索引
    :param epoch: 当前训练的 epoch，用于保存文件名
    :param dataset_name: 数据集名称，用于文件名标注
    :param past_step: 回顾步长，用于文件名标注
    :param pred_step: 预测步长，用于文件名标注
    :param save_path: 热力图保存路径
    """
    # 提取指定层、头、样本的注意力权重
    attn_weights = attentions[layer_idx][time_step, head_idx].cpu().detach().numpy()

    # 绘制热力图
    plt.figure(figsize=(8, 6))
    sns.heatmap(attn_weights, cmap='YlGnBu', cbar=True)
    plt.title(f"Attention Heatmap - Layer {layer_idx}, Head {head_idx}, Epoch {epoch}")
    plt.xlabel("Time Steps")
    plt.ylabel("Time Steps")
    plt.tight_layout()

    # 创建保存路径
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 保存热力图
    file_name = f"attention_epoch_{epoch}_layer_{layer_idx}_head_{head_idx}_past_{past_step}_pred_{pred_step}.png"
    plt.savefig(os.path.join(save_path, file_name))
    plt.close()




def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)
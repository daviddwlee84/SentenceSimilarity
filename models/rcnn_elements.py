# shared element for rcnn
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.functions import *


class FCSubtract(nn.Module):
    def __init__(self, D_in, D_out):
        super(FCSubtract, self).__init__()
        self.dense = nn.Linear(D_in, D_out)

    def forward(self, input_1, input_2):
        res_sub = torch.sub(input_1, input_2)
        res_sub_mul = torch.mul(res_sub, res_sub)
        out = self.dense(res_sub_mul)
        return F.relu(out)


class FCMultiply(nn.Module):
    def __init__(self, D_in, D_out):
        super(FCMultiply, self).__init__()
        self.dense = nn.Linear(D_in, D_out)

    def forward(self, input_1, input_2):
        res_mul = torch.mul(input_1, input_2)
        out = self.dense(res_mul)
        return F.relu(out)


class Inception1(nn.Module):
    def __init__(self, input_dim, conv_dim=64):
        super(Inception1, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, conv_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(conv_dim, conv_dim, kernel_size=2),
            nn.ReLU()
        )
        self.global_avg_pool = nn.AvgPool1d(input_dim)
        self.global_max_pool = nn.MaxPool1d(input_dim)

    def forward(self, x):
        out = self.cnn(x)
        avg_pool, max_pool = mean_max(x)
        return torch.cat((avg_pool, max_pool), dim=1)


class Inception2(nn.Module):
    def __init__(self, input_dim, conv_dim=64):
        super(Inception2, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, conv_dim, kernel_size=1),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.cnn(x)
        avg_pool, max_pool = mean_max(x)
        return torch.cat((avg_pool, max_pool), dim=1)


class Inception3(nn.Module):
    def __init__(self, input_dim, conv_dim=64):
        super(Inception3, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, conv_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(conv_dim, conv_dim, kernel_size=3),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.cnn(x)
        avg_pool, max_pool = mean_max(x)
        return torch.cat((avg_pool, max_pool), dim=1)

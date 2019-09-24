import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SingleSiameseCNN(nn.Module):
    def __init__(self, embedding_matrix, max_len, output_size, freeze_embed=False):
        super(SingleSiameseCNN, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix, freeze=freeze_embed)
        cnn_feature_num = 256
        cnn_filter_size = 7
        padding_size = math.floor(cnn_filter_size/2)
        pools_size = 3
        self.cnn_layers = nn.Sequential(
            nn.Conv1d(max_len, cnn_feature_num,
                      cnn_filter_size, stride=1, padding=padding_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(pools_size),
            nn.Conv1d(cnn_feature_num, cnn_feature_num,
                      cnn_filter_size, stride=1, padding=padding_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(pools_size),
            nn.Conv1d(cnn_feature_num, cnn_feature_num,
                      cnn_filter_size, stride=1, padding=padding_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(cnn_feature_num, cnn_feature_num,
                      cnn_filter_size, stride=1, padding=padding_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(cnn_feature_num, cnn_feature_num,
                      cnn_filter_size, stride=1, padding=padding_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(cnn_feature_num, cnn_feature_num,
                      cnn_filter_size, stride=1, padding=padding_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(pools_size)
        )
        fcnn_units = 1024
        nums_of_pools = 3
        feature_left = math.floor(
            self.embedding.embedding_dim / pools_size**nums_of_pools)
        self.dense = nn.Sequential(
            nn.Linear(cnn_feature_num*feature_left, fcnn_units),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(fcnn_units, fcnn_units),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(fcnn_units, output_size),
            nn.Sigmoid()
        )

    def forward(self, sentence):
        sent_embed = self.embedding(sentence)
        dense_input = self.cnn_layers(sent_embed)
        output = self.dense(dense_input.view(dense_input.shape[0], -1))
        return output


class SingleSiameseRNN(nn.Module):
    def __init__(self, embedding_matrix, max_len, output_size, bidirectional=False, num_layers=2, linear_size=128, freeze_embed=False):
        super(SingleSiameseRNN, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix, freeze=freeze_embed)
        self.rnn = nn.RNN(self.embedding.embedding_dim,
                          linear_size, num_layers, bidirectional=bidirectional)
        direction = 1 + int(bidirectional)
        self.dense = nn.Sequential(
            nn.Linear(linear_size*direction, output_size),
            nn.Sigmoid()
        )

    def forward(self, sentence):
        # batch_size, max_len, embed_dim
        sent_embed = self.embedding(sentence)
        # batch_size, max_len, linear_size
        rnn_output, hn = self.rnn(sent_embed)
        # batch_size, linear_size
        last_hidden_state = rnn_output[:, -1, :]
        # batch_size, output_size
        output = self.dense(last_hidden_state)
        return output


class SingleSiameseLSTM(nn.Module):
    def __init__(self, embedding_matrix, max_len, output_size, bidirectional=False, num_layers=2, linear_size=128, freeze_embed=False):
        super(SingleSiameseLSTM, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix, freeze=freeze_embed)
        self.rnn = nn.LSTM(self.embedding.embedding_dim,
                           linear_size, num_layers, bidirectional=bidirectional)
        direction = 1 + int(bidirectional)
        self.dense = nn.Sequential(
            nn.Linear(linear_size*direction, output_size),
            nn.Sigmoid()
        )

    def forward(self, sentence):
        # batch_size, max_len, embed_dim
        sent_embed = self.embedding(sentence)
        # batch_size, max_len, linear_size
        lstm_output, (hn, cn) = self.rnn(sent_embed)
        # batch_size, linear_size
        last_hidden_state = lstm_output[:, -1, :]
        # batch_size, output_size
        output = self.dense(last_hidden_state)
        return output


class SingleSiameseTextCNN(nn.Module):
    """ deprecated """

    def __init__(self, embedding_matrix, max_len, output_size, device, linear_size=128, windows=[3, 4, 5], freeze_embed=False):
        super(SingleSiameseTextCNN, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix, freeze=freeze_embed)
        # somehow the model in list can't be auto connect `to(device)`
        self.cnn_filters = [
            nn.Sequential(
                nn.Conv1d(max_len, linear_size, kernel_wide),
                nn.ReLU()
            ).to(device) for kernel_wide in windows]
        self.dense = nn.Sequential(
            nn.Linear(linear_size*len(windows), output_size),
            nn.Sigmoid()
        )

    def forward(self, sentence):
        sent_embed = self.embedding(sentence)
        output_of_cnns = [
            cnn_filter(sent_embed)
            for cnn_filter in self.cnn_filters]
        global_pooled_output = list(map(
            lambda x: torch.max(x, dim=2)[0], output_of_cnns))
        # equivalent to
        # global_pooled_output = [F.max_pool1d(x, x.size(2)).squeeze(2) for x in output_of_cnns]
        dense_input = torch.cat(global_pooled_output, dim=1)
        output = self.dense(dense_input)
        return output

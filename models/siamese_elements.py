import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SingleSiameseCNN(nn.Module):
    """ Model is based on "Character-level Convolutional Networks for Text Classification" """

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
    def __init__(self, embedding_matrix, max_len, output_size, bidirectional=False, num_layers=2, linear_size=512, freeze_embed=False):
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
        # fix the warning message when using multiple GPUs
        self.rnn.flatten_parameters()
        # batch_size, max_len, linear_size
        rnn_output, hn = self.rnn(sent_embed)
        # batch_size, linear_size
        last_hidden_state = rnn_output[:, -1, :]
        # batch_size, output_size
        output = self.dense(last_hidden_state)
        return output


class SingleSiameseLSTM(nn.Module):
    """ Model is based on "Siamese Recurrent Architectures for Learning Sentence Similarity" """

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
        # fix the warning message when using multiple GPUs
        self.rnn.flatten_parameters()
        # batch_size, max_len, linear_size
        lstm_output, (hn, cn) = self.rnn(sent_embed)
        # batch_size, linear_size
        last_hidden_state = lstm_output[:, -1, :]
        # batch_size, output_size
        output = self.dense(last_hidden_state)
        return output


class SingleSiameseTextCNN(nn.Module):
    """ Model is based on "Convolutional Neural Networks for Sentence Classification" """

    def __init__(self, embedding_matrix, max_len, output_size, linear_size=128, windows=[3, 4, 5], freeze_embed=False):
        super(SingleSiameseTextCNN, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix, freeze=freeze_embed)
        self.cnn_filters = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(max_len, linear_size, kernel_wide),
                nn.ReLU()
            ) for kernel_wide in windows])
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


class SingleSiameseRCNN(nn.Module):
    """ Model is based on "Siamese Recurrent Architectures for Learning Sentence Similarity" """

    def __init__(self, embedding_matrix, max_len, output_size, num_layers=2, freeze_embed=False):
        super(SingleSiameseRCNN, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix, freeze=freeze_embed)
        hidden_layer_size = 100
        context_window_size = 7
        self.rnn = nn.RNN(self.embedding.embedding_dim, hidden_layer_size,
                          num_layers, bidirectional=True)
        self.cnn = nn.Conv1d(max_len, hidden_layer_size,
                             context_window_size, padding=context_window_size//2)
        self.dense = nn.Sequential(
            nn.Linear(hidden_layer_size*2, output_size),
            nn.Sigmoid()
        )

    def forward(self, sentence):
        # batch_size, max_len, embed_dim
        sent_embed = self.embedding(sentence)
        # fix the warning message when using multiple GPUs
        self.rnn.flatten_parameters()
        # batch_size, max_len, hidden_layer_size*2
        rnn_output, hn = self.rnn(sent_embed)
        # batch_size, max_len, hidden_layer_size*2
        cnn_output = self.cnn(rnn_output)
        represent_for_words = F.relu(cnn_output)
        # batch_size, hidden_layer_size*2
        last_hidden_state = torch.max(represent_for_words, dim=1)[0]
        # batch_size, output_size
        output = self.dense(last_hidden_state)
        return output


class SingleSiameseAttentionRNN(nn.Module):
    """ Model is based on "Text Classification Research with Attention-based Recurrent Neural Networks" """

    def __init__(self, embedding_matrix, max_len, output_size, freeze_embed=False):
        super(SingleSiameseAttentionRNN, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix, freeze=freeze_embed)

        self.hidden_layer_size = 100
        self.seq_len = max_len
        self.num_direction = 2
        num_rnn_layers = 1

        self.rnn = nn.GRU(self.embedding.embedding_dim, self.hidden_layer_size,
                          num_rnn_layers, bidirectional=True)
        self.attention = Attention(
            self.embedding.embedding_dim*self.num_direction, max_len)

        dense_size = 100
        self.dense = nn.Sequential(
            nn.Linear(self.hidden_layer_size*self.num_direction*2, dense_size),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(dense_size, output_size),
            nn.Sigmoid()
        )

    def forward(self, sentence):
        # batch_size, max_len, embed_dim
        sent_embed = self.embedding(sentence)
        # fix the warning message when using multiple GPUs
        self.rnn.flatten_parameters()
        # max_len, batch_size, num_directions, hidden_layer_size
        rnn_output = self.rnn(sent_embed)[0].view(
            self.seq_len, -1, self.num_direction, self.hidden_layer_size)
        # batch_size, max_len, hidden_layer_size
        rnn_forward = rnn_output[:, :, 0, :].permute(1, 0, 2)
        # batch_size, max_len, hidden_layer_size
        rnn_backward = rnn_output[:, :, 1, :].permute(1, 0, 2)

        # batch_size, max_len, hidden_layer_size*2
        rnn_cat = torch.cat((rnn_forward, rnn_backward), dim=-1)

        # batch_size, hidden_layer_size*2
        attention = self.attention(rnn_cat)
        # batch_size, hidden_layer_size
        forward_last_state = rnn_forward[:, -1, :]
        # batch_size, hidden_layer_size
        backward_last_state = rnn_backward[:, -1, :]

        # batch_size, hidden_layer_size*4
        dense_input = torch.cat(
            (attention, forward_last_state, backward_last_state), dim=-1)

        # batch_size, output_size
        output = self.dense(dense_input)
        return output


# Sub-Layers

class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True):
        super(Attention, self).__init__()

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0

        weight = torch.zeros(feature_dim, step_dim)
        nn.init.kaiming_uniform_(weight)
        self.weight = nn.Parameter(weight)

        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))

    def forward(self, x):
        eij = torch.bmm(x, self.weight.unsqueeze(0).repeat(x.shape[0], 1, 1))

        if self.bias:
            eij = eij + self.b

        a = F.softmax(eij, dim=1)

        return torch.sum(torch.bmm(a, x), dim=1)

import torch
import torch.nn as nn
import torch.nn.functional as F


def mean_max(x):
    return torch.mean(x, dim=1), torch.max(x, dim=1)[0]

# define custom module (layer)


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


# main module


class EnhancedRCNN(nn.Module):
    def __init__(self, embeddings_matrix, max_len, padding_idx, num_class=2, lstm_dim=192, dropout_rate=0.2, linear_size=384, conv_dim=64):
        super(EnhancedRCNN, self).__init__()

        self.max_len = max_len
        fc_out_dims = int(linear_size//2)
        # define basic network layers
        self.embedding = nn.Embedding.from_pretrained(embeddings_matrix)
        # self.embedding.padding_idx = padding_idx  # 0
        self.batchnrom = nn.BatchNorm1d(max_len)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.global_avg_pool = nn.AvgPool1d(max_len)
        self.global_max_pool = nn.MaxPool1d(max_len)
        self.cnn1 = Inception1(lstm_dim*2)
        self.cnn2 = Inception2(lstm_dim*2)
        self.cnn3 = Inception3(lstm_dim*2)
        self.q1_rnn = nn.GRU(self.embedding.embedding_dim, lstm_dim,
                             num_layers=2, dropout=dropout_rate, bidirectional=True)
        self.q2_rnn = nn.GRU(self.embedding.embedding_dim, lstm_dim,
                             num_layers=2, dropout=dropout_rate, bidirectional=True)
        self.fc_sub = FCSubtract(max_len*2*3 + lstm_dim*2*2, fc_out_dims)
        self.fc_mul = FCMultiply(max_len*2*3 + lstm_dim*2*2, fc_out_dims)
        self.dense = nn.Sequential(
            nn.Linear(fc_out_dims*2, num_class),
            nn.Sigmoid()
        )

    # define complex network connection
    def soft_align(self, input_1, input_2):
        attention = torch.bmm(input_1, input_2.permute(0, 2, 1))
        w_att_1 = F.softmax(attention, dim=1)
        w_att_2 = F.softmax(attention, dim=2).permute(0, 2, 1)
        in1_aligned = torch.bmm(w_att_1, input_1)
        in2_aligned = torch.bmm(w_att_2, input_2)
        return in1_aligned, in2_aligned

    def forward(self, q1, q2):
        # connect network structure
        q1_embed = self.batchnrom(self.embedding(q1))
        q2_embed = self.batchnrom(self.embedding(q2))

        x1 = self.dropout(q1_embed)
        x2 = self.dropout(q2_embed)

        # batch_size, max_len, lstm_dim*2
        q1_output, q1_hn = self.q1_rnn(x1)
        q2_output, q2_hn = self.q2_rnn(x2)

        # batch_size, lstm_dim*2, max_len
        q1_encoded_permute = q1_output.permute(0, 2, 1)
        q2_encoded_permute = q2_output.permute(0, 2, 1)

        # batch_size, max_len, lstm_dim*2
        q1_aligned, q2_aligned = self.soft_align(q1_output, q2_output)

        # batch_size, lstm_dim*2
        sentence1_att_mean, sentence1_att_max = mean_max(q1_aligned)
        sentence2_att_mean, sentence2_att_max = mean_max(q2_aligned)

        sentence1_cnn = torch.cat((self.cnn1(q1_encoded_permute), self.cnn2(
            q1_encoded_permute), self.cnn3(q1_encoded_permute)), dim=1)
        sentence2_cnn = torch.cat((self.cnn1(q2_encoded_permute), self.cnn2(
            q2_encoded_permute), self.cnn3(q2_encoded_permute)), dim=1)

        sentence1 = torch.cat(
            (sentence1_att_mean, sentence1_cnn, sentence1_att_max), dim=1)
        sentence2 = torch.cat(
            (sentence2_att_mean, sentence2_cnn, sentence2_att_max), dim=1)

        res_sub = self.fc_sub(sentence1, sentence2)
        res_mul = self.fc_mul(sentence1, sentence2)
        res = torch.cat((res_sub, res_mul), dim=1)

        output = self.dense(res)

        return output

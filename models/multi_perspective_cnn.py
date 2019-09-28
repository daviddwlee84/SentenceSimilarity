# Reference from:
# https://github.com/tuzhucheng/MP-CNN-Variants
# models/mpcnn.py & models/mpcnn_variant_base.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MPCNNBase(nn.Module):
    """ The base model for MultiPerspectiveCNN """

    def __init__(self):
        super(MPCNNBase, self).__init__()

    def _horizontal_comparison(self, sent1_block_a, sent2_block_a, pooling_types=('max', 'min', 'mean'), comparison_types=('cosine', 'euclidean')):
        comparison_feats = []
        for pool in pooling_types:
            regM1, regM2 = [], []
            for ws in self.filter_widths:
                x1 = sent1_block_a[ws][pool].unsqueeze(2)
                x2 = sent2_block_a[ws][pool].unsqueeze(2)
                if np.isinf(ws):
                    x1 = x1.expand(-1, self.n_holistic_filters, -1)
                    x2 = x2.expand(-1, self.n_holistic_filters, -1)
                regM1.append(x1)
                regM2.append(x2)

            regM1 = torch.cat(regM1, dim=2)
            regM2 = torch.cat(regM2, dim=2)

            if 'cosine' in comparison_types:
                comparison_feats.append(
                    F.cosine_similarity(regM1, regM2, dim=2))

            if 'euclidean' in comparison_types:
                pairwise_distances = []
                for x1, x2 in zip(regM1, regM2):
                    dist = F.pairwise_distance(x1, x2).view(1, -1)
                    pairwise_distances.append(dist)
                comparison_feats.append(torch.cat(pairwise_distances))

            if 'abs' in comparison_types:
                comparison_feats.append(
                    torch.abs(regM1 - regM2).view(regM1.size(0), -1))

        return torch.cat(comparison_feats, dim=1)

    def _vertical_comparison(self, sent1_block_a, sent2_block_a, sent1_block_b, sent2_block_b, holistic_pooling_types=('max', 'min', 'mean'), per_dim_pooling_types=('max', 'min'), comparison_types=('cosine', 'euclidean', 'abs')):
        comparison_feats = []
        ws_no_inf = [w for w in self.filter_widths if not np.isinf(w)]
        for pool in holistic_pooling_types:
            for ws1 in self.filter_widths:
                x1 = sent1_block_a[ws1][pool]
                for ws2 in self.filter_widths:
                    x2 = sent2_block_a[ws2][pool]
                    if (not np.isinf(ws1) and not np.isinf(ws2)) or (np.isinf(ws1) and np.isinf(ws2)):
                        if 'cosine' in comparison_types:
                            comparison_feats.append(
                                F.cosine_similarity(x1, x2).unsqueeze(1))
                        if 'euclidean' in comparison_types:
                            comparison_feats.append(
                                F.pairwise_distance(x1, x2).unsqueeze(1))
                        if 'abs' in comparison_types:
                            comparison_feats.append(torch.abs(x1 - x2))

        for pool in per_dim_pooling_types:
            for ws in ws_no_inf:
                oG_1B = sent1_block_b[ws][pool]
                oG_2B = sent2_block_b[ws][pool]
                for i in range(0, self.n_per_dim_filters):
                    x1 = oG_1B[:, :, i]
                    x2 = oG_2B[:, :, i]
                    if 'cosine' in comparison_types:
                        comparison_feats.append(
                            F.cosine_similarity(x1, x2).unsqueeze(1))
                    if 'euclidean' in comparison_types:
                        comparison_feats.append(
                            F.pairwise_distance(x1, x2).unsqueeze(1))
                    if 'abs' in comparison_types:
                        comparison_feats.append(torch.abs(x1 - x2))

        return torch.cat(comparison_feats, dim=1)

    def forward(self, q1, q2):
        # to be implemented in child class
        pass


class MultiPerspectiveCNN(MPCNNBase):
    """ Model is based on "Multi-Perspective Sentence Similarity Modeling with Convolutional Neural Networks" """

    def __init__(self, embedding_matrix, max_len, num_class=1, freeze_embed=False):
        super(MultiPerspectiveCNN, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix, freeze=freeze_embed)

        self.n_word_dim = self.embedding.embedding_dim
        self.n_holistic_filters = 300
        self.n_per_dim_filters = 20
        self.filter_widths = [1, 2, 3, np.inf]
        self.wide_conv = True

        # input channels for CNN
        self.in_channels = max_len

        self._add_layers()

        # compute number of inputs to first hidden layer
        n_feats = self._get_n_feats()

        dropout = 0.2
        hidden_layer_units = 512

        self.final_layers = nn.Sequential(
            nn.Linear(n_feats, hidden_layer_units),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_layer_units, num_class),
            # nn.LogSoftmax(1)
            nn.Sigmoid()
        )

    def _add_layers(self):
        """ construct CNN layers for block A (holistic) and block B (per-dim) """
        holistic_conv_layers_max = []
        holistic_conv_layers_min = []
        holistic_conv_layers_mean = []
        per_dim_conv_layers_max = []
        per_dim_conv_layers_min = []

        for ws in self.filter_widths:
            if np.isinf(ws):  # if window size = infinity that means no convolution
                continue

            padding = ws-1 if self.wide_conv else 0

            holistic_conv_layers_max.append(nn.Sequential(
                nn.Conv1d(self.in_channels, self.n_holistic_filters,
                          ws, padding=padding),
                nn.Tanh()
            ))

            holistic_conv_layers_min.append(nn.Sequential(
                nn.Conv1d(self.in_channels, self.n_holistic_filters,
                          ws, padding=padding),
                nn.Tanh()
            ))

            holistic_conv_layers_mean.append(nn.Sequential(
                nn.Conv1d(self.in_channels, self.n_holistic_filters,
                          ws, padding=padding),
                nn.Tanh()
            ))

            per_dim_conv_layers_max.append(nn.Sequential(
                nn.Conv1d(self.in_channels, self.in_channels * self.n_per_dim_filters,
                          ws, padding=padding, groups=self.in_channels),
                nn.Tanh()
            ))

            per_dim_conv_layers_min.append(nn.Sequential(
                nn.Conv1d(self.in_channels, self.in_channels * self.n_per_dim_filters,
                          ws, padding=padding, groups=self.in_channels),
                nn.Tanh()
            ))

        self.holistic_conv_layers_max = nn.ModuleList(holistic_conv_layers_max)
        self.holistic_conv_layers_min = nn.ModuleList(holistic_conv_layers_min)
        self.holistic_conv_layers_mean = nn.ModuleList(
            holistic_conv_layers_mean)
        self.per_dim_conv_layers_max = nn.ModuleList(per_dim_conv_layers_max)
        self.per_dim_conv_layers_min = nn.ModuleList(per_dim_conv_layers_min)

    def _get_n_feats(self):
        """ calculate the feature number for the final dense layer input size """
        COMP_1_COMPONENTS_HOLISTIC, COMP_1_COMPONENTS_PER_DIM, COMP_2_COMPONENTS = 2 + \
            self.n_holistic_filters, 2 + self.in_channels, 2
        n_feats_h = 3 * self.n_holistic_filters * COMP_2_COMPONENTS
        n_feats_v = (
            # comparison units from holistic conv for min, max, mean pooling for non-infinite widths
            3 * ((len(self.filter_widths) - 1) ** 2) * COMP_1_COMPONENTS_HOLISTIC +
            # comparison units from holistic conv for min, max, mean pooling for infinite widths
            3 * 3 +
            # comparison units from per-dim conv
            2 * (len(self.filter_widths) - 1) * \
            self.n_per_dim_filters * COMP_1_COMPONENTS_PER_DIM
        )
        n_feats = n_feats_h + n_feats_v
        return n_feats

    def _get_blocks_for_sentence(self, sent):
        """ computing CNNs => Poolings results for a sentence """
        block_a = {}
        block_b = {}
        for ws in self.filter_widths:
            if np.isinf(ws):  # if window size = infinity that means no convolution
                sent_flattened, sent_flattened_size = sent.contiguous().view(
                    sent.size(0), 1, -1), sent.size(1) * sent.size(2)
                block_a[ws] = {
                    'max': F.max_pool1d(sent_flattened, sent_flattened_size).view(sent.size(0), -1),
                    'min': F.max_pool1d(-1 * sent_flattened, sent_flattened_size).view(sent.size(0), -1),
                    'mean': F.avg_pool1d(sent_flattened, sent_flattened_size).view(sent.size(0), -1)
                }
                continue

            holistic_conv_out_max = self.holistic_conv_layers_max[ws - 1](sent)
            holistic_conv_out_min = self.holistic_conv_layers_min[ws - 1](sent)
            holistic_conv_out_mean = self.holistic_conv_layers_mean[ws - 1](
                sent)
            block_a[ws] = {
                'max': F.max_pool1d(holistic_conv_out_max, holistic_conv_out_max.size(2)).contiguous().view(-1, self.n_holistic_filters),
                'min': F.max_pool1d(-1 * holistic_conv_out_min, holistic_conv_out_min.size(2)).contiguous().view(-1, self.n_holistic_filters),
                'mean': F.avg_pool1d(holistic_conv_out_mean, holistic_conv_out_mean.size(2)).contiguous().view(-1, self.n_holistic_filters)
            }

            per_dim_conv_out_max = self.per_dim_conv_layers_max[ws - 1](sent)
            per_dim_conv_out_min = self.per_dim_conv_layers_min[ws - 1](sent)
            block_b[ws] = {
                'max': F.max_pool1d(per_dim_conv_out_max, per_dim_conv_out_max.size(2)).contiguous().view(-1, self.in_channels, self.n_per_dim_filters),
                'min': F.max_pool1d(-1 * per_dim_conv_out_min, per_dim_conv_out_min.size(2)).contiguous().view(-1, self.in_channels, self.n_per_dim_filters)
            }
        return block_a, block_b

    def _algo_1_horiz_comp(self, sent1_block_a, sent2_block_a):
        """ call the horizontal comparison (in parent class) """
        return self._horizontal_comparison(sent1_block_a, sent2_block_a)

    def _algo_2_vert_comp(self, sent1_block_a, sent2_block_a, sent1_block_b, sent2_block_b):
        """ call the vertical comparison (in parent class) """
        return self._vertical_comparison(sent1_block_a, sent2_block_a, sent1_block_b, sent2_block_b)

    def forward(self, q1, q2):
        sent1 = self.embedding(q1)
        sent2 = self.embedding(q2)

        # Sentence modeling module
        sent1_block_a, sent1_block_b = self._get_blocks_for_sentence(sent1)
        sent2_block_a, sent2_block_b = self._get_blocks_for_sentence(sent2)

        # Similarity measurement layer
        feat_h = self._algo_1_horiz_comp(sent1_block_a, sent2_block_a)
        feat_v = self._algo_2_vert_comp(
            sent1_block_a, sent2_block_a, sent1_block_b, sent2_block_b)
        feat_all = torch.cat([feat_h, feat_v], dim=1)

        preds = self.final_layers(feat_all)
        return preds

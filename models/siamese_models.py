# Siamese-CNN
import torch
import torch.nn as nn


class SiameseModel(nn.Module):
    def __init__(self, embedding_matrix, max_len, SingleModel, model_output_size=100, freeze_embed=True):
        super(SiameseModel, self).__init__()
        embedding_layer = nn.Embedding.from_pretrained(
            embedding_matrix, freeze=freeze_embed)
        self.half_model = SingleModel(
            embedding_layer, max_len, model_output_size)

    def forward(self, q1, q2):
        output1 = self.half_model(q1)
        output2 = self.half_model(q2)

        return output1, output2

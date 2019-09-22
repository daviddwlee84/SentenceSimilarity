import torch
import torch.nn as nn


class SiameseModel(nn.Module):
    def __init__(self, single_model):
        super(SiameseModel, self).__init__()
        self.half_model = single_model

    def forward(self, q1, q2):
        output1 = self.half_model(q1)
        output2 = self.half_model(q2)

        return output1, output2

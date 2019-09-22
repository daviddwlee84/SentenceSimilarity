import torch
import torch.nn as nn


class SiameseModel(nn.Module):
    def __init__(self, single_model, linear_size, num_class=1):
        super(SiameseModel, self).__init__()
        self.half_model = single_model
        self.dense = nn.Sequential(
            nn.Linear(linear_size, num_class),
            nn.Sigmoid()
        )

    def forward(self, q1, q2):
        sent_embed_1 = self.half_model(q1)
        sent_embed_2 = self.half_model(q2)
        l1_distance = torch.abs(sent_embed_1 - sent_embed_2)
        output = self.dense(l1_distance)

        return output


class SiameseModelWithoutDense(nn.Module):
    """ deprecated """

    def __init__(self, single_model):
        super(SiameseModel, self).__init__()
        self.half_model = single_model

    def forward(self, q1, q2):
        output1 = self.half_model(q1)
        output2 = self.half_model(q2)

        return output1, output2

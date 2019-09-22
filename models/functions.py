import torch
import torch.nn.functional as F


def mean_max(x):
    return torch.mean(x, dim=1), torch.max(x, dim=1)[0]


def contrastive_loss(output1, output2, label, margin=2.0):
    """ calculate loss for siamese models """
    # Find the pairwise distance or eucledian distance of two output feature vectors
    euclidean_distance = F.pairwise_distance(output1, output2)
    # perform contrastive loss calculation with the distance
    loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                  (label) * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2))

    return loss_contrastive

# Adopted from a code base from https://github.com/cetinsamet
# --------------------------------------------------

import random
random.seed(123)

import numpy as np
np.random.seed(123)

import torch
torch.manual_seed(123)

criterion   = torch.nn.CrossEntropyLoss(reduction='sum')    # <-- Loss Function


class Network(torch.nn.Module):
    """ Zero-Shot model """

    def __init__(self, feature_dim, vector_dim):

        super(Network, self).__init__()
        self.wHidden1   = torch.nn.Linear(feature_dim, vector_dim)

    def forward(self, imageFeatures, classVectors):

        imageFeatures   = self.wHidden1(imageFeatures)
        out             = torch.matmul(imageFeatures, torch.t(classVectors))

        return out


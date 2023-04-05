import numpy as np

import paddle
from paddle import nn
from paddle.nn.initializer import Constant, Normal, TruncatedNormal

trunc_normal_ = TruncatedNormal(std=0.02)
normal_ = Normal(mean=0, std=0.01)
zeros_ = Constant(value=0.0)
ones_ = Constant(value=1.0)


class LinearClassifier(nn.Layer):
    """Linear layer to train on top of frozen features"""

    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        normal_(self.linear.weight)
        zeros_(self.linear.bias)

    def forward(self, x):
        # flatten
        x = x.reshape((x.shape[0], -1))

        # linear layer
        return self.linear(x)

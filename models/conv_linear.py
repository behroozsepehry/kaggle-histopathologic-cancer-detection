import numpy as np
from torch import nn

from utilities import general_utilities as g_util
from utilities import nn_utilities as n_util


class Model(nn.Module):
    def __init__(self, in_size, in_channels, mid_channels, out_channels, **kwargs):
        super(Model, self).__init__()
        self.ngpu = kwargs.get('ngpu', 1)
        self.kernel_size = kernel_size = kwargs.get('kernel_size', 7)

        activation_setting = kwargs.get('activation')
        activation = []
        if activation_setting:
            activation_args = activation_setting.get('args', {})
            activation.append(getattr(nn, activation_setting['name'])(**activation_args))

        n_mid_layers = in_size // (kernel_size - 1) - 2
        last_kernel_size = in_size - (n_mid_layers+1) * (kernel_size - 1)
        mid_layers = []
        for i in range(1, n_mid_layers+1):
            mid_layers += [nn.Conv2d(i * mid_channels, (i+1) * mid_channels, kernel_size, bias=False),
                           nn.BatchNorm2d((i+1) * mid_channels),
                           nn.LeakyReLU(0.2, inplace=True)]

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            *mid_layers,
            nn.Conv2d((n_mid_layers+1) * mid_channels, out_channels, last_kernel_size, bias=False),
            *activation
        )

    def forward(self, x):
        output = n_util.data_parallel_model(self.cnn, x, self.ngpu)
        return output.view(x.size(0), -1)


if __name__ == '__main__':
    import torch
    model = Model(96, 3, 20, 1, activation=dict(name='Sigmoid'), kernel_size=4)
    print(model)
    x = torch.randn(2, 3, 96, 96)
    y = model(x)
    print(y)
    print(y.size())

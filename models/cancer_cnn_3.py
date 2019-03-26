from torchvision import models
from torch import nn
import torch
import torch.nn.functional as F
from collections import OrderedDict

from torchvision.models import densenet
from torchvision import transforms

from models import base


class Model(base.ModelBase):
    def __init__(self, *args, **kwargs):
        super(Model, self).__init__(*args, **kwargs)
        self.cnn = models.densenet121(num_classes=1)
        self.cnn._modules['conv0'] = nn.Conv2d(6, self.cnn._modules['conv0'].out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        nn.init.kaiming_normal(self.cnn._modules['conv0'].weight.data)

        activation_setting = kwargs.get('activation')
        if activation_setting:
            activation_args = activation_setting.get('args', {})
            self.activation = getattr(nn, activation_setting['name'])(**activation_args)


        self.resize_1 = transforms.Compose([
            transforms.Resize(224)
        ])
        self.resize_2 = transforms.Compose([
            transforms.CenterCrop(32),
            transforms.Resize(224)
        ])

    def forward(self, x):
        x1 = self.resize_1(x)
        x2 = self.resize_2(x)
        xx = torch.cat((x1, x2), dim=1)
        y = self.cnn(xx).view(xx.size(0), -1)
        if self.activation:
            y = self.activation(y)
        return 

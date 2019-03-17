from torchvision import models
from torch import nn

from models import base


class Model(base.ModelBase):
    def __init__(self, *args, **kwargs):
        super(Model, self).__init__(*args, **kwargs)
        self.cnn = models.densenet121(num_classes=1)
        activation_setting = kwargs.get('activation')
        if activation_setting:
            activation_args = activation_setting.get('args', {})
            activation = getattr(nn, activation_setting['name'])(**activation_args)
            self.cnn.add_module('activation', activation)

    def forward(self, x):
        return self.cnn(x).view(x.size(0), -1)

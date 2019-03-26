from torchvision import models
from torch import nn

from models import base


class Model(base.ModelBase):
    def __init__(self, *args, **kwargs):
        super(Model, self).__init__(*args, **kwargs)
        self.cnn = models.densenet161(num_classes=1)
        self.activation = None
        activation_setting = kwargs.get('activation')
        if activation_setting:
            activation_args = activation_setting.get('args', {})
            self.activation = getattr(nn, activation_setting['name'])(**activation_args)


    def forward(self, x):
        y = self.cnn(x).view(x.size(0), -1)
        if self.activation:
            y = self.activation(y)
        return y

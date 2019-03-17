from models import conv_linear
from models import base


class Model(base.ModelBase):
    def __init__(self, *args, **kwargs):
        super(Model, self).__init__(*args, **kwargs)
        self.cnn = conv_linear.Model(*args, **kwargs)

    def forward(self, *input):
        return self.cnn(*input)

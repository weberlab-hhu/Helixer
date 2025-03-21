import torch
from torch import nn

# todo:rename this, options: common_model_modules?
class TransposeDimsOneTwo(nn.Module):
    def __init__(self):
        super(TransposeDimsOneTwo, self).__init__()

    def forward(self, x):
        return torch.transpose(x, 1, 2)  # swap class * len dimensions, not batch


class Reshape(nn.Module):
    def __init__(self, new_shape, *args, **kwargs):
        super(Reshape, self).__init__()
        super().__init__(*args, **kwargs)
        self.new_shape = new_shape

    def forward(self, x):
        return x.view(new_shape=self.new_shape)


class bLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, layers, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layer = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                             num_layers=layers, batch_first=True, bidirectional=True)

    def forward(self, x):
        x, _ = self.layer(x)
        return x

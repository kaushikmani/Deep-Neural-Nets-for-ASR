import torch
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self, feature_dim, label_dim, num_filters, kernel_size, drop_rate):
        super(CNNModel, self).__init__()
        self.conv1d = nn.Conv1d(feature_dim, num_filters, kernel_size, padding=(int)((kernel_size-1)/2))
        self.dense =  nn.Linear(num_filters, label_dim)
        self.dropout = nn.Dropout(p=drop_rate)
        self.mse_loss = nn.MSELoss()

    def forward(self, input):
        input = input.transpose(1,2)
        input = self.conv1d(input)
        input = input.transpose(1,2)
        input = torch.tanh(input)
        input = self.dropout(input)
        output = self.dense(input)

        return output

    def loss(self, predict, target):
        return self.mse_loss(predict, target)


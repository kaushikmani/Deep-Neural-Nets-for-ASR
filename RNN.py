import torch.nn as nn

class RNNModel(nn.Module):

    def __init__(self, feature_dim, hidden_size, num_layers, drop_rate, label_dim, biDirectional=True):
        super(RNNModel, self).__init__()
        self.rnn = nn.LSTM(feature_dim, hidden_size,num_layers,batch_first=True,bidirectional=biDirectional)
        self.dense = nn.Linear(2*hidden_size,label_dim)
        self.dropout = nn.Dropout(p=drop_rate)
        self.mseloss = nn.MSELoss()

    def forward(self, input):
        input, _ = self.rnn(input)
        input = self.dropout(input)
        output = self.dense(input)
        return output

    def loss(self, predict, target):
        return self.mseloss(predict, target)



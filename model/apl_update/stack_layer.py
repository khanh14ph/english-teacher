import torch.nn as nn

class CNN_Stack(nn.Module):
    def __init__(self, num_features, p=0.1):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(num_features=num_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=p)

    def forward(self, x):
        # x shape: batch_size x time x num_features
        x = x.unsqueeze(1)  # -> batch_size x 1 x time x num_features
        x = self.conv2d(x)  # -> batch_size x 1 x time x num_features
        x = x.squeeze(1)    # -> batch_size x time x num_features
        x = x.transpose(1, 2)
        x = self.bn(x)      # -> batch_size x time x num_features
        x = x.transpose(1, 2)
        x = self.relu(x)    # -> batch_size x time x num_features
        x = self.dropout(x) # -> batch_size x time x num_features
        return x
    
class RNN_Stack(nn.Module):
    def __init__(self, num_features_in, num_features_out, p=0.1):
        super().__init__()
        assert num_features_out % 2 == 0, 'num_features_out must be divided by 2'
        self.bi_lstm = nn.LSTM(
            input_size=num_features_in, hidden_size=num_features_out//2, bidirectional=True, batch_first=True
        )
        self.bn = nn.BatchNorm1d(num_features_out)
        self.dropout = nn.Dropout(p=p)

    def forward(self, x):
        x, _    = self.bi_lstm(x)   # batch_size x time x num_features_out
        x       = x.transpose(1, 2)
        x       = self.bn(x)        # batch_size x time x num_features_out
        x       = x.transpose(1, 2)
        x       = self.dropout(x)   # batch_size x time x num_features_out
        return x

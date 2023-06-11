import torch.nn as nn
from stack_layer import CNN_Stack, RNN_Stack

class AcousticEncoder(nn.Module):
    def __init__(self, num_features_in=81, num_features_out=256):
        super().__init__()
        self.cnn_stacks = nn.Sequential(
            CNN_Stack(num_features=num_features_in), CNN_Stack(num_features=num_features_in),
        )
        self.rnn_stacks = nn.Sequential(
            RNN_Stack(num_features_in=num_features_in, num_features_out=num_features_out),
            *[RNN_Stack(num_features_in=num_features_out, num_features_out=num_features_out) for _ in range(3)]
        )

    def forward(self, x):
        x = self.cnn_stacks(x)
        x = self.rnn_stacks(x)
        return x

class PhoneticEncoder(nn.Module):
    def __init__(self, num_features_in=1024, num_features_out=512):
        super().__init__()
        self.cnn_stacks = nn.Sequential(
            CNN_Stack(num_features=num_features_in), CNN_Stack(num_features=num_features_in),
        )
        self.rnn_stack = RNN_Stack(num_features_in=num_features_in, num_features_out=num_features_out)

    def forward(self, x):
        x = self.cnn_stacks(x)
        x = self.rnn_stack(x)
        return x

class LinguisticEncoder(nn.Module):
    def __init__(self, num_features_out=768, vocab_size=40):
        super().__init__()
        self.embedding  = nn.Embedding(vocab_size, 64)
        self.bi_lstm    = nn.LSTM(input_size=64, hidden_size=num_features_out, bidirectional=True, batch_first=True)
        self.linear     = nn.Linear(num_features_out, num_features_out)

    def forward(self, x):
        # x shape : batch_size x length_phoneme
        x           = self.embedding(x)     # batch_size x length_phoneme x 64
        _, (hk, hv) = self.bi_lstm(x)       
        hk          = hk.transpose(0, 1)
        hv          = hv.transpose(0, 1)
        hk          = self.linear(hk)       
        return hk, hv
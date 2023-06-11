import torch.nn as nn
import torch


class CNN_Stack1(nn.Module):
    def __init__(self):
        super().__init__()


        self.bn = nn.BatchNorm1d(81)
        self.Conv2d = nn.Conv2d(1,1,3,1,1)
        self.reLU = nn.ReLU()
        self.drop_out = nn.Dropout(p=0.2)


    def forward(self, x):
        # print(x.shape)
        x = (self.Conv2d(x))
        x = x.squeeze(0)
        x = torch.t(x)
        x = x.unsqueeze(0)
        x = self.bn(x)
        x = self.reLU(x)
        x = self.drop_out(x)
        
        return x

class CNN_Stack(nn.Module):
    def __init__(self):
        super().__init__()


        self.bn = nn.BatchNorm1d(81)
        self.Conv2d = nn.Conv2d(1,1,3,1,1)
        self.reLU = nn.ReLU()
        self.drop_out = nn.Dropout(p=0.2)


    def forward(self, x):
        x = (self.Conv2d(x))
        x = self.bn(x)
        x = self.reLU(x)
        x = self.drop_out(x)
        
        return x

class RNN_Stack_1(nn.Module):
    def __init__(self):
        super().__init__()
 
        self.reLU = nn.ReLU()
        self.drop_out = nn.Dropout(p=0.2)
        self.bilstm = nn.LSTM(input_size= 81, hidden_size = 384,bidirectional = True)
        self.bn = nn.BatchNorm1d(768)

    def forward(self, x):
        x = x.squeeze(0)
        x = torch.t(x)
        x = x.unsqueeze(1)
        x = self.bilstm(x)
        x = self.drop_out(x[0])
        x = x.squeeze(1)
        x = torch.t(x)
        x = x.unsqueeze(0)
        x = self.bn(x)
        return x.unsqueeze(0)


class RNN_Stack(nn.Module):
    def __init__(self):
        super().__init__()
 
        self.reLU = nn.ReLU()
        self.drop_out = nn.Dropout(p=0.2)
        self.bilstm = nn.LSTM(input_size= 768, hidden_size = 384,bidirectional = True)
        self.bn = nn.BatchNorm1d(768)

    def forward(self, x):
        x = x.squeeze(0).squeeze(0)
        x = torch.t(x)
        x = x.unsqueeze(1)
        x = self.bilstm(x)
        x = self.drop_out(x[0])
        x = x.squeeze(1)
        x = torch.t(x)
        x = x.unsqueeze(0)
        x = self.bn(x)
        return x.unsqueeze(0)
        
        
        


class Acoustic_encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.CNN1 = CNN_Stack1()
        self.CNN = CNN_Stack()
        self.RNN = RNN_Stack()
        self.RNN1 = RNN_Stack_1()
        self.RNN2 = RNN_Stack()
        self.RNN3 = RNN_Stack()

        


    def forward(self, x):
        x = self.CNN1(x)
        x = self.CNN(x)
        x = self.RNN1(x)
        x = self.RNN(x)
        x = self.RNN2(x)
        x = self.RNN3(x)
        return x
        

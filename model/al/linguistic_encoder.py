import torch.nn as nn
import torch.nn.functional as F
import torch        


class Linguistic_encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(40,64)
        # self.fc1 = nn.Linear(64, 1536)
        # self.fc2 = nn.Linear(1,64) 
        self.bilstm = nn.LSTM(input_size= 64, hidden_size = 768,bidirectional = True)
        self.fc3 = nn.Linear(768,768)
        self.bn = nn.BatchNorm1d(768)
        # self.Embedding_Layer = nn.Embedding(200, 64)

        


    def forward(self, x):
        # x = self.fc2(x)
        x = torch.t(x)  #(len(canonical) x 1)
        # x = torch.tensor(x, dtype=torch.float)
        x = self.embedding(x).squeeze(1)
        o, (h_n, c_n) = self.bilstm(x)
        # y = self.fc1(h_n)
        y = h_n
        # print(h_n.shape)
        x = self.fc3(h_n)
        return x,y
    


        
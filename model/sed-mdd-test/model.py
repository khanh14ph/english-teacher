import torch.nn as nn
import torch

class PreNet(nn.Module):
    def __init__(self, n_in, out_pre_net=512):
        super().__init__()
        self.linear1 = nn.Sequential(
            nn.Linear(n_in, out_pre_net),
            nn.Dropout(p=0.5),
            nn.ReLU()
        )
        self.linear2 = nn.Sequential(
            nn.Linear(out_pre_net, out_pre_net),
            nn.Dropout(p=0.5),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x
    

class CNN_RNN(nn.Module):
    def __init__(self, n_in, out_pre_net=512, gru_hidden_size=1024):
        super().__init__()
        self.pre_net = PreNet(n_in=n_in, out_pre_net=out_pre_net)
        self.conv1d = nn.Conv1d(in_channels=out_pre_net, out_channels=out_pre_net, kernel_size=5)
        self.bn = nn.BatchNorm1d(out_pre_net)
        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=1)
        self.gru = nn.GRU(
            input_size=out_pre_net, hidden_size=gru_hidden_size, 
            num_layers=2, batch_first=True, bidirectional=True
        )

    def forward(self, x):
        # x shape: batch_size x length x n_in
        x = self.pre_net(x) # batch_size x length x out_pre_net
        x = torch.permute(x, (0, 2, 1)) # batch_size x out_pre_net x length
        x = self.conv1d(x) # batch_size x out_pre_net x (length - 4)
        x = self.bn(x) # batch_size x out_pre_net x (length - 4)
        x = self.max_pool(x) # batch_size x out_pre_net x (length - 5)
        x = torch.permute(x, (0, 2, 1)) # batch_size x (length - 5) x out_pre_net
        x, h_n = self.gru(x) # Hk, Hv shape: 4 x batch_size x n_out
        return x
    
class SequenceLabeling(nn.Module):
    def __init__(self, n_mel, vocab_size, out_pre_net=512, n_heads_attention=16):
        super().__init__()
        self.pre_net = PreNet(n_mel, out_pre_net=out_pre_net)
        self.gru_encoder = nn.GRU(input_size=out_pre_net, hidden_size=out_pre_net, num_layers=2, batch_first=True, bidirectional=True)
        self.bn = nn.BatchNorm1d(out_pre_net*2)
        self.multihead_attention    = nn.MultiheadAttention(out_pre_net*2, n_heads_attention, dropout=0.1, batch_first=True)
        self.gru_decoder = nn.GRU(input_size=out_pre_net*4, hidden_size=out_pre_net, num_layers=2, batch_first=True, bidirectional=True)
        self.post_processing = nn.Sequential(
            CNN_RNN(out_pre_net*2, out_pre_net, out_pre_net),
            nn.Linear(out_pre_net*2, vocab_size + 1)
        )

    def forward(self, mel_spectrogram, Hk, Hv):
        # mel_spectrogram shape: batch_size x time x n_mel
        x = self.pre_net(mel_spectrogram) # batch_size x time x out_pre_net
        x, _ = self.gru_encoder(x) # batch_size x time x (out_pre_net * 2)
        x = torch.permute(x, (0, 2, 1))
        x = self.bn(x) # batch_size x (out_pre_net * 2) x time
        Hq = torch.permute(x, (0, 2, 1))
        x, _ = self.multihead_attention(Hq, Hk, Hv)
        x = torch.concat([x, Hq], dim=2)
        x, _ = self.gru_decoder(x)
        logits = self.post_processing(x)
        return logits
    
class MY_SED_MDD(nn.Module):
    def __init__(self, vocab_size, n_mel=80, n_embed=64, out_cnn_rnn_prenet=512, cnn_rnn_hidden_size=512, n_heads_attention=16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size+1, n_embed, padding_idx=vocab_size) # The last index is padding_idx
        self.cnn_rnn = CNN_RNN(n_embed, out_pre_net=out_cnn_rnn_prenet, gru_hidden_size=cnn_rnn_hidden_size)
        self.seq_labeling = SequenceLabeling(n_mel=n_mel, vocab_size=vocab_size, out_pre_net=cnn_rnn_hidden_size, n_heads_attention=n_heads_attention)
        self.linear = nn.Linear(cnn_rnn_hidden_size*2, cnn_rnn_hidden_size*2)

    def forward(self, mel_spectrogram, linguistic):
        x = self.embed(linguistic)
        x = self.cnn_rnn(x)
        Hk = self.linear(x)
        Hv = x
        logits = self.seq_labeling(mel_spectrogram, Hk, Hv)
        return logits
    
if __name__ == '__main__':
    model = MY_SED_MDD(40, n_mel=100, cnn_rnn_hidden_size=768, out_cnn_rnn_prenet=256)
    batch_size = 13
    mel_spec = torch.rand(batch_size, 300, 100)
    ling = torch.randint(0, 39, (batch_size, 30))
    a = model(mel_spec, ling)
    print(a.shape)
    print(a)
        



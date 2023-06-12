import torch.nn as nn
import torch
from encoder import AcousticEncoder, LinguisticEncoder, PhoneticEncoder
from decoder import Decoder

class APL(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.acoustic_encoder   = AcousticEncoder(81, 256)
        self.linguistic_encoder  = LinguisticEncoder(768, vocab_size)
        self.phonetic_encoder   = PhoneticEncoder(1024, 512)
        self.decoder            = Decoder(768, vocab_size)

    def forward(self, acoustic, phonetic, linguistic):
        Ha      = self.acoustic_encoder(acoustic)
        Hp      = self.phonetic_encoder(phonetic)
        Hq      = torch.concat([Ha, Hp], dim=2)
        Hk, Hv  = self.linguistic_encoder(linguistic)
        logits  = self.decoder(Hq, Hk, Hv)
        return logits



if __name__ == '__main__':
    batch_size  = 3
    acoustic    = torch.rand(batch_size, 168, 81)
    phonetic    = torch.rand(batch_size, 168, 1024)
    linguistic  = torch.randint(1, 39, size=(batch_size, 40))
    model       = APL(40)
    o           = model(acoustic, phonetic, linguistic)
    print(o)
    print(o.shape)
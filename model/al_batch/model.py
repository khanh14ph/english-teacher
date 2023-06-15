import torch.nn as nn
import torch
from encoder import AcousticEncoder, LinguisticEncoder
from decoder import Decoder

class AL(nn.Module):
    def __init__(self, vocab_size, num_features_in=81):
        super().__init__()
        self.acoustic_encoder   = AcousticEncoder(num_features_in, 768)
        self.linguistic_encoder  = LinguisticEncoder(768, vocab_size)
        self.decoder            = Decoder(768, vocab_size)

    def forward(self, acoustic, linguistic):
        Hq      = self.acoustic_encoder(acoustic)
        Hk, Hv  = self.linguistic_encoder(linguistic)
        logits  = self.decoder(Hq, Hk, Hv)
        return logits



if __name__ == '__main__':
    import time
    t1 = time.time()
    batch_size  = 1
    acoustic    = torch.rand(batch_size, 1000, 81)
    linguistic  = torch.randint(1, 39, size=(batch_size, 40))
    model       = AL(40)
    o           = model(acoustic, linguistic)
    t2 = time.time()
    print(o)
    print(o.shape)
    print(t2 - t1)
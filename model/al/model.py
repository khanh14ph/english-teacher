import torch.nn as nn
import torch
from .linguistic_encoder import Linguistic_encoder
from .acoustic_encoder import Acoustic_encoder


class Acoustic_Linguistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.Acoustic_encoder = Acoustic_encoder()
        self.Linguistic_encoder = Linguistic_encoder()
        self.fc1 = nn.Linear(1536,41, bias = True)     
        self.multihead_attn = nn.MultiheadAttention(768, 16, batch_first=True)
        

    def forward(self, acoustic, linguistic):
  
        acoustic = self.Acoustic_encoder(acoustic) #batch x time x 768
        linguistic = self.Linguistic_encoder(linguistic) # shape [0]: 1536 x len(canon)
        Hk = linguistic[0] 
        Hv = linguistic[1] 
        acoustic = acoustic.squeeze(0).squeeze(0)
        acoustic = torch.t(acoustic)
        acoustic = acoustic.unsqueeze(0)

        Hq = acoustic
        Hk = Hk.unsqueeze(0)
        Hv = Hv.unsqueeze(0)
        attn_output, attn_output_weights = self.multihead_attn(Hq, Hk, Hv)
        c = attn_output
        before_Linear = torch.cat((c,Hq), 2)
        output = self.fc1(before_Linear)

        return output.squeeze(0)
    
if __name__ == '__main__':
    import os
    current_folder = os.path.dirname(os.path.realpath(__file__))
    cp_path = os.path.join(current_folder, "checkpoints", "checkpoint.pth")
    cp_state_dict_path = os.path.join(current_folder, "checkpoints", "checkpoint_state_dict.pth")
    model = torch.load(cp_path, map_location=torch.device('cpu'))
    torch.save(model.state_dict(), cp_state_dict_path)
        
        
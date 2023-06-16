from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import os, torch, librosa, numpy as np
from python_speech_features import fbank
import torch.nn.functional as F
from .model import Acoustic_Linguistic
from util.util import text_to_phonemes, tokenizer_phonemes

current_folder = os.path.dirname(os.path.realpath(__file__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ModelInference:
    def __init__(self):
        print("AL model init")
        self.model = Acoustic_Linguistic().to(device)
        state_dict_path = os.path.join(current_folder, 'checkpoints', 'checkpoint_state_dict.pth')
        self.model.load_state_dict(torch.load(state_dict_path))
        self.model.eval()

    def infer(self, text, audio_path):
        canonical, word_phoneme_in = text_to_phonemes(text)
        linguistic = torch.tensor(tokenizer_phonemes(canonical), device=device).unsqueeze(0)
        audio, sr = librosa.load(audio_path, sr=16000)
        signal, energy = fbank(audio, sr, winlen=0.02, winstep=0.02, nfilt=80)
        fb = np.concatenate([signal, energy.reshape(-1, 1)], axis=1)
        acoustic = torch.tensor(fb, device=device, dtype=torch.float).unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(acoustic, linguistic).unsqueeze(0)
            log_proba = F.log_softmax(outputs,dim=2).squeeze(0)
        return log_proba, canonical, word_phoneme_in



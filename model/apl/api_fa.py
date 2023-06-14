from metric import align_for_force_alignment, Correct_Rate, Align
from g2p_en import G2p
from fastapi import FastAPI, Request
from pyngrok import ngrok
import os, torch, librosa, uvicorn, nest_asyncio, gc
import numpy as np
from python_speech_features import fbank
import scipy.io.wavfile as wav
from pyctcdecode import build_ctcdecoder
import torch.nn.functional as F
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from force_alignment import calculate_score

current_folder = os.path.dirname(os.path.realpath(__file__))

wav2vec2_large_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60")
wav2vec2_large_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60")

phonemes_70 = [
    'AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0',
    'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH', 'D', 'DH',
    'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1',
    'EY2', 'F', 'G', 'HH',
    'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2', 'JH', 'K', 'L',
    'M', 'N', 'NG', 'OW0', 'OW1',
    'OW2', 'OY0', 'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH',
    'UH0', 'UH1', 'UH2', 'UW',
    'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH'
]

ipa_mapping = {
    'y': 'j', 'ng': 'ŋ', 'dh': 'ð', 'w': 'w', 'er': 'ɝ', 'r': 'ɹ', 'm': 'm', 'p': 'p',
    'k': 'k', 'ah': 'ʌ', 'sh': 'ʃ', 't': 't', 'aw': 'aʊ', 'hh': 'h', 'ey': 'eɪ', 'oy': 'ɔɪ',
    'zh': 'ʒ', 'n': 'n', 'th': 'θ', 'z': 'z', 'aa': 'ɑ', 'ao': 'aʊ', 'f': 'f', 'b': 'b', 'ih': 'ɪ',
    'jh': 'dʒ', 's': 's', 'err': '', 'iy': 'i', 'uh': 'ʊ', 'ch': 'tʃ', 'g': 'g', 'ay': 'aɪ', 'l': 'l',
    'ae': 'æ', 'd': 'd', 'v': 'v', 'uw': 'u', 'eh': 'ɛ', 'ow': 'oʊ'
}

map_39 = {}
for phoneme in phonemes_70:
    phoneme_39 = phoneme.lower()
    if phoneme_39[-1].isnumeric():
        phoneme_39 = phoneme_39[:-1]
    map_39[phoneme] = phoneme_39

def text_to_phonemes(text):
    g2p = G2p()
    phonemes = g2p(text)
    word_phoneme_in = []
    phonemes_result = []
    n_word = 0
    for phoneme in phonemes:
        if map_39.get(phoneme, None) is not None:
            phonemes_result.append(map_39[phoneme])
            word_phoneme_in.append(n_word)
        elif len(phoneme.strip()) == 0:
            n_word += 1
    return ' '.join(phonemes_result), word_phoneme_in


dict_vocab = {
    "y": 0, "ng": 1, "dh": 2, "w": 3, "er": 4, "r": 5, "m": 6, "p": 7, "k": 8, "ah": 9, "sh": 10, 
    "t": 11, "aw": 12, "hh": 13, "ey": 14, "oy": 15, "zh": 16, "n": 17, "th": 18, "z": 19, "aa": 20, 
    "ao": 21, "f": 22, "b": 23, "ih": 24, "jh": 25, "s": 26, "err": 27, "iy": 28, "uh": 29, "ch": 30, 
    "g": 31, "ay": 32, "l": 33, "ae": 34, "d": 35, "v": 36, "uw": 37, "eh": 38, "ow": 39
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path_temp = os.path.join(current_folder, 'temp_audio.wav')
model = torch.load(os.path.join(current_folder, 'checkpoints', 'checkpoint.pth'))
model = model.to(device)
model.eval()

labels = sorted([w for w in list(dict_vocab.keys())], key=lambda x : dict_vocab[x])
labels = [f'{w} ' for w in labels]

def text_to_tensor(text):
    text = text.lower()
    text = text.split(" ")
    text_list = []
    for idex in text:
        text_list.append(dict_vocab[idex])
    return text_list

def get_filterbank(path):
    (rate,sig) = wav.read(path)
    filter, energy = fbank(sig,rate, winlen=0.032, winstep = 0.02, nfilt=80)
    filter = filter.reshape(80, -1)
    energy = energy.reshape(1,-1)
    data = np.concatenate((filter,energy))
    return data

wav2vec2_large_submodel = torch.nn.Sequential(*(list(wav2vec2_large_model.children())[:-2])).to(device)
wav2vec2_large_submodel.eval()

def en_phonetic_extract(path):
    with torch.no_grad():
        path = path
        wav, sr = librosa.load(path, sr=16000)
        input_values = wav2vec2_large_processor(wav, return_tensors="pt",sampling_rate=16000, padding="longest").input_values
        input_values = input_values.to(device)     
        outputs = wav2vec2_large_submodel(input_values)
    return outputs.last_hidden_state.squeeze(0).detach().cpu().numpy()

def run_model(text, audio_path):
    with torch.no_grad():
        phonemes, word_phoneme_in = text_to_phonemes(text)

        linguistic = torch.tensor(text_to_tensor(phonemes), device=device).unsqueeze(0)
        phonetic = torch.tensor(en_phonetic_extract(audio_path), device=device, dtype=torch.float)
        fbank = get_filterbank(audio_path)
        rshape_fbank = fbank[:,:phonetic.shape[0]]
        acoustic = torch.tensor(rshape_fbank.T, device=device, dtype=torch.float).unsqueeze(0)
        phonetic = phonetic.unsqueeze(0)
        
        outputs = model(acoustic, phonetic, linguistic)
        x = F.log_softmax(outputs,dim=2).squeeze(0)
        x1 = x.detach().cpu()
        x = x1.numpy()
        torch.cuda.empty_cache()
        gc.collect()
        decoder = build_ctcdecoder(
            labels = labels,
            kenlm_model_path = os.path.join(current_folder, 'text.arpa')
        )
        hypothesis = str(decoder.decode(x)).strip()
        hyp_score = [(p, 1) for p in hypothesis.split()]
        pho_score = calculate_score(x1, phonemes.split(), dict_vocab)
        cnt, l, temp = Correct_Rate(phonemes.split(), hypothesis.split())
        correct_rate = 1 - cnt/l
        print(phonemes)
        print(pho_score)
        print(hypothesis)
        print(hyp_score)
        return pho_score, hyp_score, word_phoneme_in, correct_rate

app = FastAPI()

@app.post('/phonemes')
async def get_phoneme(request: Request):
    form_data: bytes = await request.form()
    text = form_data['text']
    phonemes, word_phoneme_in = text_to_phonemes(text)
    phonemes = phonemes.split()
    result = ''
    for i in range(len(phonemes)):
        if i > 0 and word_phoneme_in[i] > word_phoneme_in[i - 1]:
            result += ' '
        result += ipa_mapping[phonemes[i]]
    return {'phonetics':f'{result}'}

@app.post('/predict')
async def predict(request: Request):
    form_data: bytes = await request.form()
    text = form_data['text']
    byte_content = await form_data['audio'].read()
    with open(path_temp, 'wb') as f:
        f.write(byte_content)
    pho_score, hyp_score, word_phoneme_in, correct_rate = run_model(text, path_temp)
    pho_score, hyp_score = align_for_force_alignment(pho_score, hyp_score)

    result = [] # right_phoneme, model_predict_phoneme, right_phoneme_score, predict_score
    n = -1
    for i in range(len(pho_score)):
        if pho_score[i] != '<eps>':
            phoneme, score = pho_score[i]
            n += 1
            if n == 0 or word_phoneme_in[n] > word_phoneme_in[n - 1]:
                result.append([])
            if isinstance(hyp_score[i], tuple):
                pred, predict_score = hyp_score[i]
            else:
                pred, predict_score = "<unk>", 0
            result[-1].append((
                ipa_mapping.get(phoneme, ''),
                ipa_mapping.get(pred, ''),
                score,
                predict_score
            ))
       
    return {'correct_rate': str(correct_rate), 'phoneme_result': str(result)}

def run_api(auth_token=None):
    if auth_token is not None:
        ngrok.set_auth_token(auth_token)
        ngrok_tunnel = ngrok.connect(8085) 
        print("public url: ", ngrok_tunnel.public_url)
        nest_asyncio.apply()
        print(f"{'-'*50}RUN API ONLINE{'-'*50}")
    else:
        print(f"{'-'*50}RUN API LOCAL{'-'*50}")

    uvicorn.run(app, port=8085)
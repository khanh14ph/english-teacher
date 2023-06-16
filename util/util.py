from g2p_en import G2p
from pyctcdecode import build_ctcdecoder
from .force_alignment import calculate_score
from .metric import Correct_Rate, align_for_force_alignment

dict_vocab = {
    "y": 0, "ng": 1, "dh": 2, "w": 3, "er": 4, "r": 5, "m": 6, "p": 7, "k": 8, "ah": 9, "sh": 10, 
    "t": 11, "aw": 12, "hh": 13, "ey": 14, "oy": 15, "zh": 16, "n": 17, "th": 18, "z": 19, "aa": 20, 
    "ao": 21, "f": 22, "b": 23, "ih": 24, "jh": 25, "s": 26, "err": 27, "iy": 28, "uh": 29, "ch": 30, 
    "g": 31, "ay": 32, "l": 33, "ae": 34, "d": 35, "v": 36, "uw": 37, "eh": 38, "ow": 39
}


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


labels = sorted([w for w in list(dict_vocab.keys())], key=lambda x : dict_vocab[x])
labels = [f'{w} ' for w in labels]


def text_to_phonemes(text):
    g2p = G2p()
    phonemes = g2p(text.lower())
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


def get_phoneme_ipa_form(text):
    phonemes, word_phoneme_in = text_to_phonemes(text.lower())
    phonemes = phonemes.split()
    result = ''
    for i in range(len(phonemes)):
        if i > 0 and word_phoneme_in[i] > word_phoneme_in[i - 1]:
            result += ' '
        result += ipa_mapping[phonemes[i]]
    return {'phonetics': result}


def tokenizer_phonemes(phonemes):
    text = phonemes.lower()
    text = text.split(" ")
    text_list = []
    for idex in text:
        text_list.append(dict_vocab[idex])
    return text_list


decoder = build_ctcdecoder(
    labels = labels,
)

def decode(log_proba):
    return str(decoder.decode(log_proba)).strip()

def generate_mdd_for_app(log_proba, canonical, word_phoneme_in):
    emission = log_proba.detach().cpu()
    hypothesis = decode(emission.numpy()).split()
    canonical = canonical.split()

    hypothesis_score = calculate_score(emission, hypothesis, dict_vocab)
    canonical_score = calculate_score(emission, canonical, dict_vocab)
    hypothesis_score, canonical_score = align_for_force_alignment(hypothesis_score, canonical_score)

    cnt, l, temp = Correct_Rate(canonical, hypothesis)
    correct_rate = 1 - cnt/l if l != 0 else 0

    result = [] # canonical, predict_phoneme, canonical_score, predict_score
    n = -1
    for i in range(len(canonical_score)):
        if canonical_score[i] != '<eps>':
            phoneme, score = canonical_score[i]
            n += 1
            if n == 0 or word_phoneme_in[n] > word_phoneme_in[n - 1]:
                result.append([])
            if isinstance(hypothesis_score[i], tuple):
                pred, predict_score = hypothesis_score[i]
            else:
                pred, predict_score = "<unk>", 0
            result[-1].append((
                ipa_mapping.get(phoneme, ''),
                ipa_mapping.get(pred, ''),
                score,
                predict_score
            ))
    
    return {
        'correct_rate': str(correct_rate),
        'phoneme_result': str(result)
    }
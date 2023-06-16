from fastapi import FastAPI, Request
import uvicorn, nest_asyncio, os, argparse
from util.util import get_phoneme_ipa_form, generate_mdd_for_app
from pyngrok import ngrok, conf

parser = argparse.ArgumentParser(description='Model Live')
parser.add_argument("--model", default="apl", type=str)
parser.add_argument("--auth_token", default="", type=str)
parser.add_argument("--region", default="ap", type=str)
args = parser.parse_args()

model_name = args.model.lower().strip()
if model_name == 'apl':
    from model.apl.infer import ModelInference
    model = ModelInference()
else:
    from model.al.infer import ModelInference
    model = ModelInference()

current_folder = os.path.dirname(os.path.realpath(__file__))
audio_folder = os.path.join(current_folder, 'audio')
if not os.path.exists(audio_folder):
    os.mkdir(audio_folder)

audio_path = os.path.join(audio_folder, 'audio.wav')

app = FastAPI()
@app.post('/phonemes')
async def get_phoneme(request: Request):
    form_data: bytes = await request.form()
    text = form_data['text']
    return get_phoneme_ipa_form(text)

@app.post('/predict')
async def predict(request: Request):
    form_data: bytes = await request.form()
    text = form_data['text']
    byte_content = await form_data['audio'].read()
    with open(audio_path, 'wb') as f:
        f.write(byte_content)
    log_proba, canonical, word_phoneme_in = model.infer(text, audio_path)
    return generate_mdd_for_app(log_proba, canonical, word_phoneme_in)

ngrok.set_auth_token(args.auth_token.strip())
conf.get_default().region = args.region
ngrok_tunnel = ngrok.connect(8085) 
print("public url: ", ngrok_tunnel.public_url)
uvicorn.run(app, port=8085)
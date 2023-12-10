import logging
import os
import subprocess

import langid
import torch
import torchaudio
import whisper

langid.set_languages(["en", "zh", "ja"])

import numpy as np

from vallex.data.collation import get_text_token_collater
from vallex.data.tokenizer import AudioTokenizer, tokenize_audio
from vallex.macros import *
from vallex.utils.g2p import PhonemeBpeTokenizer

DIR_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__)))

text_tokenizer = PhonemeBpeTokenizer(tokenizer_path=f"{DIR_PATH}/g2p/bpe_69.json")
text_collater = get_text_token_collater()

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda", 0)
if torch.backends.mps.is_available():
    device = torch.device("mps")
codec = AudioTokenizer(device)

if not os.path.exists(f"{DIR_PATH}/../whisper/"):
    os.mkdir(f"{DIR_PATH}/../whisper/")
whisper_model = None


@torch.no_grad()
def transcribe_one(model, audio_path):
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")
    lang = max(probs, key=probs.get)
    # decode the audio
    options = whisper.DecodingOptions(
        temperature=1.0,
        best_of=5,
        fp16=False if device == torch.device("cpu") else True,
        sample_len=150,
    )
    result = whisper.decode(model, mel, options)

    # print the recognized text
    print(result.text)

    text_pr = result.text
    if text_pr.strip(" ")[-1] not in "?!.,。，？！。、":
        text_pr += "."
    return lang, text_pr


def make_prompt(name, input_audio_path, output_prompt_path, transcript=None):
    global model, text_collater, text_tokenizer, codec
    wav_pr, sr = torchaudio.load(input_audio_path)
    # check length
    if wav_pr.size(-1) / sr > 15:
        raise ValueError(
            f"Prompt too long, expect length below 15 seconds, got {wav_pr / sr} seconds."
        )
    if wav_pr.size(0) == 2:
        wav_pr = wav_pr.mean(0, keepdim=True)
    text_pr, lang_pr = make_transcript(name, wav_pr, sr, transcript)

    # tokenize audio
    encoded_frames = tokenize_audio(codec, (wav_pr, sr))
    audio_tokens = encoded_frames[0][0].transpose(2, 1).cpu().numpy()

    # tokenize text
    phonemes, langs = text_tokenizer.tokenize(text=f"{text_pr}".strip())
    text_tokens, enroll_x_lens = text_collater([phonemes])

    message = f"Detected language: {lang_pr}\n Detected text {text_pr}\n"

    # save as npz file
    np.savez(
        output_prompt_path,
        audio_tokens=audio_tokens,
        text_tokens=text_tokens,
        lang_code=lang2code[lang_pr],
    )
    logging.info(f"Successful. Prompt saved to {output_prompt_path}")


def make_transcript(name, wav, sr, transcript=None):
    if not isinstance(wav, torch.FloatTensor):
        wav = torch.tensor(wav)
    if wav.abs().max() > 1:
        wav /= wav.abs().max()
    if wav.size(-1) == 2:
        wav = wav.mean(-1, keepdim=False)
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    assert wav.ndim and wav.size(0) == 1
    if transcript is None or transcript == "":
        logging.info("Transcript not given, using Whisper...")
        global whisper_model
        if whisper_model is None:
            whisper_model = whisper.load_model(
                "medium", download_root=os.path.join(os.getcwd(), "whisper")
            )
        whisper_model.to(device)
        torchaudio.save(f"{DIR_PATH}/../prompts/{name}.wav", wav, sr)
        lang, text = transcribe_one(whisper_model, f"{DIR_PATH}/../prompts/{name}.wav")
        lang_token = lang2token[lang]
        text = lang_token + text + lang_token
        os.remove(f"{DIR_PATH}/../prompts/{name}.wav")
        whisper_model.cpu()
    else:
        text = transcript
        lang, _ = langid.classify(text)
        lang_token = lang2token[lang]
        text = lang_token + text + lang_token

    torch.cuda.empty_cache()
    return text, lang

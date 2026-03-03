# Voice Cloning Prototype

This folder is a focused voice-cloning extraction from `Qwen3-TTS`, containing only:

- `app.py`: CLI app to clone voice from a reference audio clip
- `download_model.py`: downloads only the Base voice-clone model + tokenizer

## Setup

```bash
cd voice-cloning
uv venv .venv
uv pip install -U pip
uv pip install -r requirements.txt
```

## Download Model Weights

```bash
uv run --python .venv/bin/python download_model.py
```

This downloads to:

- `models/Qwen3-TTS-12Hz-1.7B-Base`
- `models/Qwen3-TTS-Tokenizer-12Hz`

## Run GUI

Launch the web UI:

```bash
uv run --python .venv/bin/python gui_app.py --host 127.0.0.1 --port 7860
```

Then open:

- `http://127.0.0.1:7860`

GUI flow:

- Upload or record reference audio
- Add reference transcript (optional, but improves quality)
- Enter target text to synthesize
- Generate cloned speech
- Swipe/seek in waveform player and download the output WAV

## Run Voice Cloning

With transcript (best quality):

```bash
uv run --python .venv/bin/python app.py \
  --ref-audio inputs/my_voice.m4a \
  --ref-text "Hello, this is my reference voice sample." \
  --text "This sentence is generated in my cloned voice." \
  --language English \
  --output outputs/my_clone.wav
```

Without transcript (automatic x-vector-only fallback):

```bash
uv run --python .venv/bin/python app.py \
  --ref-audio inputs/my_voice.m4a \
  --text "This is a fast clone using speaker embedding only." \
  --language English \
  --output outputs/my_clone_xvec.wav
```

## Standalone Notes

- This project is standalone and does not require a sibling `Qwen3-TTS` checkout.
- Runtime dependency is installed from PyPI via `qwen-tts` in `requirements.txt`.
- Models are loaded from `models/` if present, otherwise from Hugging Face model id fallback.

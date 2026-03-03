# Voice Cloning Prototype

This folder is a focused voice-cloning extraction from `Qwen3-TTS`, containing only:

- `app.py`: CLI app to clone voice from a reference audio clip
- `download_model.py`: downloads Base voice-clone model + tokenizer + Whisper STT model

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
- `models/whisper-small`

Optional flags:

- `--skip-stt` to skip Whisper download
- `--stt-model-id openai/whisper-small` to choose a different STT repo

## Run GUI

Launch the web UI:

```bash
uv run --python .venv/bin/python gui_app.py --host 127.0.0.1 --port 7860
```

Then open:

- `http://127.0.0.1:7860`

GUI flow:

- Upload or record reference audio
- Set **Language** (for most English clips, keep it as `English`; use `Auto` to detect language)
- Click **Auto Transcribe Reference Audio** to fill transcript from the audio
- Edit the transcript manually if needed (it stays editable)
- Enter target text to synthesize
- Generate cloned speech
- Swipe/seek in waveform player and download the output WAV

Notes:

- Auto-transcribe defaults to local `models/whisper-small` if it exists, otherwise `openai/whisper-small`.
- For long audio (>30s), timestamps are enabled automatically for Whisper long-form transcription.
- For `.m4a` and other compressed formats, make sure `ffmpeg` is installed (`ffmpeg`/`ffprobe` in `PATH`).

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

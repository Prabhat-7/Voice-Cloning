from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

BASE_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
TOKENIZER_ID = "Qwen/Qwen3-TTS-Tokenizer-12Hz"
STT_MODEL_ID = "openai/whisper-small"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download voice-cloning model + tokenizer locally")
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Directory where model repos are downloaded.",
    )
    parser.add_argument(
        "--skip-tokenizer",
        action="store_true",
        help="Skip tokenizer download.",
    )
    parser.add_argument(
        "--stt-model-id",
        default=STT_MODEL_ID,
        help="STT model repo id to download for auto transcription.",
    )
    parser.add_argument(
        "--skip-stt",
        action="store_true",
        help="Skip STT model download.",
    )
    return parser.parse_args()


def target_dir(models_dir: Path, repo_id: str) -> Path:
    return models_dir / repo_id.split("/")[-1]


def download_repo(repo_id: str, models_dir: Path) -> Path:
    out_dir = target_dir(models_dir, repo_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo_id, local_dir=out_dir.as_posix())
    return out_dir


def main() -> None:
    args = parse_args()
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {BASE_MODEL_ID} ...")
    model_path = download_repo(BASE_MODEL_ID, models_dir)
    print(f"Saved model to: {model_path.resolve()}")

    if not args.skip_tokenizer:
        print(f"Downloading {TOKENIZER_ID} ...")
        tokenizer_path = download_repo(TOKENIZER_ID, models_dir)
        print(f"Saved tokenizer to: {tokenizer_path.resolve()}")

    if not args.skip_stt:
        print(f"Downloading {args.stt_model_id} ...")
        stt_path = download_repo(args.stt_model_id, models_dir)
        print(f"Saved STT model to: {stt_path.resolve()}")


if __name__ == "__main__":
    main()

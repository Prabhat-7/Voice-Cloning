from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf
import torch
try:
    from qwen_tts import Qwen3TTSModel
    from qwen_tts.inference.mlx_hybrid import MLXHybridConfig, enable_mlx_hybrid_decoder
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'qwen-tts'. Install with: pip install -r requirements.txt"
    ) from exc

DEFAULT_HF_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
DEFAULT_MODEL_DIR = Path("models") / "Qwen3-TTS-12Hz-1.7B-Base"
DEFAULT_OUTPUT = Path("outputs") / "voice_clone.wav"


def detect_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def resolve_dtype(dtype_arg: str, device: str) -> torch.dtype:
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    if dtype_arg == "float32":
        return torch.float32

    if device.startswith("cuda") or device == "mps":
        return torch.float16
    return torch.float32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3-TTS voice-cloning prototype")
    parser.add_argument("--text", required=True, help="Target text to synthesize using cloned voice.")
    parser.add_argument("--language", default="English", help="Language, e.g. English/Chinese/Auto")
    parser.add_argument(
        "--ref-audio",
        "--input-audio",
        dest="ref_audio",
        required=True,
        help="Reference audio path/URL/base64 for voice cloning.",
    )
    parser.add_argument(
        "--ref-text",
        default="",
        help="Transcript of reference audio. If omitted, x-vector-only mode is enabled automatically.",
    )
    parser.add_argument(
        "--voice-description",
        "--instruct",
        dest="voice_description",
        default="",
        help="Optional style instruction for generation.",
    )
    parser.add_argument(
        "--x-vector-only-mode",
        action="store_true",
        help="Force speaker-embedding-only mode (reference transcript not required).",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT.as_posix(), help="Path for generated wav.")
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR.as_posix(),
        help="Local model folder. If missing, falls back to Hugging Face model id.",
    )
    parser.add_argument("--hf-model", default=DEFAULT_HF_MODEL, help="Hugging Face model id fallback.")
    parser.add_argument("--device", default="auto", help="auto, mps, cpu, cuda:0, ...")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument(
        "--mlx-hybrid",
        action="store_true",
        help="Enable MLX hybrid decoder acceleration (Apple Silicon, experimental).",
    )
    parser.add_argument(
        "--mlx-disable-quantizer",
        action="store_true",
        help="When --mlx-hybrid is enabled, keep quantizer on PyTorch and use MLX for decoder blocks only.",
    )
    parser.add_argument(
        "--eris-src-dir",
        default="eris-voice/src",
        help="Path to eris-voice src directory for MLX modules.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = detect_device() if args.device == "auto" else args.device
    dtype = resolve_dtype(args.dtype, device)
    ref_text = args.ref_text.strip()
    use_x_vector_only_mode = args.x_vector_only_mode or ref_text == ""

    local_model_path = Path(args.model_dir)
    model_source = local_model_path.as_posix() if local_model_path.exists() else args.hf_model

    print(f"Loading model from: {model_source}")
    print(f"Device: {device} | DType: {dtype}")
    if use_x_vector_only_mode and ref_text == "":
        print("No --ref-text provided. Falling back to x-vector-only mode.")

    model = Qwen3TTSModel.from_pretrained(
        model_source,
        device_map=device,
        dtype=dtype,
    )
    if args.mlx_hybrid:
        try:
            message = enable_mlx_hybrid_decoder(
                model,
                config=MLXHybridConfig(
                    use_mlx_quantizer=not args.mlx_disable_quantizer,
                    eris_src_dir=args.eris_src_dir,
                ),
            )
            print(message)
        except Exception as exc:
            raise SystemExit(f"Failed to enable MLX hybrid decoder: {type(exc).__name__}: {exc}") from exc

    wavs, sample_rate = model.generate_voice_clone(
        text=args.text,
        language=args.language,
        ref_audio=args.ref_audio,
        ref_text=ref_text if ref_text else None,
        instruct=args.voice_description.strip() if args.voice_description else None,
        x_vector_only_mode=use_x_vector_only_mode,
        non_streaming_mode=True,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path.as_posix(), wavs[0], sample_rate)
    print(f"Saved audio to: {output_path.resolve()}")


if __name__ == "__main__":
    main()

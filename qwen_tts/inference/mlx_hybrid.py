from __future__ import annotations

import importlib
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class MLXHybridConfig:
    use_mlx_quantizer: bool = True
    eris_src_dir: str = "eris-voice/src"


def _resolve_eris_src_dir(eris_src_dir: str) -> Path:
    candidate = Path(eris_src_dir).expanduser()
    if not candidate.is_absolute():
        # .../qwen_tts/inference/mlx_hybrid.py -> project root
        project_root = Path(__file__).resolve().parents[2]
        candidate = project_root / candidate
    return candidate.resolve()


def _validate_platform() -> None:
    if platform.system() != "Darwin":
        raise RuntimeError("MLX hybrid decode is only supported on macOS.")
    machine = platform.machine().lower()
    if machine not in {"arm64", "aarch64"}:
        raise RuntimeError("MLX hybrid decode requires Apple Silicon (arm64).")


def _import_eris_modules(eris_src_dir: Path) -> dict[str, Any]:
    if not eris_src_dir.exists():
        raise RuntimeError(
            f"Eris source directory not found: {eris_src_dir}. "
            "Clone https://github.com/eris-ths/eris-voice and point to its src directory."
        )

    eris_src = eris_src_dir.as_posix()
    if eris_src not in sys.path:
        sys.path.insert(0, eris_src)

    try:
        import mlx.core as mx
    except Exception as exc:
        raise RuntimeError("Python package 'mlx' is required. Install it with: pip install mlx") from exc

    try:
        decoder_mod = importlib.import_module("mlx_decoder_v2")
        quantizer_mod = importlib.import_module("mlx_quantizer")
        converter_mod = importlib.import_module("weight_converter")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to import Eris MLX modules from {eris_src_dir}. "
            "Expected files: mlx_decoder_v2.py, mlx_quantizer.py, weight_converter.py."
        ) from exc

    return {
        "mx": mx,
        "decoder_mod": decoder_mod,
        "quantizer_mod": quantizer_mod,
        "converter_mod": converter_mod,
    }


def enable_mlx_hybrid_decoder(
    tts_model: Any,
    *,
    config: MLXHybridConfig | None = None,
) -> str:
    """
    Patch Qwen3TTSModel speech tokenizer decoder:
    - quantizer decode -> MLX (optional)
    - decoder blocks -> MLX

    This preserves the existing generation APIs, including voice cloning.
    """
    if config is None:
        config = MLXHybridConfig()

    existing = getattr(tts_model, "_mlx_hybrid_state", None)
    if existing and existing.get("enabled"):
        same_quantizer = bool(existing.get("use_mlx_quantizer")) == bool(config.use_mlx_quantizer)
        same_src = str(existing.get("eris_src_dir")) == str(config.eris_src_dir)
        if same_quantizer and same_src:
            return existing.get("message", "MLX hybrid decode already enabled.")
        raise RuntimeError(
            "MLX hybrid decode is already enabled with different settings. "
            "Reload the model instance to change configuration."
        )

    _validate_platform()

    tokenizer_model = tts_model.model.speech_tokenizer.model
    get_model_type = getattr(tokenizer_model, "get_model_type", None)
    tokenizer_type = get_model_type() if callable(get_model_type) else None
    if tokenizer_type != "qwen3_tts_tokenizer_12hz":
        raise RuntimeError(
            f"MLX hybrid decode currently supports tokenizer type 'qwen3_tts_tokenizer_12hz' only (got {tokenizer_type!r})."
        )

    eris_src_dir = _resolve_eris_src_dir(config.eris_src_dir)
    modules = _import_eris_modules(eris_src_dir)
    mx = modules["mx"]
    decoder_mod = modules["decoder_mod"]
    quantizer_mod = modules["quantizer_mod"]
    converter_mod = modules["converter_mod"]

    pt_decoder = tokenizer_model.decoder

    # Convert and load decoder weights from current model.
    mlx_decoder = decoder_mod.Qwen3TTSDecoderMLX()
    decoder_weights = converter_mod.extract_decoder_weights(tts_model)
    mlx_decoder.load_weights(decoder_weights)

    mlx_quantizer = None
    if config.use_mlx_quantizer:
        mlx_quantizer = quantizer_mod.SplitResidualVectorQuantizerMLX(
            n_q_semantic=1,
            total_quantizers=pt_decoder.config.num_quantizers,
            codebook_size=pt_decoder.config.codebook_size,
            input_dim=pt_decoder.config.codebook_dim,
            codebook_dim=pt_decoder.config.codebook_dim // 2,
        )
        quantizer_weights = converter_mod.extract_quantizer_weights(tts_model)
        mlx_quantizer.load_weights(quantizer_weights)

    decoder_device = pt_decoder.pre_conv.conv.weight.device
    output_dtype = pt_decoder.pre_conv.conv.weight.dtype
    original_forward = pt_decoder.forward

    def hybrid_forward(codes: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if config.use_mlx_quantizer and mlx_quantizer is not None:
                codes_np = codes.detach().to(device="cpu", dtype=torch.int64).numpy()
                hidden_mlx = mlx_quantizer.decode(mx.array(codes_np))
                mx.eval(hidden_mlx)
                hidden = torch.from_numpy(np.array(hidden_mlx))
            else:
                hidden = pt_decoder.quantizer.decode(codes)

            hidden = hidden.to(device=decoder_device, dtype=output_dtype)
            hidden = pt_decoder.pre_conv(hidden).transpose(1, 2)

            if pt_decoder.pre_transformer is not None:
                hidden = pt_decoder.pre_transformer(inputs_embeds=hidden).last_hidden_state
                hidden = hidden.permute(0, 2, 1)

            for blocks in pt_decoder.upsample:
                for block in blocks:
                    hidden = block(hidden)

        hidden_cpu = hidden.detach().to(device="cpu", dtype=torch.float32)
        hidden_mlx = mx.array(hidden_cpu.numpy())
        wav_mlx = mlx_decoder.decoder_conv0(hidden_mlx)
        for block in mlx_decoder.decoder_blocks:
            wav_mlx = block(wav_mlx)
        wav_mlx = mlx_decoder.final_act(wav_mlx)
        wav_mlx = mlx_decoder.final_conv(wav_mlx)
        wav_mlx = mx.clip(wav_mlx, -1.0, 1.0)
        mx.eval(wav_mlx)

        wav = torch.from_numpy(np.array(wav_mlx))
        return wav.to(device=codes.device, dtype=output_dtype)

    pt_decoder.forward = hybrid_forward

    message = (
        "Enabled MLX hybrid decode "
        f"(decoder=mlx, quantizer={'mlx' if config.use_mlx_quantizer else 'pytorch'}) "
        f"from {eris_src_dir}"
    )
    tts_model._mlx_hybrid_state = {
        "enabled": True,
        "message": message,
        "original_forward": original_forward,
        "use_mlx_quantizer": bool(config.use_mlx_quantizer),
        "eris_src_dir": config.eris_src_dir,
        "resolved_eris_src_dir": eris_src_dir.as_posix(),
    }
    return message

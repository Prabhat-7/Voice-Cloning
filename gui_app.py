from __future__ import annotations

import argparse
import socket
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import gradio as gr
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
DEFAULT_OUTPUT_DIR = Path("outputs")
DEFAULT_STT_HF_MODEL = "openai/whisper-small"
DEFAULT_STT_MODEL_DIR = Path("models") / "whisper-small"
NETWORK_ERROR_HINTS = (
    "failed to resolve",
    "max retries exceeded",
    "connection error",
    "temporarily unavailable",
)
VOICE_CLONE_PROMPT_CACHE_MAX = 8
VOICE_CLONE_PROMPT_CACHE: dict[tuple[Any, ...], Any] = {}


def resolve_default_stt_model() -> str:
    local_stt_path = DEFAULT_STT_MODEL_DIR.expanduser()
    if local_stt_path.exists():
        return local_stt_path.as_posix()
    return DEFAULT_STT_HF_MODEL


def detect_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def resolve_dtype_name(dtype_arg: str, device: str) -> str:
    dtype_arg = dtype_arg.strip().lower()
    if dtype_arg in {"float16", "bfloat16", "float32"}:
        return dtype_arg
    if device.startswith("cuda") or device == "mps":
        return "float16"
    return "float32"


def dtype_from_name(dtype_name: str) -> torch.dtype:
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    return torch.float32


@lru_cache(maxsize=4)
def load_model(
    model_source: str,
    device: str,
    dtype_name: str,
    use_mlx_hybrid: bool,
    use_mlx_quantizer: bool,
    eris_src_dir: str,
) -> Qwen3TTSModel:
    model = Qwen3TTSModel.from_pretrained(
        model_source,
        device_map=device,
        dtype=dtype_from_name(dtype_name),
    )
    if use_mlx_hybrid:
        enable_mlx_hybrid_decoder(
            model,
            config=MLXHybridConfig(
                use_mlx_quantizer=use_mlx_quantizer,
                eris_src_dir=eris_src_dir,
            ),
        )
    return model


def resolve_asr_device(device: str) -> str | int:
    if device.startswith("cuda"):
        if ":" in device:
            return int(device.split(":", 1)[1])
        return 0
    # Whisper on MPS is less stable than CPU on common local setups.
    if device == "mps":
        return -1
    if device == "cpu":
        return -1
    return device


def resolve_asr_dtype_name(dtype_arg: str, device: str) -> str:
    if device in {"cpu", "mps"}:
        return "float32"
    return resolve_dtype_name(dtype_arg, device)


def has_hf_connectivity(timeout_seconds: float = 1.0) -> bool:
    try:
        with socket.create_connection(("huggingface.co", 443), timeout=timeout_seconds):
            return True
    except OSError:
        return False


def create_stt_pipeline(stt_model_source: str, device: str, dtype_name: str):
    from transformers import pipeline

    asr_device = resolve_asr_device(device)
    dtype_value = dtype_from_name(dtype_name)
    try:
        return pipeline(
            task="automatic-speech-recognition",
            model=stt_model_source,
            device=asr_device,
            dtype=dtype_value,
        )
    except TypeError:
        return pipeline(
            task="automatic-speech-recognition",
            model=stt_model_source,
            device=asr_device,
            torch_dtype=dtype_value,
        )


@lru_cache(maxsize=4)
def load_stt_pipeline(stt_model_source: str, device: str, dtype_name: str):
    try:
        import transformers  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Missing dependency 'transformers'. Install with: pip install -r requirements.txt") from exc

    stt_source = stt_model_source.strip()
    source_path = Path(stt_source).expanduser()

    # Load from local HF cache first to avoid long retry loops when offline.
    if not source_path.exists() and "/" in stt_source and not stt_source.startswith(("http://", "https://")):
        try:
            from huggingface_hub import snapshot_download

            stt_source = snapshot_download(repo_id=stt_source, local_files_only=True)
        except Exception:
            if not has_hf_connectivity():
                raise RuntimeError(
                    f"STT model '{stt_source}' is not in local cache and Hugging Face is unreachable."
                )

    if not Path(stt_source).expanduser().exists() and not has_hf_connectivity():
        try:
            from huggingface_hub import snapshot_download

            # If this succeeds, it returns a cached path and avoids network requests.
            stt_source = snapshot_download(repo_id=stt_source, local_files_only=True)
        except Exception as exc:
            raise RuntimeError(
                f"STT model '{stt_source}' is not in local cache and Hugging Face is unreachable."
            ) from exc

    return create_stt_pipeline(stt_source, device, dtype_name)


def normalize_audio_path(ref_audio_path: Any) -> str | None:
    if isinstance(ref_audio_path, str):
        return ref_audio_path
    if isinstance(ref_audio_path, dict):
        path_value = ref_audio_path.get("path")
        if isinstance(path_value, str):
            return path_value
    return None


def make_audio_signature(audio_path: str | None) -> str:
    if not audio_path:
        return ""
    p = Path(audio_path).expanduser()
    if not p.exists():
        return audio_path
    try:
        stat = p.stat()
    except OSError:
        return p.resolve().as_posix()
    return f"{p.resolve().as_posix()}::{stat.st_mtime_ns}::{stat.st_size}"


def get_cached_voice_clone_prompt(
    model: Qwen3TTSModel,
    ref_audio_path: str,
    clean_ref_text: str,
    use_x_vector_only_mode: bool,
):
    ref_text_for_prompt = None if use_x_vector_only_mode else clean_ref_text
    cache_key = (
        id(model),
        make_audio_signature(ref_audio_path),
        ref_text_for_prompt or "",
        bool(use_x_vector_only_mode),
    )
    cached = VOICE_CLONE_PROMPT_CACHE.get(cache_key)
    if cached is not None:
        return cached, True

    prompt_items = model.create_voice_clone_prompt(
        ref_audio=ref_audio_path,
        ref_text=ref_text_for_prompt,
        x_vector_only_mode=use_x_vector_only_mode,
    )

    if len(VOICE_CLONE_PROMPT_CACHE) >= VOICE_CLONE_PROMPT_CACHE_MAX:
        oldest_key = next(iter(VOICE_CLONE_PROMPT_CACHE))
        VOICE_CLONE_PROMPT_CACHE.pop(oldest_key, None)
    VOICE_CLONE_PROMPT_CACHE[cache_key] = prompt_items
    return prompt_items, False


def resolve_transcription_language(language: str) -> str | None:
    normalized = (language or "").strip().lower()
    if not normalized:
        return "english"
    if normalized in {"auto", "automatic", "autodetect", "auto-detect", "detect"}:
        return None

    alias_map = {
        "en": "english",
        "en-us": "english",
        "en-gb": "english",
        "zh": "chinese",
        "zh-cn": "chinese",
        "zh-tw": "chinese",
        "ja": "japanese",
        "ko": "korean",
        "de": "german",
        "fr": "french",
        "es": "spanish",
        "pt": "portuguese",
        "ru": "russian",
        "it": "italian",
    }
    return alias_map.get(normalized, normalized)


def load_audio_for_stt(audio_path: str) -> tuple[Any, float]:
    try:
        import numpy as np
        from pydub import AudioSegment
    except ImportError as exc:
        raise RuntimeError("Missing dependency 'pydub'. Install with: pip install pydub") from exc

    audio = AudioSegment.from_file(audio_path)
    duration_seconds = float(len(audio)) / 1000.0

    # Normalize to Whisper-native format to avoid unstable internal resampling paths.
    normalized = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
    samples = np.array(normalized.get_array_of_samples(), dtype="float32")
    if samples.size == 0:
        raise RuntimeError("Reference audio is empty after decoding.")

    samples = samples / float(1 << 15)
    samples = np.clip(samples, -1.0, 1.0)
    return np.ascontiguousarray(samples, dtype=np.float32), duration_seconds


def format_transcription_error(exc: Exception) -> str:
    message = f"{type(exc).__name__}: {exc}"
    lower_message = message.lower()

    if "ffmpeg" in lower_message or "ffprobe" in lower_message:
        return (
            "Transcription failed because audio decoding tools are missing. "
            "Install ffmpeg (so ffmpeg/ffprobe are in PATH) and try again."
        )

    if any(hint in lower_message for hint in NETWORK_ERROR_HINTS):
        return (
            "Transcription failed because the STT model could not be downloaded from Hugging Face. "
            "Check internet access, or set STT Model to a local Whisper model folder."
        )

    return f"Transcription failed: {message}"


def is_whisper_long_form_timestamp_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "return_timestamps=true" in text
        or "long-form generation" in text
        or "predict timestamp tokens" in text
        or "more than 3000 mel input features" in text
    )


def transcribe_reference_audio(
    ref_audio_path: Any,
    stt_model_source: str,
    language: str,
    device: str,
    dtype: str,
    progress=gr.Progress(),
):
    progress(0.05, desc="Preparing transcription...")
    audio_path = normalize_audio_path(ref_audio_path)
    if not audio_path:
        return gr.update(), "Reference audio is required before transcription."

    effective_device = detect_device() if device == "auto" else device
    effective_dtype_name = resolve_asr_dtype_name(dtype, effective_device)
    stt_source = stt_model_source.strip() or resolve_default_stt_model()
    whisper_language = resolve_transcription_language(language)

    errors: list[Exception] = []
    transcriber = None
    transcribe_device = effective_device
    progress(0.2, desc="Loading Whisper model...")
    try:
        transcriber = load_stt_pipeline(
            stt_model_source=stt_source,
            device=transcribe_device,
            dtype_name=effective_dtype_name,
        )
    except Exception as exc:
        errors.append(exc)
        if effective_device != "cpu":
            transcribe_device = "cpu"
            try:
                transcriber = load_stt_pipeline(
                    stt_model_source=stt_source,
                    device=transcribe_device,
                    dtype_name="float32",
                )
            except Exception as fallback_exc:
                errors.append(fallback_exc)
                return gr.update(), format_transcription_error(fallback_exc)
        else:
            return gr.update(), format_transcription_error(exc)

    if transcriber is None:
        return gr.update(), format_transcription_error(errors[-1])

    language_warning = ""
    try:
        progress(0.5, desc="Decoding audio...")
        audio_input, duration_seconds = load_audio_for_stt(audio_path)
        use_timestamps = duration_seconds > 30.0
        generate_kwargs = {"task": "transcribe"}
        if whisper_language:
            generate_kwargs["language"] = whisper_language

        progress(0.65, desc="Running speech-to-text...")
        result = transcriber(
            audio_input,
            return_timestamps=use_timestamps,
            generate_kwargs=generate_kwargs,
        )
    except Exception as exc:
        if not use_timestamps and is_whisper_long_form_timestamp_error(exc):
            try:
                progress(0.7, desc="Retrying in long-audio mode...")
                use_timestamps = True
                result = transcriber(
                    audio_input,
                    return_timestamps=True,
                    generate_kwargs=generate_kwargs,
                )
            except Exception as retry_exc:
                return gr.update(), format_transcription_error(retry_exc)
        else:
            # If a custom language hint is invalid, retry with auto language detection.
            if whisper_language and "language" in str(exc).lower():
                try:
                    result = transcriber(
                        audio_input,
                        return_timestamps=use_timestamps,
                        generate_kwargs={"task": "transcribe"},
                    )
                    language_warning = (
                        f"Requested transcription language '{(language or '').strip()}' was not accepted; "
                        "used automatic language detection."
                    )
                except Exception:
                    return gr.update(), format_transcription_error(exc)
            else:
                return gr.update(), format_transcription_error(exc)
    transcript = ""
    if isinstance(result, dict):
        transcript = (result.get("text") or "").strip()
        if not transcript and isinstance(result.get("chunks"), list):
            transcript = " ".join(
                str(chunk.get("text", "")).strip()
                for chunk in result["chunks"]
                if isinstance(chunk, dict) and str(chunk.get("text", "")).strip()
            ).strip()
    if not transcript:
        return gr.update(), "Transcription completed but returned empty text."

    progress(0.95, desc="Finalizing transcript...")
    details = []
    if use_timestamps:
        details.append("long-audio mode enabled")
    if language_warning:
        details.append(language_warning)

    detail_text = f" ({'; '.join(details)})" if details else ""
    if transcribe_device != effective_device:
        return transcript, f"Reference transcript auto-filled{detail_text}. STT fell back to CPU for compatibility."
    return transcript, f"Reference transcript auto-filled{detail_text}. You can edit it before generating."


def clone_voice(
    ref_audio_path: str | None,
    ref_text: str,
    target_text: str,
    language: str,
    voice_description: str,
    x_vector_only_mode: bool,
    model_dir: str,
    hf_model: str,
    device: str,
    dtype: str,
    use_mlx_hybrid: bool,
    use_mlx_quantizer: bool,
    eris_src_dir: str,
):
    if not ref_audio_path:
        return None, None, "Reference audio is required."
    if not target_text or not target_text.strip():
        return None, None, "Target text is required."

    effective_device = detect_device() if device == "auto" else device
    effective_dtype_name = resolve_dtype_name(dtype, effective_device)

    local_model_path = Path(model_dir).expanduser()
    model_source = local_model_path.as_posix() if local_model_path.exists() else hf_model
    try:
        model = load_model(
            model_source,
            effective_device,
            effective_dtype_name,
            use_mlx_hybrid,
            use_mlx_quantizer,
            eris_src_dir,
        )
    except Exception as exc:
        return None, None, f"Model initialization failed: {type(exc).__name__}: {exc}"

    clean_ref_text = ref_text.strip()
    use_x_vector_only_mode = x_vector_only_mode or clean_ref_text == ""
    try:
        voice_clone_prompt, cache_hit = get_cached_voice_clone_prompt(
            model=model,
            ref_audio_path=ref_audio_path,
            clean_ref_text=clean_ref_text,
            use_x_vector_only_mode=use_x_vector_only_mode,
        )
    except Exception as exc:
        return None, None, f"Reference prompt preparation failed: {type(exc).__name__}: {exc}"

    try:
        wavs, sample_rate = model.generate_voice_clone(
            text=target_text.strip(),
            language=language.strip() or "English",
            voice_clone_prompt=voice_clone_prompt,
            instruct=voice_description.strip() if voice_description.strip() else None,
            non_streaming_mode=True,
        )
    except Exception as exc:
        return None, None, f"{type(exc).__name__}: {exc}"

    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = (DEFAULT_OUTPUT_DIR / f"voice_clone_{timestamp}.wav").resolve()
    sf.write(output_path.as_posix(), wavs[0], sample_rate)

    mode_text = "x-vector only mode" if use_x_vector_only_mode else "reference transcript mode"
    backend_text = "mlx-hybrid" if use_mlx_hybrid else "pytorch"
    prompt_text = "prompt cache hit" if cache_hit else "prompt cache miss"
    status = (
        f"Generated successfully on {effective_device} ({effective_dtype_name}) "
        f"using {mode_text} [{backend_text}, fast mode, {prompt_text}]. Saved to: {output_path}"
    )
    return output_path.as_posix(), output_path.as_posix(), status


def build_ui(
    default_model_dir: str,
    default_hf_model: str,
    default_stt_model: str,
    default_device: str,
    default_dtype: str,
    default_use_mlx_hybrid: bool,
    default_use_mlx_quantizer: bool,
    default_eris_src_dir: str,
) -> gr.Blocks:
    with gr.Blocks(title="Voice Cloning GUI") as demo:
        gr.Markdown(
            """
# Voice Cloning GUI
1. Upload or record a reference audio sample.
2. Auto-transcribe reference audio (or type/edit transcript manually).
3. Enter the new text to synthesize in the cloned voice.
4. Click **Create Voice Clone**.

Use the output waveform player to drag/swipe through the audio, then download the WAV file.
"""
        )

        with gr.Row():
            with gr.Column(scale=1):
                ref_audio = gr.Audio(
                    label="1) Reference Audio",
                    type="filepath",
                    sources=["upload", "microphone"],
                )
                ref_text = gr.Textbox(
                    label="2) Reference Transcript (optional)",
                    lines=3,
                    placeholder="Exact words spoken in the reference audio for best cloning quality.",
                    interactive=True,
                )
                transcribe_btn = gr.Button("Auto Transcribe Reference Audio")
                target_text = gr.Textbox(
                    label="3) Target Text to Clone",
                    lines=4,
                    placeholder="Type what you want the cloned voice to say.",
                )
                language = gr.Textbox(
                    label="Language (used for TTS and Auto Transcribe)",
                    value="English",
                    placeholder="Examples: English, Chinese, Auto",
                )
                voice_description = gr.Textbox(
                    label="Optional Style / Voice Instruction",
                    lines=2,
                    placeholder="e.g., Speak slowly and warmly.",
                )
                x_vector_only_mode = gr.Checkbox(
                    label="Force x-vector only mode (no transcript guidance)",
                    value=False,
                )
                generate_btn = gr.Button("Create Voice Clone", variant="primary")

            with gr.Column(scale=1):
                preview_audio = gr.Audio(
                    label="4) Cloned Audio Preview (swipe/drag to seek)",
                    type="filepath",
                )
                download_audio = gr.File(label="Download Cloned Audio (.wav)")
                status = gr.Textbox(label="Status", lines=4, interactive=False)

                with gr.Accordion("Advanced Settings", open=False):
                    model_dir = gr.Textbox(
                        label="Local Model Directory",
                        value=default_model_dir,
                    )
                    hf_model = gr.Textbox(
                        label="HF Model Fallback",
                        value=default_hf_model,
                    )
                    stt_model = gr.Textbox(
                        label="STT Model (for Auto Transcribe)",
                        value=default_stt_model,
                    )
                    device = gr.Dropdown(
                        label="Device",
                        choices=["auto", "cpu", "mps", "cuda:0"],
                        value=default_device,
                    )
                    dtype = gr.Dropdown(
                        label="DType (used for TTS and STT)",
                        choices=["auto", "float16", "bfloat16", "float32"],
                        value=default_dtype,
                    )
                    use_mlx_hybrid = gr.Checkbox(
                        label="Enable MLX hybrid decoder acceleration (Apple Silicon, experimental)",
                        value=default_use_mlx_hybrid,
                    )
                    use_mlx_quantizer = gr.Checkbox(
                        label="Use MLX quantizer too (faster, more experimental)",
                        value=default_use_mlx_quantizer,
                    )
                    eris_src_dir = gr.Textbox(
                        label="Eris MLX source directory",
                        value=default_eris_src_dir,
                    )

        transcribe_btn.click(
            fn=transcribe_reference_audio,
            inputs=[ref_audio, stt_model, language, device, dtype],
            outputs=[ref_text, status],
        )

        generate_btn.click(
            fn=clone_voice,
            inputs=[
                ref_audio,
                ref_text,
                target_text,
                language,
                voice_description,
                x_vector_only_mode,
                model_dir,
                hf_model,
                device,
                dtype,
                use_mlx_hybrid,
                use_mlx_quantizer,
                eris_src_dir,
            ],
            outputs=[preview_audio, download_audio, status],
        )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a GUI for Qwen3 voice cloning.")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR.as_posix())
    parser.add_argument("--hf-model", default=DEFAULT_HF_MODEL)
    parser.add_argument("--stt-model", default=resolve_default_stt_model())
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--mlx-hybrid", action="store_true")
    parser.add_argument("--mlx-disable-quantizer", action="store_true")
    parser.add_argument("--eris-src-dir", default="eris-voice/src")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    demo = build_ui(
        default_model_dir=args.model_dir,
        default_hf_model=args.hf_model,
        default_stt_model=args.stt_model,
        default_device=args.device,
        default_dtype=args.dtype,
        default_use_mlx_hybrid=args.mlx_hybrid,
        default_use_mlx_quantizer=not args.mlx_disable_quantizer,
        default_eris_src_dir=args.eris_src_dir,
    )
    demo.queue(default_concurrency_limit=1).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()

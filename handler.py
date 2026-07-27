"""Runpod queue worker for KIVO's Chatterbox Multilingual TTS."""

import base64
import binascii
import io
import tempfile
import threading
from pathlib import Path

import runpod
import soundfile as sf
import torch
from chatterbox.mtl_tts import ChatterboxMultilingualTTS


MAX_TEXT_LENGTH = 4_000
MAX_REFERENCE_AUDIO_BYTES = 12 * 1024 * 1024
SUPPORTED_LANGUAGES = {
    "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it",
    "ja", "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh",
}

_model = None
_model_lock = threading.Lock()
_generation_lock = threading.Lock()


def get_model():
    """Load the model once per warm worker."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                if not torch.cuda.is_available():
                    raise RuntimeError("A CUDA GPU is required")
                _model = ChatterboxMultilingualTTS.from_pretrained(
                    device="cuda",
                    t3_model="v3",
                )
    return _model


def decode_reference_audio(value):
    """Decode a raw base64 string or data URI to a temporary audio file."""
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("reference_audio must be a non-empty base64 string")

    encoded = value.split(",", 1)[1] if value.startswith("data:") and "," in value else value
    try:
        payload = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("reference_audio is not valid base64") from error
    if len(payload) > MAX_REFERENCE_AUDIO_BYTES:
        raise ValueError("reference_audio exceeds 12 MB")

    temp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        temp.write(payload)
        return temp.name
    finally:
        temp.close()


def handler(job):
    data = job.get("input") or {}
    text = data.get("text") or data.get("prompt")
    if not isinstance(text, str) or not text.strip():
        return {"error": "input.text must be a non-empty string"}
    text = text.strip()
    if len(text) > MAX_TEXT_LENGTH:
        return {"error": f"input.text exceeds {MAX_TEXT_LENGTH} characters"}

    language = data.get("language") or data.get("language_id") or "en"
    if language not in SUPPORTED_LANGUAGES:
        return {"error": f"unsupported language: {language}"}

    try:
        exaggeration = float(data.get("exaggeration", 0.5))
        cfg_weight = float(data.get("cfg_weight", 0.5))
    except (TypeError, ValueError):
        return {"error": "exaggeration and cfg_weight must be numbers"}
    if not 0.0 <= exaggeration <= 2.0 or not 0.0 <= cfg_weight <= 1.0:
        return {"error": "exaggeration must be 0-2 and cfg_weight must be 0-1"}

    reference_path = None
    try:
        reference_path = decode_reference_audio(
            data.get("reference_audio") or data.get("voice_audio")
        )
        model = get_model()
        generate_args = {
            "language_id": language,
            "exaggeration": exaggeration,
            "cfg_weight": cfg_weight,
        }
        if reference_path:
            generate_args["audio_prompt_path"] = reference_path

        # One model instance should not generate two clips concurrently.
        with _generation_lock, torch.inference_mode():
            wav = model.generate(text, **generate_args)

        samples = wav.squeeze().detach().float().cpu().numpy()
        audio = io.BytesIO()
        sf.write(audio, samples, model.sr, format="WAV", subtype="PCM_16")
        payload = audio.getvalue()
        return {
            "audio_base64": base64.b64encode(payload).decode("ascii"),
            "content_type": "audio/wav",
            "sample_rate": model.sr,
            "duration_seconds": round(len(samples) / model.sr, 3),
        }
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {"error": "GPU ran out of memory while generating audio"}
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"}
    finally:
        if reference_path:
            Path(reference_path).unlink(missing_ok=True)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

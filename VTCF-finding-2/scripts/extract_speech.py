"""Speech extraction pipeline: download → ffmpeg → VAD → Demucs → Whisper + OCR."""

from __future__ import annotations

import logging
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

WHISPER_HF_MODEL = "bengaliAI/tugstugi_bengaliai-regional-asr_whisper-medium"
WHISPER_FALLBACK_MODEL = "medium"
WAV2VEC_MODEL = "ai4bharat/indicwav2vec_v1_bengali"
DEMUCS_TIMEOUT_SECONDS = 60
MAX_TRANSCRIBE_SECONDS: float | None = None
_vad_model: Any | None = None
_ocr_reader: Any | None = None


def save_wav(path: Path, waveform: torch.Tensor, sample_rate: int) -> None:
    """Save mono WAV via soundfile (avoids torchcodec dependency in torchaudio 2.11+)."""
    import soundfile as sf

    audio = waveform.detach().cpu().numpy()
    if audio.ndim > 1:
        audio = audio.squeeze()
    sf.write(str(path), audio, sample_rate)


def get_ocr_reader() -> Any:
    """Reuse a single EasyOCR reader across videos."""
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr

        _ocr_reader = easyocr.Reader(["bn"], gpu=torch.cuda.is_available(), verbose=False)
    return _ocr_reader


@dataclass
class PipelineFailure:
    video_id: str
    step: str
    error_message: str


def build_youtube_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def _yt_dlp_command() -> list[str]:
    """Resolve yt-dlp executable (venv Scripts on Windows)."""
    venv_bin = Path(sys.executable).resolve().parent / ("yt-dlp.exe" if sys.platform == "win32" else "yt-dlp")
    if venv_bin.exists():
        return [str(venv_bin), "--js-runtimes", "node"]
    return ["yt-dlp", "--js-runtimes", "node"]


def download_video_360p(
    video_id: str,
    temp_dir: Path,
    cookies_file: Path | None = None,
) -> Path | None:
    """Download mp4 at <=360p via yt-dlp."""
    temp_dir.mkdir(parents=True, exist_ok=True)
    output_path = temp_dir / f"{video_id}.mp4"
    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path

    command = [
        *_yt_dlp_command(),
        "-f",
        "best[height<=360][ext=mp4]/best[height<=360]/bv*[height<=360]+ba/b",
        "-o",
        str(output_path),
        "--quiet",
        "--no-progress",
        build_youtube_url(video_id),
    ]
    if cookies_file and cookies_file.exists():
        command.extend(["--cookies", str(cookies_file)])

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=300)
        if result.returncode == 0 and output_path.exists():
            return output_path
        logger.warning("yt-dlp failed for %s: %s", video_id, (result.stderr or result.stdout)[:300])
    except (subprocess.SubprocessError, OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("yt-dlp exception for %s: %s", video_id, exc)
    return None


def extract_audio_wav(
    video_path: Path,
    output_wav: Path,
    sample_rate: int = 16000,
) -> Path | None:
    """Convert video to mono 16 kHz WAV using ffmpeg."""
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        str(output_wav),
    ]
    try:
        subprocess.run(command, capture_output=True, text=True, check=True, timeout=120)
        if output_wav.exists():
            return output_wav
    except (subprocess.SubprocessError, OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("ffmpeg failed for %s: %s", video_path.stem, exc)
    return None


def load_audio_mono(path: Path, sample_rate: int = 16000) -> tuple[torch.Tensor, int] | None:
    """Load WAV as float tensor [samples] using soundfile."""
    try:
        import soundfile as sf

        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        waveform = torch.from_numpy(data.mean(axis=1))
        if sr != sample_rate:
            import torchaudio

            waveform = torchaudio.functional.resample(waveform.unsqueeze(0), sr, sample_rate).squeeze(0)
            sr = sample_rate
        return waveform.contiguous(), sr
    except Exception as exc:
        logger.warning("Audio load failed for %s: %s", path.stem, exc)
        return None


def get_vad_model() -> Any:
    """Load Silero VAD once via silero-vad package."""
    global _vad_model
    if _vad_model is None:
        from silero_vad import load_silero_vad

        _vad_model = load_silero_vad()
    return _vad_model


def run_vad(
    waveform: torch.Tensor,
    sample_rate: int,
    threshold: float = 0.5,
    min_speech_ms: int = 250,
) -> tuple[list[tuple[int, int]], float]:
    """
    Detect speech segments via Silero VAD.

    Returns (segments, speech_coverage_percent).
    """
    try:
        from silero_vad import get_speech_timestamps

        model = get_vad_model()
        if waveform.dtype != torch.float32:
            waveform = waveform.float()
        timestamps = get_speech_timestamps(
            waveform,
            model,
            sampling_rate=sample_rate,
            threshold=threshold,
            min_speech_duration_ms=min_speech_ms,
            return_seconds=False,
        )
        segments = [(int(item["start"]), int(item["end"])) for item in timestamps]
        total_samples = max(int(waveform.numel()), 1)
        speech_samples = sum(end - start for start, end in segments)
        coverage_percent = 100.0 * speech_samples / total_samples
        return segments, coverage_percent
    except Exception as exc:
        logger.warning("VAD failed: %s", exc)
        return [], 0.0


def concatenate_speech(
    waveform: torch.Tensor,
    segments: list[tuple[int, int]],
) -> torch.Tensor:
    if not segments:
        return waveform.new_zeros(0)
    return torch.cat([waveform[start:end] for start, end in segments], dim=0)


def apply_demucs_vocals(wav_path: Path, work_dir: Path) -> Path | None:
    """Separate vocals with Demucs (60s timeout). Returns vocals path or None."""
    work_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "demucs",
        "--two-stems",
        "vocals",
        "-o",
        str(work_dir),
        str(wav_path),
    ]
    try:
        subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            timeout=DEMUCS_TIMEOUT_SECONDS,
        )
        stem_path = work_dir / "htdemucs" / wav_path.stem / "vocals.wav"
        if stem_path.exists():
            return stem_path
    except (subprocess.SubprocessError, OSError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
        logger.warning("Demucs failed/skipped for %s: %s", wav_path.stem, exc)
    return None


_whisper_model_cache: dict[str, Any] = {}


def _load_whisper_hf(model_id: str | None = None) -> Any | None:
    if "hf" in _whisper_model_cache:
        return _whisper_model_cache["hf"]
    model_name = model_id or WHISPER_HF_MODEL
    try:
        from transformers import pipeline

        device = 0 if torch.cuda.is_available() else -1
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            device=device,
            dtype=dtype,
        )
        _whisper_model_cache["hf"] = pipe
        _whisper_model_cache["hf_model_name"] = model_name
        logger.info("Loaded HuggingFace ASR model: %s", model_name)
        return pipe
    except Exception as exc:
        logger.warning("HF Whisper unavailable (%s): %s", model_name, exc)
        _whisper_model_cache["hf"] = None
        return None


def _load_whisper_openai() -> Any:
    if "openai" not in _whisper_model_cache:
        import whisper

        _whisper_model_cache["openai"] = whisper.load_model(WHISPER_FALLBACK_MODEL)
        logger.info("Loaded OpenAI Whisper model: %s", WHISPER_FALLBACK_MODEL)
    return _whisper_model_cache["openai"]


def transcribe_wav2vec2(audio_path: Path, model_id: str | None = None) -> str:
    import soundfile as sf
    from transformers import AutoProcessor, Wav2Vec2ForCTC

    model_name = model_id or WAV2VEC_MODEL
    audio, sr = sf.read(str(audio_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if "wav2vec_processor" not in _whisper_model_cache:
        _whisper_model_cache["wav2vec_processor"] = AutoProcessor.from_pretrained(model_name)
        model = Wav2Vec2ForCTC.from_pretrained(model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _whisper_model_cache["wav2vec_model"] = model.to(device).eval()
        _whisper_model_cache["wav2vec_device"] = device
        logger.info("Loaded Wav2Vec2 ASR model: %s on %s", model_name, device)

    processor = _whisper_model_cache["wav2vec_processor"]
    model = _whisper_model_cache["wav2vec_model"]
    device = _whisper_model_cache["wav2vec_device"]
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)
    with torch.no_grad():
        logits = model(input_values).logits
    pred_ids = torch.argmax(logits, dim=-1)
    return str(processor.batch_decode(pred_ids)[0]).strip()


def transcribe_audio(
    audio_path: Path,
    language: str = "bn",
    max_seconds: float | None = None,
    hf_model: str | None = None,
    fallback_model: str = WHISPER_FALLBACK_MODEL,
    asr_backend: str = "hf_whisper",
    wav2vec_model: str | None = None,
) -> tuple[str, float | None]:
    """
    Transcribe VAD-trimmed audio with configured Bengali ASR backend.

    Returns (transcript_text, average_log_prob_or_none).
    """
    asr_path = audio_path
    if max_seconds is not None:
        loaded = load_audio_mono(audio_path)
        if loaded is not None:
            waveform, sr = loaded
            max_samples = int(max_seconds * sr)
            if waveform.numel() > max_samples:
                trimmed = audio_path.parent / f"{audio_path.stem}_trim.wav"
                save_wav(trimmed, waveform[:max_samples], sr)
                asr_path = trimmed

    backend = str(asr_backend or "hf_whisper").lower()
    if backend == "wav2vec2":
        try:
            text = transcribe_wav2vec2(asr_path, model_id=wav2vec_model)
            if text:
                return text, None
        except Exception as exc:
            logger.warning("Wav2Vec2 transcription failed, falling back to HF Whisper: %s", exc)
            backend = "hf_whisper"

    if backend == "openai_whisper":
        try:
            import whisper

            if "openai" not in _whisper_model_cache:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                _whisper_model_cache["openai"] = whisper.load_model(fallback_model, device=device)
                _whisper_model_cache["openai_device"] = device
                logger.info("Loaded OpenAI Whisper model: %s on %s", fallback_model, device)
            model = _whisper_model_cache["openai"]
            use_fp16 = torch.cuda.is_available()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = model.transcribe(
                    str(asr_path),
                    language=language,
                    fp16=use_fp16,
                    condition_on_previous_text=False,
                )
            text = str(result.get("text", "")).strip()
            if text:
                segments = result.get("segments") or []
                log_probs = [
                    float(seg.get("avg_logprob", 0.0)) for seg in segments if "avg_logprob" in seg
                ]
                avg_logprob = float(np.mean(log_probs)) if log_probs else None
                return text, avg_logprob
        except Exception as exc:
            logger.warning("OpenAI Whisper failed, falling back to HF: %s", exc)

    hf_pipe = _load_whisper_hf(hf_model)
    if hf_pipe is not None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = hf_pipe(
                    str(asr_path),
                    return_timestamps=False,
                    chunk_length_s=30,
                    stride_length_s=5,
                )
            text = str(result.get("text", "")).strip()
            if text:
                return text, None
        except Exception as exc:
            logger.warning("HF transcription failed, falling back to openai-whisper: %s", exc)

    try:
        import whisper

        if "openai" not in _whisper_model_cache:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _whisper_model_cache["openai"] = whisper.load_model(fallback_model, device=device)
            _whisper_model_cache["openai_device"] = device
            logger.info("Loaded OpenAI Whisper model: %s on %s", fallback_model, device)
        model = _whisper_model_cache["openai"]
        use_fp16 = torch.cuda.is_available()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.transcribe(
                str(asr_path),
                language=language,
                fp16=use_fp16,
                condition_on_previous_text=False,
                no_speech_threshold=0.6,
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0,
            )

        text = str(result.get("text", "")).strip()
        segments = result.get("segments") or []
        log_probs = [float(seg.get("avg_logprob", 0.0)) for seg in segments if "avg_logprob" in seg]
        avg_logprob = float(np.mean(log_probs)) if log_probs else None
        return text, avg_logprob
    except Exception as exc:
        logger.warning("Whisper transcription failed: %s", exc)
        return "", None


def extract_ocr_bn(frame_path: Path) -> tuple[str, bool, int]:
    """Run EasyOCR (Bangla) on hook frame. Returns (text, detected, char_count)."""
    if not frame_path.exists():
        return "", False, 0
    try:
        reader = get_ocr_reader()
        detections = reader.readtext(str(frame_path))
        texts = [str(text).strip() for _bbox, text, _conf in detections if str(text).strip()]
        combined = " ".join(texts)
        return combined, bool(combined), len(combined)
    except Exception as exc:
        logger.warning("OCR failed for %s: %s", frame_path.parent.name, exc)
        return "", False, 0


def cleanup_temp_dir(temp_dir: Path) -> None:
    """Remove temp audio/video files for one video."""
    if not temp_dir.exists():
        return
    try:
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)
    except OSError as exc:
        logger.warning("Temp cleanup failed for %s: %s", temp_dir.name, exc)


def process_video_pipeline(
    video_id: str,
    frames_dir: Path,
    temp_root: Path,
    cookies_file: Path | None,
    sample_rate: int = 16000,
    vad_threshold: float = 0.5,
    min_speech_ms: int = 250,
    use_demucs: bool = True,
    language: str = "bn",
    max_transcribe_seconds: float | None = None,
    hf_model: str | None = None,
    fallback_model: str = WHISPER_FALLBACK_MODEL,
    asr_backend: str = "hf_whisper",
    wav2vec_model: str | None = None,
) -> tuple[dict[str, Any], list[PipelineFailure]]:
    """
    Run full ASR+OCR pipeline for one video.

    Each external step is wrapped in try/except; failures are collected but
    processing continues with fallbacks where possible.
    """
    failures: list[PipelineFailure] = []
    temp_dir = temp_root / video_id
    temp_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    speech_coverage_percent = 0.0
    transcript = ""
    whisper_confidence: float | None = None
    ocr_text = ""
    ocr_text_detected = False
    ocr_char_count = 0
    video_duration_sec = 0.0

    video_path: Path | None = None
    try:
        video_path = download_video_360p(video_id, temp_dir, cookies_file=cookies_file)
        if video_path is None:
            failures.append(PipelineFailure(video_id, "yt-dlp", "Video download failed"))
    except Exception as exc:
        failures.append(PipelineFailure(video_id, "yt-dlp", str(exc)))

    raw_wav = temp_dir / f"{video_id}.wav"
    asr_input: Path | None = None

    if video_path is not None:
        try:
            wav_out = extract_audio_wav(video_path, raw_wav, sample_rate=sample_rate)
            if wav_out is None:
                failures.append(PipelineFailure(video_id, "ffmpeg", "Audio extraction failed"))
            else:
                asr_input = raw_wav
        except Exception as exc:
            failures.append(PipelineFailure(video_id, "ffmpeg", str(exc)))

    if asr_input is not None:
        loaded = load_audio_mono(asr_input, sample_rate=sample_rate)
        if loaded is None:
            failures.append(PipelineFailure(video_id, "audio_load", "Could not load extracted WAV"))
        else:
            waveform, sr = loaded
            video_duration_sec = float(waveform.numel()) / sr

            try:
                segments, speech_coverage_percent = run_vad(
                    waveform,
                    sr,
                    threshold=vad_threshold,
                    min_speech_ms=min_speech_ms,
                )
            except Exception as exc:
                failures.append(PipelineFailure(video_id, "vad", str(exc)))
                segments, speech_coverage_percent = [], 0.0

            speech_waveform = concatenate_speech(waveform, segments)
            if speech_waveform.numel() == 0:
                speech_waveform = waveform
                speech_coverage_percent = 100.0 if waveform.numel() > 0 else 0.0

            speech_wav = temp_dir / f"{video_id}_speech.wav"
            try:
                save_wav(speech_wav, speech_waveform, sr)
                asr_input = speech_wav
            except Exception as exc:
                failures.append(PipelineFailure(video_id, "vad_save", str(exc)))

            if use_demucs and asr_input is not None:
                try:
                    vocals = apply_demucs_vocals(asr_input, temp_dir / "demucs")
                    if vocals is not None:
                        asr_input = vocals
                except Exception as exc:
                    failures.append(PipelineFailure(video_id, "demucs", str(exc)))

    if asr_input is not None and asr_input.exists():
        try:
            transcript, whisper_confidence = transcribe_audio(
                asr_input,
                language=language,
                max_seconds=max_transcribe_seconds,
                hf_model=hf_model,
                fallback_model=fallback_model,
                asr_backend=asr_backend,
                wav2vec_model=wav2vec_model,
            )
        except Exception as exc:
            failures.append(PipelineFailure(video_id, "whisper", str(exc)))

    hook_frame = frames_dir / "frame_0.png"
    try:
        ocr_text, ocr_text_detected, ocr_char_count = extract_ocr_bn(hook_frame)
    except Exception as exc:
        failures.append(PipelineFailure(video_id, "ocr", str(exc)))

    processing_time_seconds = time.perf_counter() - started
    word_count = len(transcript.split()) if transcript else 0

    result = {
        "video_id": video_id,
        "speech_coverage_percent": round(speech_coverage_percent, 2),
        "transcript": transcript,
        "transcript_word_count": word_count,
        "transcript_is_empty": word_count == 0,
        "ocr_text": ocr_text,
        "ocr_text_detected": ocr_text_detected,
        "ocr_char_count": ocr_char_count,
        "processing_time_seconds": round(processing_time_seconds, 2),
        "whisper_confidence": whisper_confidence,
        "video_duration_sec": round(video_duration_sec, 2),
    }

    cleanup_temp_dir(temp_dir)
    return result, failures

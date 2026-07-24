"""
Compare Bengali ASR candidates on the same VAD speech audio.

Models:
1. bengaliAI/tugstugi_bengaliai-regional-asr_whisper-medium (Whisper/HF)
2. ai4bharat/indicwav2vec-v1-bengali (Wav2Vec2 CTC)
3. openai/whisper-medium (baseline)
"""

from __future__ import annotations

import argparse
import sys
import textwrap
import warnings
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._paths import load_config, resolve_path
from scripts.extract_speech import (
    concatenate_speech,
    download_video_360p,
    extract_audio_wav,
    load_audio_mono,
    run_vad,
    save_wav,
)

def log(msg: str) -> None:
    print(msg, flush=True)


BENGALI_WHISPER_HF = "bengaliAI/tugstugi_bengaliai-regional-asr_whisper-medium"
INDICWAV2VEC = "ai4bharat/indicwav2vec_v1_bengali"
OPENAI_WHISPER = "medium"
DEFAULT_VIDEO_IDS = ("17M9XIMGApE", "95vimI2OIUk", "7p_OX_Dtmkw")


def prepare_speech_wav(
    video_id: str,
    cache_dir: Path,
    cookies_file: Path | None,
    sample_rate: int = 16000,
    vad_threshold: float = 0.35,
    min_speech_ms: int = 250,
) -> Path | None:
    """Download once and cache `{video_id}_speech.wav` for benchmark reuse."""
    out_wav = cache_dir / f"{video_id}_speech.wav"
    if out_wav.exists() and out_wav.stat().st_size > 0:
        return out_wav

    work = cache_dir / video_id
    work.mkdir(parents=True, exist_ok=True)
    video_path = download_video_360p(video_id, work, cookies_file=cookies_file)
    if video_path is None:
        return None

    raw_wav = work / f"{video_id}.wav"
    if extract_audio_wav(video_path, raw_wav, sample_rate=sample_rate) is None:
        return None

    loaded = load_audio_mono(raw_wav, sample_rate=sample_rate)
    if loaded is None:
        return None
    waveform, sr = loaded
    segments, _ = run_vad(waveform, sr, threshold=vad_threshold, min_speech_ms=min_speech_ms)
    speech = concatenate_speech(waveform, segments)
    if speech.numel() == 0:
        speech = waveform
    save_wav(out_wav, speech, sr)
    return out_wav


def transcribe_hf_whisper(audio_path: Path, model_id: str) -> str:
    from transformers import pipeline

    log(f"    Loading HF model: {model_id} ...")
    device = 0 if torch.cuda.is_available() else -1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=device,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        log(f"    Running HF inference on {audio_path.name} ...")
        result = pipe(
            str(audio_path),
            return_timestamps=False,
            chunk_length_s=30,
            stride_length_s=5,
        )
    return str(result.get("text", "")).strip()


def transcribe_openai_whisper(audio_path: Path, model_name: str = OPENAI_WHISPER) -> str:
    import whisper

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(model_name, device=device)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.transcribe(
            str(audio_path),
            language="bn",
            fp16=torch.cuda.is_available(),
            condition_on_previous_text=False,
        )
    return str(result.get("text", "")).strip()


def transcribe_indicwav2vec(audio_path: Path, model_id: str = INDICWAV2VEC) -> str:
    import soundfile as sf
    from transformers import AutoProcessor, Wav2Vec2ForCTC

    log(f"    Loading Wav2Vec2 model: {model_id} ...")
    audio, sr = sf.read(str(audio_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    processor = AutoProcessor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    inputs = processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)
    with torch.no_grad():
        logits = model(input_values).logits
    pred_ids = torch.argmax(logits, dim=-1)
    text = processor.batch_decode(pred_ids)[0]
    return str(text).strip()


def wrap(text: str, width: int = 88) -> str:
    cleaned = " ".join(str(text).split())
    if not cleaned:
        return "(empty)"
    return "\n".join(textwrap.wrap(cleaned, width=width))


def print_comparison(video_id: str, title: str, results: dict[str, str]) -> None:
    print("\n" + "=" * 96)
    print(f"VIDEO: {video_id}")
    print(f"TITLE (reference): {title}")
    print("-" * 96)
    for model_name, transcript in results.items():
        print(f"\n[{model_name}]")
        print(wrap(transcript))
    print("=" * 96)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bengali ASR model shootout")
    parser.add_argument(
        "--videos",
        nargs="+",
        default=list(DEFAULT_VIDEO_IDS),
        help="YouTube video IDs to benchmark",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "asr_benchmark" / "audio",
    )
    return parser.parse_args()


def main() -> None:
    if sys.platform == "win32":
        import io

        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    args = parse_args()
    config = load_config()
    cookies = resolve_path(config["data"]["youtube_cookies"])
    cookies_file = cookies if cookies.exists() else None
    speech_cfg = config.get("speech", {})

    titles = {}
    spike_csv = resolve_path(config["data"]["spike_csv"])
    if spike_csv.exists():
        df = pd.read_csv(spike_csv)
        titles = dict(zip(df["video_id"].astype(str), df["title"].astype(str)))

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    log(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}")

    runners = {
        f"HF {BENGALI_WHISPER_HF}": lambda p: transcribe_hf_whisper(p, BENGALI_WHISPER_HF),
        f"Wav2Vec2 {INDICWAV2VEC}": lambda p: transcribe_indicwav2vec(p, INDICWAV2VEC),
        f"OpenAI whisper-{OPENAI_WHISPER}": lambda p: transcribe_openai_whisper(p, OPENAI_WHISPER),
    }

    for video_id in args.videos:
        video_id = str(video_id)
        title = titles.get(video_id, "(title not in spike_csv)")
        speech_wav = prepare_speech_wav(
            video_id,
            args.cache_dir,
            cookies_file=cookies_file,
            sample_rate=int(speech_cfg.get("sample_rate", 16000)),
            vad_threshold=float(speech_cfg.get("vad_threshold", 0.35)),
            min_speech_ms=int(speech_cfg.get("min_speech_segment_ms", 250)),
        )
        if speech_wav is None:
            log(f"\nSKIP {video_id}: could not prepare speech audio (download/VAD failed)")
            continue

        log(f"\nPrepared speech audio: {speech_wav} ({speech_wav.stat().st_size // 1024} KB)")
        results: dict[str, str] = {}
        for model_name, fn in runners.items():
            try:
                log(f"  Transcribing with {model_name}...")
                results[model_name] = fn(speech_wav)
                log(f"  Done ({len(results[model_name])} chars)")
            except Exception as exc:
                results[model_name] = f"(ERROR: {exc})"
        print_comparison(video_id, title, results)


if __name__ == "__main__":
    main()

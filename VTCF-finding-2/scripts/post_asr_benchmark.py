"""
Wait for asr_benchmark.py to finish, pick ASR winner, re-transcribe spike set,
re-summarize, and recompute GO/NO-GO.

Run in background while the benchmark is in progress:
  python -u scripts/post_asr_benchmark.py
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._env import load_dotenv

LOG_PATH = PROJECT_ROOT / "outputs" / "logs" / "post_asr_benchmark_runner.log"
BENCHMARK_LOG = PROJECT_ROOT / "outputs" / "logs" / "asr_benchmark.log"
WINNER_PATH = PROJECT_ROOT / "outputs" / "asr_benchmark" / "winner.json"

MODEL_KEYS = {
    "HF bengaliAI/tugstugi_bengaliai-regional-asr_whisper-medium": {
        "asr_backend": "hf_whisper",
        "whisper_hf_model": "bengaliAI/tugstugi_bengaliai-regional-asr_whisper-medium",
    },
    "Wav2Vec2 ai4bharat/indicwav2vec-v1-bengali": {
        "asr_backend": "wav2vec2",
        "wav2vec_model": "ai4bharat/indicwav2vec-v1-bengali",
    },
    "OpenAI whisper-medium": {
        "asr_backend": "openai_whisper",
        "whisper_fallback_model": "medium",
    },
}


def setup_logging() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def benchmark_process_running() -> bool:
    if sys.platform == "win32":
        cmd = [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
            "Where-Object { $_.CommandLine -like '*scripts\\asr_benchmark.py*' -or "
            "$_.CommandLine -like '*scripts/asr_benchmark.py*' } | "
            "Select-Object -ExpandProperty ProcessId",
        ]
    else:
        cmd = ["pgrep", "-f", "asr_benchmark.py"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=30)
        return bool(result.stdout.strip())
    except (subprocess.SubprocessError, OSError):
        return False


def wait_for_benchmark(poll_seconds: int = 60, max_hours: float = 8.0) -> None:
    deadline = time.time() + max_hours * 3600
    logging.info("Waiting for asr_benchmark.py to finish...")
    while time.time() < deadline:
        if not benchmark_process_running():
            logging.info("Benchmark process exited.")
            return
        if BENCHMARK_LOG.exists():
            text = BENCHMARK_LOG.read_text(encoding="utf-8", errors="replace")
            # 3 videos x 3 models = 9 successful "Done" lines expected
            done_count = len(re.findall(r"^\s*Done \(\d+ chars\)", text, flags=re.MULTILINE))
            if done_count >= 9 and text.count("=" * 96) >= 3:
                logging.info("Benchmark log looks complete (%s Done lines).", done_count)
                return
        logging.info("Still running... (poll every %ss)", poll_seconds)
        time.sleep(poll_seconds)
    raise TimeoutError("Timed out waiting for ASR benchmark")


def bengali_score(text: str) -> float:
    cleaned = str(text or "").strip()
    if not cleaned or cleaned.startswith("(ERROR:"):
        return -1.0
    bn = len(re.findall(r"[\u0980-\u09FF]", cleaned))
    if bn == 0:
        return 0.0
    # Favor readable Bangla over mixed-script gibberish.
    latin = len(re.findall(r"[A-Za-z]", cleaned))
    return bn - 0.15 * latin


def parse_benchmark_log(log_text: str) -> dict[str, list[str]]:
    results: dict[str, list[str]] = {key: [] for key in MODEL_KEYS}
    current: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        nonlocal buffer, current
        if current and buffer:
            results[current].append(" ".join(buffer))
        buffer = []

    for line in log_text.splitlines():
        header = line.strip()
        if header.startswith("[") and header.endswith("]"):
            flush()
            label = header[1:-1]
            current = label if label in MODEL_KEYS else None
            continue
        if current and line.strip() and not line.startswith("=") and not line.startswith("-"):
            if line.startswith("VIDEO:") or line.startswith("TITLE"):
                continue
            buffer.append(line.strip())
    flush()
    return results


def pick_winner(log_text: str) -> tuple[str, dict]:
    parsed = parse_benchmark_log(log_text)
    scores: dict[str, float] = {}
    for label, transcripts in parsed.items():
        if not transcripts:
            scores[label] = -999.0
            continue
        scores[label] = sum(bengali_score(t) for t in transcripts) / len(transcripts)
        logging.info("Model score %s: %.1f (%s videos)", label, scores[label], len(transcripts))

    best_label = max(scores, key=scores.get)
    if scores[best_label] < 0:
        logging.warning("No valid transcripts in benchmark log; defaulting to BengaliAI HF.")
        best_label = "HF bengaliAI/tugstugi_bengaliai-regional-asr_whisper-medium"

    winner = {
        "label": best_label,
        "score": scores.get(best_label),
        **MODEL_KEYS[best_label],
    }
    return best_label, winner


def apply_winner_to_config(winner: dict) -> None:
    import yaml

    config_path = PROJECT_ROOT / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    speech = config.setdefault("speech", {})
    speech["asr_backend"] = winner["asr_backend"]
    if "whisper_hf_model" in winner:
        speech["whisper_hf_model"] = winner["whisper_hf_model"]
    if "wav2vec_model" in winner:
        speech["wav2vec_model"] = winner["wav2vec_model"]
    if "whisper_fallback_model" in winner:
        speech["whisper_fallback_model"] = winner["whisper_fallback_model"]
    config_path.write_text(yaml.dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    logging.info("Updated config.yaml speech settings: %s", winner)


def run_script(name: str, *args: str) -> None:
    cmd = [sys.executable, "-u", str(PROJECT_ROOT / "scripts" / name), *args]
    logging.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    setup_logging()
    load_dotenv()
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    wait_for_benchmark()
    time.sleep(5)

    log_text = BENCHMARK_LOG.read_text(encoding="utf-8", errors="replace") if BENCHMARK_LOG.exists() else ""
    label, winner = pick_winner(log_text)
    WINNER_PATH.parent.mkdir(parents=True, exist_ok=True)
    WINNER_PATH.write_text(json.dumps(winner, indent=2, ensure_ascii=False), encoding="utf-8")
    logging.info("Winner: %s -> %s", label, WINNER_PATH)
    apply_winner_to_config(winner)

    run_script("retranscribe_spike.py")
    run_script("resummarize_spike.py")
    logging.info("Post-benchmark pipeline complete.")


if __name__ == "__main__":
    import os

    main()

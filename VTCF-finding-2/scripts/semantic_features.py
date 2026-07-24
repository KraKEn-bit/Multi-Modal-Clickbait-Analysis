"""BanglaBERT embeddings, semantic divergence, and LLM transcript summarization."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any
from urllib import error, request

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

SUMMARY_PROMPT = (
    "Summarize what this video actually discusses in 2-3 sentences, in Bangla, "
    "based on this transcript: {transcript}"
)

_bert_bundle: dict[str, Any] | None = None


def _get_bert(model_name: str) -> dict[str, Any]:
    global _bert_bundle
    if _bert_bundle is None:
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        _bert_bundle = {"tokenizer": tokenizer, "model": model}
    return _bert_bundle


def encode_text(text: str, model_name: str, max_length: int = 512) -> torch.Tensor | None:
    """Mean-pooled BanglaBERT embedding for non-empty text."""
    cleaned = str(text or "").strip()
    if not cleaned:
        return None
    try:
        bundle = _get_bert(model_name)
        tokenizer = bundle["tokenizer"]
        model = bundle["model"]
        encoded = tokenizer(
            cleaned,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        with torch.no_grad():
            outputs = model(**encoded)
            hidden = outputs.last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return pooled.squeeze(0)
    except Exception as exc:
        logger.warning("BanglaBERT encoding failed: %s", exc)
        return None


def is_usable_hook_ocr(
    text: str,
    *,
    min_chars: int = 8,
    min_letter_ratio: float = 0.4,
) -> bool:
    """
    True when frame-0 EasyOCR text looks like real hook copy (not empty/noise).

    Raw output is always kept in hook_ocr.txt; this gate decides whether to
    include it in the promise side of semantic_divergence.
    """
    cleaned = str(text or "").strip()
    if len(cleaned) < min_chars:
        return False
    bn = len(re.findall(r"[\u0980-\u09FF]", cleaned))
    latin = len(re.findall(r"[A-Za-z]", cleaned))
    letters = bn + latin
    if letters / len(cleaned) < min_letter_ratio:
        return False
    symbol_runs = re.findall(r"[^\w\s\u0980-\u09FF.,!?;:\-'\"()]", cleaned)
    if len(symbol_runs) > len(cleaned) * 0.25:
        return False
    return True


def resolve_promise_text(title: str, hook_ocr: str) -> tuple[str, str]:
    """
    Build promise text for divergence: title + usable OCR, else title only.

    Returns (promise_text, hook_promise_source) where source is ``title`` or
    ``title+ocr``.
    """
    title_clean = str(title or "").strip()
    ocr_clean = str(hook_ocr or "").strip()
    if title_clean and is_usable_hook_ocr(ocr_clean):
        return f"{title_clean}\n{ocr_clean}".strip(), "title+ocr"
    return title_clean, "title"


def semantic_divergence(promise_text: str, delivery_text: str, model_name: str) -> float | None:
    """
    Diagnostic only: 1 - cosine_similarity(BanglaBERT(promise), BanglaBERT(delivery)).
    """
    promise_vec = encode_text(promise_text, model_name=model_name)
    delivery_vec = encode_text(delivery_text, model_name=model_name)
    if promise_vec is None or delivery_vec is None:
        return None
    similarity = F.cosine_similarity(
        promise_vec.unsqueeze(0),
        delivery_vec.unsqueeze(0),
        dim=-1,
    ).item()
    return float(1.0 - similarity)


def _extractive_fallback(transcript: str, max_chars: int = 600) -> str:
    """Fallback when no LLM API key is configured."""
    text = re.sub(r"\s+", " ", transcript.strip())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rsplit(" ", 1)[0] + "..."


def get_gemini_model_rotation(llm_cfg: dict[str, Any]) -> list[tuple[str, int]]:
    """Ordered Gemini models with per-model daily limits for quota rotation."""
    rotation = llm_cfg.get("gemini_model_rotation")
    if rotation:
        chain: list[tuple[str, int]] = []
        for entry in rotation:
            if isinstance(entry, dict):
                model = str(entry.get("model", "")).strip()
                limit = int(entry.get("daily_limit", 0))
            else:
                model = str(entry).strip()
                limit = 0
            if model:
                chain.append((model, limit))
        if chain:
            return chain

    primary = str(llm_cfg.get("gemini_model", "gemini-3.1-flash-lite")).strip()
    fallback = str(llm_cfg.get("gemini_fallback_model", "")).strip()
    chain = [(primary, 480)]
    if fallback and fallback != primary:
        chain.append((fallback, 20))
    return chain


def _http_post_json(url: str, payload: dict, headers: dict, max_retries: int = 5) -> dict:
    body = json.dumps(payload).encode("utf-8")
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        req = request.Request(url, data=body, headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            last_exc = exc
            if exc.code in {429, 500, 502, 503, 504} and attempt < max_retries - 1:
                delay = min(60, 2 ** attempt)
                logger.warning("HTTP %s from LLM API; retry in %ss", exc.code, delay)
                time.sleep(delay)
                continue
            raise
        except (error.URLError, TimeoutError, OSError) as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise
    if last_exc:
        raise last_exc
    raise RuntimeError("HTTP request failed")


def _summarize_gemini(
    transcript: str,
    api_key: str,
    model: str,
    *,
    max_retries: int = 3,
) -> str:
    prompt = SUMMARY_PROMPT.format(transcript=transcript[:12000])
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        f"?key={api_key}"
    )
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    data = _http_post_json(
        url,
        payload,
        headers={"Content-Type": "application/json"},
        max_retries=max_retries,
    )
    candidates = data.get("candidates") or []
    if not candidates:
        raise RuntimeError("Gemini returned no candidates")
    parts = candidates[0].get("content", {}).get("parts") or []
    text = " ".join(str(part.get("text", "")) for part in parts).strip()
    if not text:
        raise RuntimeError("Gemini returned empty summary")
    return text


def _summarize_openai(transcript: str, api_key: str, model: str) -> str:
    prompt = SUMMARY_PROMPT.format(transcript=transcript[:12000])
    url = "https://api.openai.com/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    data = _http_post_json(url, payload, headers=headers)
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("OpenAI returned no choices")
    text = str(choices[0].get("message", {}).get("content", "")).strip()
    if not text:
        raise RuntimeError("OpenAI returned empty summary")
    return text


def summarize_transcript(
    transcript: str,
    llm_cfg: dict[str, Any],
    *,
    model_filter: list[str] | None = None,
    can_use_model: Any | None = None,
) -> tuple[str, str, str]:
    """
    Summarize transcript via rotating Gemini models, then OpenAI if configured.

    Returns (summary_text, summary_source, model_used).
    Falls back to extractive truncation if all providers fail.
    """
    cleaned = transcript.strip()
    if not cleaned:
        primary = str(llm_cfg.get("gemini_model", "gemini-3.1-flash-lite"))
        return "", "empty_transcript", primary

    provider = str(llm_cfg.get("provider", "auto")).lower()
    gemini_key = os.environ.get(str(llm_cfg.get("gemini_api_key_env", "GEMINI_API_KEY")), "")
    openai_key = os.environ.get(str(llm_cfg.get("openai_api_key_env", "OPENAI_API_KEY")), "")

    if provider in {"auto", "gemini"} and gemini_key:
        rotation = get_gemini_model_rotation(llm_cfg)
        if model_filter:
            rotation = [(m, lim) for m, lim in rotation if m in model_filter]
        for model, _limit in rotation:
            if can_use_model is not None and not can_use_model(model):
                logger.debug("Skipping %s (daily quota reached)", model)
                continue
            try:
                summary = _summarize_gemini(cleaned, api_key=gemini_key, model=model)
                logger.info("LLM summary via gemini model=%s (%s chars)", model, len(summary))
                return summary, "gemini", model
            except error.HTTPError as exc:
                if exc.code == 429:
                    logger.warning(
                        "Gemini rate limit on model=%s; rotating to next model.",
                        model,
                    )
                    continue
                logger.warning("Gemini summarization failed (model=%s): %s", model, exc)
            except (error.URLError, RuntimeError, json.JSONDecodeError, OSError) as exc:
                logger.warning("Gemini summarization failed (model=%s): %s", model, exc)

    if provider in {"auto", "openai"} and openai_key:
        openai_model = str(llm_cfg.get("openai_model", "gpt-4o-mini"))
        try:
            summary = _summarize_openai(cleaned, api_key=openai_key, model=openai_model)
            logger.info("LLM summary via openai model=%s (%s chars)", openai_model, len(summary))
            return summary, "openai", openai_model
        except (error.URLError, error.HTTPError, RuntimeError, json.JSONDecodeError, OSError) as exc:
            logger.warning("OpenAI summarization failed (model=%s): %s", openai_model, exc)

    fallback = _extractive_fallback(cleaned)
    logger.warning("Using extractive summary fallback (no LLM API or all providers failed)")
    return fallback, "extractive_fallback", "extractive"

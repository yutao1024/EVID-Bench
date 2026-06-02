from __future__ import annotations

from typing import Any

TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")


def empty_token_counter() -> dict[str, int]:
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def normalize_token_counter(raw: Any) -> dict[str, int]:
    out = empty_token_counter()
    if not isinstance(raw, dict):
        return out
    for key in TOKEN_FIELDS:
        try:
            out[key] = int(raw.get(key, 0) or 0)
        except Exception:
            out[key] = 0
    return out


def add_token_counter(dst: dict[str, int], raw: Any) -> dict[str, int]:
    norm = normalize_token_counter(raw)
    for key in TOKEN_FIELDS:
        dst[key] = int(dst.get(key, 0) or 0) + norm[key]
    return dst


def add_model_token_usage(
    by_model: dict[str, dict[str, int]],
    model: str,
    raw: Any,
) -> dict[str, dict[str, int]]:
    model_name = (model or "").strip() or "unknown"
    bucket = by_model.setdefault(model_name, empty_token_counter())
    add_token_counter(bucket, raw)
    return by_model


def merge_model_token_usage(
    dst: dict[str, dict[str, int]],
    src: Any,
) -> dict[str, dict[str, int]]:
    if not isinstance(src, dict):
        return dst
    for model_name, usage in src.items():
        if not isinstance(usage, dict):
            continue
        add_model_token_usage(dst, str(model_name), usage)
    return dst

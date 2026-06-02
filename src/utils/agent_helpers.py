import base64
import json
import re
import time
from pathlib import Path
from typing import Any


def as_bool(val: Any, default: bool = False) -> bool:
    """Safely coerce a value to bool.

    Handles VLM outputs that may return ``"false"`` / ``"true"`` as strings
    (where ``bool("false")`` would wrongly be ``True``).
    """
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in ("true", "1", "yes")
    if isinstance(val, (int, float)):
        return bool(val)
    return default


def _coerce_to_dict(obj: Any) -> dict[str, Any] | None:
    """Best-effort: return obj if dict, or first dict element if obj is a
    non-empty list of dicts. Otherwise None.

    This handles models (e.g. gemini-3.1-pro) that occasionally wrap their
    JSON-object response in a top-level array like ``[{...}]``.
    """
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, list) and obj:
        first = obj[0]
        if isinstance(first, dict):
            return first
    return None


def extract_json_from_text(text: str) -> dict[str, Any]:
    """Extract a JSON object from raw model text.

    Accepts (in order of preference):
    - A pure JSON object ``{...}``
    - A pure JSON array of objects ``[{...}, ...]`` (returns first element)
    - Markdown-fenced object or array
    - Object/array substring embedded in surrounding prose
    """
    if not text:
        raise ValueError("Empty text; cannot extract JSON.")

    cleaned = text.strip()
    fenced = re.search(
        r"```(?:json)?\s*([\[{][\s\S]*[\]}])\s*```",
        cleaned,
        flags=re.IGNORECASE,
    )
    if fenced:
        cleaned = fenced.group(1).strip()

    try:
        obj = json.loads(cleaned)
        coerced = _coerce_to_dict(obj)
        if coerced is None:
            raise ValueError("Parsed JSON is not an object or array of objects.")
        return coerced
    except json.JSONDecodeError:
        pass

    candidates: list[tuple[int, str]] = []
    obj_first = cleaned.find("{")
    obj_last = cleaned.rfind("}")
    if obj_first != -1 and obj_last > obj_first:
        candidates.append((obj_first, cleaned[obj_first : obj_last + 1]))
    arr_first = cleaned.find("[")
    arr_last = cleaned.rfind("]")
    if arr_first != -1 and arr_last > arr_first:
        candidates.append((arr_first, cleaned[arr_first : arr_last + 1]))
    if not candidates:
        raise ValueError("No JSON object or array found in model output.")
    candidates.sort(key=lambda x: x[0])

    last_err: Exception | None = None
    for _, snippet in candidates:
        try:
            obj = json.loads(snippet)
        except json.JSONDecodeError as e:
            last_err = e
            continue
        coerced = _coerce_to_dict(obj)
        if coerced is not None:
            return coerced
        last_err = ValueError("Parsed JSON is not an object or array of objects.")
    raise last_err if last_err else ValueError(
        "Parsed JSON is not an object or array of objects."
    )


def sanitize_query_text(query: str) -> str:
    """Normalize model-generated search query text for YouTube retrieval."""
    q = str(query or "").strip()
    if not q:
        return ""
    q = re.sub(r"^(new\s+)?query\s*\d*\s*:\s*", "", q, flags=re.IGNORECASE).strip()
    q = re.sub(r"\s+", " ", q).strip()
    return q


def sanitize_queries(raw_queries: Any, *, limit: int = 3) -> list[str]:
    """Sanitize, deduplicate, and clip query list."""
    if not isinstance(raw_queries, list):
        return []
    out: list[str] = []
    for item in raw_queries:
        q = sanitize_query_text(str(item))
        if q and q not in out:
            out.append(q)
        if len(out) >= limit:
            break
    return out


def _build_multimodal_content(
    text_prompt: str,
    image_paths: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Build OpenAI multimodal message content: text + optional images (base64).

    Returns a list suitable for the 'content' field of a user message.
    """
    parts: list[dict[str, Any]] = [{"type": "text", "text": text_prompt}]
    if image_paths:
        for p in image_paths:
            with open(p, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_b64}",
                    },
                }
            )
    return parts


def call_vlm_with_retry(
    client: Any,
    model: str,
    contents: list[Any],
    *,
    max_retries: int = 3,
    temperature: float = 0.0,
    json_mode: bool = True,
    retry_backoff_sec: float = 1.5,
    logger: Any = None,
    log_prefix: str = "",
) -> tuple[str, dict[str, int]]:
    """Call OpenAI-compatible chat completions API (e.g. OpenRouter) with retry.

    `contents` is the list returned by _build_multimodal_content():
      [{"type": "text", "text": "..."}, {"type": "image_url", ...}, ...]

    Returns (content_str, token_usage_dict).
    token_usage_dict has keys: prompt_tokens, completion_tokens, total_tokens (all int, 0 on failure).
    """
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": [{"role": "user", "content": contents}],
                "temperature": temperature,
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            response = client.chat.completions.create(**kwargs)

            # Defensive: providers (OpenRouter / upstream Gemini) can return
            # responses with no choices, or with a choice that has no message,
            # or with message.content == None. Classify all of these as
            # vlm_no_content so retry can kick in instead of crashing the
            # whole agent turn with "'NoneType' object is not subscriptable".
            choices = getattr(response, "choices", None)
            if not choices:
                raise ValueError(
                    f"vlm_no_content: response has no choices (response={response!r})"
                )
            msg = getattr(choices[0], "message", None)
            if msg is None:
                raise ValueError(
                    "vlm_no_content: response.choices[0] has no message field"
                )
            content = getattr(msg, "content", None)
            if content is None:
                # Some providers wrap an error object inside choices[0]; surface
                # it in the error message so the log shows what actually came back.
                err_obj = getattr(choices[0], "error", None) or getattr(response, "error", None)
                raise ValueError(
                    f"vlm_no_content: message.content is None (provider_error={err_obj!r})"
                )
            if not str(content).strip():
                raise ValueError("vlm_no_content: VLM returned empty content string.")

            # Extract token usage
            usage = getattr(response, "usage", None)
            tokens = {
                "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0,
                "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0,
                "total_tokens": int(getattr(usage, "total_tokens", 0) or 0) if usage else 0,
            }
            if callable(logger):
                logger(
                    f"{log_prefix}VLM call success on attempt {attempt + 1}/{max_retries} "
                    f"(tokens: {tokens['prompt_tokens']}p + {tokens['completion_tokens']}c = {tokens['total_tokens']}t)"
                )
            return content, tokens
        except Exception as exc:
            last_error = exc
            if callable(logger):
                logger(
                    f"{log_prefix}VLM call failed on attempt {attempt + 1}/{max_retries}: {exc}"
                )
            if attempt == max_retries - 1:
                break
            sleep_s = retry_backoff_sec * (2**attempt)
            if callable(logger):
                logger(f"{log_prefix}Retrying after {sleep_s:.1f}s")
            time.sleep(sleep_s)
    raise RuntimeError(f"VLM call failed after retries: {last_error}") from last_error


import cv2

from src.utils.frame_sampling import sample_frames_uniform


def uniform_sample_frames(
    video_path: str,
    *,
    num_frames: int = 64,
    output_dir: str | Path,
    prefix: str = "frame",
    jpeg_quality: int = 85,
    target_height: int | None = None,
    cache_dir: str | Path | None = None,
    max_workers: int | None = None,
    log_fn=None,
) -> tuple[list[str], bool]:
    """Uniformly sample `num_frames` from `video_path` via ffmpeg (single sequential decode).

    If `target_height` is set, frames are resized to that height in parallel after extract.

    If `cache_dir` is set, cached ``{prefix}_{frame_idx}.jpg`` files are reused on hit.

    Returns ``(frame_paths, cache_hit)``. Raises on missing ffmpeg or decode failure.
    """
    video = Path(video_path)
    if not video.is_file():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    if cache_dir is not None:
        cached = sorted(str(p) for p in Path(cache_dir).glob(f"{prefix}_*.jpg"))
        if cached:
            return cached, True

    out_dir = Path(cache_dir) if cache_dir is not None else Path(output_dir)
    saved = sample_frames_uniform(
        str(video),
        num_frames,
        out_dir,
        prefix=prefix,
        target_height=target_height,
        jpeg_quality=jpeg_quality,
        max_workers=max_workers,
        log_fn=log_fn,
    )
    if not saved:
        raise ValueError(f"No frames extracted from video: {video_path}")
    return saved, False


def score_shot_informativeness(frame_path: str) -> dict[str, float]:
    """Quick offline information-density score for a single shot frame.

    Returns: {"sharpness": float, "brightness": float, "color_var": float}.
    All zeros on read failure.
    """
    try:
        img = cv2.imread(frame_path)
        if img is None:
            return {"sharpness": 0.0, "brightness": 0.0, "color_var": 0.0}
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(gray.mean())
        color_var = float(img.astype("float32").std())
        return {
            "sharpness": sharpness,
            "brightness": brightness,
            "color_var": color_var,
        }
    except Exception:
        return {"sharpness": 0.0, "brightness": 0.0, "color_var": 0.0}


def _transnetv2_boundaries(video_path: str) -> list[int] | None:
    """Try TransNetV2 from common Python package shapes."""
    try:
        from transnetv2 import TransNetV2  # type: ignore
    except Exception:
        return None

    model = TransNetV2()
    predictions = model.predict_video(video_path)
    if predictions is None:
        return None

    boundaries: list[int] = []
    if isinstance(predictions, tuple) and predictions:
        scores = predictions[0]
    else:
        scores = predictions

    for idx, score in enumerate(scores):
        score_value = float(score[0] if hasattr(score, "__len__") else score)
        if score_value >= 0.5:
            boundaries.append(idx)
    return sorted(set(boundaries))


def _naive_boundaries(video_path: str, threshold: float = 28.0) -> list[int]:
    """Fallback boundary detector based on frame-diff if TransNetV2 unavailable."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    prev_gray = None
    frame_idx = 0
    boundaries: list[int] = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            mean_diff = float(diff.mean())
            if mean_diff > threshold:
                boundaries.append(frame_idx)
        prev_gray = gray
        frame_idx += 1

    cap.release()
    return boundaries


def detect_shots(video_path: str) -> list[dict[str, Any]]:
    """
    Detect shot boundaries and return shot metadata.

    Returns:
      [
        {
          "shot_id": int,
          "start_frame": int,
          "end_frame": int,
          "start_sec": float,
          "end_sec": float
        },
        ...
      ]
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    cap.release()

    if total_frames <= 0:
        raise ValueError(f"Invalid frame count for video: {video_path}")

    boundaries = _transnetv2_boundaries(video_path)
    if boundaries is None:
        boundaries = _naive_boundaries(video_path)

    boundaries = sorted({b for b in boundaries if 0 < b < total_frames})

    shots: list[dict[str, Any]] = []
    start = 0
    shot_id = 0
    for boundary in boundaries:
        end = max(start, boundary - 1)
        shots.append(
            {
                "shot_id": shot_id,
                "start_frame": start,
                "end_frame": end,
                "start_sec": start / fps,
                "end_sec": end / fps,
            }
        )
        shot_id += 1
        start = boundary

    shots.append(
        {
            "shot_id": shot_id,
            "start_frame": start,
            "end_frame": total_frames - 1,
            "start_sec": start / fps,
            "end_sec": (total_frames - 1) / fps,
        }
    )
    return shots

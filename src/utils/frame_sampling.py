"""Frame extraction: single-pass ffmpeg decode; optional parallel resize to target height."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2

LogFn = Callable[[str], None] | None


def default_frame_resize_workers() -> int:
    """Default thread count for parallel frame resize (not ffmpeg decode)."""
    raw = os.environ.get("FRAME_RESIZE_WORKERS", "16").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 16


def _log_msg(log_fn: LogFn, msg: str) -> None:
    if log_fn is not None:
        log_fn(msg)
    else:
        print(msg, flush=True)


def find_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    for candidate in (
        Path.home() / "miniconda3/bin/ffmpeg",
    ):
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(
        "ffmpeg not found. Install ffmpeg or add it to PATH "
        "(e.g. conda install -c conda-forge ffmpeg)."
    )


def find_ffprobe() -> str:
    exe = shutil.which("ffprobe")
    if exe:
        return exe
    for candidate in (
        Path.home() / "miniconda3/bin/ffprobe",
    ):
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError("ffprobe not found (install alongside ffmpeg).")


def _parse_fps(rate: str) -> float:
    rate = (rate or "").strip()
    if not rate:
        return 25.0
    if "/" in rate:
        num, den = rate.split("/", 1)
        try:
            den_f = float(den)
            return float(num) / den_f if den_f else 25.0
        except ValueError:
            return 25.0
    try:
        return float(rate)
    except ValueError:
        return 25.0


def probe_video_frame_count(video_path: str) -> tuple[int, float]:
    """Return ``(frame_count, fps)`` using ffprobe metadata (fast, no full decode)."""
    ffprobe = find_ffprobe()
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_frames,r_frame_rate",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(video_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(proc.stdout or "{}")
    streams = data.get("streams") or []
    fmt = data.get("format") or {}
    stream = streams[0] if streams else {}
    fps = _parse_fps(str(stream.get("r_frame_rate", "25/1")))
    nb = stream.get("nb_frames")
    if nb is not None and str(nb).isdigit() and int(nb) > 0:
        return int(nb), fps
    duration = float(fmt.get("duration") or 0.0)
    if duration > 0 and fps > 0:
        return max(1, int(round(duration * fps))), fps
    raise ValueError(f"Cannot determine frame count for video: {video_path}")


def compute_uniform_indices(total: int, num_frames: int) -> list[int]:
    total = int(total)
    if total <= 0:
        raise ValueError("total must be positive")
    n = max(1, min(int(num_frames), total))
    if n == 1:
        return [total // 2]
    return [round(i * (total - 1) / (n - 1)) for i in range(n)]


def _jpeg_qscale(jpeg_quality: int) -> int:
    """Map OpenCV-style JPEG quality (0–100) to ffmpeg ``-q:v`` (1–31, lower is better)."""
    q = 31 - (max(0, min(100, int(jpeg_quality))) / 100.0) * 30
    return max(1, min(31, int(round(q))))


def _resize_frame_file(path: str, target_height: int, jpeg_quality: int) -> None:
    """Resize one JPEG in place to ``target_height`` (width even, proportional)."""
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"Cannot read frame image: {path}")
    h, w = img.shape[:2]
    th = int(target_height)
    if h > th:
        new_w = max(2, int(round(w * th / h)))
        if new_w % 2:
            new_w += 1
        img = cv2.resize(img, (new_w, th), interpolation=cv2.INTER_AREA)
    q = max(0, min(100, int(jpeg_quality)))
    if not cv2.imwrite(path, img, [int(cv2.IMWRITE_JPEG_QUALITY), q]):
        raise RuntimeError(f"Failed to write resized frame: {path}")


def compress_frames_parallel(
    frame_paths: list[str],
    target_height: int,
    *,
    jpeg_quality: int = 85,
    max_workers: int | None = None,
    log_fn: LogFn = None,
) -> None:
    """Resize multiple JPEGs in parallel (used after single-pass ffmpeg extract)."""
    if not frame_paths:
        return
    workers = max_workers if max_workers is not None else default_frame_resize_workers()
    pool_size = 1 if len(frame_paths) == 1 else min(workers, len(frame_paths))
    _log_msg(
        log_fn,
        f"[Sampling] parallel resize: {len(frame_paths)} frames to height<={target_height}, "
        f"workers={pool_size}",
    )
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=pool_size) as executor:
        futures = {
            executor.submit(_resize_frame_file, p, target_height, jpeg_quality): p
            for p in frame_paths
        }
        for fut in as_completed(futures):
            path = futures[fut]
            try:
                fut.result()
            except Exception as exc:
                errors.append(f"{path}: {exc}")
    if errors:
        raise RuntimeError(
            f"parallel frame resize failed ({len(errors)}/{len(frame_paths)}): "
            + "; ".join(errors[:5])
            + (" ..." if len(errors) > 5 else "")
        )


def sample_frames_at_indices(
    video_path: str,
    indices: list[int],
    out_dir: str | Path,
    prefix: str = "frame",
    *,
    target_height: int | None = None,
    jpeg_quality: int = 85,
    max_workers: int | None = None,
    log_fn: LogFn = None,
) -> list[str]:
    """Extract frames at given indices with one ffmpeg pass (sequential decode).

    When ``target_height`` is set, frames are extracted at source resolution, then
    resized in parallel via ``max_workers`` threads (``FRAME_RESIZE_WORKERS``).
    """
    if not indices:
        return []
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ffmpeg = find_ffmpeg()

    select_parts = "+".join(f"eq(n\\,{idx})" for idx in indices)
    vf = f"select='{select_parts}'"

    tmp_pattern = str(out / f"_tmp_{prefix}_%08d.jpg")
    for old in out.glob(f"_tmp_{prefix}_*.jpg"):
        old.unlink(missing_ok=True)

    _log_msg(
        log_fn,
        f"[Sampling] ffmpeg sequential extract: {len(indices)} frames from {video_path}",
    )

    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        vf,
        "-vsync",
        "vfr",
        "-q:v",
        str(_jpeg_qscale(jpeg_quality)),
        tmp_pattern,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise RuntimeError(f"ffmpeg failed to run: {exc}") from exc
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"ffmpeg frame extract failed (code {proc.returncode}): {err}")

    tmp_files = sorted(out.glob(f"_tmp_{prefix}_*.jpg"))
    if len(tmp_files) < len(indices):
        for f in tmp_files:
            f.unlink(missing_ok=True)
        raise RuntimeError(
            f"ffmpeg extracted {len(tmp_files)} frames, expected {len(indices)}"
        )

    saved: list[str] = []
    for tmp, idx in zip(tmp_files, indices, strict=True):
        dest = out / f"{prefix}_{idx:08d}.jpg"
        if dest.exists():
            dest.unlink()
        tmp.replace(dest)
        saved.append(str(dest))

    if target_height is not None:
        compress_frames_parallel(
            saved,
            int(target_height),
            jpeg_quality=jpeg_quality,
            max_workers=max_workers,
            log_fn=log_fn,
        )

    return saved


def resolve_frame_cache_dir(
    cache_root: str | Path,
    video_id: str,
    *,
    height: int,
    num_frames: int,
    start_sec: float | None = None,
    end_sec: float | None = None,
    source: str = "",
) -> Path:
    """Persistent frame cache directory.

    Example: ``.frame_cache/input/NX7QNWEGcNI_h480_n64/`` or ``.frame_cache/candidate/ySqunOZDDMo_h480_n16/``.

    ``source`` creates a subdirectory (e.g. ``input/``, ``candidate/``) to avoid
    collisions when a forged video and its YouTube source share the same ID.
    """
    root = Path(cache_root) / source if source else Path(cache_root)
    name = f"{video_id}_h{int(height)}"
    if start_sec is not None and end_sec is not None and end_sec > start_sec:
        name += f"_s{int(start_sec)}_{int(end_sec)}"
    name += f"_n{int(num_frames)}"
    return root / name


def sample_frames_uniform(
    video_path: str,
    num_frames: int,
    out_dir: str | Path,
    prefix: str = "frame",
    *,
    target_height: int | None = None,
    jpeg_quality: int = 85,
    total_frames: int | None = None,
    fps: float | None = None,
    max_workers: int | None = None,
    log_fn: LogFn = None,
) -> list[str]:
    """Uniformly sample ``num_frames`` from ``video_path`` via ffmpeg."""
    if total_frames is None or fps is None:
        total_frames, fps = probe_video_frame_count(video_path)
    del fps  # reserved for windowed sampling callers
    indices = compute_uniform_indices(int(total_frames), num_frames)
    return sample_frames_at_indices(
        video_path,
        indices,
        out_dir,
        prefix=prefix,
        target_height=target_height,
        jpeg_quality=jpeg_quality,
        max_workers=max_workers,
        log_fn=log_fn,
    )

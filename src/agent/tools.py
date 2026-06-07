from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import urllib.request
from pathlib import Path
from typing import Any

import cv2

from src.agent.oracle_eval import extract_youtube_id
from src.agent.session_state import CandidateRecord, SessionState
from src.utils.agent_helpers import (
    _build_multimodal_content,
    as_bool,
    call_vlm_with_retry,
    extract_json_from_text,
    sanitize_queries,
)
from src.agent.prompts import (
    PROMPT_COARSE_RELEVANCE,
    PROMPT_DEEPSEARCH_NEXT_STEP,
    PROMPT_FINE_FORGERY_POINTS,
)
from src.utils.config import resolve_yt_dlp_cookies_files

# yt-dlp leaves intermediate fragment files named like "<id>.f251.webm" (audio-only)
# or "<id>.f137.mp4" (video-only) when a merge step fails.
_DURATION_PATTERN = re.compile(r"^(\d+):(\d+)(?::(\d+))?$")
_FRAGMENT_FILE_PATTERN = re.compile(r"\.f\d{2,4}\.(?:webm|mp4|m4a|mka|opus)$", re.IGNORECASE)


# Backwards-compat shim so existing call sites elsewhere keep working.
def _extract_youtube_id_from_url(url: str) -> str | None:
    return extract_youtube_id(url)


_DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) "
    "Version/17.5 Safari/605.1.15"
)
# `tv` is the most reliable client for unauthenticated/bot-challenged downloads in
# 2026 yt-dlp; mweb / web_safari serve as fallback shapes. Avoid `android` here as
# it has been hard-blocked in 2025.
_DEFAULT_PLAYER_CLIENTS = "web,android"
# yt-dlp 2026 requires a JS runtime to solve player challenges; without one, every
# YouTube download trips "Sign in to confirm you're not a bot" / 403. We try deno
# first (default), then node/bun.
_JS_RUNTIME_CANDIDATES = ("deno", "node", "bun")
_DOWNLOAD_TIMEOUT_SECONDS = 240
_COOKIE_RETRY_REASONS = frozenset(
    {"http_403_bot_check", "http_429_rate_limited", "age_restricted"}
)
_COOKIE_STOP_REASONS = frozenset(
    {"unavailable", "private_video", "geo_block", "http_404", "invalid_candidate_url"}
)
_cookies_rr_lock = threading.Lock()
_cookies_rr_index = 0


def _next_cookies_file(cookies_files: list[str]) -> str | None:
    """Round-robin pick the next cookies file (thread-safe across workers)."""
    global _cookies_rr_index
    if not cookies_files:
        return None
    with _cookies_rr_lock:
        idx = _cookies_rr_index % len(cookies_files)
        _cookies_rr_index += 1
    return cookies_files[idx]


def _cookies_attempt_order(cookies_files: list[str]) -> list[str]:
    """Primary cookie first (round-robin), then the rest for fallback retries."""
    if not cookies_files:
        return []
    primary = _next_cookies_file(cookies_files)
    if primary is None:
        return []
    order = [primary]
    order.extend(path for path in cookies_files if path != primary)
    return order


def _detect_js_runtime() -> tuple[str, str] | None:
    """Return (runtime_name, absolute_path) of the first available JS runtime, or None."""
    for name in _JS_RUNTIME_CANDIDATES:
        path = shutil.which(name)
        if path:
            return name, path
    return None


def _parse_duration_seconds(text: str) -> int | None:
    """Parse duration string like '1:23', '12:34', '1:23:45' or plain seconds to int seconds."""
    if not text:
        return None
    s = str(text).strip()
    if not s:
        return None
    if s.isdigit():
        try:
            return int(s)
        except Exception:
            return None
    m = _DURATION_PATTERN.match(s)
    if not m:
        return None
    parts = [int(g) for g in m.groups() if g is not None]
    if len(parts) == 2:
        mm, ss = parts
        return mm * 60 + ss
    if len(parts) == 3:
        hh, mm, ss = parts
        return hh * 3600 + mm * 60 + ss
    return None


def _classify_ytdlp_failure(stderr: str) -> str:
    """Classify yt-dlp failure stderr into a coarse reason_code."""
    s = (stderr or "").lower()
    if not s:
        return "unknown"
    if "no supported javascript runtime" in s or "js runtime" in s:
        return "missing_js_runtime"
    if "sign in to confirm your age" in s or ("age" in s and "restrict" in s):
        return "age_restricted"
    if "sign in to confirm" in s or "confirm you" in s or "not a bot" in s:
        return "http_403_bot_check"
    if "429" in s or "too many requests" in s:
        return "http_429_rate_limited"
    if "403" in s and "forbidden" in s:
        return "http_403_bot_check"
    if "private video" in s or "members-only" in s:
        return "private_video"
    if "video unavailable" in s or "this video has been removed" in s:
        return "unavailable"
    if "geo" in s or "not available in your country" in s:
        return "geo_block"
    if "timed out" in s or "timeout" in s:
        return "timeout"
    if "404" in s:
        return "http_404"
    return "unknown"


def _is_video_file_healthy(path: str) -> bool:
    """Quick OpenCV-based check that a file actually decodes to >0 frames at >0x0."""
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            cap.release()
            return False
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return total > 0 and width > 0 and height > 0
    except Exception:
        return False


def _pick_downloaded_video(
    output_dir: str,
    *,
    expected_video_id: str | None = None,
) -> tuple[str | None, str | None]:
    """Return (chosen_path, reason_code). reason_code only set on failure.

    If expected_video_id is provided, only files with that id prefix are considered.
    """
    extensions_priority = [".mp4", ".mkv", ".mov", ".webm"]
    files: list[Path] = []
    for p in Path(output_dir).glob("*"):
        if not p.is_file():
            continue
        if expected_video_id and not p.name.startswith(f"{expected_video_id}."):
            continue
        if p.suffix.lower() not in extensions_priority:
            continue
        if _FRAGMENT_FILE_PATTERN.search(p.name):
            # yt-dlp leftover audio-/video-only fragment when merging failed.
            continue
        files.append(p)
    if not files:
        return None, "no_video_file"
    files.sort(
        key=lambda p: (
            extensions_priority.index(p.suffix.lower())
            if p.suffix.lower() in extensions_priority
            else len(extensions_priority),
            -p.stat().st_size,
        )
    )
    for p in files:
        if _is_video_file_healthy(str(p)):
            return str(p), None
    return None, "decode_failed"


class AgentTools:
    def __init__(
        self,
        *,
        api_config: dict[str, Any],
        model: str,
        candidate_sample_frames: int,
        candidate_video_height: int = 480,
        prompts: dict[str, str],
        logger,
        coarse_frames: int = 16,
        dense_frames: int = 16,
        query_temperature: float = 0.0,
        frame_resize_workers: int | None = None,
    ) -> None:
        self.api_config = api_config
        self.model = model
        if frame_resize_workers is None:
            from src.utils.frame_sampling import default_frame_resize_workers

            self.frame_resize_workers = default_frame_resize_workers()
        else:
            self.frame_resize_workers = max(1, int(frame_resize_workers))
        self.candidate_sample_frames = candidate_sample_frames
        self.candidate_video_height = max(1, int(candidate_video_height))
        self.prompts = prompts
        self.log = logger
        self.coarse_frames = coarse_frames
        self.dense_frames = dense_frames
        self.query_temperature = query_temperature
        self._cookies_files = resolve_yt_dlp_cookies_files()
        if self._cookies_files:
            names = ", ".join(Path(path).name for path in self._cookies_files)
            self.log(
                f"[yt-dlp] cookies rotation enabled ({len(self._cookies_files)} files): {names}"
            )
        elif os.getenv("YT_DLP_COOKIES_FILE", "").strip() or os.getenv("YT_DLP_COOKIES_FILES", "").strip() or os.getenv("YT_DLP_COOKIES_DIR", "").strip():
            self.log("[yt-dlp] cookies configured but no valid cookie files found")
        self._cookies_browser = (os.getenv("YT_DLP_COOKIES_FROM_BROWSER") or "").strip() or None
        if self._cookies_browser:
            self.log(
                f"[yt-dlp] cookies-from-browser fallback enabled: {self._cookies_browser}"
            )
        self._js_runtime = _detect_js_runtime()
        if self._js_runtime:
            self.log(
                f"[yt-dlp] JS runtime detected: {self._js_runtime[0]} at {self._js_runtime[1]}"
            )
        else:
            self.log(
                "[yt-dlp] WARNING: no JS runtime found (deno/node/bun). "
                "YouTube downloads will likely 403 'not a bot'. "
                "Install one e.g. `brew install deno` or `brew install node`."
            )

    def _ytdlp_common_args(
        self,
        *,
        use_browser_cookies: bool = False,
        cookies_file: str | None = None,
    ) -> list[str]:
        """Common yt-dlp anti-bot flags for download."""
        args = [
            "--user-agent",
            _DEFAULT_USER_AGENT,
            "--extractor-args",
            f"youtube:player_client={_DEFAULT_PLAYER_CLIENTS}",
            "--retries",
            "3",
            "--fragment-retries",
            "3",
        ]
        if self._js_runtime:
            name, path = self._js_runtime
            args.extend(["--js-runtimes", f"{name}:{path}"])
        if use_browser_cookies and self._cookies_browser:
            args.extend(["--cookies-from-browser", self._cookies_browser])
        elif cookies_file:
            args.extend(["--cookies", cookies_file])
        return args

    def _vlm_json_from_frames(
        self,
        frame_paths: list[str],
        prompt: str,
        *,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        contents = _build_multimodal_content(prompt, image_paths=frame_paths)
        raw, tokens = call_vlm_with_retry(
            self.api_config,
            self.model,
            contents,
            image_paths=frame_paths,
            max_retries=3,
            temperature=temperature,
            json_mode=True,
            logger=self.log,
            log_prefix="[VLM] ",
        )
        return extract_json_from_text(raw)

    def extract_clues(self, shot_frame_paths: list[str]) -> dict[str, Any]:
        data = self._vlm_json_from_frames(
            shot_frame_paths, self.prompts["extract"], temperature=self.query_temperature
        )
        raw_forbidden = data.get("forbidden_overlay_text") or []
        if not isinstance(raw_forbidden, list):
            raw_forbidden = []
        forbidden = [str(x).strip() for x in raw_forbidden if str(x).strip()]
        return {
            "reasoning": str(data.get("reasoning") or "").strip(),
            "forbidden_overlay_text": forbidden,
            "physical_observations": str(data.get("physical_observations") or "").strip(),
            "queries": sanitize_queries(data.get("queries"), limit=4),
        }

    def _cluster_batch(
        self,
        shot_id_list: list[int],
        rep_frame_paths: list[str],
        max_image_dim: int,
        batch_idx: int,
        target_groups: int = 5,
    ) -> list[dict[str, Any]]:
        """Run a single batch clustering VLM call.

        Returns [{shot_ids, physical_observations, queries}, ...].
        On JSON failure, returns empty list.
        """
        prompt_template = self.prompts.get("cluster") or ""
        if not prompt_template:
            raise RuntimeError("cluster prompt missing")

        mini_paths: list[str] = []
        for sid, src_path in zip(shot_id_list, rep_frame_paths):
            img = cv2.imread(src_path)
            if img is None:
                continue
            h, w = img.shape[:2]
            scale = float(max_image_dim) / float(max(h, w))
            if scale < 1.0:
                new_w = max(1, int(round(w * scale)))
                new_h = max(1, int(round(h * scale)))
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            mini_paths.append(img)  # keep as ndarray to avoid tmp disk writes

        # Write resized images to a temp dir (needed for PIL-based VLM).
        tmp_root = Path(tempfile.mkdtemp(prefix=f"cluster_batch_{batch_idx}_"))
        try:
            saved_paths: list[str] = []
            for sid, img in zip(shot_id_list, mini_paths):
                out_path = str(tmp_root / f"shot_{sid:04d}.jpg")
                cv2.imwrite(out_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                saved_paths.append(out_path)

            shot_id_block = ", ".join(str(s) for s in shot_id_list)
            prompt = (
                prompt_template
                .replace("__SHOT_ID_LIST__", shot_id_block)
                .replace("__N_SHOTS__", str(len(shot_id_list)))
                .replace("__TARGET_GROUPS__", str(int(target_groups)))
            )
            contents = _build_multimodal_content(prompt, image_paths=saved_paths)
            raw_text, _tokens = call_vlm_with_retry(
                self.api_config,
                self.model,
                contents,
                image_paths=saved_paths,
                max_retries=3,
                temperature=self.query_temperature,
                json_mode=True,  # Helps enforce JSON format
                logger=self.log,
                log_prefix=f"[ClusterBatch-{batch_idx}] ",
            )
            data = extract_json_from_text(raw_text)
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)

        raw_groups = data.get("groups")
        groups_out: list[dict[str, Any]] = []
        if isinstance(raw_groups, list):
            seen: set[int] = set()
            for raw in raw_groups:
                if not isinstance(raw, dict):
                    continue
                sid_raw = raw.get("shot_ids")
                if not isinstance(sid_raw, list):
                    continue
                shot_ids: list[int] = []
                for s in sid_raw:
                    try:
                        s_int = int(s)
                    except Exception:
                        continue
                    if s_int in seen or s_int not in shot_id_list:
                        continue
                    shot_ids.append(s_int)
                    seen.add(s_int)
                if not shot_ids:
                    continue
                raw_forbidden = raw.get("forbidden_overlay_text") or []
                if not isinstance(raw_forbidden, list):
                    raw_forbidden = []
                forbidden = [str(x).strip() for x in raw_forbidden if str(x).strip()]
                groups_out.append(
                    {
                        "shot_ids": shot_ids,
                        "reasoning": str(raw.get("reasoning") or "").strip(),
                        "forbidden_overlay_text": forbidden,
                        "physical_observations": str(raw.get("physical_observations") or "").strip(),
                        "queries": sanitize_queries(raw.get("queries"), limit=4),
                    }
                )
        # Orphans within this batch (not assigned by VLM) generate singleton groups.
        assigned = {s for g in groups_out for s in g["shot_ids"]}
        for sid in shot_id_list:
            if sid not in assigned:
                groups_out.append(
                    {
                        "shot_ids": [sid],
                        "reasoning": "",
                        "forbidden_overlay_text": [],
                        "physical_observations": "",
                        "queries": [],
                    }
                )
        return groups_out

    def cluster_shots_with_vlm(
        self,
        shot_frame_map: dict[int, list[str]],
        *,
        max_image_dim: int = 192,
        batch_size: int = 20,
        target_groups: int = 5,
        force_single_batch: bool = False,
    ) -> dict[str, Any]:
        """Cluster shots by visual source using batched VLM calls.

        When there are many shots (e.g. 64), we split them into batches of
        `batch_size` to avoid putting too many images in one VLM call (which
        degrades JSON accuracy). After all batches finish, a second lightweight
        pass merges similar groups across batches by comparing their
        representative frames.

        Returns: {"groups": [{"shot_ids": [...], "physical_observations": str,
                              "queries": [str, str, str]}, ...]}
        """
        non_empty: list[tuple[int, str]] = []
        for sid in sorted(shot_frame_map.keys()):
            frames = shot_frame_map.get(sid) or []
            if not frames:
                continue
            mid_path = frames[len(frames) // 2]
            non_empty.append((int(sid), mid_path))

        if not non_empty:
            return {"groups": []}

        # Force single batch when requested: all shots in one VLM call so the
        # 'AT MOST N groups' hard constraint can actually bind globally.
        effective_batch_size = (
            len(non_empty) if force_single_batch else batch_size
        )

        # Batch by consecutive shot_id order (they are time-ordered).
        all_batch_groups: list[list[dict[str, Any]]] = []
        for batch_start in range(0, len(non_empty), effective_batch_size):
            batch = non_empty[batch_start : batch_start + effective_batch_size]
            batch_sids = [sid for sid, _ in batch]
            batch_paths = [p for _, p in batch]
            batch_idx = batch_start // effective_batch_size
            self.log(
                f"[Cluster] batch {batch_idx + 1}/"
                f"{max(1, (len(non_empty) + effective_batch_size - 1)//effective_batch_size)}: "
                f"shots {batch_sids[0]}-{batch_sids[-1]} ({len(batch_sids)} shots)"
            )
            try:
                bg = self._cluster_batch(
                    batch_sids,
                    batch_paths,
                    max_image_dim,
                    batch_idx,
                    target_groups=target_groups,
                )
                all_batch_groups.append(bg)
            except Exception as exc:
                self.log(f"[Cluster] batch {batch_idx} failed: {exc}")
                # If single-batch attempt failed, retry once with the original
                # batched route (smaller batches, more robust JSON).
                if force_single_batch and batch_size < len(non_empty):
                    self.log(
                        f"[Cluster] single-batch failed; falling back to batched "
                        f"({batch_size} shots/batch)"
                    )
                    return self.cluster_shots_with_vlm(
                        shot_frame_map,
                        max_image_dim=max_image_dim,
                        batch_size=batch_size,
                        target_groups=target_groups,
                        force_single_batch=False,
                    )
                # Fallback: each shot in batch as its own group.
                fallback = [
                    {
                        "shot_ids": [sid],
                        "reasoning": "",
                        "forbidden_overlay_text": [],
                        "physical_observations": "",
                        "queries": [],
                    }
                    for sid in batch_sids
                ]
                all_batch_groups.append(fallback)

        # Flatten all groups from all batches.
        all_groups = [g for batch in all_batch_groups for g in batch]
        return {"groups": all_groups}

    def reflect_queries(
        self, shot_frame_paths: list[str], prev_queries: list[str], wrong_titles: list[str]
    ) -> dict[str, Any]:
        prompt = self.prompts["reflect"].replace(
            "{prev_queries}", json.dumps(prev_queries, ensure_ascii=False)
        ).replace("{candidate_titles}", json.dumps(wrong_titles, ensure_ascii=False))
        data = self._vlm_json_from_frames(
            shot_frame_paths, prompt, temperature=self.query_temperature
        )
        raw_neg = data.get("negative_keywords")
        negative_keywords: list[str] = []
        if isinstance(raw_neg, list):
            for item in raw_neg:
                s = str(item).strip()
                if s and s not in negative_keywords:
                    negative_keywords.append(s)
        return {
            "reasoning": str(data.get("reasoning") or "").strip(),
            "reflection": str(data.get("reflection") or "").strip(),
            "negative_keywords": negative_keywords[:8],
            "new_queries": sanitize_queries(data.get("new_queries"), limit=3),
        }

    def verify_match(self, shot_frame_paths: list[str], candidate_frame_paths: list[str]) -> dict[str, Any]:
        prompt = self.prompts["verify"].replace("{num_target}", str(len(shot_frame_paths)))
        data = self._vlm_json_from_frames(shot_frame_paths + candidate_frame_paths, prompt)
        try:
            confidence = float(data.get("confidence", 0.0) or 0.0)
        except Exception:
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        seg_raw = data.get("most_similar_segment") or {}
        if isinstance(seg_raw, dict):
            try:
                start_sec = float(seg_raw.get("start_sec", 0.0) or 0.0)
            except Exception:
                start_sec = 0.0
            try:
                end_sec = float(seg_raw.get("end_sec", 0.0) or 0.0)
            except Exception:
                end_sec = 0.0
            if end_sec < start_sec:
                end_sec = start_sec
            most_similar_segment = {"start_sec": start_sec, "end_sec": end_sec}
        else:
            most_similar_segment = {"start_sec": 0.0, "end_sec": 0.0}
        return {
            "reasoning": str(data.get("reasoning") or "").strip(),
            "is_match": bool(data.get("is_match", False)),
            "confidence": confidence,
            "most_similar_segment": most_similar_segment,
        }

    def analyze_manipulation(
        self, shot_frame_paths: list[str], candidate_frame_paths: list[str], candidate_id: str
    ) -> dict[str, Any]:
        prompt = self.prompts["manipulation"].replace("__SOURCE_ID__", candidate_id)
        data = self._vlm_json_from_frames(shot_frame_paths + candidate_frame_paths, prompt)
        raw_types = data.get("manipulation_types")
        mtypes = [str(x).strip() for x in raw_types] if isinstance(raw_types, list) else []
        regions = data.get("suspect_regions")
        suspect_regions = [str(x).strip() for x in regions] if isinstance(regions, list) else []
        return {
            "is_likely_manipulated": bool(data.get("is_likely_manipulated", False)),
            "manipulation_types": [x for x in mtypes if x],
            "evidence": str(data.get("evidence") or "").strip(),
            "suspect_regions": [x for x in suspect_regions if x],
        }

    def search_candidates(
        self,
        session: SessionState,
        query: str,
        top_k: int,
    ) -> dict[str, Any]:
        query = query.strip()
        results = self._search_serper(query, top_k)
        source = "serper"

        existing_ids = {
            _extract_youtube_id_from_url(rec.url)
            for rec in session.candidates.values()
        }
        existing_ids.discard(None)

        # Stage 1: dedup + register, but as PENDING (not yet committed if rank drops them)
        registered: list[tuple[str, dict[str, Any]]] = []
        skipped_dup = 0
        for item in results:
            url = item["url"]
            yid = _extract_youtube_id_from_url(url)
            if yid and yid in existing_ids:
                skipped_dup += 1
                continue
            ref = session.next_candidate_ref()
            rec = CandidateRecord(
                ref=ref,
                url=url,
                title=item.get("title", ""),
                channel=item.get("channel", ""),
                duration=item.get("duration", ""),
                upload_date=item.get("upload_date", ""),
                source=source,
                query=query,
            )
            session.register_candidate(rec)
            registered.append((ref, item))
            if yid:
                existing_ids.add(yid)

        refs = [ref for ref, _ in registered][:top_k]
        observation = {
            "query": query,
            "source": source,
            "candidate_refs": refs,
            "count": len(refs),
            "skipped_duplicates": skipped_dup,
        }
        session.retrieval_events.append(observation)
        return observation

    def download_candidate(self, session: SessionState, candidate_ref: str, output_dir: str) -> dict[str, Any]:
        candidate = session.candidates.get(candidate_ref)
        if candidate is None:
            raise ValueError(f"Unknown candidate_ref: {candidate_ref}")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        video_id = _extract_youtube_id_from_url(candidate.url)
        if not video_id:
            return {
                "ok": False,
                "reason": "invalid_candidate_url",
                "reason_code": "invalid_candidate_url",
            }
        # Reuse already-downloaded video if it exists and decodes successfully.
        existing, existing_reason = _pick_downloaded_video(
            output_dir, expected_video_id=video_id
        )
        if existing:
            candidate.downloaded_video_path = existing
            return {"ok": True, "video_path": existing, "reused": True, "attempts": []}
        out_template = str(Path(output_dir) / "%(id)s.%(ext)s")

        # Single-file mp4 first to avoid merge dependency; merged formats as fallback.
        max_height = self.candidate_video_height
        format_selectors = [
            (
                "primary",
                f"best[ext=mp4][height<={max_height}]/best[ext=mp4]/"
                f"bv*[height<={max_height}]+ba/best[height<={max_height}]/best",
            ),
            (
                "fallback_lowres",
                f"worst[ext=mp4][height<={max_height}]/worst[height<={max_height}]/worst",
            ),
        ]

        def _build_cmd(
            use_browser_cookies: bool,
            format_selector: str,
            cookies_file: str | None = None,
        ) -> list[str]:
            cmd = [
                "yt-dlp",
                "--no-playlist",
                "--quiet",
                "--no-warnings",
                "-f",
                format_selector,
                "--remux-video",
                "mp4",
                "-o",
                out_template,
            ]
            cmd.extend(
                self._ytdlp_common_args(
                    use_browser_cookies=use_browser_cookies,
                    cookies_file=cookies_file,
                )
            )
            cmd.append(candidate.url)
            return cmd

        attempts: list[dict[str, str]] = []
        err: subprocess.CalledProcessError | None = None
        stderr_tail = ""
        reason_code = "unknown"
        success = False
        cookie_attempt_order = _cookies_attempt_order(self._cookies_files)
        if not cookie_attempt_order:
            cookie_attempt_order = [None]

        stop_all = False
        for cookies_file in cookie_attempt_order:
            cookie_label = Path(cookies_file).name if cookies_file else "none"
            for selector_name, format_selector in format_selectors:
                try:
                    subprocess.run(
                        _build_cmd(
                            use_browser_cookies=False,
                            format_selector=format_selector,
                            cookies_file=cookies_file,
                        ),
                        capture_output=True,
                        text=True,
                        check=True,
                        timeout=_DOWNLOAD_TIMEOUT_SECONDS,
                    )
                    success = True
                    break
                except subprocess.CalledProcessError as exc:
                    err = exc
                    stderr_tail = (exc.stderr or "").strip()[-500:]
                    reason_code = _classify_ytdlp_failure(stderr_tail)
                    attempts.append(
                        {
                            "strategy": f"cookies_file:{cookie_label}:{selector_name}",
                            "reason_code": reason_code,
                        }
                    )
                    if reason_code in _COOKIE_STOP_REASONS:
                        stop_all = True
                        break
                except subprocess.TimeoutExpired:
                    attempts.append(
                        {
                            "strategy": f"cookies_file:{cookie_label}:{selector_name}",
                            "reason_code": "timeout",
                        }
                    )
                    reason_code = "timeout"
                    continue

                if (
                    self._cookies_browser
                    and reason_code in {"http_403_bot_check", "age_restricted", "private_video"}
                ):
                    self.log(
                        f"[yt-dlp] retrying download with --cookies-from-browser={self._cookies_browser} "
                        f"after reason_code={reason_code} selector={selector_name} cookie={cookie_label}"
                    )
                    try:
                        subprocess.run(
                            _build_cmd(
                                use_browser_cookies=True,
                                format_selector=format_selector,
                            ),
                            capture_output=True,
                            text=True,
                            check=True,
                            timeout=_DOWNLOAD_TIMEOUT_SECONDS,
                        )
                        success = True
                        break
                    except subprocess.CalledProcessError as exc2:
                        err = exc2
                        stderr_tail = (exc2.stderr or "").strip()[-500:]
                        reason_code = _classify_ytdlp_failure(stderr_tail)
                        attempts.append(
                            {
                                "strategy": f"cookies_from_browser:{selector_name}",
                                "reason_code": reason_code,
                            }
                        )
                    except subprocess.TimeoutExpired:
                        attempts.append(
                            {
                                "strategy": f"cookies_from_browser:{selector_name}",
                                "reason_code": "timeout",
                            }
                        )
                        reason_code = "timeout"

            if success or stop_all:
                break
            if reason_code not in _COOKIE_RETRY_REASONS:
                break

        if not success:
            return {
                "ok": False,
                "reason": "download_timeout" if reason_code == "timeout" else "download_failed",
                "reason_code": reason_code,
                "stderr_tail": stderr_tail,
                "attempts": attempts,
            }

        chosen, fail_reason = _pick_downloaded_video(
            output_dir, expected_video_id=video_id
        )
        if chosen is None:
            return {
                "ok": False,
                "reason": fail_reason or "no_video_file",
                "reason_code": fail_reason or "no_video_file",
                "attempts": attempts,
            }
        candidate.downloaded_video_path = chosen
        return {"ok": True, "video_path": chosen, "attempts": attempts}

    def sample_candidate_frames(
        self,
        candidate: CandidateRecord,
        *,
        output_dir: str,
        num_frames: int,
        start_sec: float | None = None,
        end_sec: float | None = None,
        prefix: str = "cand",
        cache_root: str | None = None,
    ) -> dict[str, Any]:
        """Sample frames from a downloaded candidate video. Optional [start_sec, end_sec] window.

        If `cache_root` is given, frames are cached under
        ``<cache_root>/candidate/{youtube_id}_h{height}_n{num_frames}/``.
        Downloaded mp4 files live under ``download_output_dir`` and are not
        removed when the temp workspace is cleaned."""
        if not candidate.downloaded_video_path:
            return {"ok": False, "reason": "candidate_not_downloaded", "reason_code": "candidate_not_downloaded"}

        # Reuse persistent cached frames if available.
        cache_dir: Path | None = None
        if cache_root:
            yid = _extract_youtube_id_from_url(candidate.url) or ""
            if yid:
                from src.utils.frame_sampling import resolve_frame_cache_dir

                cache_dir = resolve_frame_cache_dir(
                    cache_root,
                    yid,
                    height=self.candidate_video_height,
                    num_frames=num_frames,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    source="candidate",
                )
                if cache_dir.is_dir():
                    cached = sorted(str(p) for p in cache_dir.glob("frame_*.jpg"))
                    if cached:
                        self.log(
                            f"[Sampling] frame_cache hit ({prefix}): {len(cached)} frames "
                            f"from {cache_dir} (skipped ffmpeg)"
                        )
                        return {
                            "ok": True,
                            "frame_paths": cached,
                            "count": len(cached),
                            "fps": 0.0,
                            "start_idx": 0,
                            "end_idx": 0,
                            "cached": True,
                        }

        from src.utils.frame_sampling import (
            compute_uniform_indices,
            probe_video_frame_count,
            sample_frames_at_indices,
        )

        try:
            total, fps = probe_video_frame_count(candidate.downloaded_video_path)
        except (FileNotFoundError, ValueError, subprocess.SubprocessError) as exc:
            return {
                "ok": False,
                "reason": f"candidate_probe_failed: {exc}",
                "reason_code": "decode_failed",
            }

        start_idx = 0
        end_idx = total - 1
        if start_sec is not None and end_sec is not None and end_sec > start_sec:
            start_idx = max(0, int(start_sec * fps))
            end_idx = min(total - 1, int(end_sec * fps))
            if end_idx <= start_idx:
                end_idx = min(total - 1, start_idx + max(1, int(fps)))

        window = max(1, end_idx - start_idx + 1)
        frame_count = min(num_frames, window)
        if frame_count <= 1:
            indices = [start_idx]
        else:
            rel = compute_uniform_indices(window, frame_count)
            indices = [start_idx + i for i in rel]

        target_dir = cache_dir if cache_dir is not None else Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        try:
            out_paths = sample_frames_at_indices(
                candidate.downloaded_video_path,
                indices,
                target_dir,
                prefix="frame",
                target_height=self.candidate_video_height,
                max_workers=self.frame_resize_workers,
                log_fn=self.log,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            return {
                "ok": False,
                "reason": f"candidate_sample_failed: {exc}",
                "reason_code": "decode_failed",
            }
        if not out_paths:
            return {"ok": False, "reason": "no_frames_extracted", "reason_code": "decode_failed"}
        self.log(
            f"[Sampling] frame_cache miss ({prefix}): ffmpeg extracted {len(out_paths)} frames "
            f"-> {target_dir}"
        )
        return {
            "ok": True,
            "frame_paths": out_paths,
            "count": len(out_paths),
            "fps": fps,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "cached": False,
        }

    def _search_serper(
        self, query: str, top_k: int, max_page: int = 3,
    ) -> list[dict[str, Any]]:
        """Search YouTube via Serper API (google.serper.dev)."""
        import urllib.error
        import urllib.request

        serper_key = os.getenv("SERPER_KEY", "").strip()
        if not serper_key:
            self.log("[serper] SERPER_KEY not set, cannot search")
            return []

        sanitized = re.sub(r'[^\w\s]', '', query)
        final_query = f"{sanitized} youtube"

        candidates: list[dict[str, Any]] = []
        seen: set[str] = set()
        skipped = {"shorts": 0, "live": 0}

        for page in range(1, max_page + 1):
            if len(candidates) >= top_k:
                break

            payload = json.dumps({
                "q": final_query,
                "num": 10,
                "page": page,
                "hl": "en",
                "gl": "us",
            }).encode("utf-8")

            headers = {
                "X-API-KEY": serper_key,
                "Content-Type": "application/json",
            }
            req = urllib.request.Request(
                "https://google.serper.dev/search",
                data=payload,
                headers=headers,
                method="POST",
            )

            try:
                with urllib.request.urlopen(req, timeout=15) as resp:
                    results = json.loads(resp.read().decode("utf-8"))
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
                self.log(f"[serper] request failed (page {page}): {exc}")
                break

            organic = results.get("organic") or results.get("organic_results") or []
            if not organic:
                break

            for item in organic:
                if len(candidates) >= top_k:
                    break
                url = str(item.get("link") or "").strip()
                if not url or url in seen:
                    continue
                if "youtube.com" not in url and "youtu.be" not in url:
                    continue
                if "watch?v=" not in url and "/shorts/" not in url:
                    continue

                title = str(item.get("title") or "").strip()
                title_lower = title.lower()
                if "/shorts/" in url.lower() or "#shorts" in title_lower:
                    skipped["shorts"] += 1
                    continue
                if " live" in title_lower or "[live]" in title_lower or "🔴" in title:
                    skipped["live"] += 1
                    continue

                seen.add(url)
                candidates.append({
                    "url": url,
                    "title": title,
                    "channel": "",
                    "duration": "",
                    "upload_date": "",
                })

        if any(skipped.values()):
            self.log(f"[serper] filtered candidates: {skipped}")
        return candidates

    # ------------------------------------------------------------------
    # Deepsearch: async search & download
    # ------------------------------------------------------------------
    async def search_one_async(
        self,
        session: SessionState,
        query: str,
    ) -> dict[str, Any]:
        """Async wrapper: search YouTube for one keyword, return top-1 candidate."""
        import asyncio

        results = await asyncio.to_thread(self._search_serper, query, 1)
        if not results:
            return {"ok": False, "query": query, "candidate_ref": None, "reason": "no_results"}

        item = results[0]
        url = item["url"]
        yid = extract_youtube_id(url)

        # Dedup check
        existing_ids = {
            extract_youtube_id(rec.url)
            for rec in session.candidates.values()
        }
        existing_ids.discard(None)
        if yid and yid in existing_ids:
            return {"ok": False, "query": query, "candidate_ref": None, "reason": "duplicate"}

        ref = session.next_candidate_ref()
        rec = CandidateRecord(
            ref=ref,
            url=url,
            title=item.get("title", ""),
            channel=item.get("channel", ""),
            duration=item.get("duration", ""),
            upload_date=item.get("upload_date", ""),
            source="serper",
            query=query,
        )
        session.register_candidate(rec)
        return {"ok": True, "query": query, "candidate_ref": ref, "title": rec.title, "url": url}

    async def download_candidate_async(
        self,
        session: SessionState,
        candidate_ref: str,
        output_dir: str,
    ) -> dict[str, Any]:
        """Async wrapper: download a candidate video."""
        import asyncio

        return await asyncio.to_thread(
            self.download_candidate, session, candidate_ref, output_dir
        )

    # ------------------------------------------------------------------
    # Deepsearch: VLM-based coarse/fine/sufficiency checks
    # ------------------------------------------------------------------
    def check_coarse_relevance(
        self,
        forged_frame_paths: list[str],
        candidate_frame_paths: list[str],
        *,
        physical_observations: str = "",
        logical_analysis: str = "",
        search_intent: str = "",
    ) -> dict[str, Any]:
        """Coarse relevance over two frame groups: forged vs candidate."""
        prompt = PROMPT_COARSE_RELEVANCE.format(
            physical_observations=physical_observations or "(not available)",
            logical_analysis=logical_analysis or "(not available)",
            search_intent=search_intent or "(not available)",
        )
        contents = _build_multimodal_content(
            prompt + "\n\nFrame groups order:\n- First all images are Group A (forged).\n- Then all images are Group B (candidate).",
            image_paths=[*forged_frame_paths, *candidate_frame_paths],
        )
        all_paths = [*forged_frame_paths, *candidate_frame_paths]
        raw, _tokens = call_vlm_with_retry(
            self.api_config,
            self.model,
            contents,
            image_paths=all_paths,
            max_retries=3,
            temperature=0.0,
            json_mode=True,
            logger=self.log,
            log_prefix="[VLM] ",
        )
        data = extract_json_from_text(raw)
        return {
            "reasoning": str(data.get("reasoning") or "").strip(),
            "is_relevant": as_bool(data.get("is_relevant"), default=False),
            "tokens": _tokens or {},
        }

    def extract_fine_forgery_points(
        self,
        forged_frame_paths: list[str],
        candidate_frame_paths: list[str],
        *,
        physical_observations: str = "",
        logical_analysis: str = "",
        search_intent: str = "",
        entities: dict[str, list[str]] | None = None,
    ) -> dict[str, Any]:
        """Fine-grained narrative forgery extraction over forged vs candidate."""
        entities_summary = json.dumps(entities or {}, ensure_ascii=False)
        prompt = PROMPT_FINE_FORGERY_POINTS.format(
            physical_observations=physical_observations or "(not available)",
            logical_analysis=logical_analysis or "(not available)",
            search_intent=search_intent or "(not available)",
            entities_summary=entities_summary,
        )
        contents = _build_multimodal_content(
            prompt + "\n\nFrame groups order:\n- First all images are Group A (forged).\n- Then all images are Group B (candidate).",
            image_paths=[*forged_frame_paths, *candidate_frame_paths],
        )
        all_paths = [*forged_frame_paths, *candidate_frame_paths]
        raw, _tokens = call_vlm_with_retry(
            self.api_config,
            self.model,
            contents,
            image_paths=all_paths,
            max_retries=3,
            temperature=0.0,
            json_mode=True,
            logger=self.log,
            log_prefix="[VLM] ",
        )
        data = extract_json_from_text(raw)
        raw_points = data.get("points") or []
        points = []
        if isinstance(raw_points, list):
            for p in raw_points:
                if isinstance(p, dict):
                    desc = str(p.get("description") or p.get("zh") or "").strip()
                    if desc:
                        points.append({"description": desc})
        return {
            "source_description": str(data.get("source_description") or "").strip(),
            "points": points,
            "tokens": _tokens or {},
        }

    def deepsearch_next_step(
        self,
        collected_points: list[dict[str, Any]],
        examined_videos: list[dict[str, str]],
        source_descriptions: list[str],
        prev_queries: list[str],
        *,
        current_round: int | None = None,
        max_rounds: int | None = None,
        physical_observations: str = "",
        logical_analysis: str = "",
        search_intent: str = "",
        entities: dict[str, list[str]] | None = None,
    ) -> dict[str, Any]:
        """Combined sufficiency judgment + next keyword generation.

        Text-only VLM call (no frames).  Decides: (1) is evidence sufficient,
        (2) if not, what keyword to search next.
        """
        if (
            current_round is not None
            and max_rounds is not None
            and current_round > max_rounds
        ):
            return {
                "reasoning": "Exceeded max_deepsearch_rounds; hard stop.",
                "is_sufficient": True,
                "missing_description": "",
                "next_keyword": "",
                "tokens": {},
                "forced_stop": True,
            }

        # Format collected points
        if collected_points:
            pts_lines = []
            for i, p in enumerate(collected_points):
                pts_lines.append(f"  [{i+1}] {p.get('description', '')}")
            points_str = "\n".join(pts_lines)
        else:
            points_str = "  (none collected yet)"

        # Format examined videos
        if examined_videos:
            vid_lines = []
            for v in examined_videos:
                vid_lines.append(f"  - {v.get('title', 'unknown')} ({v.get('url', '')})")
            videos_str = "\n".join(vid_lines)
        else:
            videos_str = "  (none examined yet)"

        # Format source descriptions
        src_str = "\n".join(f"  - {s}" for s in source_descriptions) if source_descriptions else "  (none)"

        # Format previous queries
        q_str = "\n".join(f"  - {q}" for q in prev_queries) if prev_queries else "  (none)"
        entities_summary = json.dumps(entities or {}, ensure_ascii=False)
        if current_round is not None and max_rounds is not None:
            if current_round == max_rounds:
                round_status = (
                    f"Current round: {current_round}/{max_rounds}. "
                    "This is the FINAL round. You MUST set is_sufficient=true, "
                    "summarize the evidence collected so far, and leave next_keyword empty. "
                    "Do not request another search."
                )
            else:
                round_status = (
                    f"Current round: {current_round}/{max_rounds}. "
                    f"There are {max_rounds - current_round} round(s) remaining after this decision."
                )
        else:
            round_status = "Round status unavailable."

        prompt = PROMPT_DEEPSEARCH_NEXT_STEP.format(
            round_status=round_status,
            physical_observations=physical_observations or "(not available)",
            logical_analysis=logical_analysis or "(not available)",
            search_intent=search_intent or "(not available)",
            entities_summary=entities_summary,
            source_descriptions=src_str,
            collected_points=points_str,
            examined_videos=videos_str,
            prev_queries=q_str,
        )

        # Text-only call — no images, saves ~70k tokens per round
        from src.utils.agent_helpers import call_vlm_with_retry

        raw, _tokens = call_vlm_with_retry(
            self.api_config,
            self.model,
            [{"type": "text", "text": prompt}],
            temperature=0.0,
            logger=self.log,
        )
        data = extract_json_from_text(raw)
        return {
            "reasoning": str(data.get("reasoning") or "").strip(),
            "is_sufficient": as_bool(data.get("is_sufficient"), default=False),
            "missing_description": str(data.get("missing_description") or "").strip(),
            "next_keyword": str(data.get("next_keyword") or "").strip(),
            "tokens": _tokens or {},
            "forced_stop": False,
        }

"""Oracle-eval helpers: load truth IDs / GT lists from evid_bench.json manifests.

Expected dataset layout:
    <dataset_root>/evid_bench.json
    <dataset_root>/videos/<forged_id>.mp4

Manifest entry schema:
    {
      "id":          "<youtube id of the forged video>",
      "source":      ["<original youtube id>", ...],
      "topic":       "daily_life" | "sports" | ...,
      "task":        {"category": "single_source_editing", "type": "selective_omission"},
      "groundtruth": ["GT point text", ...]              # optional
    }
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


_YOUTUBE_ID_RE = re.compile(r"(?:v=|youtu\.be/|/shorts/|/embed/)([A-Za-z0-9_-]{8,})")


def _snake_label(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return " ".join(part.capitalize() for part in text.split("_"))


def format_task_label(entry: dict[str, Any]) -> str:
    """Human-readable task label from manifest ``task`` (snake_case values)."""
    task = entry.get("task")
    if not isinstance(task, dict):
        return ""
    category = _snake_label(str(task.get("category") or ""))
    subtype = _snake_label(str(task.get("type") or ""))
    if category and subtype:
        return f"{category} / {subtype}"
    return category or subtype


def extract_youtube_id(url: str) -> str | None:
    """Return the 11-char-ish YouTube id from any youtube URL form, or None."""
    if not url:
        return None
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if "youtu.be" in host:
        vid = parsed.path.strip("/").split("/")[0]
        return vid or None
    if "youtube.com" in host:
        qs = parse_qs(parsed.query or "")
        if qs.get("v"):
            return (qs["v"][0] or "").strip() or None
        parts = [p for p in parsed.path.split("/") if p]
        if len(parts) >= 2 and parts[0] in {"shorts", "embed"}:
            return parts[1]
    m = _YOUTUBE_ID_RE.search(url)
    if m:
        return m.group(1)
    return None


def _expected_ids_from_entry(entry: dict[str, Any]) -> list[str]:
    """Pull expected original-source YouTube ids from manifest ``source``."""
    raw = entry.get("source")
    if not isinstance(raw, list):
        return []
    ids: list[str] = []
    for item in raw:
        val = str(item or "").strip()
        if val and val not in ids:
            ids.append(val)
    return ids


def load_oracle_map(manifest_path: str | Path | None) -> dict[str, list[str]]:
    """Return {forged_video_id: [expected source youtube ids]}.

    Missing manifest returns {}.
    """
    if not manifest_path:
        return {}
    p = Path(manifest_path)
    if not p.exists():
        return {}
    try:
        rows = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(rows, list):
        return {}
    out: dict[str, list[str]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = str(row.get("id") or "").strip()
        if not key:
            continue
        out[key] = _expected_ids_from_entry(row)
    return out


def load_manifest_entries(manifest_path: str | Path) -> list[dict[str, Any]]:
    """Load a list of Edit-style manifest entries. Returns [] on any error."""
    p = Path(manifest_path)
    if not p.exists():
        return []
    try:
        rows = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(rows, list):
        return []
    return [r for r in rows if isinstance(r, dict)]


def extract_groundtruth(entry: dict[str, Any]) -> list[str]:
    """Return GT forgery-point strings from a manifest entry (up to five).

    Missing / wrong-typed ``groundtruth`` returns ``[]``.
    Items may be plain strings or dicts with ``description`` / ``zh``.
    """
    raw = entry.get("groundtruth")
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for item in raw:
        if isinstance(item, dict):
            s = str(item.get("description") or item.get("zh") or "").strip()
        else:
            s = str(item or "").strip()
        if s:
            out.append(s)
    return out[:5]


def evaluate_oracle(
    *,
    input_video_id: str,
    expected_source_ids: list[str],
    action_trace: list[dict[str, Any]],
    verification_events: list[dict[str, Any]],
    group_to_shot_ids: dict[int, list[int]] | None = None,
    matched_group_ids: list[int] | None = None,
) -> dict[str, Any]:
    """Roll up SEARCH / VERIFY events into recall metrics against expected_source_ids."""
    expected = set(expected_source_ids)
    hit_ids: set[str] = set()
    search_hit_shots: set[int] = set()
    verified_hit_shots: set[int] = set()
    visual_same_source_hit_shots: set[int] = set()
    visual_same_source_hit_ids: set[str] = set()
    search_recall_hits: set[str] = set()
    verify_misses_on_truth = 0

    for evt in action_trace:
        if evt.get("action") != "SEARCH":
            continue
        shot_id = evt.get("args", {}).get("shot_id")
        obs = evt.get("observation", {}) or {}
        for url in (obs.get("candidate_urls") or []):
            vid = extract_youtube_id(str(url))
            if vid and vid in expected:
                hit_ids.add(vid)
                search_recall_hits.add(vid)
                if shot_id is not None:
                    search_hit_shots.add(int(shot_id))

    for evt in verification_events:
        try:
            shot_id = int(evt.get("shot_id", -1))
        except Exception:
            shot_id = -1
        url = str(evt.get("candidate_url") or "")
        vid = extract_youtube_id(url)
        verification = evt.get("verification") or {}
        if bool(verification.get("is_match")):
            visual_same_source_hit_shots.add(shot_id)
            if vid:
                visual_same_source_hit_ids.add(vid)
        if not vid or vid not in expected:
            continue
        if bool(verification.get("is_match")):
            verified_hit_shots.add(shot_id)
        else:
            verify_misses_on_truth += 1

    g2s = {int(k): [int(s) for s in v] for k, v in (group_to_shot_ids or {}).items()}
    matched_groups = [int(g) for g in (matched_group_ids or [])]
    total_groups = len(g2s)
    resolved_groups = len(matched_groups)
    groups_with_visual_hit: set[int] = set()
    groups_with_truth_verify: set[int] = set()
    for gid, shots in g2s.items():
        sset = set(shots)
        if sset & visual_same_source_hit_shots:
            groups_with_visual_hit.add(gid)
        if sset & verified_hit_shots:
            groups_with_truth_verify.add(gid)

    return {
        "input_video_id": input_video_id,
        "expected_source_ids": list(expected_source_ids),
        "search_hit_shots": len(search_hit_shots),
        "verified_hit_shots": len(verified_hit_shots),
        "hit_source_ids": sorted(hit_ids),
        "search_recall_hits": sorted(search_recall_hits),
        "verify_misses_on_truth": verify_misses_on_truth,
        "search_recall_count": len(search_recall_hits),
        "expected_count": len(expected),
        "visual_same_source_hit_shots": len(visual_same_source_hit_shots),
        "visual_same_source_hit_ids": sorted(visual_same_source_hit_ids),
        "visual_same_source_hit_count": len(visual_same_source_hit_ids),
        "total_groups": total_groups,
        "resolved_groups": resolved_groups,
        "matched_group_ids": sorted(matched_groups),
        "group_to_shot_ids": {str(k): list(v) for k, v in g2s.items()},
        "groups_with_visual_hit": sorted(groups_with_visual_hit),
        "groups_with_truth_verify": sorted(groups_with_truth_verify),
    }

"""Oracle-eval helpers: load truth IDs / GT 3-5 point lists from Edit-style manifests.

Manifest entry schema (Edit.json / example.json):
    {
      "id":        "<youtube id of the forged video>",
      "topic":     "...",
      "task":      "2.1" | "2.2" | "2.3" | "3.1" | "3.2" | "3.3",
      "video2":    "<source youtube id>" | "",   # task 3.x has at least one
      "video3":    "<source youtube id>" | "",   # task 3.x sometimes has two
      "videoN":    ...                            # forward-compat: any videoN
      "groundtruth": ["GT point text", ...]      # optional, may be missing
    }
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


_VIDEO_FIELD_RE = re.compile(r"^video\d+$")
_YOUTUBE_ID_RE = re.compile(r"(?:v=|youtu\.be/|/shorts/|/embed/)([A-Za-z0-9_-]{8,})")


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
    """Pull all source-truth YouTube ids out of an Edit-style manifest entry.

    Rules:
    - For task 2.x (single source): the source IS the forged id itself (`entry["id"]`).
    - For task 3.x (multi source):  `video2`, `video3`, ... videoN.
    - Order preserved (id first, then video2, video3, ...).
    """
    ids: list[str] = []
    primary = str(entry.get("id") or "").strip()
    if primary:
        ids.append(primary)
    for key in entry.keys():
        if not _VIDEO_FIELD_RE.match(key):
            continue
        val = str(entry.get(key) or "").strip()
        if val and val not in ids:
            ids.append(val)
    return ids


def load_oracle_map(manifest_path: str | Path | None) -> dict[str, list[str]]:
    """Return {forged_video_id: [expected source youtube ids]}.

    Accepts any number of `videoN` columns. Missing manifest returns {}.
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

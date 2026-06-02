"""Batch orchestrator: Stage A (retrieval + deepsearch) -> Stage C (judge).

Usage::

    python -m src.cli <folder-or-manifest>

Common flags::

    [--manifest Edit.json]
    [--output _results_YYYYMMDD_HHMMSS]
    [--limit N]
    [--skip-judge]
    [--resume / --no-resume]
    [--judge-model ...]
    ... plus retrieval hyperparams (--top-k, --max-deepsearch-rounds, ...)

Outputs:
    <folder>/<output>/results.jsonl
    <folder>/<output>/run_trace.jsonl   (only with --save-run-trace)
    <folder>/<output>/summary.json
    <folder>/<output>/summary.md
    <folder>/<output>/logs/run_<ts>_<vid>.log
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.agent.judge import DEFAULT_JUDGE_MODELS, judge_points  # noqa: E402
from src.agent.oracle_eval import (  # noqa: E402
    extract_groundtruth,
    load_manifest_entries,
)
from src.agent_pipeline_v2 import VisualRetrievalAgentV2  # noqa: E402
from src.utils.config import get_llm_client  # noqa: E402
from src.utils.pipeline_log import default_log_path, tee_stdout  # noqa: E402
from src.utils.token_usage import (  # noqa: E402
    add_token_counter,
    empty_token_counter,
    merge_model_token_usage,
    normalize_token_counter,
)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run deepsearch retrieval + judge on a folder of forged videos.",
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to a folder containing forged .mp4 files, or directly to a manifest JSON file. Video files are expected in the same directory as the manifest.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="Edit.json",
        help="Manifest filename inside <folder> (default: Edit.json).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="_results",
        help="Output subdirectory under <folder>. Default `_results` is expanded to `_results_<timestamp>` on each run.",
    )
    parser.add_argument(
        "--results-jsonl",
        type=str,
        default=None,
        help="Path for appended results JSONL. Absolute paths are used as-is; relative paths are resolved under --output. Default: <output>/results.jsonl.",
    )
    parser.add_argument(
        "--trace-jsonl",
        type=str,
        default=None,
        help="Path for appended run-trace JSONL. Absolute paths are used as-is; relative paths are resolved under --output. Default: <output>/run_trace.jsonl.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N manifest entries (for debugging).",
    )
    parser.add_argument(
        "--only-ids",
        type=str,
        default="",
        help="Comma-separated list of video ids to restrict the run to.",
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Run retrieval but skip judge (useful when GT is not yet filled in).",
    )
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="If a per-video result already exists, reuse it (default).",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Force re-run even when a per-video result exists.",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Legacy single-judge override for Stage C. If set, only this judge model is used.",
    )
    parser.add_argument(
        "--judge-models",
        type=str,
        default="",
        help=(
            "Comma-separated judge models for Stage C voting. "
            f"Default: {', '.join(DEFAULT_JUDGE_MODELS)}"
        ),
    )
    parser.add_argument(
        "--search-only",
        action="store_true",
        help=(
            "Fast retrieval eval: skip visual verify. Only checks "
            "whether the candidate YouTube id is in manifest expected_source_ids."
        ),
    )
    # Stage A hyperparameters
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--total-sample-frames",
        type=int,
        default=64,
        help="Frames sampled from the input video (default: 64).",
    )
    parser.add_argument("--query-temperature", type=float, default=0.0)
    parser.add_argument(
        "--download-dir",
        type=str,
        default="downloads",
        help="Where to save downloaded candidate videos (relative path resolved from CWD).",
    )
    parser.add_argument(
        "--candidate-video-height",
        type=int,
        default=480,
        help="Max height for searched/downloaded candidate videos in pixels (default: 480). Input videos are unchanged.",
    )
    parser.add_argument("--quiet", action="store_true")
    # Deepsearch-specific hyperparameters
    parser.add_argument(
        "--max-reflect-rounds",
        type=int,
        default=3,
        help="Max REFLECT rounds when search fails (default 3).",
    )
    parser.add_argument(
        "--candidate-sample-frames",
        type=int,
        default=None,
        help="Frames sampled from each candidate video for fine extraction (default: same as --total-sample-frames).",
    )
    parser.add_argument(
        "--max-deepsearch-rounds",
        type=int,
        default=6,
        help="Max search rounds before giving up (default 6).",
    )
    parser.add_argument(
        "--coarse-sample-frames",
        type=int,
        default=16,
        help="Frames sampled for coarse relevance filter (default 16).",
    )
    parser.add_argument(
        "--reasoning-model",
        type=str,
        default="google/gemini-2.5-pro",
        help="Model for all VLM calls (default: google/gemini-2.5-pro).",
    )
    parser.add_argument(
        "--frame-resize-workers",
        type=int,
        default=None,
        help=(
            "Thread pool size for parallel JPEG resize after single-pass ffmpeg extract "
            "(default: 16, or FRAME_RESIZE_WORKERS env). Does not spawn multiple decoders."
        ),
    )
    parser.add_argument(
        "--use-cot",
        dest="use_cot",
        action="store_true",
        default=True,
        help="Use COT reasoning for keyword generation (default).",
    )
    parser.add_argument(
        "--no-cot",
        dest="use_cot",
        action="store_false",
        help="Skip COT, use filename as fallback keyword (ablation mode).",
    )
    parser.add_argument(
        "--parallel-videos",
        type=int,
        default=1,
        help="Number of videos to process in parallel (default 1 = sequential).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Alias for --parallel-videos. If provided, overrides --parallel-videos.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Override log directory path (default: <output>/logs).",
    )
    parser.add_argument(
        "--save-run-trace",
        action="store_true",
        help="Save a structured per-video run trace with COT summaries, queries, and candidate titles/URLs (default: off).",
    )
    parser.add_argument(
        "--skip-summary",
        action="store_true",
        help="Do not write summary.json or summary.md. Useful when only JSONL outputs are needed.",
    )
    return parser


def _filter_entries(
    entries: list[dict[str, Any]],
    *,
    only_ids: list[str] | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    if only_ids:
        keep = set(only_ids)
        entries = [e for e in entries if str(e.get("id") or "").strip() in keep]
    if limit is not None and limit > 0:
        entries = entries[:limit]
    return entries


def _gather_local_sources(
    *,
    retrieved_truth_ids: list[str],
    source_video_paths: list[str],
    downloads_dir: Path,
    dataset_folder: Path,
) -> tuple[list[str], list[str]]:
    """Return (final_source_paths, final_truth_ids_used).

    Prefer the per-truth-id YouTube id when we have it (downloaded by the agent
    or already lying next to the manifest in the dataset folder). The agent
    already returns local paths via `source_video_paths`; we just verify them
    and supplement with any expected truth ids that happen to be present locally.
    """
    out_paths: list[str] = []
    out_ids: list[str] = []

    # 1. Trust the agent's reported paths first.
    for p in source_video_paths:
        if p and Path(p).is_file():
            if p not in out_paths:
                out_paths.append(p)

    # 2. For any retrieved_truth_id that maps to a known file (downloads or dataset
    #    folder) also include it. This is mostly redundant but bullet-proofs the
    #    case where the agent forgot to record the local path.
    for yid in retrieved_truth_ids or []:
        out_ids.append(yid)
        candidates = [
            downloads_dir / f"{yid}.mp4",
            downloads_dir / f"{yid}.mkv",
            downloads_dir / f"{yid}.webm",
            dataset_folder / f"{yid}.mp4",
            dataset_folder / f"{yid}.mkv",
            dataset_folder / f"{yid}.webm",
        ]
        for c in candidates:
            if c.is_file() and str(c) not in out_paths:
                out_paths.append(str(c))
                break
    return out_paths, out_ids


def _write_summary_md(summary: dict[str, Any], out_md: Path) -> None:
    """Write a human-readable markdown summary."""
    lines: list[str] = []
    lines.append(f"# Forgery Pipeline Summary")
    lines.append("")
    lines.append(f"- Folder: `{summary['folder']}`")
    lines.append(f"- Manifest: `{summary['manifest']}`")
    lines.append(f"- Videos processed: **{summary['n_videos']}**")
    lines.append(f"- Videos with GT (judged): **{summary['n_judged']}**")
    lines.append("")
    r = summary["retrieval"]
    lines.append("## Stage A: retrieval")
    lines.append("")
    lines.append(f"- Videos with at least one truth hit: **{r['videos_with_any_truth_hit']}** / {summary['n_videos']}")
    lines.append(f"- Retrieval rate (video-level): **{r['retrieval_rate']:.3f}**")
    lines.append(f"- Per-task retrieval rate: `{r['per_task_breakdown']}`")
    lines.append("")
    s = summary["scoring"]
    lines.append("## Stage C: forgery-point scoring")
    lines.append("")
    if s["total_points"] == 0:
        lines.append("_No GT-bearing videos in this run; scoring skipped._")
    else:
        lines.append(f"- Hits: **{s['hit_points']}** / {s['total_points']}")
        lines.append(f"- Accuracy: **{s['accuracy']:.3f}**")
        lines.append(f"- Per-task accuracy: `{s['per_task_breakdown']}`")
    lines.append("")
    t = summary.get("tokens") or {}
    lines.append("## Token usage")
    lines.append("")
    lines.append(
        f"- Total: **{int(t.get('total_tokens', 0))}** "
        f"(prompt={int(t.get('total_prompt_tokens', 0))}, completion={int(t.get('total_completion_tokens', 0))})"
    )
    by_stage = t.get("by_stage") or {}
    retrieval_stage = by_stage.get("retrieval") or {}
    judge_stage = by_stage.get("judge") or {}
    lines.append(
        f"- Stage A retrieval: {int(retrieval_stage.get('total_tokens', 0))} "
        f"(prompt={int(retrieval_stage.get('prompt_tokens', 0))}, completion={int(retrieval_stage.get('completion_tokens', 0))})"
    )
    lines.append(
        f"- Stage C judge: {int(judge_stage.get('total_tokens', 0))} "
        f"(prompt={int(judge_stage.get('prompt_tokens', 0))}, completion={int(judge_stage.get('completion_tokens', 0))})"
    )
    by_model = t.get("by_model") or {}
    if by_model:
        lines.append("- Per-model token totals:")
        for model_name in sorted(by_model.keys()):
            usage = by_model.get(model_name) or {}
            lines.append(
                f"  - `{model_name}`: {int(usage.get('total_tokens', 0))} "
                f"(prompt={int(usage.get('prompt_tokens', 0))}, completion={int(usage.get('completion_tokens', 0))})"
            )
    lines.append("")
    lines.append("## Per-video table")
    lines.append("")
    lines.append("| Video ID | Task | Truth hit | Resolved groups | Score |")
    lines.append("|---|---|---|---|---|")
    for row in summary["per_video"]:
        score_str = '-' if row.get('score') is None else f'{row["score"]:.3f}'
        lines.append(
            f"| `{row['video_id']}` | {row.get('task', '')} | "
            f"{'YES' if row.get('truth_hit') else 'no'} | "
            f"{row.get('resolved_groups', 0)}/{row.get('total_groups', 0)} | "
            f"{score_str} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_partial_record(*, entry: dict[str, Any], video_id: str, error: str) -> dict[str, Any]:
    """Build a minimal record when a video is interrupted before completion."""
    return {
        "video_id": video_id,
        "topic": str(entry.get("topic") or ""),
        "task": str(entry.get("task") or ""),
        "retrieval": {"matched_urls": [], "retrieved_truth_ids": [], "truth_hit": False},
        "deepsearch": {"collected_forgery_points": [], "evidence_videos": []},
        "tokens": {
            **empty_token_counter(),
            "by_model": {},
            "stages": {
                "retrieval": {**empty_token_counter(), "by_model": {}},
                "judge": {**empty_token_counter(), "by_model": {}},
            },
        },
        "analysis": {"mode": "interrupted", "points": []},
        "score": None,
        "error": error,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }


def _merge_token_usage(
    retrieval_tokens_raw: Any,
    judge_tokens_raw: Any,
) -> dict[str, Any]:
    retrieval_tokens = normalize_token_counter(retrieval_tokens_raw)
    judge_tokens = normalize_token_counter(judge_tokens_raw)

    retrieval_by_model: dict[str, dict[str, int]] = {}
    judge_by_model: dict[str, dict[str, int]] = {}
    if isinstance(retrieval_tokens_raw, dict):
        merge_model_token_usage(retrieval_by_model, retrieval_tokens_raw.get("by_model"))
    if isinstance(judge_tokens_raw, dict):
        merge_model_token_usage(judge_by_model, judge_tokens_raw.get("by_model"))

    overall = empty_token_counter()
    add_token_counter(overall, retrieval_tokens)
    add_token_counter(overall, judge_tokens)

    overall_by_model: dict[str, dict[str, int]] = {}
    merge_model_token_usage(overall_by_model, retrieval_by_model)
    merge_model_token_usage(overall_by_model, judge_by_model)

    return {
        **overall,
        "by_model": overall_by_model,
        "stages": {
            "retrieval": {**retrieval_tokens, "by_model": retrieval_by_model},
            "judge": {**judge_tokens, "by_model": judge_by_model},
        },
    }


def _append_jsonl(path: Path, row: dict[str, Any], *, lock: threading.Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, ensure_ascii=False) + "\n"
    with lock:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line)


def _load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _dedupe_records_by_video_id(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    ordered_ids: list[str] = []
    for row in records:
        video_id = str(row.get("video_id") or "").strip()
        if not video_id:
            continue
        if video_id not in by_id:
            ordered_ids.append(video_id)
        by_id[video_id] = row
    return [by_id[video_id] for video_id in ordered_ids]


def _build_trace_row(*, record: dict[str, Any], run_trace: dict[str, Any]) -> dict[str, Any]:
    return {
        "video_id": record.get("video_id"),
        "topic": record.get("topic"),
        "task": record.get("task"),
        "timestamp": record.get("timestamp"),
        "run_trace": run_trace,
    }


def _resolve_output_path(base_dir: Path, raw_path: str | None, default_name: str) -> Path:
    if not raw_path:
        return base_dir / default_name
    p = Path(raw_path)
    if p.is_absolute():
        return p
    return base_dir / p


def _resolve_judge_models(args) -> list[str]:
    if getattr(args, "judge_models", "").strip():
        return [s.strip() for s in args.judge_models.split(",") if s.strip()]
    if getattr(args, "judge_model", None):
        return [str(args.judge_model).strip()]
    return list(DEFAULT_JUDGE_MODELS)


def _process_one(
    *,
    entry: dict[str, Any],
    dataset_folder: Path,
    output_dir: Path,
    downloads_dir: Path,
    agent: VisualRetrievalAgentV2,
    judge_models: list[str],
    judge_client,
    skip_judge: bool,
    save_run_trace: bool,
    resume: bool,
    completed_ids: set[str],
    completed_ids_lock: threading.Lock,
    log_dir: Path,
    results_jsonl_path: Path,
    results_jsonl_lock: threading.Lock,
    trace_jsonl_path: Path | None,
    trace_jsonl_lock: threading.Lock | None,
    parallel_mode: bool = False,
) -> dict[str, Any] | None:
    """Run A->C on one video. Returns the per-video result dict or None if skipped."""
    video_id = str(entry.get("id") or "").strip()
    if not video_id:
        return None
    video_path = dataset_folder / f"{video_id}.mp4"
    if not video_path.is_file():
        print(f"[Orchestrator] skip {video_id}: video file not found ({video_path})", flush=True)
        return None

    with completed_ids_lock:
        already_done = video_id in completed_ids
    if resume and already_done:
        print(f"[Orchestrator] reuse cached jsonl result for {video_id}", flush=True)
        return None

    log_path = default_log_path(log_dir, video_id)
    print(f"[Orchestrator] {video_id}: log -> {log_path}", flush=True)

    if parallel_mode:
        # In parallel mode, tee_stdout modifies global sys.stdout and would
        # interleave all threads.  Instead, let the agent write its own log
        # file via _log → _log_fh.
        try:
            if hasattr(agent, "set_log_file"):
                agent.set_log_file(str(log_path))
            return _process_one_inner(
                entry=entry, video_id=video_id, video_path=video_path,
                dataset_folder=dataset_folder, output_dir=output_dir,
                downloads_dir=downloads_dir, agent=agent,
                judge_models=judge_models, judge_client=judge_client,
                skip_judge=skip_judge,
                save_run_trace=save_run_trace,
                completed_ids=completed_ids,
                completed_ids_lock=completed_ids_lock,
                results_jsonl_path=results_jsonl_path,
                results_jsonl_lock=results_jsonl_lock,
                trace_jsonl_path=trace_jsonl_path,
                trace_jsonl_lock=trace_jsonl_lock,
            )
        finally:
            if hasattr(agent, "set_log_file"):
                agent.set_log_file(None)
    else:
        with tee_stdout(log_path):
            return _process_one_inner(
                entry=entry, video_id=video_id, video_path=video_path,
                dataset_folder=dataset_folder, output_dir=output_dir,
                downloads_dir=downloads_dir, agent=agent,
                judge_models=judge_models, judge_client=judge_client,
                skip_judge=skip_judge,
                save_run_trace=save_run_trace,
                completed_ids=completed_ids,
                completed_ids_lock=completed_ids_lock,
                results_jsonl_path=results_jsonl_path,
                results_jsonl_lock=results_jsonl_lock,
                trace_jsonl_path=trace_jsonl_path,
                trace_jsonl_lock=trace_jsonl_lock,
            )


def _process_one_inner(
    *,
    entry: dict[str, Any],
    video_id: str,
    video_path: Path,
    dataset_folder: Path,
    output_dir: Path,
    downloads_dir: Path,
    agent: VisualRetrievalAgentV2,
    judge_models: list[str],
    judge_client,
    skip_judge: bool,
    save_run_trace: bool,
    completed_ids: set[str],
    completed_ids_lock: threading.Lock,
    results_jsonl_path: Path,
    results_jsonl_lock: threading.Lock,
    trace_jsonl_path: Path | None,
    trace_jsonl_lock: threading.Lock | None,
) -> dict[str, Any]:
    """Core logic of _process_one, extracted so tee_stdout wrapping is clean."""
    print(f"=== Stage A: retrieval ({video_id}) ===", flush=True)
    try:
        retrieval = agent.run_retrieval(str(video_path))
    except KeyboardInterrupt:
        # Save partial result so user can see progress
        print(f"[Orchestrator] {video_id}: interrupted during retrieval; saving partial result", flush=True)
        partial = _build_partial_record(entry=entry, video_id=video_id, error="interrupted_during_retrieval")
        _append_jsonl(results_jsonl_path, partial, lock=results_jsonl_lock)
        with completed_ids_lock:
            completed_ids.add(video_id)
        return partial
    except Exception as exc:
        # Save partial result on any error so we can debug
        print(f"[Orchestrator] {video_id}: retrieval failed: {exc}", flush=True)
        partial = _build_partial_record(entry=entry, video_id=video_id, error=f"retrieval_failed: {exc}")
        _append_jsonl(results_jsonl_path, partial, lock=results_jsonl_lock)
        with completed_ids_lock:
            completed_ids.add(video_id)
        return partial

    retrieved_truth_ids = retrieval.get("retrieved_truth_ids") or []
    local_sources, used_ids = _gather_local_sources(
        retrieved_truth_ids=retrieved_truth_ids,
        source_video_paths=retrieval.get("source_video_paths") or [],
        downloads_dir=downloads_dir,
        dataset_folder=dataset_folder,
    )

    # Forgery points are collected during Stage A (deepsearch).
    # Skip Stage B entirely and use collected_forgery_points directly.
    collected_points = retrieval.get("collected_forgery_points") or []
    analysis = {
        "mode": "deepsearch",
        "summary_zh": "",
        "summary_en": "",
        "points": collected_points,
    }
    print(
        f"[Orchestrator] {video_id}: using {len(collected_points)} "
        f"collected forgery points (skipping Stage B)",
        flush=True,
    )

    gt_points = extract_groundtruth(entry)
    score_result: dict[str, Any] | None = None
    if not skip_judge and gt_points and collected_points:
        print(f"=== Stage C: judging ({video_id}) ===", flush=True)
        try:
            score_result = judge_points(
                video_id=video_id,
                gt_points=gt_points,
                pred_points=analysis.get("points") or [],
                client=judge_client,
                models=judge_models,
                logger=lambda m: print(m, flush=True),
            )
        except Exception as exc:
            print(f"[Orchestrator] {video_id}: Stage C failed: {exc}", flush=True)
            score_result = {
                "video_id": video_id,
                "judge_models": judge_models,
                "n_gt": len(gt_points),
                "passed_gt_count": 0,
                "hits": 0,
                "score": 0.0,
                "gt_vote_results": [],
                "judge_results": [],
                "comment": "",
                "ok": False,
                "error": f"judge_failed: {exc}",
                "token_usage": {
                    **empty_token_counter(),
                    "by_model": {m: empty_token_counter() for m in judge_models},
                },
            }
    elif not skip_judge and gt_points and not collected_points:
        print(f"[Orchestrator] {video_id}: no predicted forgery points; skipping Stage C judge", flush=True)
        score_result = {
            "video_id": video_id,
            "judge_models": judge_models,
            "n_gt": len(gt_points),
            "passed_gt_count": 0,
            "hits": 0,
            "score": 0.0,
            "gt_vote_results": [],
            "judge_results": [],
            "comment": "",
            "ok": False,
            "error": None,
            "skipped_reason": "no_pred_points",
            "token_usage": {
                **empty_token_counter(),
                "by_model": {m: empty_token_counter() for m in judge_models},
            },
        }
    elif skip_judge:
        print(f"[Orchestrator] {video_id}: --skip-judge in effect; skipping Stage C", flush=True)
    else:
        print(f"[Orchestrator] {video_id}: no groundtruth in manifest; Stage C skipped", flush=True)

    expected_ids = retrieval.get("oracle_eval", {}).get("expected_source_ids", []) or []
    merged_tokens = _merge_token_usage(
        retrieval.get("tokens") or {},
        (score_result or {}).get("token_usage") or {},
    )
    judge_overview = {
        "judge_models": (score_result or {}).get("judge_models", judge_models),
        "n_gt": int((score_result or {}).get("n_gt", len(gt_points)) or 0),
        "passed_gt_count": int((score_result or {}).get("passed_gt_count", 0) or 0),
        "score": float((score_result or {}).get("score", 0.0) or 0.0) if score_result else None,
        "skipped_reason": (score_result or {}).get("skipped_reason"),
    }
    record = {
        "video_id": video_id,
        "task": str(entry.get("task") or ""),
        "topic": str(entry.get("topic") or ""),
        "judge_overview": judge_overview,
        "manifest_entry_snapshot": {
            "id": entry.get("id"),
            "topic": entry.get("topic"),
            "task": entry.get("task"),
            "video2": entry.get("video2", ""),
            "video3": entry.get("video3", ""),
        },
        "retrieval": {
            "input_video_id": retrieval.get("input_video_id"),
            "matched_urls": retrieval.get("matched_urls", []),
            "matched_youtube_ids": retrieval.get("matched_youtube_ids", []),
            "retrieved_truth_ids": retrieved_truth_ids,
            "source_video_paths": local_sources,
            "forgery_type": retrieval.get("forgery_type"),
            "is_multi_source": retrieval.get("is_multi_source"),
            "stats": retrieval.get("stats", {}),
            "expected_source_ids": expected_ids,
            "truth_hit": bool(retrieved_truth_ids),
            "search_only": bool(retrieval.get("search_only", False)),
        },
        "deepsearch": {
            "collected_forgery_points": retrieval.get("collected_forgery_points", []),
            "evidence_videos": retrieval.get("evidence_videos", []),
        },
        "tokens": merged_tokens,
        "analysis": analysis,
        "ground_truth": gt_points,
        "score": score_result,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    _append_jsonl(results_jsonl_path, record, lock=results_jsonl_lock)
    if save_run_trace and trace_jsonl_path and trace_jsonl_lock:
        _append_jsonl(
            trace_jsonl_path,
            _build_trace_row(record=record, run_trace=retrieval.get("run_trace", {})),
            lock=trace_jsonl_lock,
        )
    with completed_ids_lock:
        completed_ids.add(video_id)
    print(f"[Orchestrator] {video_id}: appended to {results_jsonl_path}", flush=True)
    return record


def _build_summary(
    *,
    folder: Path,
    manifest: str,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    n_videos = len(records)
    n_judged = sum(1 for r in records if (r.get("score") or {}).get("ok"))

    retrieval_hits = sum(1 for r in records if r.get("retrieval", {}).get("truth_hit"))
    per_task_retr: dict[str, list[bool]] = defaultdict(list)
    per_task_score: dict[str, list[float]] = defaultdict(list)
    hit_points = 0
    total_points = 0
    per_video_rows = []
    for r in records:
        task = r.get("task") or "?"
        truth_hit = bool(r.get("retrieval", {}).get("truth_hit"))
        per_task_retr[task].append(truth_hit)
        score_obj = r.get("score") or {}
        if score_obj.get("ok"):
            sc = float(score_obj.get("score", 0.0))
            per_task_score[task].append(sc)
            hits = int(score_obj.get("hits", 0))
            n_gt = int(score_obj.get("n_gt", 0) or 0)
            hit_points += hits
            total_points += n_gt
        per_video_rows.append(
            {
                "video_id": r.get("video_id"),
                "task": task,
                "truth_hit": truth_hit,
                "resolved_groups": r.get("retrieval", {}).get("stats", {}).get("resolved_groups", 0),
                "total_groups": r.get("retrieval", {}).get("stats", {}).get("total_groups", 0),
                "score": (score_obj.get("score") if score_obj.get("ok") else None),
                "matched_urls": r.get("retrieval", {}).get("matched_urls", []),
            }
        )

    retrieval_rate = retrieval_hits / n_videos if n_videos else 0.0
    accuracy = hit_points / total_points if total_points else 0.0

    # Token aggregation
    total_tokens = empty_token_counter()
    retrieval_stage_tokens = empty_token_counter()
    judge_stage_tokens = empty_token_counter()
    tokens_by_model: dict[str, dict[str, int]] = {}
    for r in records:
        t = r.get("tokens") or {}
        add_token_counter(total_tokens, t)
        merge_model_token_usage(tokens_by_model, t.get("by_model"))
        stages = t.get("stages") or {}
        add_token_counter(retrieval_stage_tokens, stages.get("retrieval"))
        add_token_counter(judge_stage_tokens, stages.get("judge"))

    return {
        "folder": str(folder),
        "manifest": manifest,
        "n_videos": n_videos,
        "n_judged": n_judged,
        "retrieval": {
            "videos_with_any_truth_hit": retrieval_hits,
            "retrieval_rate": round(retrieval_rate, 4),
            "per_task_breakdown": {
                t: round(sum(v) / len(v), 4) if v else 0.0
                for t, v in sorted(per_task_retr.items())
            },
        },
        "scoring": {
            "total_points": total_points,
            "hit_points": hit_points,
            "accuracy": round(accuracy, 4),
            "per_task_breakdown": {
                t: round(sum(v) / len(v), 4) if v else 0.0
                for t, v in sorted(per_task_score.items())
            },
        },
        "tokens": {
            "total_prompt_tokens": total_tokens["prompt_tokens"],
            "total_completion_tokens": total_tokens["completion_tokens"],
            "total_tokens": total_tokens["total_tokens"],
            "per_video_avg_tokens": round(total_tokens["total_tokens"] / n_videos) if n_videos else 0,
            "by_model": tokens_by_model,
            "by_stage": {
                "retrieval": retrieval_stage_tokens,
                "judge": judge_stage_tokens,
            },
        },
        "per_video": per_video_rows,
    }


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    input_path = Path(args.input_path).resolve()
    if input_path.is_file():
        folder = input_path.parent
        manifest_path = input_path
    else:
        folder = input_path
        if not folder.is_dir():
            raise SystemExit(f"Input path not found: {input_path}")
        manifest_path = folder / args.manifest
    if not manifest_path.is_file():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    output_name = args.output
    if output_name == "_results":
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"_results_{ts}"
    output_dir = folder / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.log_dir:
        log_dir = Path(args.log_dir).resolve()
    else:
        log_subdir = "logs"
        log_dir = output_dir / log_subdir
    log_dir.mkdir(parents=True, exist_ok=True)
    results_jsonl_path = _resolve_output_path(output_dir, args.results_jsonl, "results.jsonl")
    trace_jsonl_path = _resolve_output_path(output_dir, args.trace_jsonl, "run_trace.jsonl")
    results_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    if args.save_run_trace:
        trace_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    results_jsonl_lock = threading.Lock()
    trace_jsonl_lock = threading.Lock()

    downloads_dir = Path(args.download_dir).resolve()
    downloads_dir.mkdir(parents=True, exist_ok=True)

    existing_records = _dedupe_records_by_video_id(_load_jsonl_records(results_jsonl_path))
    completed_ids = {
        str(r.get("video_id") or "").strip()
        for r in existing_records
        if str(r.get("video_id") or "").strip()
    }
    completed_ids_lock = threading.Lock()

    entries_all = load_manifest_entries(manifest_path)
    only_ids = [s.strip() for s in (args.only_ids or "").split(",") if s.strip()]
    entries = _filter_entries(entries_all, only_ids=only_ids or None, limit=args.limit)
    requested_workers = args.num_workers if args.num_workers is not None else args.parallel_videos
    n_workers = max(1, int(requested_workers))
    print(
        f"[Orchestrator] folder={folder} manifest={manifest_path.name} "
        f"entries={len(entries)}/{len(entries_all)} output={output_dir} log_dir={log_dir} "
        f"results_jsonl={results_jsonl_path} "
        f"{'trace_jsonl=' + str(trace_jsonl_path) if args.save_run_trace else 'trace_jsonl=OFF'} "
        f"resume_hits={len(completed_ids) if args.resume else 0}",
        flush=True,
    )

    agent = VisualRetrievalAgentV2(
        top_k=args.top_k,
        total_sample_frames=args.total_sample_frames,
        candidate_sample_frames=args.candidate_sample_frames,
        candidate_video_height=args.candidate_video_height,
        max_reflect_rounds=args.max_reflect_rounds,
        download_output_dir=str(downloads_dir),
        oracle_manifest_path=str(manifest_path),
        search_only=args.search_only,
        query_temperature=args.query_temperature,
        infra_consecutive_threshold=3,
        verbose=not args.quiet,
        max_deepsearch_rounds=args.max_deepsearch_rounds,
        coarse_sample_frames=args.coarse_sample_frames,
        use_cot=args.use_cot,
        save_run_trace=args.save_run_trace,
        reasoning_model=args.reasoning_model,
        frame_resize_workers=args.frame_resize_workers,
    )
    print(
        f"[Orchestrator] deepsearch hyperparameters: "
        f"total_sample_frames={args.total_sample_frames}, "
        f"top_k={args.top_k}, "
        f"candidate_sample_frames={agent.candidate_sample_frames}, "
        f"candidate_video_height={args.candidate_video_height}, "
        f"coarse_sample_frames={args.coarse_sample_frames}, "
        f"max_deepsearch_rounds={args.max_deepsearch_rounds}, "
        f"max_reflect_rounds={args.max_reflect_rounds}, "
        f"query_temperature={args.query_temperature}, "
        f"use_cot={args.use_cot}, "
        f"reasoning_model={agent.reasoning_model}, "
        f"vlm_model={agent.model}, "
        f"frame_resize_workers={agent.frame_resize_workers}",
        flush=True,
    )

    judge_models = _resolve_judge_models(args)
    print(f"[Orchestrator] judge models: {judge_models}", flush=True)
    client = get_llm_client()

    records: list[dict[str, Any]] = list(existing_records) if args.resume else []

    def _process_one_wrapper(entry):
        """Wrapper that catches exceptions per-video for thread safety."""
        try:
            return _process_one(
                entry=entry,
                dataset_folder=folder,
                output_dir=output_dir,
                downloads_dir=downloads_dir,
                agent=agent,
                judge_models=judge_models,
                judge_client=client,
                skip_judge=args.skip_judge,
                save_run_trace=args.save_run_trace,
                resume=args.resume,
                completed_ids=completed_ids,
                completed_ids_lock=completed_ids_lock,
                log_dir=log_dir,
                results_jsonl_path=results_jsonl_path,
                results_jsonl_lock=results_jsonl_lock,
                trace_jsonl_path=(trace_jsonl_path if args.save_run_trace else None),
                trace_jsonl_lock=(trace_jsonl_lock if args.save_run_trace else None),
                parallel_mode=(n_workers > 1),
            )
        except KeyboardInterrupt:
            vid = str(entry.get("id") or "?")
            print(f"[Orchestrator] entry {vid!r}: interrupted", flush=True)
            return None
        except Exception as exc:
            vid = str(entry.get("id") or "?")
            print(f"[Orchestrator] entry {vid!r}: fatal: {exc}", flush=True)
            return None

    if n_workers <= 1:
        # Sequential processing (original behavior)
        for entry in entries:
            try:
                result = _process_one_wrapper(entry)
            except KeyboardInterrupt:
                print("\n[Orchestrator] Interrupted by user; saving partial summary.", flush=True)
                break
            if result is not None:
                records.append(result)
    else:
        # Parallel processing with ThreadPoolExecutor
        import concurrent.futures

        print(f"[Orchestrator] parallel mode: {n_workers} workers", flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_process_one_wrapper, entry): entry
                for entry in entries
            }
            try:
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    if result is not None:
                        records.append(result)
            except KeyboardInterrupt:
                print("\n[Orchestrator] Interrupted by user; cancelling remaining tasks", flush=True)
                for f in futures:
                    f.cancel()
                executor.shutdown(wait=False, cancel_futures=True)

    # Save summary even if some videos were interrupted
    if args.skip_summary:
        print("[Orchestrator] --skip-summary in effect; summary.json and summary.md not written", flush=True)
    else:
        summary = _build_summary(folder=folder, manifest=manifest_path.name, records=_dedupe_records_by_video_id(records))
        summary_path = output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        _write_summary_md(summary, output_dir / "summary.md")
        print(f"[Orchestrator] wrote {summary_path}", flush=True)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

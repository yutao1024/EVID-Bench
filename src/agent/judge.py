"""Stage C: multi-judge LLM voting for forgery-point scoring.

Each judge scores GT 3-5 points against predicted 3-5 points. Final passage for
each GT point is decided by majority vote across judges.
"""

from __future__ import annotations

from typing import Any

from src.agent.prompts import PROMPT_JUDGE_POINTS
from src.utils.agent_helpers import (
    _build_multimodal_content,
    call_vlm_with_retry,
    extract_json_from_text,
)
from src.utils.token_usage import (
    add_model_token_usage,
    add_token_counter,
    empty_token_counter,
)


DEFAULT_JUDGE_MODELS = [
    "gpt-5.5",
    "gemini-3.1-pro",
    "claude-sonnet-4.6",
]


def _format_points_block(points: list[dict[str, Any]] | list[str], *, label: str) -> str:
    """Render either:
      - a list of {"zh": ..., "en": ..., "manipulation_type": ...} dicts (predictions), or
      - a list of {"description": ...} dicts (deepsearch v2 predictions), or
      - a list of plain strings (GT)
    as a numbered block for the judge prompt."""
    lines: list[str] = []
    for i, p in enumerate(points):
        if isinstance(p, str):
            lines.append(f"[{label}-{i + 1}] {p.strip()}")
        elif isinstance(p, dict):
            # v2 deepsearch uses "description"; v1 uses "zh"
            text = str(p.get("description") or p.get("zh") or "").strip()
            mt = str(p.get("manipulation_type") or "").strip()
            mp = str(p.get("misleading_point") or "").strip()
            if mt or mp:
                lines.append(
                    f"[{label}-{i + 1}] {text}\n"
                    f"    (declared_manipulation_type={mt!r}, declared_misleading_point={mp!r})"
                )
            else:
                lines.append(f"[{label}-{i + 1}] {text}")
        else:
            lines.append(f"[{label}-{i + 1}] {p!r}")
    return "\n".join(lines)


def _judge_points_single(
    *,
    video_id: str,
    gt_points: list[str],
    pred_points: list[dict[str, Any]],
    client,
    model: str,
    temperature: float = 0.0,
    logger=None,
) -> dict[str, Any]:
    """Run one judge model. Always emits a result dict, even on errors.

    Returns:
        {
          "video_id": str,
          "hits": int in {0..n_gt},
          "score": float in [0,1],
          "matches": [{"gt_idx": int, "pred_idx": int|None, "verdict": 0/1,
                       "matched_dim_method": bool, "matched_dim_misleading": bool,
                       "reason": str}, ...],
          "comment": str,
          "ok": bool,
          "error": str | None,
        }
    """
    gt = [g for g in (gt_points or []) if g and g.strip()]
    if not gt:
        return {
            "video_id": video_id,
            "hits": 0,
            "score": 0.0,
            "matches": [],
            "comment": "no_groundtruth",
            "ok": False,
            "error": "no_groundtruth",
            "token_usage": {
                **empty_token_counter(),
                "by_model": {model: empty_token_counter()},
            },
        }

    gt_block = _format_points_block(gt, label="GT")
    pred_block = _format_points_block(pred_points or [], label="PRED")
    prompt = PROMPT_JUDGE_POINTS.format(gt_block=gt_block, pred_block=pred_block)

    try:
        raw, judge_tokens = call_vlm_with_retry(
            client,
            model,
            _build_multimodal_content(prompt),
            max_retries=2,
            temperature=temperature,
            json_mode=True,
            logger=logger,
            log_prefix="[Judge] ",
        )
        data = extract_json_from_text(raw)
    except Exception as exc:
        return {
            "video_id": video_id,
            "hits": 0,
            "score": 0.0,
            "matches": [],
            "comment": "",
            "ok": False,
            "error": f"judge_call_failed: {exc}",
            "token_usage": {
                **empty_token_counter(),
                "by_model": {model: empty_token_counter()},
            },
        }

    # Parse robustly. We trust judge's `hits` but clamp to [0, n_gt] and
    # cross-check against `matches`.
    raw_matches = data.get("matches") if isinstance(data, dict) else None
    matches: list[dict[str, Any]] = []
    if isinstance(raw_matches, list):
        for item in raw_matches:
            if not isinstance(item, dict):
                continue
            try:
                gt_idx = int(item.get("gt_idx", -1))
            except Exception:
                gt_idx = -1
            try:
                p_raw = item.get("pred_idx")
                pred_idx = None if p_raw is None else int(p_raw)
            except Exception:
                pred_idx = None
            try:
                verdict = 1 if int(item.get("verdict", 0)) >= 1 else 0
            except Exception:
                verdict = 0
            matches.append(
                {
                    "gt_idx": gt_idx,
                    "pred_idx": pred_idx,
                    "verdict": verdict,
                    "matched_dim_method": bool(item.get("matched_dim_method", False)),
                    "matched_dim_misleading": bool(item.get("matched_dim_misleading", False)),
                    "reason": str(item.get("reason") or "").strip(),
                }
            )
    n_gt = max(1, len(gt))
    # If judge gave a `hits` field trust it (clamped); else derive from matches.
    if isinstance(data, dict) and "hits" in data:
        try:
            hits = int(data.get("hits", 0))
        except Exception:
            hits = sum(m["verdict"] for m in matches)
    else:
        hits = sum(m["verdict"] for m in matches)
    hits = max(0, min(hits, n_gt))
    score = hits / float(n_gt)
    comment = ""
    if isinstance(data, dict):
        comment = str(data.get("comment") or "").strip()

    return {
        "video_id": video_id,
        "hits": hits,
        "score": round(score, 4),
        "n_gt": n_gt,
        "matches": matches,
        "comment": comment,
        "ok": True,
        "error": None,
        "token_usage": {
            **judge_tokens,
            "by_model": {model: judge_tokens},
        },
    }


def judge_points(
    *,
    video_id: str,
    gt_points: list[str],
    pred_points: list[dict[str, Any]],
    client,
    models: list[str],
    temperature: float = 0.0,
    logger=None,
) -> dict[str, Any]:
    """Run multi-judge voting. Majority vote decides each GT point."""
    gt = [g for g in (gt_points or []) if g and g.strip()]
    n_gt = len(gt)
    if not gt:
        return {
            "video_id": video_id,
            "judge_models": list(models or []),
            "n_gt": 0,
            "passed_gt_count": 0,
            "hits": 0,
            "score": 0.0,
            "gt_vote_results": [],
            "judge_results": [],
            "comment": "no_groundtruth",
            "ok": False,
            "error": "no_groundtruth",
            "token_usage": {
                **empty_token_counter(),
                "by_model": {},
            },
        }

    active_models = [str(m).strip() for m in (models or []) if str(m).strip()]
    if not active_models:
        active_models = list(DEFAULT_JUDGE_MODELS)

    judge_results: list[dict[str, Any]] = []
    vote_token_usage = empty_token_counter()
    vote_token_usage_by_model: dict[str, dict[str, int]] = {}
    for model in active_models:
        result = _judge_points_single(
            video_id=video_id,
            gt_points=gt,
            pred_points=pred_points,
            client=client,
            model=model,
            temperature=temperature,
            logger=logger,
        )
        result["judge_model"] = model
        judge_results.append(result)
        add_token_counter(vote_token_usage, result.get("token_usage"))
        add_model_token_usage(
            vote_token_usage_by_model,
            model,
            (result.get("token_usage") or {}).get("by_model", {}).get(model, result.get("token_usage") or {}),
        )

    majority_threshold = len(active_models) / 2.0
    gt_vote_results: list[dict[str, Any]] = []
    passed_gt_count = 0

    for gt_idx, gt_text in enumerate(gt):
        judge_votes: list[dict[str, Any]] = []
        pass_votes = 0
        matched_pred_indices: list[int] = []
        reasons: list[str] = []
        for jr in judge_results:
            match_obj = None
            for m in jr.get("matches") or []:
                if int(m.get("gt_idx", -1)) == gt_idx:
                    match_obj = m
                    break
            verdict = int(match_obj.get("verdict", 0)) if isinstance(match_obj, dict) else 0
            pred_idx = match_obj.get("pred_idx") if isinstance(match_obj, dict) else None
            reason = str(match_obj.get("reason") or "").strip() if isinstance(match_obj, dict) else ""
            if verdict >= 1:
                pass_votes += 1
            if isinstance(pred_idx, int):
                matched_pred_indices.append(pred_idx)
            if reason:
                reasons.append(reason)
            judge_votes.append(
                {
                    "judge_model": jr.get("judge_model"),
                    "verdict": 1 if verdict >= 1 else 0,
                    "pred_idx": pred_idx,
                    "reason": reason,
                }
            )

        passed = pass_votes >= majority_threshold
        if passed:
            passed_gt_count += 1
        gt_vote_results.append(
            {
                "gt_idx": gt_idx,
                "gt_text": gt_text,
                "pass_votes": pass_votes,
                "judge_count": len(active_models),
                "passed": passed,
                "matched_pred_indices": matched_pred_indices,
                "judge_votes": judge_votes,
                "reasons": reasons,
            }
        )

    ok = any(jr.get("ok") for jr in judge_results)
    errors = [str(jr.get("error") or "").strip() for jr in judge_results if jr.get("error")]
    score = passed_gt_count / float(n_gt) if n_gt else 0.0
    comment = "; ".join(
        f"{jr.get('judge_model')}: {jr.get('comment')}"
        for jr in judge_results
        if str(jr.get("comment") or "").strip()
    )

    return {
        "video_id": video_id,
        "judge_models": active_models,
        "n_gt": n_gt,
        "passed_gt_count": passed_gt_count,
        "hits": passed_gt_count,
        "score": round(score, 4),
        "gt_vote_results": gt_vote_results,
        "judge_results": judge_results,
        "comment": comment,
        "ok": ok,
        "error": "; ".join(errors) if errors else None,
        "token_usage": {
            **vote_token_usage,
            "by_model": vote_token_usage_by_model,
        },
    }


def sanity_self_judge(
    *,
    client,
    model: str,
    sample_gt: list[str],
    logger=None,
) -> dict[str, Any]:
    """Feed a GT list to the judge AS THE PREDICTION. Expect score == 1.0.

    Useful for catching prompt regressions / model drift before a real run.
    """
    pred = [
        {
            "zh": s,
            "en": "(self-test, same as GT)",
            "manipulation_type": "",
            "misleading_point": "",
        }
        for s in sample_gt
    ]
    result = judge_points(
        video_id="__sanity__",
        gt_points=sample_gt,
        pred_points=pred,
        client=client,
        models=[model],
        logger=logger,
    )
    result["passed"] = result.get("ok") and result.get("score", 0.0) >= 0.99
    return result

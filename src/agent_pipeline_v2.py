"""Stage A v2 (deepsearch): iterative deep-search retrieval agent.

Replaces the flat search+verify loop with a two-stage retrieval pipeline:
  1. Uniform 64-frame sampling (no shot detection)
  2. Single VLM COT call -> overlay triage + temporal analysis + keyword generation
  3. Deepsearch loop (per source group, serial):
     a. Search 1 keyword -> 1 candidate video
     b. Coarse filter: 16 input + 16 candidate frames, VLM judges relevance
     c. Fine filter: sample 64 frames, VLM extracts 3-5 forgery points
     d. Sufficiency check: VLM judges if evidence is complete
     e. If not sufficient, reflect -> new keyword -> repeat (max N rounds)
  4. Return results + collected forgery points

StageC (judge.py) is unchanged. StageB (forgery_analyzer.py) is skipped
since forgery points are already extracted during fine retrieval.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from src.agent.oracle_eval import (
    evaluate_oracle,
    extract_youtube_id,
    load_oracle_map,
)
from src.agent.prompts import (
    PROMPT_COT_RETRIEVAL_V3,
    PROMPT_REFLECT_REFINE,
    PROMPT_VERIFY_MATCH,
)
from src.agent.session_state import CandidateRecord, SessionState
from src.agent.tools import AgentTools
from src.utils.agent_helpers import (
    _build_multimodal_content,
    call_vlm_with_retry,
    extract_json_from_text,
    sanitize_queries,
    uniform_sample_frames,
)


class VisualRetrievalAgentV2:
    """Deepsearch visual retrieval agent (iterative coarse->fine pipeline).

    Public API: `run_retrieval(video_path)` -> dict with `matched_urls`,
    `retrieved_truth_ids`, `source_video_paths`, stats, etc.
    """

    def __init__(
        self,
        *,
        top_k: int = 10,
        total_sample_frames: int = 64,
        candidate_sample_frames: int | None = None,
        candidate_video_height: int = 480,
        max_reflect_rounds: int = 3,
        download_output_dir: str = "downloads",
        oracle_manifest_path: str | None = None,
        search_only: bool = False,
        query_temperature: float = 0.0,
        infra_consecutive_threshold: int = 3,
        verbose: bool = True,
        # Deepsearch-specific params
        max_deepsearch_rounds: int = 6,
        coarse_sample_frames: int = 16,
        use_cot: bool = True,
        save_run_trace: bool = False,
        reasoning_model: str | None = None,
        frame_resize_workers: int | None = None,
    ) -> None:
        from src.utils.config import get_llm_client

        self.client = get_llm_client()
        self.reasoning_model = reasoning_model
        self.model = self.reasoning_model
        if frame_resize_workers is None:
            from src.utils.frame_sampling import default_frame_resize_workers

            self.frame_resize_workers = default_frame_resize_workers()
        else:
            self.frame_resize_workers = max(1, int(frame_resize_workers))
        self.top_k = top_k
        self.total_sample_frames = max(1, int(total_sample_frames))
        if candidate_sample_frames is None:
            self.candidate_sample_frames = self.total_sample_frames
        else:
            self.candidate_sample_frames = max(1, int(candidate_sample_frames))
        self.candidate_video_height = max(1, int(candidate_video_height))
        self.max_reflect_rounds = max_reflect_rounds
        self.download_output_dir = str(Path(download_output_dir).resolve())
        self.oracle_map = load_oracle_map(oracle_manifest_path)
        self.search_only = bool(search_only)
        self.query_temperature = query_temperature
        self.infra_consecutive_threshold = infra_consecutive_threshold
        self.verbose = verbose
        self.max_deepsearch_rounds = max_deepsearch_rounds
        self.coarse_sample_frames = coarse_sample_frames
        self.use_cot = use_cot
        self.save_run_trace = bool(save_run_trace)
        self._tls = threading.local()

    def set_log_file(self, path: str | None) -> None:
        """Attach/detach a per-video log file for the current thread."""
        old = getattr(self._tls, "log_fh", None)
        if old:
            try:
                old.close()
            except OSError:
                pass
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self._tls.log_fh = open(path, "w", encoding="utf-8")
        else:
            self._tls.log_fh = None

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def _log(self, msg: str) -> None:
        if not self.verbose:
            return
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        fh = getattr(self._tls, "log_fh", None)
        if fh:
            fh.write(line + "\n")
            fh.flush()

    # ------------------------------------------------------------------
    # COT reasoning via VLM
    # ------------------------------------------------------------------
    def _fallback_source_queries(
        self,
        *,
        input_video_id: str,
        cot_entities: dict[str, list[str]] | None = None,
        cot_search_intent: str = "",
        source_descriptions: list[str] | None = None,
        prev_queries: list[str] | None = None,
    ) -> list[str]:
        """Build lightweight fallback queries when VLM planning fails."""
        out: list[str] = []
        seen: set[str] = set()

        def _add(q: str) -> None:
            sq = " ".join(str(q or "").replace("_", " ").split()).strip()
            if not sq:
                return
            key = sq.lower()
            if key in seen:
                return
            seen.add(key)
            out.append(sq)

        entities = cot_entities or {}
        for bucket in ("people", "events", "locations", "text_claims"):
            for item in entities.get(bucket) or []:
                _add(str(item))
                if len(out) >= 4:
                    break
            if len(out) >= 4:
                break
        for desc in source_descriptions or []:
            _add(str(desc).split(":")[-1].strip())
            if len(out) >= 4:
                break
        _add(cot_search_intent)
        _add(input_video_id)

        prev = {str(q).strip().lower() for q in (prev_queries or []) if str(q).strip()}
        return [q for q in out if q.lower() not in prev][:4]

    def _cot_reasoning(self, frame_paths: list[str]) -> dict[str, Any]:
        """Single VLM call: V3 content understanding + retrieval planning."""
        self._log(f"[COT] sending {len(frame_paths)} frames to VLM for chain-of-thought analysis")
        contents = _build_multimodal_content(PROMPT_COT_RETRIEVAL_V3, image_paths=frame_paths)
        raw, tokens = call_vlm_with_retry(
            self.client,
            self.reasoning_model,
            contents,
            max_retries=3,
            temperature=self.query_temperature,
            json_mode=True,
            logger=self._log,
            log_prefix="[COT] ",
        )
        data = extract_json_from_text(raw)

        estimated_sources = int(data.get("estimated_sources") or 1)

        raw_entities = data.get("entities") if isinstance(data.get("entities"), dict) else {}
        entities = {
            "people": [str(x).strip() for x in (raw_entities.get("people") or []) if str(x).strip()],
            "locations": [str(x).strip() for x in (raw_entities.get("locations") or []) if str(x).strip()],
            "events": [str(x).strip() for x in (raw_entities.get("events") or []) if str(x).strip()],
            "text_claims": [str(x).strip() for x in (raw_entities.get("text_claims") or []) if str(x).strip()],
        }

        raw_source_queries = data.get("source_queries")
        source_queries: list[dict[str, Any]] = []
        if isinstance(raw_source_queries, list) and raw_source_queries:
            for sq in raw_source_queries:
                if not isinstance(sq, dict):
                    continue
                label = str(sq.get("source_label") or "unknown_source").strip()
                qs = sanitize_queries(sq.get("queries"), limit=1)
                if qs:
                    source_queries.append({"source_label": label, "queries": qs})
        elif data.get("initial_query"):
            qs = sanitize_queries([data.get("initial_query")], limit=1)
            if qs:
                source_queries.append({"source_label": "source_1", "queries": qs})

        if not source_queries:
            source_queries.append({"source_label": "source_1", "queries": []})

        result = {
            "reasoning": str(data.get("reasoning") or "").strip(),
            "physical_observations": str(data.get("physical_observations") or "").strip(),
            "logical_analysis": str(data.get("logical_analysis") or "").strip(),
            "search_intent": str(data.get("search_intent") or "").strip(),
            "entities": entities,
            "estimated_sources": estimated_sources,
            "source_queries": source_queries,
            "tokens": tokens,
        }

        all_queries = [q for sq in source_queries for q in sq["queries"]]
        self._log(f"[COT] estimated_sources={estimated_sources}, total_queries={len(all_queries)}")
        for sq in source_queries:
            self._log(f"[COT]   {sq['source_label']}: {sq['queries']}")
        if result["physical_observations"]:
            self._log(f"[COT] physical_observations: {result['physical_observations'][:300]}")
        if result["logical_analysis"]:
            self._log(f"[COT] logical_analysis: {result['logical_analysis'][:300]}")
        if result["search_intent"]:
            self._log(f"[COT] search_intent: {result['search_intent'][:200]}")

        return result

    # ------------------------------------------------------------------
    # REFLECT: generate new keyword when search fails
    # ------------------------------------------------------------------
    def _reflect_query(
        self,
        frame_paths: list[str],
        prev_queries: list[str],
        wrong_titles: list[str],
        *,
        cot_physical_observations: str = "",
        cot_logical_analysis: str = "",
        cot_search_intent: str = "",
        cot_entities: dict[str, list[str]] | None = None,
    ) -> tuple[str | None, dict[str, int]]:
        """Generate a single fresh keyword when previous one failed."""
        entities = cot_entities or {}
        entities_summary = json.dumps(entities, ensure_ascii=False)
        prompt = PROMPT_REFLECT_REFINE.replace(
            "{prev_queries}", json.dumps(prev_queries, ensure_ascii=False)
        ).replace("{candidate_titles}", json.dumps(wrong_titles, ensure_ascii=False))
        prompt = prompt.replace("{physical_observations}", cot_physical_observations or "(not available)")
        prompt = prompt.replace("{logical_analysis}", cot_logical_analysis or "(not available)")
        prompt = prompt.replace("{search_intent}", cot_search_intent or "(not available)")
        prompt = prompt.replace("{entities_summary}", entities_summary)

        contents = _build_multimodal_content(prompt, image_paths=frame_paths)
        raw, tokens = call_vlm_with_retry(
            self.client,
            self.reasoning_model,
            contents,
            max_retries=3,
            temperature=self.query_temperature,
            json_mode=True,
            logger=self._log,
            log_prefix="[Reflect] ",
        )
        data = extract_json_from_text(raw)
        new_queries = sanitize_queries(data.get("new_queries"), limit=1)
        self._log(f"[Reflect] new_queries: {new_queries}")
        if data.get("reflection"):
            self._log(f"[Reflect] reflection: {str(data['reflection'])[:300]}")
        return (new_queries[0] if new_queries else None), (tokens or {})

    def _safe_reflect_query(
        self,
        frame_paths: list[str],
        prev_queries: list[str],
        wrong_titles: list[str],
        *,
        input_video_id: str,
        cot_physical_observations: str = "",
        cot_logical_analysis: str = "",
        cot_search_intent: str = "",
        cot_entities: dict[str, list[str]] | None = None,
        source_descriptions: list[str] | None = None,
    ) -> tuple[str | None, dict[str, int]]:
        """Reflect with a rule-based fallback when the VLM is unavailable."""
        try:
            return self._reflect_query(
                frame_paths,
                prev_queries,
                wrong_titles,
                cot_physical_observations=cot_physical_observations,
                cot_logical_analysis=cot_logical_analysis,
                cot_search_intent=cot_search_intent,
                cot_entities=cot_entities,
            )
        except Exception as exc:
            self._log(f"[Reflect] fallback due to failure: {exc}")
            fallback_queries = self._fallback_source_queries(
                input_video_id=input_video_id,
                cot_entities=cot_entities,
                cot_search_intent=cot_search_intent,
                source_descriptions=source_descriptions,
                prev_queries=prev_queries,
            )
            return (fallback_queries[0] if fallback_queries else None), {}

    def _looks_like_related_not_same_source(
        self,
        candidate_title: str,
        points: list[dict[str, Any]],
        source_description: str = "",
    ) -> bool:
        """Heuristic guardrail against stopping on merely related evidence.

        We keep this lightweight and intentionally conservative: if the returned
        forgery points mostly emphasize different shows/eras/networks/contexts,
        we should continue searching even if the VLM says the evidence is
        otherwise "sufficient".
        """
        text_parts = [candidate_title, source_description]
        text_parts.extend(str(p.get("description") or "") for p in points)
        blob = " ".join(text_parts).lower()
        mismatch_markers = [
            "different era",
            "different show",
            "different season",
            "different production",
            "different period",
            "different context",
            "different network",
            "different program",
            "not the same show",
            "not the same season",
            "not the same program",
            "velocity",
            "tlc",
            "anachronistic",
            "broadcast history",
            "network affiliation",
        ]
        hits = sum(1 for marker in mismatch_markers if marker in blob)
        return hits >= 2

    # ------------------------------------------------------------------
    # Deepsearch loop: unified single-loop
    # ------------------------------------------------------------------
    async def _deepsearch(
        self,
        *,
        input_video_id: str,
        initial_keyword: str,
        source_descriptions: list[str],
        session: SessionState,
        tools: AgentTools,
        input_frame_paths: list[str],
        input_coarse_frame_paths: list[str],
        cache_root: str,
        cot_physical_observations: str = "",
        cot_logical_analysis: str = "",
        cot_search_intent: str = "",
        cot_entities: dict[str, list[str]] | None = None,
        total_tokens: dict[str, int] | None = None,
        run_trace: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run a single unified deepsearch loop.

        Each round: keyword → search → download → coarse → fine → next_step.
        The VLM self-decides what keyword to use next based on all context
        (COT source descriptions, collected points, examined videos).
        """
        if input_frame_paths:
            self._log(
                f"[Sampling] input frames for deepsearch: {len(input_frame_paths)} reused "
                f"from {Path(input_frame_paths[0]).parent} (fine/sufficiency/reflect)"
            )
        if input_coarse_frame_paths:
            self._log(
                f"[Sampling] input coarse frames: {len(input_coarse_frame_paths)} reused "
                f"from {Path(input_coarse_frame_paths[0]).parent}"
            )
        collected_points: list[dict[str, Any]] = []
        evidence_videos: list[dict[str, str]] = []
        source_like_points: list[dict[str, Any]] = []
        source_like_videos: list[dict[str, str]] = []
        supporting_urls: list[str] = []
        matched_urls: list[str] = []

        current_keyword = initial_keyword
        wrong_titles: list[str] = []
        all_prev_queries: list[str] = []

        round_num = 0
        while True:
            round_num += 1
            if round_num > self.max_deepsearch_rounds:
                self._log("[DeepSearch] exceeded max_deepsearch_rounds; hard stop")
                break

            round_trace_entry: dict[str, Any] | None = None
            if run_trace is not None:
                round_trace_entry = {"round": round_num, "query": current_keyword or ""}
                run_trace.setdefault("rounds", []).append(round_trace_entry)
            if not current_keyword:
                self._log("[DeepSearch] no keyword available; stopping")
                if round_trace_entry is not None:
                    round_trace_entry["stop_reason"] = "no_keyword_available"
                break

            self._log(
                f"[DeepSearch] round {round_num}/{self.max_deepsearch_rounds} "
                f"keyword={current_keyword!r}"
            )
            all_prev_queries.append(current_keyword)

            # --- Step 1: Search (async) ---
            search_result = await tools.search_one_async(session, current_keyword)
            if not search_result.get("ok"):
                reason = search_result.get("reason", "unknown")
                self._log(f"[DeepSearch] search failed: {reason}")
                if round_trace_entry is not None:
                    round_trace_entry["search"] = {"ok": False, "reason": reason}
                if round_num < self.max_deepsearch_rounds:
                    current_keyword, reflect_tokens = self._safe_reflect_query(
                        input_frame_paths,
                        all_prev_queries,
                        wrong_titles[-8:],
                        input_video_id=input_video_id,
                        cot_physical_observations=cot_physical_observations,
                        cot_logical_analysis=cot_logical_analysis,
                        cot_search_intent=cot_search_intent,
                        cot_entities=cot_entities,
                        source_descriptions=source_descriptions,
                    )
                    if total_tokens is not None:
                        for k in total_tokens:
                            total_tokens[k] += reflect_tokens.get(k, 0)
                else:
                    current_keyword = None
                    if round_trace_entry is not None:
                        round_trace_entry["stop_reason"] = "search_failed_final_round"
                continue

            cand_ref = search_result["candidate_ref"]
            cand_title = search_result.get("title", "")
            self._log(f"[DeepSearch] found: {cand_title!r} ref={cand_ref}")
            if round_trace_entry is not None:
                round_trace_entry["search"] = {
                    "ok": True,
                    "candidate_ref": cand_ref,
                    "candidate_title": cand_title,
                    "candidate_url": search_result.get("url", ""),
                }

            # --- Step 2: Download (async) ---
            dl_result = await tools.download_candidate_async(
                session, cand_ref, self.download_output_dir
            )
            if not dl_result.get("ok"):
                code = str(dl_result.get("reason_code") or dl_result.get("reason") or "download_failed")
                stderr = str(dl_result.get("stderr_tail") or "")[-300:]
                self._log(f"[DeepSearch] download failed: {code}")
                if round_trace_entry is not None:
                    round_trace_entry["download"] = {"ok": False, "reason_code": code}
                if stderr:
                    self._log(f"[DeepSearch] download stderr: {stderr}")
                wrong_titles.append(cand_title)
                if round_num < self.max_deepsearch_rounds:
                    current_keyword, reflect_tokens = self._safe_reflect_query(
                        input_frame_paths,
                        all_prev_queries,
                        wrong_titles[-8:],
                        input_video_id=input_video_id,
                        cot_physical_observations=cot_physical_observations,
                        cot_logical_analysis=cot_logical_analysis,
                        cot_search_intent=cot_search_intent,
                        cot_entities=cot_entities,
                        source_descriptions=source_descriptions,
                    )
                    if total_tokens is not None:
                        for k in total_tokens:
                            total_tokens[k] += reflect_tokens.get(k, 0)
                else:
                    current_keyword = None
                    if round_trace_entry is not None:
                        round_trace_entry["stop_reason"] = "download_failed_final_round"
                continue
            if round_trace_entry is not None:
                round_trace_entry["download"] = {
                    "ok": True,
                    "video_path_saved": bool(dl_result.get("video_path")),
                    "reused": bool(dl_result.get("reused", False)),
                }

            candidate = session.candidates.get(cand_ref)
            if not candidate:
                wrong_titles.append(cand_title)
                if round_trace_entry is not None:
                    round_trace_entry["stop_reason"] = "candidate_missing_after_download"
                continue

            # --- Step 3: Coarse filter (input N + candidate M frames -> relevance check) ---
            coarse_result = tools.sample_candidate_frames(
                candidate,
                output_dir="",
                num_frames=self.coarse_sample_frames,
                prefix="coarse",
                cache_root=cache_root,
            )
            if not coarse_result.get("ok"):
                self._log("[DeepSearch] coarse sample failed")
                wrong_titles.append(cand_title)
                if round_trace_entry is not None:
                    round_trace_entry["coarse"] = {"sample_ok": False, "reason": "coarse_sample_failed"}
                continue

            cand_coarse_cached = bool(coarse_result.get("cached"))
            coarse_input_paths = (
                input_coarse_frame_paths if input_coarse_frame_paths else input_frame_paths
            )
            self._log(
                f"[Sampling] coarse VLM: {len(coarse_input_paths)} input frames (reused) + "
                f"{len(coarse_result['frame_paths'])} candidate frames"
                f"{' (candidate frame_cache hit)' if cand_coarse_cached else ''}"
            )

            relevance = tools.check_coarse_relevance(
                coarse_input_paths,
                coarse_result["frame_paths"],
                physical_observations=cot_physical_observations,
                logical_analysis=cot_logical_analysis,
                search_intent=cot_search_intent,
            )
            if total_tokens is not None:
                for k in total_tokens:
                    total_tokens[k] += relevance.get("tokens", {}).get(k, 0)
            self._log(
                f"[DeepSearch] coarse relevance: {relevance['is_relevant']} "
                f"({relevance['reasoning'][:100]})"
            )
            if round_trace_entry is not None:
                round_trace_entry["coarse"] = {
                    "sample_ok": True,
                    "is_relevant": bool(relevance["is_relevant"]),
                    "reasoning": str(relevance.get("reasoning") or ""),
                }

            if not relevance["is_relevant"]:
                wrong_titles.append(cand_title)
                if round_num < self.max_deepsearch_rounds:
                    current_keyword, reflect_tokens = self._safe_reflect_query(
                        input_frame_paths,
                        all_prev_queries,
                        wrong_titles[-8:],
                        input_video_id=input_video_id,
                        cot_physical_observations=cot_physical_observations,
                        cot_logical_analysis=cot_logical_analysis,
                        cot_search_intent=cot_search_intent,
                        cot_entities=cot_entities,
                        source_descriptions=source_descriptions,
                    )
                    if total_tokens is not None:
                        for k in total_tokens:
                            total_tokens[k] += reflect_tokens.get(k, 0)
                else:
                    current_keyword = None
                    if round_trace_entry is not None:
                        round_trace_entry["stop_reason"] = "coarse_not_relevant_final_round"
                continue

            # --- Step 4: Fine filter (input N + candidate M frames -> forgery points) ---
            fine_result = tools.sample_candidate_frames(
                candidate,
                output_dir="",
                num_frames=self.candidate_sample_frames,
                prefix="fine",
                cache_root=cache_root,
            )
            if not fine_result.get("ok"):
                self._log("[DeepSearch] fine sample failed")
                if round_trace_entry is not None:
                    round_trace_entry["fine"] = {"sample_ok": False, "reason": "fine_sample_failed"}
                continue

            cand_fine_cached = bool(fine_result.get("cached"))
            self._log(
                f"[Sampling] fine VLM: {len(input_frame_paths)} input frames (reused) + "
                f"{len(fine_result['frame_paths'])} candidate frames"
                f"{' (candidate frame_cache hit)' if cand_fine_cached else ''}"
            )

            forgery_result = tools.extract_fine_forgery_points(
                input_frame_paths,
                fine_result["frame_paths"],
                physical_observations=cot_physical_observations,
                logical_analysis=cot_logical_analysis,
                search_intent=cot_search_intent,
                entities=cot_entities or {},
            )
            if total_tokens is not None:
                for k in total_tokens:
                    total_tokens[k] += forgery_result.get("tokens", {}).get(k, 0)
            new_points = forgery_result.get("points") or []
            source_description = str(forgery_result.get("source_description") or "")
            unique_new: list[dict[str, Any]] = []
            candidate_is_supporting_only = False
            self._log(f"[DeepSearch] fine extraction: {len(new_points)} forgery points")
            if round_trace_entry is not None:
                round_trace_entry["fine"] = {
                    "sample_ok": True,
                    "points_count": len(new_points),
                    "source_description": source_description,
                }

            if new_points:
                # Deduplicate: skip points too similar to already-collected ones
                for pt in new_points:
                    desc = str(pt.get("description") or "").lower()
                    if not desc:
                        continue
                    desc_words = set(desc.split())
                    is_dup = False
                    for existing in collected_points:
                        ex_desc = str(existing.get("description") or "").lower()
                        ex_words = set(ex_desc.split())
                        if not ex_words:
                            continue
                        overlap = len(desc_words & ex_words) / max(len(desc_words), 1)
                        if overlap > 0.55:
                            is_dup = True
                            break
                    if not is_dup:
                        pt["evidence_video_title"] = cand_title
                        pt["evidence_video_url"] = candidate.url or ""
                        pt["evidence_video_ref"] = cand_ref
                        unique_new.append(pt)

                if unique_new:
                    self._log(f"[DeepSearch] after dedup: {len(unique_new)}/{len(new_points)} unique points")
                    candidate_is_supporting_only = self._looks_like_related_not_same_source(
                        cand_title, unique_new, source_description
                    )
                    collected_points.extend(unique_new)
                    evidence_videos.append({
                        "title": cand_title,
                        "url": candidate.url or "",
                        "ref": cand_ref,
                        "points_count": len(unique_new),
                    })
                    if candidate.url and candidate.url not in supporting_urls:
                        supporting_urls.append(candidate.url)
                    if not candidate_is_supporting_only:
                        source_like_points.extend(unique_new)
                        source_like_videos.append({
                            "title": cand_title,
                            "url": candidate.url or "",
                            "ref": cand_ref,
                            "points_count": len(unique_new),
                        })

            session.verification_events.append({
                "turn": round_num,
                "phase": "deepsearch",
                "candidate_ref": cand_ref,
                "candidate_url": candidate.url or "",
                "candidate_title": cand_title,
                "coarse_relevant": relevance["is_relevant"],
                "fine_points_count": len(new_points),
            })

            # --- Step 5: Sufficiency + next keyword (single VLM call) ---
            try:
                next_step = tools.deepsearch_next_step(
                    source_like_points,
                    [{"title": v["title"], "url": v["url"]} for v in source_like_videos],
                    source_descriptions,
                    all_prev_queries,
                    current_round=round_num,
                    max_rounds=self.max_deepsearch_rounds,
                    physical_observations=cot_physical_observations,
                    logical_analysis=cot_logical_analysis,
                    search_intent=cot_search_intent,
                    entities=cot_entities or {},
                )
                if total_tokens is not None:
                    for k in total_tokens:
                        total_tokens[k] += next_step.get("tokens", {}).get(k, 0)
            except Exception as exc:
                self._log(f"[DeepSearch] next_step VLM call failed: {exc}")
                if round_trace_entry is not None:
                    round_trace_entry["next_step"] = {"ok": False, "error": str(exc)}
                if round_num < self.max_deepsearch_rounds:
                    current_keyword, reflect_tokens = self._safe_reflect_query(
                        input_frame_paths,
                        all_prev_queries,
                        wrong_titles[-8:],
                        input_video_id=input_video_id,
                        cot_physical_observations=cot_physical_observations,
                        cot_logical_analysis=cot_logical_analysis,
                        cot_search_intent=cot_search_intent,
                        cot_entities=cot_entities,
                        source_descriptions=source_descriptions,
                    )
                    if total_tokens is not None:
                        for k in total_tokens:
                            total_tokens[k] += reflect_tokens.get(k, 0)
                else:
                    if source_like_videos and not matched_urls:
                        fallback_video = source_like_videos[-1]
                        fallback_url = str(fallback_video.get("url") or "")
                        if fallback_url:
                            matched_urls.append(fallback_url)
                            self._log("[DeepSearch] next_step failed on final round; using best-effort source-like evidence")
                            if round_trace_entry is not None:
                                round_trace_entry["resolved"] = {
                                    "candidate_ref": str(fallback_video.get("ref") or ""),
                                    "resolved_url": fallback_url,
                                    "mode": "best_effort_after_next_step_failure",
                                }
                    current_keyword = None
                    if round_trace_entry is not None:
                        round_trace_entry["stop_reason"] = "next_step_failed_final_round"
                continue

            self._log(
                f"[DeepSearch] sufficiency: {next_step['is_sufficient']} "
                f"({next_step['reasoning'][:150]})"
            )
            if round_trace_entry is not None:
                round_trace_entry["next_step"] = {
                    "ok": True,
                    "is_sufficient": bool(next_step["is_sufficient"]),
                    "reasoning": str(next_step.get("reasoning") or ""),
                    "missing_description": str(next_step.get("missing_description") or ""),
                    "next_keyword": str(next_step.get("next_keyword") or ""),
                    "forced_stop": bool(next_step.get("forced_stop")),
                }

            if next_step.get("forced_stop"):
                self._log("[DeepSearch] forced stop after exceeding max rounds")
                if round_trace_entry is not None:
                    round_trace_entry["stop_reason"] = "exceeded_max_rounds"
                break

            if next_step["is_sufficient"]:
                if not source_like_videos:
                    self._log(
                        "[DeepSearch] sufficiency overridden: no source-like evidence "
                        "has been collected yet; continuing search"
                    )
                    wrong_titles.append(cand_title)
                    current_keyword = next_step.get("next_keyword", "")
                    if not current_keyword and round_num < self.max_deepsearch_rounds:
                        current_keyword, reflect_tokens = self._safe_reflect_query(
                            input_frame_paths,
                            all_prev_queries,
                            wrong_titles[-8:],
                            input_video_id=input_video_id,
                            cot_physical_observations=cot_physical_observations,
                            cot_logical_analysis=cot_logical_analysis,
                            cot_search_intent=cot_search_intent,
                            cot_entities=cot_entities,
                            source_descriptions=source_descriptions,
                        )
                        if total_tokens is not None:
                            for k in total_tokens:
                                total_tokens[k] += reflect_tokens.get(k, 0)
                    if current_keyword:
                        self._log(f"[DeepSearch] next keyword: {current_keyword!r}")
                        if round_trace_entry is not None:
                            round_trace_entry["next_keyword_selected"] = current_keyword
                        continue
                    self._log("[DeepSearch] no stronger next keyword available; stopping")
                    if round_trace_entry is not None:
                        round_trace_entry["stop_reason"] = "no_stronger_next_keyword"
                    break

                resolved_video = source_like_videos[-1]
                resolved_ref = str(resolved_video["ref"])
                session.propagate_match_to_group(0, resolved_ref)
                session.verification_events[-1]["verification"] = {
                    "is_match": True,
                    "matched_by": "deepsearch_sufficiency",
                    "resolved_candidate_ref": resolved_ref,
                }
                resolved_url = str(resolved_video.get("url") or "")
                if resolved_url and resolved_url not in matched_urls:
                    matched_urls.append(resolved_url)
                self._log("[DeepSearch] evidence sufficient; group resolved; stopping")
                if round_trace_entry is not None:
                    round_trace_entry["resolved"] = {
                        "candidate_ref": resolved_ref,
                        "resolved_url": resolved_url,
                    }
                break

            current_keyword = next_step.get("next_keyword", "")
            if current_keyword:
                self._log(f"[DeepSearch] next keyword: {current_keyword!r}")
                if round_trace_entry is not None:
                    round_trace_entry["next_keyword_selected"] = current_keyword
            else:
                if source_like_videos and not matched_urls:
                    fallback_video = source_like_videos[-1]
                    fallback_url = str(fallback_video.get("url") or "")
                    if fallback_url:
                        matched_urls.append(fallback_url)
                        self._log("[DeepSearch] no next keyword provided; using best-effort source-like evidence")
                        if round_trace_entry is not None:
                            round_trace_entry["resolved"] = {
                                "candidate_ref": str(fallback_video.get("ref") or ""),
                                "resolved_url": fallback_url,
                                "mode": "best_effort_no_next_keyword",
                            }
                self._log("[DeepSearch] no next keyword provided; stopping")
                if round_trace_entry is not None:
                    round_trace_entry["stop_reason"] = "no_next_keyword_provided"
                break

        return {
            "collected_points": collected_points,
            "evidence_videos": evidence_videos,
            "source_like_evidence_videos": source_like_videos,
            "source_like_points": source_like_points,
            "supporting_urls": supporting_urls,
            "matched_urls": matched_urls,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run_retrieval(self, video_path: str) -> dict[str, Any]:
        """Run Stage A v2 (deepsearch) retrieval on a single forged video.

        Returns the same dict format as v1 agent so StageC works unchanged.
        """
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Input video not found: {video_path}")
        start_ts = time.time()
        temp_root = tempfile.mkdtemp(prefix="s_deepsearch_v2_")
        Path(self.download_output_dir).mkdir(parents=True, exist_ok=True)
        self._log("========== Deepsearch Agent v2 run started ==========")
        self._log(f"Input video: {video_path}")
        self._log(
            f"Config: reasoning_model={self.reasoning_model}, vlm_model={self.model}, "
            f"total_sample_frames={self.total_sample_frames}, "
            f"top_k={self.top_k}, "
            f"candidate_sample_frames={self.candidate_sample_frames}, "
            f"candidate_video_height={self.candidate_video_height}, "
            f"coarse_sample_frames={self.coarse_sample_frames}, "
            f"max_deepsearch_rounds={self.max_deepsearch_rounds}, "
            f"max_reflect_rounds={self.max_reflect_rounds}, "
            f"query_temperature={self.query_temperature}, "
            f"use_cot={self.use_cot}, "
            f"search_only={self.search_only}, "
            f"frame_resize_workers={self.frame_resize_workers}"
        )

        # Token tracking
        total_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        try:
            run_trace = None
            if self.save_run_trace:
                run_trace = {
                    "input_video_id": Path(video_path).stem,
                    "config": {
                        "total_sample_frames": self.total_sample_frames,
                        "candidate_sample_frames": self.candidate_sample_frames,
                        "candidate_video_height": self.candidate_video_height,
                        "coarse_sample_frames": self.coarse_sample_frames,
                        "max_deepsearch_rounds": self.max_deepsearch_rounds,
                        "max_reflect_rounds": self.max_reflect_rounds,
                        "query_temperature": self.query_temperature,
                        "use_cot": self.use_cot,
                    },
                    "cot": {},
                    "rounds": [],
                }
            # Step 1: Uniform frame sampling (compressed to candidate_video_height)
            input_video_id = Path(video_path).stem
            sampled_root = str(Path(temp_root) / "sampled_frames")
            # Persistent cache: .frame_cache/input/{id}_h{H}_n{N}/ and .frame_cache/candidate/{id}_h{H}_n{N}/
            from src.utils.frame_sampling import resolve_frame_cache_dir

            _project_root = Path(__file__).resolve().parent.parent
            frame_cache_root = _project_root / ".frame_cache"
            frame_cache_root.mkdir(parents=True, exist_ok=True)
            cache_root = str(frame_cache_root)
            frame_cache_dir = str(
                resolve_frame_cache_dir(
                    frame_cache_root,
                    input_video_id,
                    height=self.candidate_video_height,
                    num_frames=self.total_sample_frames,
                    source="input",
                )
            )
            self._log(f"[Sampling] persistent frame_cache: {cache_root}")
            frame_paths, frame_cache_hit = uniform_sample_frames(
                video_path,
                num_frames=self.total_sample_frames,
                output_dir=sampled_root,
                prefix="frame",
                target_height=self.candidate_video_height,
                cache_dir=frame_cache_dir,
                max_workers=self.frame_resize_workers,
                log_fn=self._log,
            )
            coarse_cache_dir = str(
                resolve_frame_cache_dir(
                    frame_cache_root,
                    input_video_id,
                    height=self.candidate_video_height,
                    num_frames=self.coarse_sample_frames,
                    source="input",
                )
            )
            input_coarse_frame_paths, coarse_cache_hit = uniform_sample_frames(
                video_path,
                num_frames=self.coarse_sample_frames,
                output_dir=sampled_root,
                prefix="frame",
                target_height=self.candidate_video_height,
                cache_dir=coarse_cache_dir,
                max_workers=self.frame_resize_workers,
                log_fn=self._log,
            )
            if coarse_cache_hit:
                self._log(
                    f"[Sampling] frame_cache hit (input coarse): {len(input_coarse_frame_paths)} frames from "
                    f"{coarse_cache_dir}"
                )
            else:
                self._log(
                    f"[Sampling] frame_cache miss (input coarse): ffmpeg extracted "
                    f"{len(input_coarse_frame_paths)} frames to {coarse_cache_dir}"
                )
            if frame_cache_hit:
                self._log(
                    f"[Sampling] frame_cache hit (input): {len(frame_paths)} frames from "
                    f"{frame_cache_dir} (height<={self.candidate_video_height}px, skipped ffmpeg)"
                )
            else:
                self._log(
                    f"[Sampling] frame_cache miss (input): ffmpeg extracted {len(frame_paths)} frames to "
                    f"{frame_cache_dir} (height<={self.candidate_video_height}px)"
                )

            # Create a minimal SessionState
            expected_ids = self.oracle_map.get(input_video_id, [])
            if expected_ids:
                self._log(f"Oracle eval expected ids: {expected_ids}")

            from src.agent.session_state import ShotGroup

            session = SessionState(
                video_path=video_path,
                input_video_id=input_video_id,
                shots=[{"shot_id": 0, "start_frame": 0, "end_frame": 0, "start_sec": 0.0, "end_sec": 0.0}],
                shot_frame_map={0: frame_paths},
                expected_source_ids=expected_ids,
                max_rounds=100,
                per_group_budget=100,
                infra_consecutive_threshold=self.infra_consecutive_threshold,
            )
            session.groups[0] = ShotGroup(
                group_id=0,
                shot_ids=[0],
                physical_observations="",
                queries=[],
            )

            tools = AgentTools(
                client=self.client,
                model=self.model,
                frame_resize_workers=self.frame_resize_workers,
                candidate_sample_frames=self.candidate_sample_frames,
                prompts={
                    "extract": "",
                    "reflect": PROMPT_REFLECT_REFINE,
                    "verify": PROMPT_VERIFY_MATCH,
                    "cluster": "",
                },
                logger=self._log,
                coarse_frames=self.coarse_sample_frames,
                dense_frames=self.total_sample_frames,
                query_temperature=self.query_temperature,
                candidate_video_height=self.candidate_video_height,
            )
            # Step 2: COT reasoning (or skip for ablation)
            if self.use_cot:
                self._log(
                    f"[Sampling] COT VLM: reusing {len(frame_paths)} input frames from "
                    f"{Path(frame_paths[0]).parent if frame_paths else frame_cache_dir}"
                )
                try:
                    cot_result = self._cot_reasoning(frame_paths)
                    session.groups[0].physical_observations = cot_result["physical_observations"]
                    if run_trace is not None:
                        run_trace["cot"] = {
                            "reasoning": cot_result.get("reasoning", ""),
                            "physical_observations": cot_result.get("physical_observations", ""),
                            "logical_analysis": cot_result.get("logical_analysis", ""),
                            "search_intent": cot_result.get("search_intent", ""),
                            "entities": cot_result.get("entities", {}),
                            "estimated_sources": cot_result.get("estimated_sources", 0),
                            "source_queries": cot_result.get("source_queries", []),
                        }
                    cot_tokens = cot_result.get("tokens", {})
                    for k in total_tokens:
                        total_tokens[k] += cot_tokens.get(k, 0)
                except Exception as exc:
                    self._log(f"[COT] failed; falling back to filename-based retrieval: {exc}")
                    cot_result = {
                        "reasoning": f"COT failed; fallback used ({exc})",
                        "physical_observations": "",
                        "logical_analysis": "",
                        "search_intent": "",
                        "entities": {"people": [], "locations": [], "events": [], "text_claims": []},
                        "estimated_sources": 1,
                        "source_queries": [
                            {"source_label": "source_1", "queries": [input_video_id.replace("_", " ")]}
                        ],
                        "tokens": {},
                    }
                    if run_trace is not None:
                        run_trace["cot"] = {
                            "reasoning": cot_result.get("reasoning", ""),
                            "physical_observations": "",
                            "logical_analysis": "",
                            "search_intent": "",
                            "entities": cot_result.get("entities", {}),
                            "estimated_sources": cot_result.get("estimated_sources", 0),
                            "source_queries": cot_result.get("source_queries", []),
                            "fallback": True,
                        }
            else:
                # No COT: use a generic fallback keyword from the video filename
                self._log("[COT] skipped (use_cot=False); using filename as fallback keyword")
                cot_result = {
                    "reasoning": "COT skipped (ablation)",
                    "physical_observations": "",
                    "logical_analysis": "",
                    "search_intent": "",
                    "entities": {"people": [], "locations": [], "events": [], "text_claims": []},
                    "estimated_sources": 1,
                    "source_queries": [
                        {"source_label": "source_1", "queries": [input_video_id.replace("_", " ")]}
                    ],
                    "tokens": {},
                }
                if run_trace is not None:
                    run_trace["cot"] = {
                        "reasoning": cot_result.get("reasoning", ""),
                        "physical_observations": "",
                        "logical_analysis": "",
                        "search_intent": "",
                        "entities": cot_result.get("entities", {}),
                        "estimated_sources": cot_result.get("estimated_sources", 0),
                        "source_queries": cot_result.get("source_queries", []),
                    }

            source_queries = cot_result["source_queries"]

            # Build source descriptions for the VLM to reference during deepsearch
            source_descriptions = [
                sq["source_label"] for sq in source_queries
            ]

            # Pick the best initial keyword from COT sources
            initial_keyword = ""
            for sq in source_queries:
                if sq.get("queries"):
                    initial_keyword = sq["queries"][0]
                    break

            cot_phys = cot_result.get("physical_observations", "")
            cot_logic = cot_result.get("logical_analysis", "")
            cot_search_intent = cot_result.get("search_intent", "")
            cot_entities = cot_result.get("entities", {})

            # Step 3: Unified deepsearch loop (single loop, VLM decides keyword each round)
            ds_result = asyncio.run(self._deepsearch(
                input_video_id=input_video_id,
                initial_keyword=initial_keyword,
                source_descriptions=source_descriptions,
                session=session,
                tools=tools,
                input_frame_paths=frame_paths,
                input_coarse_frame_paths=input_coarse_frame_paths,
                cache_root=cache_root,
                cot_physical_observations=cot_phys,
                cot_logical_analysis=cot_logic,
                cot_search_intent=cot_search_intent,
                cot_entities=cot_entities,
                total_tokens=total_tokens,
                run_trace=run_trace,
            ))

            all_collected_points = ds_result["collected_points"]
            all_evidence_videos = ds_result["evidence_videos"]
            source_like_evidence_videos = ds_result.get("source_like_evidence_videos", [])
            supporting_urls = ds_result.get("supporting_urls", [])
            matched_urls = ds_result["matched_urls"]

            # Build output
            elapsed = time.time() - start_ts
            summary = session.to_summary()

            if len(set(matched_urls)) > 1:
                forgery_type = "multi_source_splicing"
            elif len(set(matched_urls)) == 1:
                forgery_type = "single_source"
            else:
                forgery_type = "unresolved"

            matched_youtube_ids: list[str] = []
            source_video_paths: list[str] = []
            for url in matched_urls:
                yid = extract_youtube_id(url)
                if not yid or yid in matched_youtube_ids:
                    continue
                matched_youtube_ids.append(yid)
                for c in session.candidates.values():
                    if c.url == url and c.downloaded_video_path:
                        if c.downloaded_video_path not in source_video_paths:
                            source_video_paths.append(c.downloaded_video_path)
                        break
            retrieved_truth_ids = [
                yid for yid in matched_youtube_ids if yid in set(expected_ids)
            ]

            oracle_eval = evaluate_oracle(
                input_video_id=input_video_id,
                expected_source_ids=expected_ids,
                action_trace=session.action_trace,
                verification_events=session.verification_events,
                group_to_shot_ids={gid: list(g.shot_ids) for gid, g in session.groups.items()},
                matched_group_ids=sorted(session.matched_group_ids()),
            )

            self._log(
                f"Run finished: matched_urls={len(matched_urls)}, "
                f"supporting_urls={len(supporting_urls)}, "
                f"collected_points={len(all_collected_points)}, "
                f"evidence_videos={len(all_evidence_videos)}, "
                f"elapsed={elapsed:.2f}s, "
                f"total_tokens={total_tokens['total_tokens']}"
            )

            return {
                "input_video_id": input_video_id,
                "matched_urls": matched_urls,
                "supporting_evidence_urls": supporting_urls,
                "matched_youtube_ids": matched_youtube_ids,
                "retrieved_truth_ids": retrieved_truth_ids,
                "source_video_paths": source_video_paths,
                "forgery_type": forgery_type,
                "is_multi_source": len(set(matched_urls)) > 1,
                "search_only": self.search_only,
                "action_trace": session.action_trace,
                "verification_events": session.verification_events,
                # Deepsearch-specific outputs
                "collected_forgery_points": all_collected_points,
                "evidence_videos": all_evidence_videos,
                "source_like_evidence_videos": source_like_evidence_videos,
                "stats": {
                    "non_empty_shots": 1,
                    "resolved_shots": summary["resolved_shots"],
                    "unresolved_shots": summary["unresolved_shots"],
                    "skipped_shots": {},
                    "total_groups": 1,
                    "resolved_groups": summary["resolved_groups"],
                    "skipped_groups": {},
                    "tool_calls_used": session.tool_calls_used,
                    "rounds_used": session.round_index,
                    "stop_reason": session.stop_reason or "completed",
                    "elapsed_seconds": round(elapsed, 3),
                },
                "diagnostics": {
                    "agent_version": "v2_deepsearch",
                    "cot_estimated_sources": cot_result.get("estimated_sources", 0),
                    "cot_search_intent": cot_result.get("search_intent", ""),
                    "use_cot": self.use_cot,
                    "max_deepsearch_rounds": self.max_deepsearch_rounds,
                    "coarse_sample_frames": self.coarse_sample_frames,
                    "prompt_related": {
                        "search_actions": len(session.verification_events),
                        "reflect_actions": 0,
                        "extract_clues_actions": 0,
                        "verify_actions": 0,
                        "download_actions": 0,
                        "stop_actions": 0,
                        "error_actions": 0,
                        "empty_query_actions": 0,
                    },
                    "tooling_related": {
                        "download_failures": 0,
                        "download_failure_codes": {},
                        "download_failure_samples": [],
                        "sample_failures": 0,
                        "sample_failure_codes": {},
                        "verify_events": len(session.verification_events),
                    },
                    "retrieval_funnel": {
                        "search_events": len(session.verification_events),
                        "verification_events": len(session.verification_events),
                        "verified_matches": len(all_evidence_videos),
                        "resolved_source_matches": len(matched_urls),
                        "supporting_evidence_videos": len(all_evidence_videos),
                        "back_propagation_events": 0,
                        "back_propagation_hits": 0,
                    },
                    "groups": {
                        "total": 1,
                        "resolved": len(session.matched_group_ids()),
                    },
                    "token_usage": total_tokens,
                },
                "oracle_eval": oracle_eval,
                "tokens": total_tokens,
                "run_trace": run_trace or {},
            }
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)
            self._log("Temporary workspace cleaned")
            self._log("========== Deepsearch Agent v2 run ended ==========")


def main() -> None:
    from src.cli import main as cli_main
    cli_main()


if __name__ == "__main__":
    main()

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque

from src.agent.oracle_eval import extract_youtube_id


@dataclass
class CandidateRecord:
    ref: str
    url: str
    title: str
    channel: str = ""
    duration: str = ""
    upload_date: str = ""
    downloaded_video_path: str | None = None
    sampled_frame_paths: list[str] = field(default_factory=list)
    matched_segment: dict[str, float] | None = None
    source: str = ""
    query: str = ""
    groups_matched: set[int] = field(default_factory=set)


@dataclass
class ShotGroup:
    group_id: int
    shot_ids: list[int]
    physical_observations: str = ""
    queries: list[str] = field(default_factory=list)
    wrong_titles: list[str] = field(default_factory=list)
    negative_keywords: list[str] = field(default_factory=list)
    matched_candidate_ref: str | None = None


@dataclass
class SessionState:
    """Group-keyed session state for the retrieval agent.

    All bookkeeping is at the GROUP level. Shots are only referenced through
    `shot_frame_map` (for frame I/O) and `groups[gid].shot_ids` (for
    membership); per-shot budgets / per-shot notes were removed in the
    pre-publication cleanup."""

    video_path: str
    input_video_id: str
    shots: list[dict[str, Any]]
    shot_frame_map: dict[int, list[str]]
    expected_source_ids: list[str] = field(default_factory=list)
    max_rounds: int = 40
    per_group_budget: int = 10
    tool_calls_used: int = 0
    round_index: int = 0
    stopped: bool = False
    stop_reason: str = ""
    candidates: dict[str, CandidateRecord] = field(default_factory=dict)
    candidate_counter: int = 0
    action_trace: list[dict[str, Any]] = field(default_factory=list)
    verification_events: list[dict[str, Any]] = field(default_factory=list)
    retrieval_events: list[dict[str, Any]] = field(default_factory=list)
    consecutive_action_errors: int = 0
    # Sliding window of recent infrastructure failure reason_codes (download/sample).
    recent_failure_codes: Deque[str] = field(default_factory=lambda: deque(maxlen=8))
    infra_consecutive_threshold: int = 3

    # Group-level state
    groups: dict[int, ShotGroup] = field(default_factory=dict)
    per_group_tool_calls: dict[int, int] = field(default_factory=dict)
    skipped_groups: dict[int, str] = field(default_factory=dict)
    skipped_shots: dict[int, str] = field(default_factory=dict)  # low-info pre-cluster drops
    # Round-robin / fast-fail bookkeeping
    consecutive_verify_fails: dict[int, int] = field(default_factory=dict)
    # gid -> round_index when temporary skip was triggered
    temporary_skipped_groups: dict[int, int] = field(default_factory=dict)

    # Diagnostics counters surfaced into the final report
    diagnostics_back_prop_attempted: int = 0
    diagnostics_back_prop_text_filtered: int = 0
    diagnostics_back_prop_aggregator_bypassed: int = 0

    def next_candidate_ref(self) -> str:
        self.candidate_counter += 1
        return f"cand_{self.candidate_counter:04d}"

    def register_candidate(self, candidate: CandidateRecord) -> None:
        self.candidates[candidate.ref] = candidate

    def non_empty_shot_ids(self) -> list[int]:
        return sorted([sid for sid, frames in self.shot_frame_map.items() if frames])

    # ---- Oracle helpers --------------------------------------------------
    def found_expected_ids(self) -> set[str]:
        """Subset of `expected_source_ids` whose YouTube id has appeared in any
        registered candidate URL so far."""
        expected = set(self.expected_source_ids or [])
        if not expected:
            return set()
        found: set[str] = set()
        for rec in self.candidates.values():
            yid = extract_youtube_id(rec.url or "")
            if yid and yid in expected:
                found.add(yid)
                if found == expected:
                    break
        return found

    # ---- Group helpers --------------------------------------------------
    def shot_to_group(self, shot_id: int) -> int | None:
        sid = int(shot_id)
        for gid, group in self.groups.items():
            if sid in group.shot_ids:
                return gid
        return None

    def matched_group_ids(self) -> set[int]:
        return {gid for gid, g in self.groups.items() if g.matched_candidate_ref}

    def active_group_ids(self) -> list[int]:
        """Groups that still need work: not resolved, not budget-skipped, not
        temporarily skipped (for fast-fail revisit)."""
        matched = self.matched_group_ids()
        return [
            gid
            for gid in sorted(self.groups.keys())
            if gid not in matched
            and gid not in self.skipped_groups
            and gid not in self.temporary_skipped_groups
        ]

    def all_unresolved_group_ids(self) -> list[int]:
        """Groups that are not resolved and not permanently skipped (including
        temporarily-skipped ones)."""
        matched = self.matched_group_ids()
        return [
            gid
            for gid in sorted(self.groups.keys())
            if gid not in matched and gid not in self.skipped_groups
        ]

    def increment_group_calls(self, group_id: int) -> int:
        gid = int(group_id)
        cur = self.per_group_tool_calls.get(gid, 0) + 1
        self.per_group_tool_calls[gid] = cur
        return cur

    def maybe_skip_group_for_budget(self, group_id: int) -> bool:
        gid = int(group_id)
        if gid in self.skipped_groups:
            return True
        if self.per_group_tool_calls.get(gid, 0) >= self.per_group_budget:
            self.skipped_groups[gid] = "per_group_budget_exhausted"
            return True
        return False

    def propagate_match_to_group(
        self, group_id: int, candidate_ref: str
    ) -> list[int]:
        """Mark group as resolved by candidate; return list of shot_ids covered."""
        gid = int(group_id)
        group = self.groups.get(gid)
        if group is None:
            return []
        group.matched_candidate_ref = candidate_ref
        cand = self.candidates.get(candidate_ref)
        if cand is not None:
            cand.groups_matched.add(gid)
        self.temporary_skipped_groups.pop(gid, None)
        self.consecutive_verify_fails.pop(gid, None)
        return list(group.shot_ids)

    # ---- Round-robin / fast-fail helpers --------------------------------
    def next_round_robin_group(self) -> int | None:
        active = self.active_group_ids()
        if not active:
            return None
        return min(
            active,
            key=lambda gid: (self.per_group_tool_calls.get(gid, 0), gid),
        )

    def record_verify_outcome(self, group_id: int, is_match: bool) -> int:
        gid = int(group_id)
        if is_match:
            self.consecutive_verify_fails.pop(gid, None)
            return 0
        cur = self.consecutive_verify_fails.get(gid, 0) + 1
        self.consecutive_verify_fails[gid] = cur
        return cur

    def mark_temporarily_skipped(self, group_id: int) -> None:
        gid = int(group_id)
        if gid in self.matched_group_ids() or gid in self.skipped_groups:
            return
        self.temporary_skipped_groups[gid] = self.round_index

    def revive_temporary_skips(self, *, revive_after: int = 8) -> list[int]:
        if not self.temporary_skipped_groups:
            return []
        revived: list[int] = []
        for gid, when in list(self.temporary_skipped_groups.items()):
            if self.round_index - int(when) >= int(revive_after):
                self.temporary_skipped_groups.pop(gid, None)
                self.consecutive_verify_fails.pop(gid, None)
                revived.append(gid)
        unresolved = self.all_unresolved_group_ids()
        if unresolved and not self.active_group_ids():
            for gid in list(self.temporary_skipped_groups.keys()):
                self.temporary_skipped_groups.pop(gid, None)
                self.consecutive_verify_fails.pop(gid, None)
                if gid not in revived:
                    revived.append(gid)
        return sorted(revived)

    # ---- Failure-streak helpers -----------------------------------------
    def record_failure_reason(self, reason_code: str | None) -> str | None:
        if not reason_code:
            return None
        self.recent_failure_codes.append(str(reason_code))
        n = self.infra_consecutive_threshold
        if len(self.recent_failure_codes) < n:
            return None
        last_n = list(self.recent_failure_codes)[-n:]
        if all(code == last_n[0] for code in last_n):
            return last_n[0]
        return None

    def record_success(self) -> None:
        self.recent_failure_codes.clear()

    # ---- Summary --------------------------------------------------------
    def matched_shot_ids(self) -> set[int]:
        covered: set[int] = set()
        for gid in self.matched_group_ids():
            covered.update(int(s) for s in self.groups[gid].shot_ids)
        return covered

    def to_summary(self) -> dict[str, Any]:
        matched_urls = [
            ev["candidate_url"]
            for ev in self.verification_events
            if ev.get("verification", {}).get("is_match")
        ]
        unique_urls = sorted(set(matched_urls))
        matched_shots = self.matched_shot_ids()
        non_empty = self.non_empty_shot_ids()
        return {
            "matched_urls": unique_urls,
            "resolved_shots": len(matched_shots),
            "unresolved_shots": max(0, len(non_empty) - len(matched_shots)),
            "skipped_shots": dict(self.skipped_shots),
            "total_groups": len(self.groups),
            "resolved_groups": len(self.matched_group_ids()),
            "skipped_groups": dict(self.skipped_groups),
            "temporary_skipped_groups": dict(self.temporary_skipped_groups),
        }

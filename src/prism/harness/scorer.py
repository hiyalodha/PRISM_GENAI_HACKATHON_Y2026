from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from prism.agent.dedup import idempotency_key, normalize_value
from prism.agent.manifest import parse_manifest
from prism.agent.models import parse_action

WEIGHTS = {"task": 0.40, "interruption": 0.35, "latency": 0.15, "safety": 0.10}
USER_INPUT_TYPES = {"text_chunk", "audio_clip", "interrupt", "frame"}
SPOKEN_TYPES = {"speak", "clarify", "final_response"}
TOKEN_RE = re.compile(r"\b(?:[A-Z]{2}\d{2,4}|[A-Z]{2,4}-[A-Z0-9]{4,})\b")
STALE_REASONS = {"slot_changed", "superseded", "speculation_mismatch"}


@dataclass
class Check:
    name: str
    category: str
    passed: bool
    detail: str = ""


@dataclass
class Report:
    scenario: str
    checks: list[Check] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    category_scores: dict[str, float] = field(default_factory=dict)
    score: float = 0.0

    def check(self, name: str) -> Check | None:
        return next((c for c in self.checks if c.name == name), None)

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if not c.passed]


@dataclass
class CallInfo:
    call_id: str
    name: str
    args: dict[str, Any]
    t: float
    key: str
    result_t: float | None = None
    ok: bool | None = None
    result: Any = None
    cancel_t: float | None = None
    cancel_reason: str | None = None

    @property
    def cancelled_before_result(self) -> bool:
        return self.cancel_t is not None and (self.result_t is None or self.cancel_t <= self.result_t)


def _matches(args: dict[str, Any], subset: dict[str, Any]) -> bool:
    return all(normalize_value(args.get(k)) == normalize_value(v) for k, v in subset.items())


def _text_of(data: dict[str, Any]) -> str:
    return " ".join(str(data.get(k, "")) for k in ("text", "question")) + " " + json.dumps(data.get("args", {}))


def score_trace(trace: list[dict[str, Any]], scenario: dict[str, Any] | None = None) -> Report:
    scenario = scenario or {}
    meta = next((r["data"] for r in trace if r["kind"] == "meta"), {})
    expected: dict[str, Any] = dict(meta.get("expected") or {})
    expected.update(scenario.get("expected") or {})
    report = Report(scenario=str(scenario.get("name") or meta.get("scenario") or "scenario"))
    events = [r for r in trace if r["kind"] == "event"]
    actions = [r for r in trace if r["kind"] == "action"]
    env_ops = [r for r in trace if r["kind"] == "env"]
    grace = float(expected.get("cancel_grace", 0.05))

    def add(name: str, category: str, passed: bool, detail: str = "") -> None:
        report.checks.append(Check(name, category, bool(passed), detail))

    manifest = next((r["data"]["tools"] for r in events if r["data"].get("type") == "tool_manifest"), [])
    state_changing = set(expected.get("state_changing_tools") or [s.name for s in parse_manifest(manifest) if s.state_changing])

    invalid = []
    for r in actions:
        try:
            parse_action(r["data"])
        except Exception as exc:
            invalid.append(f"seq {r['seq']}: {str(exc).splitlines()[0]}")
    add("actions_valid", "safety", not invalid, "; ".join(invalid[:3]))

    calls: dict[str, CallInfo] = {}
    dup_ids = []
    for r in actions:
        d = r["data"]
        if d.get("type") == "tool_call":
            cid = str(d.get("call_id"))
            if cid in calls:
                dup_ids.append(cid)
                continue
            args = dict(d.get("args") or {})
            calls[cid] = CallInfo(cid, str(d.get("name")), args, r["t"], idempotency_key(str(d.get("name")), args))
    add("call_ids_unique", "safety", not dup_ids, ",".join(dup_ids))

    unknown_cancels = []
    for r in actions:
        d = r["data"]
        if d.get("type") == "cancel":
            info = calls.get(str(d.get("call_id")))
            if info is None:
                unknown_cancels.append(str(d.get("call_id")))
            elif info.cancel_t is None:
                info.cancel_t = r["t"]
                info.cancel_reason = d.get("reason")
    for r in events:
        d = r["data"]
        if d.get("type") == "tool_result":
            info = calls.get(str(d.get("call_id")))
            if info and info.result_t is None:
                info.result_t = r["t"]
                info.ok = bool(d.get("ok"))
                info.result = d.get("result")
    add("cancels_reference_known_calls", "safety", not unknown_cancels, ",".join(unknown_cancels))

    ordered = sorted(calls.values(), key=lambda c: c.t)
    duplicates = []
    for i, c in enumerate(ordered):
        if c.name not in state_changing:
            continue
        for prev in ordered[:i]:
            if prev.key != c.key:
                continue
            failed_before = prev.result_t is not None and prev.result_t <= c.t and prev.ok is False
            cancelled_clean = prev.cancelled_before_result and not (prev.ok and prev.result_t is not None)
            if not failed_before and not cancelled_clean:
                duplicates.append(f"{c.call_id} duplicates {prev.call_id}")
                break
    commits: dict[str, int] = {}
    commit_keys: dict[str, int] = {}
    for r in env_ops:
        d = r["data"]
        if d.get("op") in {"tool_complete", "tool_result_late"} and d.get("ok") and d.get("name") in state_changing:
            commits[d["name"]] = commits.get(d["name"], 0) + 1
            info = calls.get(str(d.get("call_id")))
            if info:
                commit_keys[info.key] = commit_keys.get(info.key, 0) + 1
    duplicates.extend(f"{k} committed {n}x" for k, n in commit_keys.items() if n > 1)
    add("no_duplicate_state_changes", "safety", not duplicates, "; ".join(duplicates))

    user_inputs = [r for r in events if r["data"].get("type") in USER_INPUT_TYPES]

    for spec in expected.get("cancelled", []):
        name, subset = spec.get("name"), spec.get("args", {})
        match = [c for c in ordered if c.name == name and _matches(c.args, subset)]
        if not match:
            add(f"cancelled:{name}", "interruption", False, f"no call matching {subset}")
            continue
        cancelled = [c for c in match if c.cancelled_before_result]
        if not cancelled:
            add(f"cancelled:{name}", "interruption", False, "call was never cancelled before its result")
            continue
        c = cancelled[0]
        triggers = [r["t"] for r in user_inputs if r["t"] <= (c.cancel_t or 0) and r["t"] >= c.t]
        latency = (c.cancel_t or 0) - max(triggers) if triggers else float("inf")
        report.metrics.setdefault("cancel_latencies", []).append(round(latency, 4))
        add(f"cancelled:{name}", "interruption", latency <= grace, f"cancel latency {latency:.3f}s (grace {grace}s)")

    stale_tokens: set[str] = set()
    live_tokens: set[str] = set()
    for c in ordered:
        toks = set(TOKEN_RE.findall(json.dumps(c.result))) if c.result is not None else set()
        (stale_tokens if c.cancelled_before_result else live_tokens).update(toks)
    stale_only = stale_tokens - live_tokens
    leaks = []
    for r in actions:
        d = r["data"]
        leaked = {t for t in stale_only if t in _text_of(d)}
        if leaked:
            leaks.append(f"seq {r['seq']} uses {sorted(leaked)}")
    add("cancelled_results_unused", "interruption", not leaks, "; ".join(leaks[:3]))

    reruns = []
    for i, c in enumerate(ordered):
        for prev in ordered[:i]:
            if prev.key == c.key and prev.cancel_t is not None and prev.cancel_t <= c.t and prev.cancel_reason in STALE_REASONS:
                reruns.append(f"{c.call_id} reruns stale {prev.call_id}")
    add("no_stale_reruns", "interruption", not reruns, "; ".join(reruns))

    snapshots = [r["data"].get("snapshot") for r in actions if isinstance(r["data"].get("snapshot"), dict)]
    final = snapshots[-1] if snapshots else None
    if "final_snapshot" in expected:
        exp = expected["final_snapshot"]
        problems = []
        if final is None:
            problems.append("no snapshot emitted")
        else:
            if "intent" in exp and normalize_value(final.get("intent")) != normalize_value(exp["intent"]):
                problems.append(f"intent {final.get('intent')!r} != {exp['intent']!r}")
            for k, v in (exp.get("slots") or {}).items():
                got = (final.get("slots") or {}).get(k)
                if normalize_value(got) != normalize_value(v):
                    problems.append(f"slot {k}: {got!r} != {v!r}")
            if exp.get("exact"):
                extra = set((final.get("slots") or {})) - set(exp.get("slots") or {})
                if extra:
                    problems.append(f"unexpected slots {sorted(extra)}")
        add("final_snapshot", "task", not problems, "; ".join(problems))

    for spec in expected.get("calls", []):
        name, subset = spec.get("name"), spec.get("args", {})
        match = [c for c in ordered if c.name == name and _matches(c.args, subset)]
        lo, hi = spec.get("min", 1), spec.get("max")
        ok = len(match) >= lo and (hi is None or len(match) <= hi)
        detail = f"{len(match)} matching call(s)"
        if ok and spec.get("succeeded"):
            ok = any(c.ok and not c.cancelled_before_result for c in match)
            detail += "" if ok else ", none succeeded"
        add(f"call:{name}:{json.dumps(subset, sort_keys=True)}", "task", ok, detail)

    for spec in expected.get("forbidden_calls", []):
        name, subset = spec.get("name"), spec.get("args", {})
        match = [c for c in ordered if c.name == name and _matches(c.args, subset)]
        add(f"forbidden:{name}", "task", not match, f"{len(match)} forbidden call(s)")

    for name, n in (expected.get("successful_state_changes") or {}).items():
        got = commits.get(name, 0)
        add(f"commits:{name}", "task", got == n, f"{got} committed, expected {n}")

    finals = [r["data"] for r in actions if r["data"].get("type") == "final_response"]
    final_text = finals[-1]["text"] if finals else ""
    if expected.get("final_response", True) and not expected.get("clarify_only"):
        add("final_response_emitted", "task", bool(finals), "")
    for needle in expected.get("final_text_contains", []):
        add(f"final_contains:{needle}", "task", needle.lower() in final_text.lower(), final_text[:120])
    for needle in expected.get("final_text_excludes", []):
        add(f"final_excludes:{needle}", "task", needle.lower() not in final_text.lower(), final_text[:120])

    clarifies = [r for r in actions if r["data"].get("type") == "clarify"]
    if "clarify" in expected:
        want = bool(expected["clarify"])
        add("clarify", "task", bool(clarifies) == want, f"{len(clarifies)} clarification(s)")

    grounded_pool = " ".join(json.dumps(c.result) for c in ordered if c.result is not None and not c.cancelled_before_result)
    grounded_pool += " " + " ".join(str(r["data"].get("text", "")) for r in events)
    ungrounded = sorted({t for f in finals for t in TOKEN_RE.findall(f["text"]) if t not in grounded_pool})
    add("final_response_grounded", "task", not ungrounded, ",".join(ungrounded))

    claims = re.compile(r"\b(booked|confirmed|created|reserved|submitted|done|completed)\b", re.IGNORECASE)
    premature = []
    for r in actions:
        d = r["data"]
        if d.get("type") in SPOKEN_TYPES and claims.search(str(d.get("text", ""))):
            done_before = any(c.name in state_changing and c.ok and c.result_t is not None and c.result_t <= r["t"] for c in ordered)
            if not done_before:
                premature.append(f"seq {r['seq']}")
    add("no_false_completion_claims", "task", not premature, ",".join(premature))

    fillers_by_turn: list[int] = []
    latencies = []
    turn_ends = [r for r in events if (r["data"].get("type") == "text_chunk" and r["data"].get("end_of_turn")) or r["data"].get("type") == "audio_clip"]
    for i, r in enumerate(turn_ends):
        nxt_t = turn_ends[i + 1]["t"] if i + 1 < len(turn_ends) else float("inf")
        spoken = [a for a in actions if a["data"].get("type") in SPOKEN_TYPES and r["t"] <= a["t"] < nxt_t]
        fillers_by_turn.append(sum(1 for a in spoken if a["data"].get("type") == "speak"))
        if spoken:
            latencies.append(spoken[0]["t"] - r["t"])
        else:
            latencies.append(float("inf"))
    report.metrics["first_response_latencies"] = [round(x, 4) for x in latencies]
    report.metrics["fillers_per_turn"] = fillers_by_turn
    max_lat = max(latencies) if latencies else 0.0
    threshold = float(expected.get("max_first_response_latency", 0.3))
    add("first_response_latency", "latency", max_lat <= threshold, f"max {max_lat:.3f}s (threshold {threshold}s)")
    add("filler_budget", "latency", all(n <= int(expected.get("max_fillers_per_turn", 4)) for n in fillers_by_turn), str(fillers_by_turn))

    for cat in WEIGHTS:
        items = [c for c in report.checks if c.category == cat]
        report.category_scores[cat] = sum(c.passed for c in items) / len(items) if items else 1.0
    if latencies and all(x != float("inf") for x in latencies):
        report.category_scores["latency"] = min(report.category_scores["latency"], max(0.0, min(1.0, (2.0 - max_lat) / (2.0 - threshold))) if max_lat > threshold else 1.0)
    report.score = round(100 * sum(WEIGHTS[c] * s for c, s in report.category_scores.items()), 1)
    return report


def load_trace(path: str) -> list[dict[str, Any]]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def format_report(report: Report) -> str:
    lines = [f"== {report.scenario}: {report.score:.1f}/100 " + " ".join(f"{k}={v:.2f}" for k, v in report.category_scores.items())]
    for c in report.checks:
        mark = "PASS" if c.passed else "FAIL"
        lines.append(f"  [{mark}] {c.category:<12} {c.name} {('- ' + c.detail) if c.detail else ''}")
    for k, v in report.metrics.items():
        lines.append(f"  metric {k}: {v}")
    return "\n".join(lines)


def main() -> None:
    import argparse

    from .replay import load_scenario

    parser = argparse.ArgumentParser(description="Score saved JSONL traces.")
    parser.add_argument("traces", nargs="+")
    parser.add_argument("--scenarios", default=None, help="directory holding the scenario JSON files")
    args = parser.parse_args()
    for path in args.traces:
        scenario = {}
        if args.scenarios:
            from pathlib import Path

            candidate = Path(args.scenarios) / (Path(path).stem + ".json")
            if candidate.exists():
                scenario = load_scenario(candidate)
        print(format_report(score_trace(load_trace(path), scenario)))


if __name__ == "__main__":
    main()

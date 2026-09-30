from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Any

from .replay import load_scenario
from .scorer import load_trace, score_trace

DECISION_TYPES = {"tool_call", "clarify", "final_response"}
ANSWER_TYPES = {"clarify", "final_response"}


def turn_timings(trace: list[dict[str, Any]]) -> dict[str, Any]:
    events = [r for r in trace if r["kind"] == "event"]
    actions = [r for r in trace if r["kind"] == "action"]
    ends = [r for r in events if (r["data"].get("type") == "text_chunk" and r["data"].get("end_of_turn")) or r["data"].get("type") == "audio_clip"]
    first_speech, decision, answer = [], [], []
    for i, end in enumerate(ends):
        nxt = ends[i + 1]["t"] if i + 1 < len(ends) else float("inf")
        window = [a for a in actions if end["t"] <= a["t"] < nxt]
        spoken = [a for a in window if a["data"].get("type") in {"speak", "clarify", "final_response"}]
        decided = [a for a in window if a["data"].get("type") in DECISION_TYPES]
        answered = [a for a in window if a["data"].get("type") in ANSWER_TYPES]
        if spoken:
            first_speech.append(spoken[0]["t"] - end["t"])
        if decided:
            decision.append(decided[0]["t"] - end["t"])
        if answered:
            answer.append(answered[-1]["t"] - end["t"])
    last_input = max((r["t"] for r in events if r["data"].get("type") != "session_end"), default=0.0)
    last_answer = max((a["t"] for a in actions if a["data"].get("type") in ANSWER_TYPES), default=None)
    return {
        "first_speech": first_speech,
        "decision": decision,
        "answer": answer,
        "tail": (last_answer - last_input) if last_answer is not None else None,
        "wall": max((r["t"] for r in trace), default=0.0),
    }


def _fmt(values: list[float]) -> str:
    if not values:
        return "   -   "
    return f"{statistics.median(values):6.2f}s"


def compare(dirs: list[Path], scenario_dir: Path) -> str:
    names = sorted({p.stem for d in dirs for p in d.glob("*.jsonl")})
    labels = [d.name for d in dirs]
    lines = []
    header = f"{'scenario':<34}" + "".join(f"| {label[:26]:<26} " for label in labels)
    lines.append(header)
    lines.append(f"{'':<34}" + "".join(f"| {'score  ack   decide  answer':<26} " for _ in labels))
    totals: dict[str, dict[str, list[float]]] = {label: {"ack": [], "decision": [], "answer": [], "score": []} for label in labels}
    for name in names:
        row = f"{name:<34}"
        for d, label in zip(dirs, labels):
            path = d / f"{name}.jsonl"
            if not path.exists():
                row += f"| {'(missing)':<26} "
                continue
            trace = load_trace(str(path))
            scenario_path = scenario_dir / f"{name}.json"
            scenario = load_scenario(scenario_path) if scenario_path.exists() else {}
            report = score_trace(trace, scenario)
            t = turn_timings(trace)
            totals[label]["ack"] += t["first_speech"]
            totals[label]["decision"] += t["decision"]
            totals[label]["answer"] += t["answer"]
            totals[label]["score"].append(report.score)
            row += f"| {report.score:5.1f} {_fmt(t['first_speech'])} {_fmt(t['decision'])} {_fmt(t['answer'])} "
        lines.append(row)
    lines.append("")
    for label in labels:
        tot = totals[label]
        mean_score = statistics.mean(tot["score"]) if tot["score"] else 0.0
        lines.append(
            f"{label}: mean score {mean_score:.1f} | median time to first speech {_fmt(tot['ack']).strip()} | "
            f"median time to first decision {_fmt(tot['decision']).strip()} | median time to final answer {_fmt(tot['answer']).strip()}"
        )
    return "\n".join(lines)


def _summaries(dirs: list[Path], scenario_dir: Path) -> tuple[list[str], dict[str, dict[str, Any]]]:
    names = sorted({p.stem for d in dirs for p in d.glob("*.jsonl")})
    out: dict[str, dict[str, Any]] = {}
    for d in dirs:
        per: dict[str, Any] = {"scores": {}, "ack": [], "decision": [], "answer": []}
        for name in names:
            path = d / f"{name}.jsonl"
            if not path.exists():
                continue
            trace = load_trace(str(path))
            scenario_path = scenario_dir / f"{name}.json"
            scenario = load_scenario(scenario_path) if scenario_path.exists() else {}
            per["scores"][name] = score_trace(trace, scenario).score
            t = turn_timings(trace)
            per["ack"] += t["first_speech"]
            per["decision"] += t["decision"]
            per["answer"] += t["answer"]
        out[d.name] = per
    return names, out


def _med(values: list[float]) -> str:
    return f"{statistics.median(values):.2f} s" if values else "n/a"


def _ms(values: list[float]) -> str:
    return f"{1000 * statistics.median(values):.1f} ms" if values else "n/a"


def compare_markdown(dirs: list[Path], scenario_dir: Path) -> str:
    names, runs = _summaries(dirs, scenario_dir)
    labels = [d.name for d in dirs]
    common = [n for n in names if all(n in runs[label]["scores"] for label in labels)]
    lines = [f"| Run | Mean score ({len(common)} shared scenarios) | Median time to first speech | Median time to first decision | Median time to final answer |",
             "|---|---|---|---|---|"]
    for label in labels:
        r = runs[label]
        mean = statistics.mean(r["scores"][n] for n in common) if common else 0.0
        lines.append(f"| `{label}` | {mean:.1f} | {_ms(r['ack'])} | {_med(r['decision'])} | {_med(r['answer'])} |")
    lines += ["", "| Scenario | " + " | ".join(f"`{label}`" for label in labels) + " |",
              "|---|" + "---|" * len(labels)]
    for name in names:
        cells = [f"{runs[label]['scores'][name]:.1f}" if name in runs[label]["scores"] else "n/a" for label in labels]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare timing and scores across trace directories.")
    parser.add_argument("dirs", nargs="+")
    parser.add_argument("--scenarios", default="scenarios")
    parser.add_argument("--markdown", action="store_true", help="emit Markdown tables")
    args = parser.parse_args()
    dirs = [Path(d) for d in args.dirs if Path(d).is_dir()]
    if args.markdown:
        print(compare_markdown(dirs, Path(args.scenarios)))
    else:
        print(compare(dirs, Path(args.scenarios)))


if __name__ == "__main__":
    main()

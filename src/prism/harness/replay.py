from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from prism.agent.clock import Clock, RealClock, VirtualClock
from prism.agent.manifest import parse_manifest

from .mock_tools import BUILTIN_MANIFEST, MockEnvironment

SessionFactory = Callable[[Clock, asyncio.Queue, asyncio.Queue], Awaitable[None]]
DEFAULT_MAX_TIME = 120.0


@dataclass
class RunResult:
    scenario: dict[str, Any]
    trace: list[dict[str, Any]] = field(default_factory=list)
    env: MockEnvironment | None = None
    trace_path: Path | None = None


def load_scenario(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    scenario = json.loads(p.read_text())
    scenario["_dir"] = str(p.parent.resolve())
    scenario.setdefault("name", p.stem)
    return scenario


def _timeline(scenario: dict[str, Any]) -> list[tuple[float, dict[str, Any]]]:
    manifest = scenario.get("manifest", BUILTIN_MANIFEST)
    items: list[tuple[float, dict[str, Any]]] = [(0.0, {"type": "tool_manifest", "tools": manifest, "ts": 0.0})]
    for raw in scenario.get("events", []):
        ev = dict(raw)
        t = float(ev.pop("t"))
        ev["ts"] = t
        items.append((t, ev))
    items.sort(key=lambda x: x[0])
    return items


def _resolve_paths(ev: dict[str, Any], base: Path) -> dict[str, Any]:
    if "path" in ev and not Path(ev["path"]).is_absolute():
        return {**ev, "path": str((base / ev["path"]).resolve())}
    return ev


def _default_factory(provider: Any, transcriber: Any, config: Any) -> SessionFactory:
    async def factory(clock: Clock, kit_in: asyncio.Queue, kit_out: asyncio.Queue) -> None:
        from prism import adapter

        await adapter.run_session(kit_in, kit_out, clock=clock, provider=provider, transcriber=transcriber, config=config)

    return factory


async def run_scenario(
    scenario: dict[str, Any] | str | Path,
    *,
    provider: Any = None,
    transcriber: Any = None,
    config: Any = None,
    trace_path: str | Path | None = None,
    session_factory: SessionFactory | None = None,
    realtime: bool = False,
    max_time: float = DEFAULT_MAX_TIME,
    quiet_period: float = 3.0,
) -> RunResult:
    if not isinstance(scenario, dict):
        scenario = load_scenario(scenario)
    clock: Clock = RealClock() if realtime else VirtualClock()
    clock.now()
    result = RunResult(scenario=scenario)
    seq = 0
    last_activity = [0.0]

    def log(kind: str, data: dict[str, Any]) -> None:
        nonlocal seq
        seq += 1
        result.trace.append({"seq": seq, "t": round(clock.now(), 6), "kind": kind, "data": data})
        if kind in {"action", "env"}:
            last_activity[0] = clock.now()

    kit_in: asyncio.Queue = asyncio.Queue()
    kit_out: asyncio.Queue = asyncio.Queue()

    async def deliver(payload: dict[str, Any]) -> None:
        log("event", payload)
        await kit_in.put(payload)

    manifest_specs = parse_manifest(scenario.get("manifest", BUILTIN_MANIFEST))
    env = MockEnvironment(clock, deliver, log, scenario.get("tools"), {s.name for s in manifest_specs if s.state_changing})
    result.env = env
    log("meta", {"scenario": scenario.get("name"), "modality": scenario.get("modality", "text"),
                 "expected": scenario.get("expected", {}), "realtime": realtime})

    base = Path(scenario.get("_dir", "."))

    async def feed() -> None:
        for t, ev in _timeline(scenario):
            delay = t - clock.now()
            if delay > 0:
                await clock.sleep(delay)
            ev["ts"] = round(clock.now(), 6) if realtime else t
            log("event", ev)
            await kit_in.put(_resolve_paths(ev, base))

    async def consume() -> None:
        while True:
            action = await kit_out.get()
            data = action if isinstance(action, dict) else json.loads(json.dumps(action, default=str))
            log("action", data)
            kind = data.get("type")
            if kind == "tool_call":
                env.start(str(data.get("call_id")), str(data.get("name")), dict(data.get("args") or {}))
            elif kind == "cancel":
                env.cancel(str(data.get("call_id")), str(data.get("reason", "")))

    factory = session_factory or _default_factory(provider, transcriber, config)
    session = asyncio.create_task(factory(clock, kit_in, kit_out))
    feeder = asyncio.create_task(feed())
    consumer = asyncio.create_task(consume())

    try:
        if isinstance(clock, VirtualClock):
            while True:
                await clock.settle()
                if session.done():
                    break
                nxt = clock.next_wake()
                if nxt is None or nxt > max_time:
                    break
                clock.advance_to(nxt)
        else:
            while not session.done() and clock.now() < max_time:
                await asyncio.sleep(0.05)
                pending = any(inv.outcome == "pending" for inv in env.invocations.values())
                if getattr(clock, "external_pending", 0):
                    last_activity[0] = clock.now()
                if feeder.done() and not pending and clock.now() - last_activity[0] > quiet_period:
                    break
        if not session.done():
            end = {"type": "session_end", "ts": round(clock.now(), 6)}
            log("event", end)
            await kit_in.put(end)
            if isinstance(clock, VirtualClock):
                await clock.settle()
            else:
                await asyncio.sleep(0.2)
    finally:
        for task in (feeder, session):
            if not task.done():
                task.cancel()
        await asyncio.gather(feeder, session, return_exceptions=True)
        while not kit_out.empty():
            await asyncio.sleep(0)
            if consumer.done():
                break
        consumer.cancel()
        await asyncio.gather(consumer, return_exceptions=True)
        env.shutdown()

    if session.done() and not session.cancelled() and session.exception() is not None:
        raise session.exception()  # type: ignore[misc]

    if trace_path is not None:
        p = Path(trace_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as fh:
            for rec in result.trace:
                fh.write(json.dumps(rec, default=str) + "\n")
        result.trace_path = p
    return result


def _collect(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            out.extend(sorted(p.glob("*.json")))
        else:
            out.append(p)
    return out


async def _main(argv: list[str]) -> int:
    from prism.agent.config import AgentConfig
    from prism.agent.llm import make_provider
    from prism.agent.perception import make_transcriber

    from .scorer import format_report, score_trace

    parser = argparse.ArgumentParser(description="Replay scenarios against the agent and score the traces.")
    parser.add_argument("scenarios", nargs="+")
    parser.add_argument("--provider", choices=["stub", "local"], default="local")
    parser.add_argument("--transcriber", choices=["sidecar", "whisper"], default="sidecar")
    parser.add_argument("--whisper-model", default=None)
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--out", default="traces")
    args = parser.parse_args(argv)

    config = AgentConfig()
    if args.whisper_model:
        config.whisper_model = args.whisper_model
    total = 0.0
    files = _collect(args.scenarios)
    for path in files:
        scenario = load_scenario(path)
        provider = make_provider(args.provider)
        transcriber = make_transcriber(args.transcriber, config)
        if hasattr(transcriber, "load"):
            await asyncio.to_thread(transcriber.load)
        result = await run_scenario(
            scenario,
            provider=provider,
            transcriber=transcriber,
            config=config,
            realtime=args.realtime,
            trace_path=Path(args.out) / f"{path.stem}.jsonl",
        )
        report = score_trace(result.trace, scenario)
        total += report.score
        print(format_report(report), flush=True)
        print(flush=True)
    if files:
        print(f"mean score: {total / len(files):.1f} over {len(files)} scenarios")
    return 0


def main() -> None:
    sys.exit(asyncio.run(_main(sys.argv[1:])))


if __name__ == "__main__":
    main()

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from . import nlu
from .call_manager import FAILED, SUCCEEDED, CallManager, CallRecord
from .clock import Clock, RealClock
from .config import AgentConfig
from .dedup import DuplicateGuard, Verdict, idempotency_key, normalize_value
from .emitter import Emitter
from .fast_path import FastPath
from .llm import LLMProvider, Plan, PlanRequest, StubProvider, failure_text
from .manifest import parse_manifest
from .models import (
    AudioClip,
    Clarify,
    FinalResponse,
    Frame,
    Interrupt,
    SessionEnd,
    StateUpdate,
    TextChunk,
    ToolManifest,
    ToolResult,
    ToolSpec,
)
from .perception import Observation, Perception, SidecarTranscriber, Transcriber, make_transcriber
from .router import EventRouter
from .slots import SlotStore
from .slow_path import SlowPath

log = logging.getLogger(__name__)


class Agent:
    def __init__(
        self,
        config: AgentConfig | None = None,
        clock: Clock | None = None,
        provider: LLMProvider | None = None,
        transcriber: Transcriber | None = None,
    ) -> None:
        self.config = config or AgentConfig()
        self.clock = clock or RealClock()
        self.provider = provider or StubProvider()
        self.provider.bind(self.clock)
        self.transcriber = transcriber or SidecarTranscriber()
        self.slots = SlotStore()
        self.guard = DuplicateGuard(self.config.max_retries)
        self.tools: dict[str, ToolSpec] = {}
        self.turns: list[dict[str, str]] = []
        self.notes: list[str] = []
        self.chunks: list[str] = []
        self.user_speaking = False
        self.turn_id = 0
        self.turn_answered = True
        self.turn_outputs: set[str] = set()
        self.iterations = 0
        self.applied_user_turns = 0
        self.plan_gen = 0
        self._dirty = False
        self._plan_task: asyncio.Task[None] | None = None
        self._spec_task: asyncio.Task[None] | None = None
        self._spec_pending: str | None = None
        self._spec_last_key: str | None = None
        self._spec_streak = 0
        self._tasks: set[asyncio.Task[Any]] = set()
        self._closed = False

    async def warmup(self) -> None:
        await asyncio.to_thread(self.transcriber.load)
        await self.provider.warmup()

    async def run(self, in_q: asyncio.Queue, out_q: asyncio.Queue) -> None:
        self.emitter = Emitter(out_q, self.clock)
        self.calls = CallManager(self.clock, self.emitter, self.guard, self.config.call_timeout, self._on_timeout)
        self.fast = FastPath(self.clock, self.emitter, self.calls, self.config)
        self.slow = SlowPath(self.provider, self.config)
        self.perception = Perception(self.clock, self.transcriber, self.provider)
        self._wakeup = asyncio.Event()
        self.router = EventRouter(
            {
                "tool_manifest": self.on_manifest,
                "text_chunk": self.on_text,
                "audio_clip": self.on_audio,
                "frame": self.on_frame,
                "interrupt": self.on_interrupt,
                "tool_result": self.on_tool_result,
                "session_end": self.on_session_end,
            }
        )
        planner = asyncio.create_task(self._planner_loop())
        try:
            await self.router.run(in_q)
        finally:
            self._closed = True
            planner.cancel()
            for task in [self._plan_task, self._spec_task, *self._tasks]:
                if task and not task.done():
                    task.cancel()
            self.fast.stop_narration()
            self.calls.shutdown()
            self.perception.shutdown()
            await asyncio.gather(planner, return_exceptions=True)

    def _spawn(self, coro: Any) -> asyncio.Task[Any]:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    @property
    def specs(self) -> list[ToolSpec]:
        return list(self.tools.values())

    def on_manifest(self, event: ToolManifest) -> None:
        for spec in parse_manifest(event.tools):
            self.tools[spec.name] = spec

    def on_session_end(self, event: SessionEnd) -> None:
        self._closed = True

    def _utterance(self) -> str:
        return " ".join(c.strip() for c in self.chunks if c.strip())

    def on_text(self, event: TextChunk) -> None:
        if event.text.strip():
            self.chunks.append(event.text)
        utterance = self._utterance()
        self.user_speaking = not event.end_of_turn
        corrections = self._fast_correct(utterance) if utterance else {}
        if event.end_of_turn:
            self.chunks = []
            self._finish_turn(utterance, corrections=corrections)
        elif self.config.speculation and utterance:
            self._speculate(utterance)

    def on_audio(self, event: AudioClip) -> None:
        self._preempt()
        self.user_speaking = False
        self.fast.new_turn()
        self.fast.ack("", self.specs, audio=True)

        async def work() -> None:
            transcript = await self.perception.transcribe(event.path)
            if not transcript.text or transcript.confidence < self.config.audio_confidence_threshold:
                self._say_clarify("Sorry, I didn't catch that. Could you say it again?")
                return
            corrections = self._fast_correct(transcript.text)
            self._finish_turn(transcript.text, corrections=corrections, acked=True)

        self.perception.track(self._spawn(work()))

    def on_frame(self, event: Frame) -> None:
        context = self.turns[-1]["text"] if self.turns and self.turns[-1]["role"] == "user" else ""

        def done(obs: Observation) -> None:
            if not self.turn_answered and not self.user_speaking and self.turns and nlu.refers_to_visual(self.turns[-1]["text"]):
                self.request_plan()

        self.perception.submit_frame(event.path, event.ts or self.clock.now(), context, done)

    def on_interrupt(self, event: Interrupt) -> None:
        self.user_speaking = True
        self._preempt()
        self.fast.stop_narration()
        if self.config.interrupt_policy == "preempt_readonly":
            self.calls.cancel_where(lambda r: not r.state_changing, "interrupted")
        marker = (self.turn_id, len(self.chunks))

        async def silence_watch() -> None:
            await self.clock.sleep(self.config.interrupt_silence)
            if self.user_speaking and (self.turn_id, len(self.chunks)) == marker:
                self.user_speaking = False
                utterance = self._utterance()
                self.chunks = []
                if utterance:
                    self._finish_turn(utterance)
                else:
                    self.request_plan()

        self._spawn(silence_watch())

    def on_tool_result(self, event: ToolResult) -> None:
        record = self.calls.handle_result(event)
        if record is None:
            late = next((r for r in self.calls.late_commits if r.call_id == event.call_id), None)
            if late is not None:
                self.notes.append(f"{late.name} {json.dumps(late.args)} completed after it was cancelled: {json.dumps(late.late_result)}")
            return
        if record.status == FAILED and self._retry(record):
            return
        if record.speculative and self.user_speaking:
            return
        self.request_plan()

    def _on_timeout(self, record: CallRecord) -> None:
        if not record.state_changing and self._retry(record):
            return
        if record.state_changing:
            self.notes.append(f"{record.name} {json.dumps(record.args)} timed out; its outcome is unknown, do not repeat it.")
        self.request_plan()

    def _retry(self, record: CallRecord) -> bool:
        spec = self.tools.get(record.name)
        if spec is None or record.speculative or not record.valid_for(self.slots.slots):
            return False
        if self.guard.check(record.key) != Verdict.ALLOW:
            return False
        self.fast.retry_notice(record)
        new = self.calls.issue(spec, record.args, self.slots.slots, turn=self.turn_id,
                               closure=self.slots.closure, attempt=record.attempt + 1)
        if new is not None:
            self.fast.on_call_started(new)
        return new is not None

    def _fast_correct(self, text: str) -> dict[str, Any]:
        live = self.calls.in_flight()
        current: dict[str, Any] = {}
        for record in live:
            current.update(record.args)
        current.update(self.slots.slots)
        changes = self.fast.detect_changes(text, current, self.specs)
        if not changes:
            return {}

        def invalid(record: CallRecord) -> bool:
            for key, value in changes.items():
                old = record.args.get(key, None)
                if key in record.args and (value is None or normalize_value(value) != normalize_value(old)):
                    return True
                if key in record.deps and (value is None or normalize_value(value) != record.deps[key]):
                    return True
            return False

        self.calls.cancel_where(invalid, "slot_changed")
        return changes

    def _finish_turn(self, utterance: str, corrections: dict[str, Any] | None = None, acked: bool = False) -> None:
        self.user_speaking = False
        self._spec_last_key = None
        self._spec_streak = 0
        if not utterance.strip():
            if self._dirty:
                self.request_plan()
            return
        if nlu.is_backchannel(utterance) and self.turn_answered and not self._awaiting_answer():
            if self._dirty:
                self.request_plan()
            return
        self.turn_id += 1
        self.iterations = 0
        self.turn_answered = False
        self.turn_outputs = set()
        self.turns.append({"role": "user", "text": utterance})
        if not acked:
            self.fast.new_turn()
            self.fast.ack(utterance, self.specs, corrections)
        self.request_plan(preempt=True)

    def _awaiting_answer(self) -> bool:
        return bool(self.turns) and self.turns[-1]["role"] == "assistant" and self.turns[-1].get("kind") == "clarify"

    def _preempt(self) -> None:
        self.plan_gen += 1
        if self._plan_task and not self._plan_task.done():
            self._plan_task.cancel()
            self._dirty = True

    def request_plan(self, preempt: bool = False) -> None:
        self._dirty = True
        if preempt:
            self._preempt()
        if hasattr(self, "_wakeup"):
            self._wakeup.set()

    async def _planner_loop(self) -> None:
        while True:
            await self._wakeup.wait()
            self._wakeup.clear()
            if not self._dirty or self.user_speaking or self._closed:
                continue
            self._dirty = False
            gen = self.plan_gen
            self._plan_task = asyncio.create_task(self._plan_once(gen))
            await asyncio.wait({self._plan_task})
            if not self._plan_task.cancelled() and self._plan_task.exception() is not None:
                log.error("planning failed", exc_info=self._plan_task.exception())
            if self._dirty:
                self._wakeup.set()

    def _observations(self) -> list[Observation]:
        return self.perception.recent(self.clock.now(), self.config.frame_window)

    def _low_confidence(self) -> bool:
        obs = self._observations()
        return bool(obs) and obs[-1].confidence < self.config.frame_confidence_threshold

    def _request(self, user_text: str, partial: bool = False) -> PlanRequest:
        slots = self.slots.slots
        results = [
            {"call_id": r.call_id, "name": r.name, "args": r.args, "status": r.status, "result": r.result,
             "error": r.error, "valid": r.valid_for(slots)}
            for r in self.calls.calls.values() if r.status != "pending"
        ][-10:]
        in_flight = [{"call_id": r.call_id, "name": r.name, "args": r.args, "status": "pending"} for r in self.calls.in_flight()]
        return PlanRequest(
            user_text=user_text,
            snapshot=self.slots.snapshot(),
            tools=self.specs,
            turns=[{"role": t["role"], "text": t["text"]} for t in self.turns[-12:]],
            results=results,
            in_flight=in_flight,
            observations=[o.as_dict() for o in self._observations()],
            notes=self.notes[-5:],
            partial=partial,
            frame_threshold=self.config.frame_confidence_threshold,
        )

    async def _plan_once(self, gen: int) -> None:
        if self.iterations >= self.config.max_plan_iterations:
            if not self.turn_answered:
                self._say_final("Sorry, I'm having trouble with that request. Could you try rephrasing it?")
            return
        self.iterations += 1
        await self.perception.wait_pending(self.config.perception_wait)
        if gen != self.plan_gen or self.user_speaking:
            return
        user_turns = [t["text"] for t in self.turns if t["role"] == "user"]
        pending = user_turns[self.applied_user_turns:] or user_turns[-1:]
        plan = await self.slow.plan(self._request(" ".join(pending)))
        if gen != self.plan_gen or self.user_speaking or self._closed:
            return
        self._apply(plan)
        self.applied_user_turns = max(self.applied_user_turns, len(user_turns))

    def _grounding(self) -> str:
        parts = [t["text"] for t in self.turns if t["role"] == "user"]
        parts += [json.dumps(r.result) for r in self.calls.calls.values() if r.status == SUCCEEDED]
        for obs in self._observations():
            parts.append(obs.description)
            parts.extend(str(v) for v in obs.entities.values())
        return " ".join(parts)

    def _derivations(self, updates: dict[str, Any]) -> dict[str, set[str]]:
        derived: dict[str, set[str]] = {}
        for key, value in updates.items():
            needle = json.dumps(normalize_value(value))
            for record in reversed(list(self.calls.calls.values())):
                if record.status != SUCCEEDED or key in record.args:
                    continue
                if needle in json.dumps(normalize_value(record.result)):
                    derived[key] = set(record.deps) | {k for k in record.args if k in self.slots.slots}
                    break
        return derived

    def _apply(self, plan: Plan) -> None:
        updates = self.slow.slot_updates(plan, self.tools)
        intent = plan.intent if plan.intent and plan.intent.lower() not in {"none", "null", "unknown"} else None
        change = self.slots.apply(intent, updates, plan.clear_slots, self._derivations(updates))
        if change.any:
            self.calls.invalidate(self.slots.slots, change.touched)
            self.emitter.emit(StateUpdate(snapshot=self.slots.snapshot()))
        plan, args = self.slow.resolve(plan, self.tools, self.slots.slots, self._low_confidence(), self._grounding())
        if plan.action == "call_tool" and plan.tool:
            adopted = self._execute(self.tools[plan.tool], args)
            self.calls.cancel_where(lambda r: r.speculative and r.key != adopted, "speculation_mismatch")
        elif plan.action == "wait":
            self._adopt_speculative()
        elif plan.action == "clarify":
            self.calls.cancel_where(lambda r: r.speculative, "speculation_mismatch")
            self._say_clarify(plan.text or "")
        elif plan.action == "respond":
            if self.calls.in_flight(include_speculative=False):
                return
            self.calls.cancel_where(lambda r: r.speculative, "speculation_mismatch")
            has_success = any(r.state_changing and r.status == SUCCEEDED for r in self.calls.calls.values())
            self._say_final(self.fast.guard_text(plan.text or "", has_success))

    def _adopt_speculative(self) -> None:
        slots = self.slots.slots
        self.calls.cancel_where(lambda r: r.speculative and not r.valid_for(slots), "speculation_mismatch")
        for record in self.calls.in_flight():
            if record.speculative:
                record.speculative = False
                self.fast.on_call_started(record)

    def _execute(self, spec: ToolSpec, args: dict[str, Any]) -> str | None:
        key = idempotency_key(spec.name, args)
        self.calls.cancel_where(lambda r: r.name == spec.name and r.key != key, "superseded")
        live = self.calls.find_in_flight(key)
        if live is not None:
            live.speculative = False
            self.fast.on_call_started(live)
            return key
        cached = self.calls.find_succeeded(key)
        if cached is not None:
            self.notes.append(f"{spec.name} with these arguments already succeeded (call {cached.call_id}); use its result.")
            self.request_plan()
            return key
        verdict = self.guard.check(key)
        if verdict == Verdict.SUCCEEDED:
            self.notes.append(f"{spec.name} {json.dumps(args)} already completed earlier; do not repeat it.")
            self.request_plan()
            return key
        if verdict in {Verdict.EXHAUSTED, Verdict.UNCERTAIN}:
            last = next((r for r in reversed(list(self.calls.calls.values())) if r.key == key), None)
            error = "the outcome could not be confirmed" if verdict == Verdict.UNCERTAIN else (last.error if last else None)
            self._say_final(failure_text(spec, error))
            return key
        if verdict == Verdict.IN_FLIGHT:
            return key
        record = self.calls.issue(spec, args, self.slots.slots, turn=self.turn_id, closure=self.slots.closure,
                                  attempt=self.guard.failures(key) + 1)
        if record is not None:
            self.fast.on_call_started(record)
        return key

    def _say_clarify(self, question: str) -> None:
        if not question.strip() or ("clarify:" + question) in self.turn_outputs:
            return
        self.turn_outputs.add("clarify:" + question)
        self.emitter.emit(Clarify(question=question, snapshot=self.slots.snapshot()))
        self.turns.append({"role": "assistant", "text": question, "kind": "clarify"})
        self.turn_answered = True

    def _say_final(self, text: str) -> None:
        if not text.strip() or ("final:" + text) in self.turn_outputs:
            return
        self.turn_outputs.add("final:" + text)
        self.emitter.emit(FinalResponse(text=text, snapshot=self.slots.snapshot()))
        self.turns.append({"role": "assistant", "text": text, "kind": "final"})
        self.turn_answered = True
        self.fast.stop_narration()

    def _speculate(self, utterance: str) -> None:
        if not self.tools or len(utterance.split()) < self.config.speculation_min_words:
            return
        if not self.provider.parallel_calls and not self.config.force_speculation:
            return
        self._spec_pending = utterance
        if self._spec_task is None or self._spec_task.done():
            self._spec_task = self._spawn(self._speculation_loop())

    async def _speculation_loop(self) -> None:
        while self._spec_pending is not None:
            utterance, self._spec_pending = self._spec_pending, None
            turn = self.turn_id
            plan = await self.slow.plan(self._request(utterance, partial=True))
            if turn != self.turn_id or not self.user_speaking or self._closed:
                return
            self._apply_speculative(plan)

    def _apply_speculative(self, plan: Plan) -> None:
        if plan.action != "call_tool" or not plan.tool or plan.confidence < self.config.speculation_min_confidence:
            self._spec_last_key, self._spec_streak = None, 0
            return
        spec = self.tools.get(plan.tool)
        if spec is None or spec.state_changing:
            return
        predicted = {**self.slots.slots, **self.slow.slot_updates(plan, self.tools)}
        plan, args = self.slow.resolve(plan, self.tools, predicted, self._low_confidence())
        if plan.action != "call_tool":
            return
        key = idempotency_key(spec.name, args)
        self._spec_streak = self._spec_streak + 1 if key == self._spec_last_key else 1
        self._spec_last_key = key
        if self._spec_streak < 2:
            return
        if self.calls.find_in_flight(key) or self.calls.find_succeeded(key) or self.guard.check(key) != Verdict.ALLOW:
            return
        self.calls.cancel_where(lambda r: r.speculative and r.key != key, "speculation_mismatch")
        self.calls.issue(spec, args, predicted, speculative=True, turn=self.turn_id, closure=self.slots.closure)


def create_agent(
    config: AgentConfig | None = None,
    clock: Clock | None = None,
    provider: LLMProvider | str | None = None,
    transcriber: Transcriber | str | None = None,
) -> Agent:
    config = config or AgentConfig()
    if isinstance(provider, str):
        from .llm import make_provider

        provider = make_provider(provider)
    if isinstance(transcriber, str):
        transcriber = make_transcriber(transcriber, config)
    return Agent(config=config, clock=clock, provider=provider, transcriber=transcriber)


async def warmup(agent: Agent | None = None, config: AgentConfig | None = None, transcriber: str = "auto") -> Agent:
    agent = agent or create_agent(config=config, transcriber=transcriber)
    await agent.warmup()
    return agent

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from .clock import Clock
from .dedup import DuplicateGuard, idempotency_key, normalize_value
from .emitter import Emitter
from .models import Cancel, ToolCall, ToolResult, ToolSpec

log = logging.getLogger(__name__)

PENDING = "pending"
SUCCEEDED = "succeeded"
FAILED = "failed"
CANCELLED = "cancelled"
TIMED_OUT = "timed_out"


@dataclass
class CallRecord:
    call_id: str
    name: str
    args: dict[str, Any]
    key: str
    state_changing: bool
    issued_at: float
    speculative: bool = False
    turn: int = 0
    attempt: int = 1
    status: str = PENDING
    deps: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    error: str | None = None
    finished_at: float | None = None
    cancel_reason: str | None = None
    late_result: Any = None

    @property
    def in_flight(self) -> bool:
        return self.status == PENDING

    def valid_for(self, slots: dict[str, Any]) -> bool:
        return all(normalize_value(slots.get(k)) == v for k, v in self.deps.items())


class CallManager:
    def __init__(
        self,
        clock: Clock,
        emitter: Emitter,
        guard: DuplicateGuard | None = None,
        timeout: float = 20.0,
        on_timeout: Callable[[CallRecord], None] | None = None,
    ) -> None:
        self.clock = clock
        self.emitter = emitter
        self.guard = guard or DuplicateGuard()
        self.timeout = timeout
        self.on_timeout = on_timeout
        self.calls: dict[str, CallRecord] = {}
        self._counter = 0
        self._timers: dict[str, asyncio.Task[None]] = {}
        self.dropped: list[ToolResult] = []
        self.late_commits: list[CallRecord] = []

    def new_call_id(self) -> str:
        self._counter += 1
        return f"call_{self._counter}"

    @staticmethod
    def dependencies(args: dict[str, Any], slots: dict[str, Any], closure: Callable[[set[str]], set[str]] | None = None) -> set[str]:
        arg_values = {json.dumps(normalize_value(v), sort_keys=True) for v in args.values()}
        deps = {k for k in args if k in slots}
        deps |= {k for k, v in slots.items() if json.dumps(normalize_value(v), sort_keys=True) in arg_values}
        return closure(deps) if closure else deps

    def issue(
        self,
        spec: ToolSpec,
        args: dict[str, Any],
        slots: dict[str, Any],
        *,
        speculative: bool = False,
        turn: int = 0,
        closure: Callable[[set[str]], set[str]] | None = None,
        attempt: int = 1,
    ) -> CallRecord | None:
        call_id = self.new_call_id()
        key = idempotency_key(spec.name, args)
        deps = self.dependencies(args, slots, closure)
        record = CallRecord(
            call_id=call_id,
            name=spec.name,
            args=dict(args),
            key=key,
            state_changing=spec.state_changing,
            issued_at=self.clock.now(),
            speculative=speculative,
            turn=turn,
            attempt=attempt,
            deps={k: normalize_value(slots.get(k)) for k in deps},
        )
        if self.emitter.emit(ToolCall(call_id=call_id, name=spec.name, args=record.args)) is None:
            return None
        self.calls[call_id] = record
        self.guard.mark_in_flight(key, call_id)
        if self.timeout > 0:
            self._timers[call_id] = asyncio.create_task(self._watch(record))
        return record

    async def _watch(self, record: CallRecord) -> None:
        await self.clock.sleep(self.timeout)
        if record.status != PENDING:
            return
        self.emitter.emit(Cancel(call_id=record.call_id, reason="timeout"))
        record.status = TIMED_OUT
        record.cancel_reason = "timeout"
        record.finished_at = self.clock.now()
        if record.state_changing:
            self.guard.mark_uncertain(record.key)
        else:
            self.guard.mark_failed(record.key)
        self._timers.pop(record.call_id, None)
        if self.on_timeout:
            self.on_timeout(record)

    def _stop_timer(self, call_id: str) -> None:
        timer = self._timers.pop(call_id, None)
        if timer and not timer.done() and timer is not asyncio.current_task():
            timer.cancel()

    def cancel(self, call_id: str, reason: str) -> bool:
        record = self.calls.get(call_id)
        if record is None or record.status != PENDING:
            return False
        self.emitter.emit(Cancel(call_id=call_id, reason=reason))
        record.status = CANCELLED
        record.cancel_reason = reason
        record.finished_at = self.clock.now()
        self.guard.release(record.key)
        self._stop_timer(call_id)
        return True

    def cancel_where(self, predicate: Callable[[CallRecord], bool], reason: str) -> list[CallRecord]:
        cancelled = []
        for record in list(self.calls.values()):
            if record.status == PENDING and predicate(record) and self.cancel(record.call_id, reason):
                cancelled.append(record)
        return cancelled

    def handle_result(self, event: ToolResult) -> CallRecord | None:
        record = self.calls.get(event.call_id)
        if record is None:
            self.dropped.append(event)
            log.warning("result for unknown call %s dropped", event.call_id)
            return None
        if record.status != PENDING:
            self.dropped.append(event)
            if record.state_changing and event.ok and record.status in {CANCELLED, TIMED_OUT}:
                record.late_result = event.result
                self.guard.mark_succeeded(record.key)
                self.late_commits.append(record)
            return None
        self._stop_timer(record.call_id)
        record.finished_at = self.clock.now()
        if event.ok:
            record.status = SUCCEEDED
            record.result = event.result
            self.guard.mark_succeeded(record.key)
        else:
            record.status = FAILED
            record.error = event.error or "unknown_error"
            self.guard.mark_failed(record.key)
        return record

    def in_flight(self, include_speculative: bool = True) -> list[CallRecord]:
        return [r for r in self.calls.values() if r.status == PENDING and (include_speculative or not r.speculative)]

    def find_in_flight(self, key: str) -> CallRecord | None:
        return next((r for r in self.calls.values() if r.status == PENDING and r.key == key), None)

    def find_succeeded(self, key: str, slots: dict[str, Any] | None = None) -> CallRecord | None:
        for record in reversed(list(self.calls.values())):
            if record.key == key and record.status == SUCCEEDED and (slots is None or record.valid_for(slots)):
                return record
        return None

    def invalidate(self, slots: dict[str, Any], touched: set[str], reason: str = "slot_changed") -> list[CallRecord]:
        if not touched:
            return []
        return self.cancel_where(lambda r: bool(set(r.deps) & touched) and not r.valid_for(slots), reason)

    def completed(self) -> list[CallRecord]:
        return [r for r in self.calls.values() if r.status in {SUCCEEDED, FAILED, TIMED_OUT}]

    def shutdown(self) -> None:
        for timer in self._timers.values():
            if not timer.done():
                timer.cancel()
        self._timers.clear()

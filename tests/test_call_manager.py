import asyncio

from prism.agent.call_manager import CANCELLED, SUCCEEDED, TIMED_OUT, CallManager
from prism.agent.clock import VirtualClock
from prism.agent.dedup import DuplicateGuard, Verdict
from prism.agent.emitter import Emitter
from prism.agent.models import ToolResult, ToolSpec

SEARCH = ToolSpec(name="flight_search", state_changing=False)
BOOK = ToolSpec(name="book_flight", state_changing=True)


def make(timeout=0.0, on_timeout=None):
    clock = VirtualClock()
    out = asyncio.Queue()
    guard = DuplicateGuard(max_retries=1)
    cm = CallManager(clock, Emitter(out, clock), guard, timeout=timeout, on_timeout=on_timeout)
    return clock, out, cm, guard


def drain(q):
    items = []
    while not q.empty():
        items.append(q.get_nowait())
    return items


async def test_issue_emits_tool_call_with_unique_ids():
    clock, out, cm, _ = make()
    a = cm.issue(SEARCH, {"destination": "Paris"}, {"destination": "Paris"})
    b = cm.issue(SEARCH, {"destination": "Rome"}, {"destination": "Rome"})
    actions = drain(out)
    assert [x.type for x in actions] == ["tool_call", "tool_call"]
    assert a.call_id != b.call_id
    assert a.deps == {"destination": "paris"}


async def test_cancel_then_drop_late_result():
    clock, out, cm, _ = make()
    rec = cm.issue(SEARCH, {"destination": "Paris"}, {"destination": "Paris"})
    assert cm.cancel(rec.call_id, "slot_changed")
    assert not cm.cancel(rec.call_id, "slot_changed")
    assert rec.status == CANCELLED
    assert cm.handle_result(ToolResult(call_id=rec.call_id, ok=True, result={"x": 1})) is None
    assert cm.dropped and rec.result is None
    types = [x.type for x in drain(out)]
    assert types == ["tool_call", "cancel"]


async def test_invalidate_cancels_only_dependent_calls():
    clock, out, cm, _ = make()
    slots = {"destination": "Paris", "passenger_name": "Ann"}
    search = cm.issue(SEARCH, {"destination": "Paris"}, slots)
    other = cm.issue(ToolSpec(name="weather", state_changing=False), {"city": "Oslo"}, slots)
    new_slots = dict(slots, destination="Rome")
    cancelled = cm.invalidate(new_slots, {"destination"})
    assert [c.call_id for c in cancelled] == [search.call_id]
    assert other.in_flight


async def test_unknown_result_is_dropped():
    _, _, cm, _ = make()
    assert cm.handle_result(ToolResult(call_id="nope", ok=True)) is None


async def test_success_marks_guard_and_blocks_duplicate():
    _, _, cm, guard = make()
    rec = cm.issue(BOOK, {"flight_id": "AF1"}, {})
    assert guard.check(rec.key) == Verdict.IN_FLIGHT
    cm.handle_result(ToolResult(call_id=rec.call_id, ok=True, result={"booking_id": "BK-1"}))
    assert rec.status == SUCCEEDED
    assert guard.check(rec.key) == Verdict.SUCCEEDED


async def test_failure_allows_bounded_retry():
    _, _, cm, guard = make()
    rec = cm.issue(BOOK, {"flight_id": "AF1"}, {})
    cm.handle_result(ToolResult(call_id=rec.call_id, ok=False, error="boom"))
    assert guard.check(rec.key) == Verdict.ALLOW
    rec2 = cm.issue(BOOK, {"flight_id": "AF1"}, {}, attempt=2)
    cm.handle_result(ToolResult(call_id=rec2.call_id, ok=False, error="boom"))
    assert guard.check(rec.key) == Verdict.EXHAUSTED


async def test_timeout_cancels_and_marks_state_changing_uncertain():
    fired = []
    clock, out, cm, guard = make(timeout=5.0, on_timeout=fired.append)
    rec = cm.issue(BOOK, {"flight_id": "AF1"}, {})
    await clock.settle()
    clock.advance_to(clock.next_wake())
    await clock.settle()
    assert rec.status == TIMED_OUT
    assert fired == [rec]
    assert guard.check(rec.key) == Verdict.UNCERTAIN
    assert [a.type for a in drain(out)] == ["tool_call", "cancel"]
    assert cm.handle_result(ToolResult(call_id=rec.call_id, ok=True, result={})) is None
    assert cm.late_commits == [rec]


async def test_result_stops_timeout_timer():
    clock, out, cm, _ = make(timeout=5.0)
    rec = cm.issue(SEARCH, {"destination": "Paris"}, {})
    await clock.settle()
    cm.handle_result(ToolResult(call_id=rec.call_id, ok=True, result={}))
    await clock.settle()
    assert clock.next_wake() is None

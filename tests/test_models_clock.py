import asyncio

import pytest
from pydantic import ValidationError

from prism.agent.clock import VirtualClock
from prism.agent.models import (
    Cancel,
    FinalResponse,
    StateSnapshot,
    TextChunk,
    ToolCall,
    parse_action,
    parse_event,
)


def test_parse_event_roundtrip():
    ev = parse_event({"type": "text_chunk", "text": "hi", "end_of_turn": True, "ts": 1.5})
    assert isinstance(ev, TextChunk)
    assert ev.end_of_turn and ev.ts == 1.5
    assert parse_event(ev) is ev


def test_parse_event_rejects_unknown_type():
    with pytest.raises(ValidationError):
        parse_event({"type": "nope"})


def test_action_validation():
    call = parse_action({"type": "tool_call", "call_id": "call_1", "name": "x", "args": {"a": 1}})
    assert isinstance(call, ToolCall)
    with pytest.raises(ValidationError):
        ToolCall(call_id="bad id!", name="x")
    with pytest.raises(ValidationError):
        ToolCall(call_id="c1", name="x", args={"f": float("nan")})
    with pytest.raises(ValidationError):
        Cancel(call_id="")
    fr = FinalResponse(text="ok", snapshot=StateSnapshot(intent="i", slots={"a": "b"}))
    assert parse_action(fr.model_dump()).snapshot.slots == {"a": "b"}


def test_snapshot_rejects_non_json():
    with pytest.raises(ValidationError):
        StateSnapshot(slots={"x": object()})


async def test_virtual_clock_orders_sleepers():
    clock = VirtualClock()
    order = []

    async def sleeper(name, d):
        await clock.sleep(d)
        order.append((name, clock.now()))

    tasks = [asyncio.create_task(sleeper("b", 2.0)), asyncio.create_task(sleeper("a", 1.0))]
    await clock.settle()
    while (t := clock.next_wake()) is not None:
        clock.advance_to(t)
        await clock.settle()
    await asyncio.gather(*tasks)
    assert order == [("a", 1.0), ("b", 2.0)]


async def test_virtual_clock_cancelled_sleeper_is_skipped():
    clock = VirtualClock()
    task = asyncio.create_task(clock.sleep(5.0))
    await clock.settle()
    task.cancel()
    await clock.settle()
    assert clock.next_wake() is None


async def test_virtual_clock_waits_for_external_work():
    clock = VirtualClock()
    result = await clock.run_blocking(lambda: 41 + 1)
    assert result == 42
    assert clock.now() == 0.0

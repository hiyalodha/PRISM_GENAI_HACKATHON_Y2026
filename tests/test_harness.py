import asyncio

from prism.harness.replay import run_scenario
from prism.harness.scorer import score_trace


def scripted_session(script):
    async def factory(clock, kit_in, kit_out):
        n = 0
        while True:
            ev = await kit_in.get()
            if ev["type"] == "session_end":
                return
            for action in script(ev, clock):
                if action.get("type") == "tool_call" and action.get("call_id") == "auto":
                    n += 1
                    action["call_id"] = f"call_{n}"
                action.setdefault("ts", clock.now())
                await kit_out.put(action)

    return factory


def _search_script(ev, clock):
    if ev["type"] == "text_chunk" and ev.get("end_of_turn"):
        return [
            {"type": "speak", "kind": "ack", "text": "One moment."},
            {"type": "tool_call", "call_id": "auto", "name": "flight_search",
             "args": {"origin": "London", "destination": "Paris", "date": "Monday"}},
        ]
    if ev["type"] == "tool_result":
        return [{"type": "final_response", "text": "Found flights.", "snapshot": {"intent": "flight_search", "slots": {}}}]
    return []


async def test_harness_runs_tool_with_virtual_latency():
    scenario = {
        "name": "t",
        "tools": {"flight_search": {"latency": 2.5}},
        "events": [{"t": 1.0, "type": "text_chunk", "text": "flights to paris", "end_of_turn": True}],
    }
    result = await run_scenario(scenario, session_factory=scripted_session(_search_script))
    results = [r for r in result.trace if r["kind"] == "event" and r["data"]["type"] == "tool_result"]
    assert len(results) == 1
    assert results[0]["t"] == 3.5
    assert results[0]["data"]["ok"] is True
    assert results[0]["data"]["result"]["flights"]
    final = [r for r in result.trace if r["kind"] == "action" and r["data"]["type"] == "final_response"]
    assert final[0]["t"] == 3.5


async def test_fail_once_fault_and_cancel():
    calls = []

    def script(ev, clock):
        if ev["type"] == "text_chunk":
            return [{"type": "tool_call", "call_id": "auto", "name": "book_flight",
                     "args": {"flight_id": "AF100", "passenger_name": "Ann Lee"}}]
        if ev["type"] == "interrupt":
            return [{"type": "cancel", "call_id": "call_2", "reason": "interrupted"}]
        if ev["type"] == "tool_result":
            calls.append(ev)
        return []

    scenario = {
        "tools": {"book_flight": {"latency": 1.0, "faults": ["fail"]}},
        "events": [
            {"t": 0.0, "type": "text_chunk", "text": "a", "end_of_turn": True},
            {"t": 2.0, "type": "text_chunk", "text": "b", "end_of_turn": True},
            {"t": 2.5, "type": "interrupt"},
        ],
    }
    result = await run_scenario(scenario, session_factory=scripted_session(script))
    assert [c["ok"] for c in calls] == [False]
    ops = [r["data"]["op"] for r in result.trace if r["kind"] == "env"]
    assert "tool_cancelled" in ops
    assert result.env.commits == []


async def test_ignore_cancel_delivers_late_result():
    seen = []

    def script(ev, clock):
        if ev["type"] == "text_chunk":
            return [{"type": "tool_call", "call_id": "auto", "name": "flight_search", "args": {"destination": "Paris"}}]
        if ev["type"] == "interrupt":
            return [{"type": "cancel", "call_id": "call_1", "reason": "interrupted"}]
        if ev["type"] == "tool_result":
            seen.append(ev["call_id"])
        return []

    scenario = {
        "tools": {"flight_search": {"latency": 2.0, "ignore_cancel": True}},
        "events": [
            {"t": 0.0, "type": "text_chunk", "text": "a", "end_of_turn": True},
            {"t": 1.0, "type": "interrupt"},
        ],
    }
    result = await run_scenario(scenario, session_factory=scripted_session(script))
    assert seen == ["call_1"]
    assert any(r["data"].get("op") == "tool_result_late" for r in result.trace if r["kind"] == "env")


async def test_timeout_fault_never_returns():
    seen = []

    def script(ev, clock):
        if ev["type"] == "text_chunk":
            return [{"type": "tool_call", "call_id": "auto", "name": "flight_search", "args": {}}]
        if ev["type"] == "tool_result":
            seen.append(ev)
        return []

    scenario = {"tools": {"flight_search": {"faults": ["timeout"]}},
                "events": [{"t": 0.0, "type": "text_chunk", "text": "a", "end_of_turn": True}]}
    await run_scenario(scenario, session_factory=scripted_session(script))
    assert seen == []


async def test_scorer_flags_duplicate_booking():
    def script(ev, clock):
        if ev["type"] == "text_chunk":
            return [{"type": "tool_call", "call_id": "auto", "name": "book_flight",
                     "args": {"flight_id": "AF100", "passenger_name": "Ann Lee"}}]
        return []

    scenario = {
        "tools": {"book_flight": {"latency": 3.0}},
        "events": [
            {"t": 0.0, "type": "text_chunk", "text": "book", "end_of_turn": True},
            {"t": 1.0, "type": "text_chunk", "text": "book again", "end_of_turn": True},
        ],
        "expected": {"successful_state_changes": {"book_flight": 1}},
    }
    result = await run_scenario(scenario, session_factory=scripted_session(script))
    report = score_trace(result.trace, scenario)
    assert not report.check("no_duplicate_state_changes").passed
    assert not report.check("commits:book_flight").passed
    assert not report.check("first_response_latency").passed


async def test_scorer_passes_clean_search():
    scenario = {
        "tools": {"flight_search": {"latency": 1.0}},
        "events": [{"t": 0.5, "type": "text_chunk", "text": "flights", "end_of_turn": True}],
        "expected": {"calls": [{"name": "flight_search", "args": {"destination": "Paris"}, "succeeded": True}]},
    }
    result = await run_scenario(scenario, session_factory=scripted_session(_search_script))
    report = score_trace(result.trace, scenario)
    assert report.failed == [], report.failed
    assert report.score == 100.0

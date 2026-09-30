from prism.agent.config import AgentConfig
from prism.agent.llm import StubProvider
from prism.agent.perception import SidecarTranscriber
from prism.harness.replay import run_scenario
from prism.harness.scorer import score_trace

SEARCH_TURN = {"t": 0.5, "type": "text_chunk", "text": "Find flights from London to Paris on Monday.", "end_of_turn": True}


async def run(scenario, config=None):
    scenario.setdefault("name", "adhoc")
    result = await run_scenario(scenario, provider=StubProvider(), transcriber=SidecarTranscriber(), config=config)
    return result, score_trace(result.trace, scenario)


def actions(result, kind=None):
    return [r for r in result.trace if r["kind"] == "action" and (kind is None or r["data"]["type"] == kind)]


async def test_repeated_booking_request_while_in_flight_is_not_duplicated():
    scenario = {
        "tools": {"flight_search": {"latency": 0.5}, "book_flight": {"latency": 4.0}},
        "events": [
            SEARCH_TURN,
            {"t": 3.0, "type": "text_chunk", "text": "Book the cheapest one for Ann Lee.", "end_of_turn": True},
            {"t": 4.0, "type": "text_chunk", "text": "Yes, book the cheapest one for Ann Lee please.", "end_of_turn": True},
        ],
        "expected": {"successful_state_changes": {"book_flight": 1}},
    }
    result, report = await run(scenario)
    books = [a for a in actions(result, "tool_call") if a["data"]["name"] == "book_flight"]
    assert len(books) == 1
    assert report.check("no_duplicate_state_changes").passed
    assert report.check("commits:book_flight").passed


async def test_booking_timeout_is_cancelled_not_retried_and_reported_honestly():
    scenario = {
        "tools": {"flight_search": {"latency": 0.5}, "book_flight": {"faults": ["timeout"]}},
        "events": [SEARCH_TURN, {"t": 3.0, "type": "text_chunk", "text": "Book the cheapest one for Ann Lee.", "end_of_turn": True}],
    }
    result, report = await run(scenario, AgentConfig(call_timeout=6.0))
    cancels = actions(result, "cancel")
    assert [c["data"]["reason"] for c in cancels] == ["timeout"]
    assert cancels[0]["t"] == 9.0
    books = [a for a in actions(result, "tool_call") if a["data"]["name"] == "book_flight"]
    assert len(books) == 1
    final = actions(result, "final_response")[-1]["data"]["text"]
    assert "couldn't" in final
    assert report.check("no_false_completion_claims").passed


async def test_read_only_retries_are_bounded():
    scenario = {"tools": {"flight_search": {"latency": 0.5, "faults": ["fail", "fail", "fail", "fail"]}}, "events": [SEARCH_TURN]}
    result, _ = await run(scenario, AgentConfig(max_retries=2))
    searches = actions(result, "tool_call")
    assert len(searches) == 3
    assert "couldn't" in actions(result, "final_response")[-1]["data"]["text"]


async def test_speculative_call_cancelled_when_user_corrects_mid_utterance():
    scenario = {
        "tools": {"flight_search": {"latency": 5.0}},
        "events": [
            {"t": 0.2, "type": "text_chunk", "text": "Find flights from London to Paris on Monday"},
            {"t": 0.6, "type": "text_chunk", "text": " in the morning"},
            {"t": 1.0, "type": "text_chunk", "text": " no wait, Tuesday not Monday."},
            {"t": 1.4, "type": "text_chunk", "text": " Thanks.", "end_of_turn": True},
        ],
        "expected": {
            "cancelled": [{"name": "flight_search", "args": {"date": "Monday"}}],
            "final_snapshot": {"slots": {"date": "Tuesday"}},
            "calls": [{"name": "flight_search", "args": {"date": "Tuesday"}, "min": 1, "max": 1, "succeeded": True}],
        },
    }
    result, report = await run(scenario)
    calls = actions(result, "tool_call")
    assert calls[0]["t"] == 0.6 and calls[0]["data"]["args"]["date"] == "Monday"
    assert report.failed == [], [(c.name, c.detail) for c in report.failed]


async def test_long_tool_progress_is_capped():
    scenario = {"tools": {"flight_search": {"latency": 15.0}}, "events": [SEARCH_TURN], "expected": {"max_fillers_per_turn": 3}}
    result, report = await run(scenario, AgentConfig(call_timeout=30.0))
    speaks = [a["data"]["kind"] for a in actions(result, "speak")]
    assert speaks.count("progress") == 2 and speaks.count("ack") == 1
    assert report.check("filler_budget").passed


async def test_unknown_and_malformed_events_are_ignored():
    scenario = {
        "tools": {"flight_search": {"latency": 1.0}},
        "events": [
            {"t": 0.1, "type": "tool_result", "call_id": "ghost_1", "ok": True, "result": {"flights": [{"flight_id": "ZZ999"}]}},
            {"t": 0.2, "type": "bogus_event", "x": 1},
            SEARCH_TURN,
        ],
    }
    result, report = await run(scenario)
    assert actions(result, "final_response")
    assert "ZZ999" not in actions(result, "final_response")[-1]["data"]["text"]
    assert report.check("actions_valid").passed


async def test_interrupt_without_followup_does_not_stall():
    scenario = {
        "tools": {"flight_search": {"latency": 3.0}},
        "events": [SEARCH_TURN, {"t": 2.0, "type": "interrupt"}],
    }
    result, _ = await run(scenario, AgentConfig(interrupt_silence=2.0))
    calls = actions(result, "tool_call")
    assert len(calls) == 2
    assert actions(result, "cancel")[0]["data"]["reason"] == "interrupted"
    assert actions(result, "final_response")[-1]["t"] == 7.0


async def test_late_commit_after_cancel_is_recorded_and_not_repeated():
    scenario = {
        "tools": {"flight_search": {"latency": 0.5}, "book_flight": {"latency": 3.0, "ignore_cancel": True}},
        "events": [
            SEARCH_TURN,
            {"t": 3.0, "type": "text_chunk", "text": "Book the cheapest one for John Smith.", "end_of_turn": True},
            {"t": 4.0, "type": "text_chunk", "text": "Sorry, the name should be Jon Smith, not John Smith.", "end_of_turn": True},
        ],
    }
    result, report = await run(scenario)
    books = [a["data"]["args"]["passenger_name"] for a in actions(result, "tool_call") if a["data"]["name"] == "book_flight"]
    assert books == ["John Smith", "Jon Smith"]
    late = [r for r in result.trace if r["kind"] == "env" and r["data"].get("op") == "tool_result_late"]
    assert len(late) == 1
    assert report.check("no_duplicate_state_changes").passed


async def test_wait_for_text_policy_keeps_calls_on_bare_interrupt():
    scenario = {"tools": {"flight_search": {"latency": 3.0}},
                "events": [SEARCH_TURN, {"t": 1.0, "type": "interrupt"}, {"t": 1.2, "type": "text_chunk", "text": "Uh-huh.", "end_of_turn": True}]}
    result, _ = await run(scenario, AgentConfig(interrupt_policy="wait_for_text"))
    assert actions(result, "cancel") == []
    assert len(actions(result, "tool_call")) == 1
    assert actions(result, "final_response")


class SlowStub(StubProvider):
    parallel_calls = False

    async def plan(self, request):
        await self.clock.sleep(5.0)
        return self._plan(request)


async def test_slow_planner_preempted_turns_are_not_lost():
    from prism.harness.replay import load_scenario
    from pathlib import Path

    scenario = load_scenario(Path(__file__).resolve().parent.parent / "scenarios" / "01_search_then_book.json")
    result = await run_scenario(scenario, provider=SlowStub(), transcriber=SidecarTranscriber())
    report = score_trace(result.trace, scenario)
    assert report.check("final_snapshot").passed, report.check("final_snapshot").detail
    assert report.check("commits:book_flight").passed
    assert not actions(result, "clarify")

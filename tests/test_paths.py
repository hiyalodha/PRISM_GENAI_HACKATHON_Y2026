import asyncio

from prism.agent import nlu
from prism.agent.call_manager import CallManager
from prism.agent.clock import VirtualClock
from prism.agent.config import AgentConfig
from prism.agent.emitter import Emitter
from prism.agent.fast_path import FastPath, activity_phrase, claims_completion
from prism.agent.llm import Plan, PlanRequest, StubProvider
from prism.agent.manifest import parse_manifest
from prism.agent.models import StateSnapshot, ToolSpec
from prism.agent.slow_path import SlowPath, coerce
from prism.harness.mock_tools import BUILTIN_MANIFEST

SPECS = parse_manifest(BUILTIN_MANIFEST)
BY_NAME = {s.name: s for s in SPECS}


def fast_path(config=None):
    clock = VirtualClock()
    out = asyncio.Queue()
    emitter = Emitter(out, clock)
    calls = CallManager(clock, emitter, timeout=0)
    return FastPath(clock, emitter, calls, config or AgentConfig()), out, clock, calls


def drain(q):
    items = []
    while not q.empty():
        items.append(q.get_nowait())
    return items


def test_nlu_entities():
    cities = nlu.find_cities("fly from New York to Paris, not Rome")
    assert [(c.value, c.role, c.negated) for c in cities] == [("New York", "origin", False), ("Paris", "destination", False), ("Rome", None, True)]
    assert nlu.last_positive(nlu.find_dates("actually Tuesday, not Monday")).value == "Tuesday"
    assert nlu.find_dates("on March 5th")[0].value == "March 5"
    assert nlu.find_name("Sorry, the name should be Jon Smith, not John Smith.") == "Jon Smith"
    assert nlu.find_name("book it for Ann Lee") == "Ann Lee"
    assert nlu.find_name("flights for New York") is None
    assert nlu.find_time("at 7:30pm") == "7:30 pm"
    assert nlu.find_quantity("table for 4 people at 7pm") == 4
    assert nlu.find_selection("the cheapest one") == ("cheapest", 0)
    assert nlu.find_selection("option 2") == ("index", 1)
    assert nlu.is_backchannel("uh-huh") and not nlu.is_backchannel("uh-huh, to Rome")


def test_tool_guessing():
    assert nlu.guess_tool("find me flights to Paris", SPECS).name == "flight_search"
    assert nlu.guess_tool("book the cheapest flight", SPECS).name == "book_flight"
    assert nlu.guess_tool("what does this blinking red light mean", SPECS).name == "manual_lookup"
    assert nlu.guess_tool("actually make it Rome", SPECS) is None


def test_ack_is_immediate_and_capped():
    fp, out, clock, _ = fast_path(AgentConfig(max_fillers_per_turn=2))
    fp.new_turn()
    assert fp.ack("find flights to Paris", SPECS)
    assert fp.ack("hello", SPECS)
    assert not fp.ack("again", SPECS)
    actions = drain(out)
    assert [a.kind for a in actions] == ["ack", "ack"]
    assert all(a.ts == 0.0 for a in actions)
    fp.new_turn()
    assert fp.ack("hi", SPECS)


def test_ack_mentions_correction():
    fp, out, _, _ = fast_path()
    fp.ack("make it Rome", SPECS, {"destination": "Rome"})
    assert drain(out)[0].text == "Got it, Rome instead."


def test_fast_path_never_claims_completion():
    fp, out, _, _ = fast_path()
    assert not fp.speak("progress", "I've booked your flight.")
    assert drain(out) == []
    assert claims_completion("Done, your flight is booked.")
    assert claims_completion("You're all set.")
    assert not claims_completion("Let me check that.")
    assert fp.guard_text("I've booked it.", has_success=False).startswith("I haven't")
    assert fp.guard_text("I've booked it.", has_success=True) == "I've booked it."


def test_detect_changes():
    fp, _, _, _ = fast_path()
    current = {"origin": "London", "destination": "Paris", "date": "Monday"}
    assert fp.detect_changes("actually make it Rome", current, SPECS) == {"destination": "Rome"}
    assert fp.detect_changes("no, Tuesday not Monday", current, SPECS) == {"date": "Tuesday"}
    assert fp.detect_changes("to Paris please", current, SPECS) == {}
    assert fp.detect_changes("sorry, from Madrid", current, SPECS) == {"origin": "Madrid"}
    assert fp.detect_changes("Rome sounds nice", current, SPECS) == {}
    assert fp.detect_changes("not London", {"origin": "London"}, SPECS) == {"origin": None}


async def test_progress_narration_capped():
    config = AgentConfig(progress_after=2.0, progress_interval=3.0, max_progress_per_call=2, max_fillers_per_turn=5)
    fp, out, clock, calls = fast_path(config)
    fp.new_turn()
    rec = calls.issue(BY_NAME["flight_search"], {"destination": "Paris"}, {})
    fp.on_call_started(rec)
    for _ in range(10):
        await clock.settle()
        nxt = clock.next_wake()
        if nxt is None or nxt > 30:
            break
        clock.advance_to(nxt)
    await clock.settle()
    progress = [a for a in drain(out) if a.type == "speak"]
    assert [p.ts for p in progress] == [2.0, 5.0]
    assert "searching" not in progress[0].text or "flight" in progress[0].text
    fp.stop_narration()


def test_activity_phrase():
    assert activity_phrase("book_flight") == "booking the flight"
    assert activity_phrase("create_ticket") == "creating the ticket"
    assert activity_phrase("flight_search") == "working on the flight search"


def test_coerce():
    assert coerce("4", {"type": "integer"}) == 4
    assert coerce("2.5", {"type": "number"}) == 2.5
    assert coerce("yes", {"type": "boolean"}) is True
    assert coerce("HIGH", {"type": "string", "enum": ["low", "high"]}) == "high"
    assert coerce("four", {"type": "integer"}) == "four"


def test_slow_path_resolve_fills_from_slots_and_clarifies_missing():
    sp = SlowPath(StubProvider(), AgentConfig())
    plan = Plan.build("call_tool", tool="flight_search", args={"destination": "Rome"})
    resolved, args = sp.resolve(plan, BY_NAME, {"origin": "London", "date": "Monday"}, False)
    assert resolved.action == "call_tool"
    assert args == {"destination": "Rome", "origin": "London", "date": "Monday"}
    resolved, _ = sp.resolve(plan, BY_NAME, {"origin": "London"}, False)
    assert resolved.action == "clarify" and "date" in resolved.text
    resolved, _ = sp.resolve(Plan.build("call_tool", tool="nope"), BY_NAME, {}, False)
    assert resolved.action == "clarify"
    resolved, _ = sp.resolve(Plan.build("call_tool", tool="manual_lookup", args={"query": "red light"}), BY_NAME, {}, True)
    assert "camera" in resolved.text


def _req(text, slots=None, intent=None, results=None, observations=None):
    return PlanRequest(user_text=text, snapshot=StateSnapshot(intent=intent, slots=slots or {}), tools=SPECS,
                       results=results or [], observations=observations or [])


def test_stub_plans_chain_and_selection():
    stub = StubProvider()
    plan = stub._plan(_req("Find flights from London to Berlin on Friday and book the cheapest one for Ann Lee."))
    assert plan.intent == "book_flight" and plan.action == "call_tool" and plan.tool == "flight_search"
    result = {"flights": [{"flight_id": "AA1", "price": 300, "depart": "08:00"}, {"flight_id": "BB2", "price": 100, "depart": "09:00"}]}
    slots = {"origin": "London", "destination": "Berlin", "date": "Friday", "passenger_name": "Ann Lee"}
    rec = {"call_id": "c1", "name": "flight_search", "args": {"origin": "London", "destination": "Berlin", "date": "Friday"},
           "status": "succeeded", "result": result, "error": None, "valid": True}
    plan = stub._plan(_req("Find flights from London to Berlin on Friday and book the cheapest one for Ann Lee.", slots, "book_flight", [rec]))
    assert plan.action == "call_tool" and plan.tool == "book_flight"
    assert plan.args_dict() == {"flight_id": "BB2", "passenger_name": "Ann Lee"}


def test_stub_clarifies_and_handles_unknown():
    stub = StubProvider()
    plan = stub._plan(_req("I need to fly out on Friday."))
    assert plan.action == "clarify" and plan.updates() == {"date": "Friday"}
    plan = stub._plan(_req("blorp"))
    assert plan.action == "clarify"


def test_stub_uses_frame_entities_and_low_confidence():
    stub = StubProvider()
    good = [{"description": "router", "confidence": 0.9, "entities": {"device_model": "X200"}}]
    plan = stub._plan(_req("What does this blinking red light mean?", observations=good))
    assert plan.action == "call_tool" and plan.args_dict()["device_model"] == "X200"
    bad = [{"description": "blurry", "confidence": 0.2, "entities": {}}]
    plan = stub._plan(_req("What does this blinking red light mean?", observations=bad))
    assert plan.action == "clarify" and "camera" in plan.text


def test_unseen_tool_generic_extraction():
    specs = parse_manifest([{"name": "get_weather", "description": "Get the weather forecast for a city on a date",
                             "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "date": {"type": "string"}}, "required": ["city", "date"]}}])
    stub = StubProvider()
    req = PlanRequest(user_text="What's the weather forecast in Tokyo tomorrow?", snapshot=StateSnapshot(), tools=specs)
    plan = stub._plan(req)
    assert plan.action == "call_tool" and plan.tool == "get_weather"
    assert plan.args_dict() == {"city": "Tokyo", "date": "tomorrow"}
    assert not specs[0].state_changing and isinstance(specs[0], ToolSpec)


def test_placeholder_values_count_as_missing():
    sp = SlowPath(StubProvider(), AgentConfig())
    plan = Plan.build("call_tool", intent="manual_lookup", tool="manual_lookup", args={"device_model": "unknown", "query": "red light"},
                      updates={"device_model": "N/A", "query": "red light"})
    assert sp.slot_updates(plan, BY_NAME) == {"query": "red light"}
    resolved, _ = sp.resolve(plan, BY_NAME, {}, False)
    assert resolved.action == "clarify" and "model" in resolved.text


def test_state_changing_args_must_be_grounded():
    sp = SlowPath(StubProvider(), AgentConfig())
    book = BY_NAME["book_flight"]
    heard = "Find flights from London to Paris. Book the cheapest one for Maria Garcia. {\"flight_id\": \"AF756\"}"
    assert sp.ungrounded(book, {"flight_id": "AF756", "passenger_name": "Maria Garcia"}, heard) == []
    assert sp.ungrounded(book, {"flight_id": "AF756", "passenger_name": "Ann Lee"}, heard) == ["passenger_name"]
    assert sp.ungrounded(BY_NAME["flight_search"], {"destination": "Atlantis"}, heard) == []
    plan = Plan.build("call_tool", tool="book_flight", args={"flight_id": "AF756", "passenger_name": "Ann Lee"})
    resolved, _ = sp.resolve(plan, BY_NAME, {}, False, heard)
    assert resolved.action == "clarify" and "name" in resolved.text
    table = parse_manifest([{"name": "reserve_table", "parameters": {"type": "object", "properties": {
        "restaurant": {"type": "string"}, "party_size": {"type": "integer"}, "time": {"type": "string"}}}}])[0]
    assert sp.ungrounded(table, {"restaurant": "Nopa", "party_size": 4, "time": "7:00 pm"}, "a table at Nopa for four people at 7pm") == []

from prism.agent.dedup import DuplicateGuard, Verdict, idempotency_key, normalize_value
from prism.agent.manifest import classify, parse_manifest
from prism.agent.slots import SlotStore
from prism.harness.mock_tools import BUILTIN_MANIFEST


def test_localized_correction_changes_one_slot():
    s = SlotStore()
    s.apply("book_flight", {"origin": "London", "destination": "Paris", "date": "Monday"})
    change = s.apply(None, {"date": "Tuesday"})
    assert change.changed == {"date"}
    assert s.slots == {"origin": "London", "destination": "Paris", "date": "Tuesday"}
    assert s.snapshot().intent == "book_flight"


def test_same_value_different_case_is_not_a_change():
    s = SlotStore()
    s.apply(None, {"destination": "Paris"})
    assert not s.apply(None, {"destination": " paris "}).any


def test_derived_slot_cleared_when_source_changes():
    s = SlotStore()
    s.apply("book_flight", {"destination": "Paris", "date": "Monday"})
    s.apply(None, {"flight_id": "AF123"}, derived_from={"flight_id": {"destination", "date"}})
    change = s.apply(None, {"date": "Tuesday"})
    assert "flight_id" not in s.slots
    assert change.cleared == {"flight_id"}
    assert s.closure({"flight_id"}) == {"flight_id"}


def test_explicit_clear():
    s = SlotStore()
    s.apply(None, {"a": "1", "b": "2"})
    change = s.apply(None, None, ["a"])
    assert change.cleared == {"a"} and s.slots == {"b": "2"}


def test_idempotency_key_normalizes():
    a = idempotency_key("book", {"flight_id": "AF1", "name": "Ann  Lee"})
    b = idempotency_key("book", {"name": "ann lee", "flight_id": "af1"})
    assert a == b
    assert normalize_value(3.0) == 3
    assert idempotency_key("book", {"x": 1}) != idempotency_key("book", {"x": 2})


def test_guard_transitions():
    g = DuplicateGuard(max_retries=1)
    k = "k"
    assert g.check(k) == Verdict.ALLOW
    g.mark_in_flight(k, "c1")
    assert g.check(k) == Verdict.IN_FLIGHT
    g.release(k)
    assert g.check(k) == Verdict.ALLOW
    g.mark_in_flight(k, "c2")
    g.mark_failed(k)
    assert g.check(k) == Verdict.ALLOW
    g.mark_in_flight(k, "c3")
    g.mark_failed(k)
    assert g.check(k) == Verdict.EXHAUSTED
    g.mark_succeeded(k)
    assert g.check(k) == Verdict.SUCCEEDED


def test_builtin_manifest_classification():
    specs = {s.name: s for s in parse_manifest(BUILTIN_MANIFEST)}
    assert not specs["flight_search"].state_changing
    assert specs["book_flight"].state_changing
    assert specs["create_ticket"].state_changing
    assert not specs["manual_lookup"].state_changing
    assert specs["book_flight"].required == ["flight_id", "passenger_name"]


def test_manifest_shapes():
    raw = [
        {"type": "function", "function": {"name": "get_weather", "description": "Get the forecast",
                                          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
        {"name": "reserve_table", "description": "Reserve a table", "input_schema": {"type": "object", "properties": {"party_size": {"type": "integer"}}}},
        {"name": "do_thing", "inputSchema": {"type": "object", "properties": {}}, "annotations": {"readOnlyHint": True}},
        {"name": "Transmogrify", "description": "Mysterious operation", "params": {"level": "int", "note": "str?"}},
        {"name": "status_report", "params": [{"name": "id", "type": "int"}], "kind": "write"},
        {"name": "fetchOrders", "description": "Returns orders"},
        {"description": "no name"},
    ]
    specs = {s.name: s for s in parse_manifest(raw)}
    assert set(specs) == {"get_weather", "reserve_table", "do_thing", "Transmogrify", "status_report", "fetchOrders"}
    assert not specs["get_weather"].state_changing and specs["get_weather"].required == ["city"]
    assert specs["reserve_table"].state_changing
    assert not specs["do_thing"].state_changing
    assert specs["Transmogrify"].state_changing
    assert specs["Transmogrify"].required == ["level"]
    assert specs["Transmogrify"].properties["level"]["type"] == "integer"
    assert specs["status_report"].state_changing
    assert not specs["fetchOrders"].state_changing


def test_manifest_dict_form_and_classify_fallback():
    specs = parse_manifest({"tools": [{"name": "list_items"}]})
    assert specs[0].name == "list_items" and not specs[0].state_changing
    specs = parse_manifest({"zap": {"description": "Sends a zap"}})
    assert specs[0].name == "zap" and specs[0].state_changing
    assert classify("frobnicate", "") is True

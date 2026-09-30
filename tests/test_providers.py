import json
from prism.agent.clock import VirtualClock
from prism.agent.llm import DEFAULT_LOCAL_MODEL, LocalProvider, PlanRequest, extract_json, make_provider
from prism.agent.manifest import parse_manifest
from prism.agent.models import StateSnapshot
from prism.harness.mock_tools import BUILTIN_MANIFEST

PLAN_JSON = json.dumps({
    "intent": "flight_search", "slot_updates": [{"name": "destination", "value": "Rome"}], "clear_slots": [],
    "action": "call_tool", "tool": "flight_search", "args": [{"name": "destination", "value": "Rome"}],
    "text": None, "confidence": 0.9,
})


def request():
    return PlanRequest(user_text="flights to Rome", snapshot=StateSnapshot(), tools=parse_manifest(BUILTIN_MANIFEST))


def test_extract_json_handles_think_and_fences():
    assert json.loads(extract_json("<think>hmm {x}</think>\n```json\n" + PLAN_JSON + "\n```"))["tool"] == "flight_search"
    assert extract_json('noise {"a": {"b": "}"}} trailing') == '{"a": {"b": "}"}}'


async def test_local_provider_format_fallback_and_think_stripping():
    p = LocalProvider(model="qwen3:8b", base_url="http://localhost:1/v1")
    p.bind(VirtualClock())
    seen = []

    def fake_post(payload):
        seen.append(payload)
        if payload.get("response_format", {}).get("type") == "json_schema":
            raise ValueError("HTTP 400: response_format json_schema not supported")
        return {"choices": [{"message": {"content": "<think>reasoning</think>" + PLAN_JSON}}]}

    p._post = fake_post
    plan = await p.plan(request())
    assert plan.tool == "flight_search"
    assert [s.get("response_format", {}).get("type") for s in seen] == ["json_schema", "json_schema", "json_object"]
    assert [("chat_template_kwargs" in s) for s in seen] == [True, False, True]
    assert p.format_mode == "json_object" and p.send_template_kwargs


async def test_local_provider_without_vision_model_returns_low_confidence():
    p = LocalProvider(model="qwen3:8b")
    p.vision_model = None
    p.bind(VirtualClock())
    obs = await p.describe_frame("missing.png")
    assert obs.confidence == 0.0


def test_make_provider_kinds(monkeypatch):
    for var in ("PRISM_LOCAL_MODEL", "PRISM_LOCAL_VISION_MODEL", "PRISM_LOCAL_URL"):
        monkeypatch.delenv(var, raising=False)
    assert type(make_provider("stub")).__name__ == "StubProvider"
    local = make_provider("local")
    assert isinstance(local, LocalProvider) and local.vision is None
    assert local.planner_model == DEFAULT_LOCAL_MODEL and local.vision_model == DEFAULT_LOCAL_MODEL
    monkeypatch.setenv("PRISM_LOCAL_VISION_MODEL", "none")
    assert make_provider("local").vision_model is None



async def test_local_provider_drops_template_kwargs_when_rejected():
    p = LocalProvider(model="qwen3-vl", base_url="http://localhost:1/v1")
    p.bind(VirtualClock())
    seen = []

    def fake_post(payload):
        seen.append(payload)
        if "chat_template_kwargs" in payload:
            raise ValueError("HTTP 422: unknown field chat_template_kwargs")
        return {"choices": [{"message": {"content": PLAN_JSON}}]}

    p._post = fake_post
    plan = await p.plan(request())
    assert plan.tool == "flight_search"
    assert [("chat_template_kwargs" in s, s["response_format"]["type"]) for s in seen] == [(True, "json_schema"), (False, "json_schema")]
    assert p.format_mode == "json_schema" and not p.send_template_kwargs

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from . import nlu
from .clock import Clock, RealClock
from .dedup import normalize_value
from .models import StateSnapshot, ToolSpec

log = logging.getLogger(__name__)


DEFAULT_LOCAL_URL = "http://127.0.0.1:8081/v1"
DEFAULT_LOCAL_MODEL = "mlx-community/Qwen3-VL-8B-Instruct-4bit"


class NameValue(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    value: str


class Plan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    intent: str | None = None
    slot_updates: list[NameValue] = Field(default_factory=list)
    clear_slots: list[str] = Field(default_factory=list)
    action: Literal["call_tool", "clarify", "respond", "wait"] = "wait"
    tool: str | None = None
    args: list[NameValue] = Field(default_factory=list)
    text: str | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    def updates(self) -> dict[str, Any]:
        return {nv.name: nv.value for nv in self.slot_updates}

    def args_dict(self) -> dict[str, Any]:
        return {nv.name: nv.value for nv in self.args}

    @classmethod
    def build(cls, action: str, *, intent: str | None = None, updates: dict[str, Any] | None = None,
              tool: str | None = None, args: dict[str, Any] | None = None, text: str | None = None,
              confidence: float = 1.0, clears: list[str] | None = None) -> "Plan":
        return cls(
            intent=intent,
            slot_updates=[NameValue(name=k, value=str(v)) for k, v in (updates or {}).items()],
            clear_slots=clears or [],
            action=action,  # type: ignore[arg-type]
            tool=tool,
            args=[NameValue(name=k, value=str(v)) for k, v in (args or {}).items()],
            text=text,
            confidence=confidence,
        )


class FrameObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    description: str
    confidence: float = Field(ge=0.0, le=1.0)
    entities: list[NameValue] = Field(default_factory=list)

    def entities_dict(self) -> dict[str, str]:
        return {nv.name: nv.value for nv in self.entities}


@dataclass
class PlanRequest:
    user_text: str
    snapshot: StateSnapshot
    tools: list[ToolSpec]
    turns: list[dict[str, str]] = field(default_factory=list)
    results: list[dict[str, Any]] = field(default_factory=list)
    in_flight: list[dict[str, Any]] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    partial: bool = False
    frame_threshold: float = 0.6

    def to_json(self) -> str:
        return json.dumps(
            {
                "new_user_speech": self.user_text,
                "user_still_speaking": self.partial,
                "state_snapshot": self.snapshot.model_dump(),
                "tools": [
                    {"name": t.name, "description": t.description, "parameters": t.parameters, "state_changing": t.state_changing}
                    for t in self.tools
                ],
                "conversation": self.turns,
                "tool_results": self.results,
                "calls_in_flight": self.in_flight,
                "camera_observations": self.observations,
                "low_confidence_threshold": self.frame_threshold,
                "notes": self.notes,
            },
            default=str,
        )


class LLMProvider(ABC):
    parallel_calls = True

    def __init__(self) -> None:
        self.clock: Clock = RealClock()

    def bind(self, clock: Clock) -> None:
        self.clock = clock

    async def warmup(self) -> None:
        return None

    @abstractmethod
    async def plan(self, request: PlanRequest) -> Plan: ...

    @abstractmethod
    async def describe_frame(self, path: str, context: str = "") -> FrameObservation: ...


def options_of(result: Any) -> list[dict[str, Any]]:
    if isinstance(result, list) and result and all(isinstance(x, dict) for x in result):
        return result
    if isinstance(result, dict):
        for value in result.values():
            if isinstance(value, list) and value and all(isinstance(x, dict) for x in value):
                return value
    return []


def _key_like(opt: dict[str, Any], words: tuple[str, ...]) -> str | None:
    return next((k for k in opt if any(w in k.lower() for w in words)), None)


def id_key(opt: dict[str, Any], param: str | None = None) -> str:
    if param and param in opt:
        return param
    return next((k for k in opt if k.lower() == "id" or k.lower().endswith("_id") or k.lower().endswith("id")), next(iter(opt)))


def choose_option(options: list[dict[str, Any]], selection: tuple[str, int]) -> dict[str, Any] | None:
    kind, idx = selection
    if not options:
        return None
    if kind == "cheapest":
        k = _key_like(options[0], ("price", "cost", "fare", "amount"))
        return min(options, key=lambda o: o.get(k, 0)) if k else options[0]
    if kind in {"earliest", "latest"}:
        k = _key_like(options[0], ("depart", "time", "start", "date"))
        if k:
            return (min if kind == "earliest" else max)(options, key=lambda o: str(o.get(k, "")))
        return options[0] if kind == "earliest" else options[-1]
    if kind == "last":
        return options[-1]
    return options[idx] if 0 <= idx < len(options) else None


def format_option(opt: dict[str, Any]) -> str:
    ik = id_key(opt)
    parts = [str(opt[ik])]
    tk = _key_like(opt, ("depart", "time", "start"))
    pk = _key_like(opt, ("price", "cost", "fare", "amount"))
    if tk:
        parts.append(f"at {opt[tk]}")
    if pk:
        parts.append(f"for ${opt[pk]}" if isinstance(opt[pk], (int, float)) else f"for {opt[pk]}")
    if not tk and not pk:
        extras = [f"{nlu.humanize(k)} {v}" for k, v in opt.items() if k != ik and isinstance(v, (str, int, float))][:2]
        parts.extend(extras)
    return " ".join(parts)


def _past(verb: str) -> str:
    irregular = {"send": "sent", "make": "made", "put": "put", "set": "set", "pay": "paid", "buy": "bought", "file": "filed"}
    if verb in irregular:
        return irregular[verb]
    if verb.endswith("e"):
        return verb + "d"
    return verb + "ed"


def summarize_result(spec: ToolSpec, args: dict[str, Any], result: Any) -> str:
    words = nlu.humanize(spec.name).split()
    if spec.state_changing:
        verb = words[0] if words else "complete"
        obj = " ".join(words[1:]) or "request"
        refs = []
        if isinstance(result, dict):
            for k, v in result.items():
                kl = k.lower()
                if isinstance(v, (str, int)) and (kl.endswith("id") or "confirmation" in kl or "reference" in kl or kl.endswith("number")) and k not in args:
                    refs.append(f"Your {nlu.humanize(k).replace(' id', ' ID')} is {v}.")
        lead = f"Done, your {obj} is {_past(verb)}"
        detail = ""
        if isinstance(args, dict):
            shown = [str(v) for k, v in args.items() if isinstance(v, (str, int)) and len(str(v)) < 40][:2]
            if shown:
                detail = " (" + ", ".join(shown) + ")"
        return (lead + detail + ". " + " ".join(refs[:2])).strip()
    opts = options_of(result)
    if opts:
        listing = "; ".join(format_option(o) for o in opts[:3])
        return f"I found {len(opts)} options: {listing}. Which one would you like?"
    if isinstance(result, dict):
        if isinstance(result.get("text"), str):
            prefix = "" if result.get("found", True) else "I couldn't find an exact match. "
            section = f" (from the {result['section']} section)" if result.get("section") else ""
            return f"{prefix}{result['text']}{section}"
        pairs = [f"{nlu.humanize(k)}: {v}" for k, v in result.items() if isinstance(v, (str, int, float, bool))][:4]
        if pairs:
            return "Here's what I found: " + ", ".join(pairs) + "."
    return f"Here's what I found: {json.dumps(result)[:200]}"


def failure_text(spec: ToolSpec, error: str | None) -> str:
    what = nlu.humanize(spec.name)
    return f"Sorry, I couldn't complete the {what} ({error or 'the service did not respond'}). Would you like me to try something else?"


def ask_for(param: str, spec: ToolSpec | None = None) -> str:
    schema = spec.properties.get(param, {}) if spec else {}
    kind = nlu.param_kind(param, schema)
    if kind == "destination":
        return "Where would you like to go?"
    if kind == "origin":
        return "Where will you be leaving from?"
    if kind == "date":
        return "What date works for you?"
    if kind == "time":
        return "What time would you like?"
    if kind == "person":
        return "What name should I put it under?"
    if kind == "number":
        return f"How many {nlu.humanize(param).split()[-1] if 'size' not in param else 'people'}?"
    if kind == "model":
        return "Which device model is it? It's usually printed on a label."
    if kind == "enum" and schema.get("enum"):
        return f"Which {nlu.humanize(param)}: " + ", ".join(map(str, schema["enum"])) + "?"
    return f"Could you tell me the {nlu.humanize(param)}?"


class StubProvider(LLMProvider):
    def __init__(self, frame_sidecar: bool = True) -> None:
        super().__init__()
        self.frame_sidecar = frame_sidecar

    async def plan(self, request: PlanRequest) -> Plan:
        await asyncio.sleep(0)
        return self._plan(request)

    async def describe_frame(self, path: str, context: str = "") -> FrameObservation:
        await asyncio.sleep(0)
        for candidate in (Path(path + ".json"), Path(path).with_suffix(".json")):
            if self.frame_sidecar and candidate.exists():
                data = json.loads(candidate.read_text())
                ents = data.get("entities", {})
                if isinstance(ents, dict):
                    data["entities"] = [{"name": k, "value": str(v)} for k, v in ents.items()]
                return FrameObservation.model_validate(data)
        return FrameObservation(description="an unclear image", confidence=0.2)

    @staticmethod
    def _producer(goal: ToolSpec, param: str, specs: dict[str, ToolSpec]) -> ToolSpec | None:
        if nlu.param_kind(param, goal.properties.get(param, {})) != "identifier":
            return None
        desc = str(goal.properties.get(param, {}).get("description", ""))
        want = nlu.content_tokens(param + " " + desc) - {"id", "identifier"}
        best, best_score = None, 0.0
        for spec in specs.values():
            if spec.name == goal.name or spec.state_changing:
                continue
            score = len(want & nlu.content_tokens(spec.name + " " + spec.description))
            if spec.name in desc:
                score += 5
            if score > best_score:
                best, best_score = spec, score
        return best

    @staticmethod
    def _latest(request: PlanRequest, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        want = normalize_value(args)
        for rec in reversed(request.in_flight + request.results):
            if rec["name"] == name and normalize_value(rec["args"]) == want:
                return rec
        return None

    def _plan(self, request: PlanRequest) -> Plan:
        text = request.user_text or ""
        specs = {s.name: s for s in request.tools}
        slots = dict(request.snapshot.slots)
        intent = request.snapshot.intent if request.snapshot.intent in specs else None
        entities: dict[str, str] = {}
        low_conf = None
        for obs in request.observations:
            if obs.get("confidence", 0) >= request.frame_threshold:
                entities.update(obs.get("entities") or {})
                low_conf = None
            else:
                low_conf = obs
        guess = nlu.guess_tool(text, request.tools)
        goal_name = intent
        if guess and guess.name != intent:
            keep = intent is not None and any(
                self._producer(specs[intent], p, specs) is specs[guess.name] for p in specs[intent].required
            )
            if not keep:
                goal_name = guess.name
        if goal_name is None:
            if not specs:
                return Plan.build("respond", text="Sorry, I don't have any tools available for that right now.")
            options = ", ".join(nlu.humanize(n) for n in list(specs)[:4])
            return Plan.build("clarify", text=f"Sorry, I'm not sure what you'd like me to do. I can help with {options}. What do you need?", confidence=0.5)
        goal = specs[goal_name]
        intent_changed = goal_name != intent
        updates = self._extract(text, goal, specs, slots, entities, intent_changed, request)
        merged = {**slots, **updates}
        confidence = min(0.95, 0.5 + 0.1 * (guess.score if guess else 2.0))
        missing = [p for p in goal.required if merged.get(p) in (None, "")]
        if not missing:
            args = {p: merged[p] for p in goal.properties if p in merged}
            rec = self._latest(request, goal.name, args)
            if rec and rec["status"] == "succeeded":
                return Plan.build("respond", intent=goal_name, updates=updates, text=summarize_result(goal, args, rec["result"]), confidence=confidence)
            if rec and rec["status"] in {"failed", "timed_out"}:
                return Plan.build("respond", intent=goal_name, updates=updates, text=failure_text(goal, rec.get("error")), confidence=confidence)
            if rec and rec["status"] == "pending":
                return Plan.build("wait", intent=goal_name, updates=updates, confidence=confidence)
            return Plan.build("call_tool", intent=goal_name, updates=updates, tool=goal.name, args=args, confidence=confidence)
        for param in missing:
            prod = self._producer(goal, param, specs)
            if prod is None:
                continue
            prod_missing = [q for q in prod.required if merged.get(q) in (None, "")]
            if prod_missing:
                return Plan.build("clarify", intent=goal_name, updates=updates, text=ask_for(prod_missing[0], prod), confidence=confidence)
            pargs = {q: merged[q] for q in prod.properties if q in merged}
            rec = self._latest(request, prod.name, pargs)
            if rec and rec["status"] == "succeeded":
                opts = options_of(rec["result"])
                if not opts:
                    return Plan.build("respond", intent=goal_name, updates=updates, text=f"I couldn't find any options for that. {summarize_result(prod, pargs, rec['result'])}", confidence=confidence)
                listing = "; ".join(format_option(o) for o in opts[:3])
                return Plan.build("clarify", intent=goal_name, updates=updates, text=f"I found {len(opts)} options: {listing}. Which one would you like?", confidence=confidence)
            if rec and rec["status"] in {"failed", "timed_out"}:
                return Plan.build("respond", intent=goal_name, updates=updates, text=failure_text(prod, rec.get("error")), confidence=confidence)
            if rec and rec["status"] == "pending":
                return Plan.build("wait", intent=goal_name, updates=updates, confidence=confidence)
            return Plan.build("call_tool", intent=goal_name, updates=updates, tool=prod.name, args=pargs, confidence=confidence)
        param = missing[0]
        if low_conf is not None and nlu.param_kind(param, goal.properties.get(param, {})) == "model":
            return Plan.build("clarify", intent=goal_name, updates=updates,
                              text="I can't make out the device clearly from the camera. Could you read me the model name on its label?",
                              confidence=confidence)
        return Plan.build("clarify", intent=goal_name, updates=updates, text=ask_for(param, goal), confidence=confidence)

    def _extract(self, text: str, goal: ToolSpec, specs: dict[str, ToolSpec], slots: dict[str, Any],
                 entities: dict[str, str], intent_changed: bool, request: PlanRequest) -> dict[str, Any]:
        params: dict[str, dict[str, Any]] = {}
        producers = [p for p in (self._producer(goal, q, specs) for q in goal.required) if p]
        for spec in [goal, *producers]:
            for name, schema in spec.properties.items():
                params.setdefault(name, schema if isinstance(schema, dict) else {})
        out: dict[str, Any] = {}
        cities = nlu.find_cities(text)
        correction = nlu.has_correction(text)
        for name, schema in params.items():
            kind = nlu.param_kind(name, schema)
            value: Any = None
            if kind == "free_text":
                if (intent_changed or name not in slots) and goal.properties.get(name) is not None:
                    value = text.strip() or None
            elif kind in {"origin", "destination"}:
                value = nlu.extract_param(name, schema, text)
                if value is None:
                    bare = nlu.last_positive([c for c in cities if c.role in (None, "location")])
                    negated = [c.value for c in cities if c.negated]
                    origin_key = next((k for k in params if nlu.param_kind(k, params[k]) == "origin"), None)
                    target = "origin" if origin_key and slots.get(origin_key) in negated else "destination"
                    if bare and target == kind and (correction or name not in slots):
                        value = bare.value
            elif kind == "identifier":
                value = nlu.extract_param(name, schema, text)
                if value is None:
                    value = self._select(text, name, goal, specs, {**slots, **out}, request)
            else:
                value = nlu.extract_param(name, schema, text, entities)
            if value is None and name in entities:
                value = entities[name]
            if value is not None and normalize_value(value) != normalize_value(slots.get(name)):
                out[name] = value
        return out

    def _select(self, text: str, param: str, goal: ToolSpec, specs: dict[str, ToolSpec],
                merged: dict[str, Any], request: PlanRequest) -> Any:
        selection = nlu.find_selection(text)
        if selection is None:
            return None
        prod = self._producer(goal, param, specs)
        if prod is None:
            return None
        pargs = {q: merged[q] for q in prod.properties if q in merged}
        rec = self._latest(request, prod.name, pargs)
        if rec is None:
            rec = next((r for r in reversed(request.results) if r["name"] == prod.name and r["status"] == "succeeded" and r.get("valid")), None)
        if rec is None or rec["status"] != "succeeded":
            return None
        opt = choose_option(options_of(rec["result"]), selection)
        return opt.get(id_key(opt, param)) if opt else None


PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "intent": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "slot_updates": {
            "type": "array",
            "items": {"type": "object", "additionalProperties": False,
                      "properties": {"name": {"type": "string"}, "value": {"type": "string"}},
                      "required": ["name", "value"]},
        },
        "clear_slots": {"type": "array", "items": {"type": "string"}},
        "action": {"type": "string", "enum": ["call_tool", "clarify", "respond", "wait"]},
        "tool": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "args": {
            "type": "array",
            "items": {"type": "object", "additionalProperties": False,
                      "properties": {"name": {"type": "string"}, "value": {"type": "string"}},
                      "required": ["name", "value"]},
        },
        "text": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "confidence": {"type": "number"},
    },
    "required": ["intent", "slot_updates", "clear_slots", "action", "tool", "args", "text", "confidence"],
}

FRAME_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "description": {"type": "string"},
        "confidence": {"type": "number"},
        "entities": PLAN_SCHEMA["properties"]["slot_updates"],
    },
    "required": ["description", "confidence", "entities"],
}

PLANNER_SYSTEM = """You are the planning module of a real-time, full-duplex voice assistant. A separate fast path has already acknowledged the user; your job is to decide the single next step and keep the state snapshot accurate.

You receive one JSON object describing the session: new_user_speech (everything the user said since your last decision; it can span several turns when the user kept talking while you were thinking), the current state snapshot (intent and slots), the tool manifest (with a state_changing flag per tool), the conversation so far, finished tool results, calls still in flight, camera observations and notes. Reply with one JSON object that matches the output schema.

Fields:
- intent: the name of the tool that fulfils the user's end goal. Choose a state-changing tool (booking, creating, reserving) only when the user explicitly asks for that action; a request to find, search or check is a read-only goal. When the user asks to book, the booking tool is the intent even if a search must run first. Use null only when no tool applies.
- slot_updates: every slot that new_user_speech adds or changes and that differs from the state snapshot. Slot names are tool parameter names. A correction such as "actually Tuesday, not Monday" updates only that slot. Keep values short and exactly as the user said them. Never invent a value the user did not say. When the user picks an option from a tool result, set the identifier slot to that option's identifier.
- clear_slots: slots the user explicitly withdrew.
- action:
  - call_tool when every required parameter of the next tool is known from the user's words, tool results or camera observations. Never fill a parameter with a guess or a placeholder; ask instead. If the goal tool needs an identifier that another read-only tool returns, call that tool first.
  - clarify when a required value is missing or ambiguous, or when the camera observation relevant to the request has confidence below low_confidence_threshold. Ask one short question.
  - respond when the goal is satisfied or cannot be satisfied. Ground every fact in tool_results. Never say a state-changing action happened unless a succeeded result for it is present.
  - wait when a call in flight will provide what is needed.
- tool and args: the tool to call and its arguments (values as strings), only for call_tool. Never repeat a state-changing call whose identical arguments already succeeded or are in flight.
- text: the spoken question or answer. Natural, spoken style, at most two short sentences, no markdown, no lists.
- confidence: 0 to 1, how sure you are about the intent and arguments.

When user_still_speaking is true the speech is partial: propose call_tool only for a read-only tool whose arguments are already unambiguous, otherwise choose wait."""

FRAME_PROMPT = """Describe this camera frame for a voice assistant that helps troubleshoot devices. Report what device is visible, any readable model name or number, and the state of lights, screens or error codes. Put readable identifiers in entities (for example name "device_model", value "X200"). Set confidence between 0 and 1 for how clearly the relevant details can be read; use a low value if the image is blurry, dark or the model cannot be read."""


def _schema_hint(schema: dict[str, Any]) -> str:
    return "Reply with only a JSON object matching this JSON schema, with no other text:\n" + json.dumps(schema)


def extract_json(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
    start = text.find("{")
    if start < 0:
        return text
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return text[start:]


class _StructuredProvider(LLMProvider):
    vision: "LLMProvider | None" = None

    async def _raw(self, model: str, system: str, content: list[dict[str, Any]], schema: dict[str, Any]) -> str | None:
        raise NotImplementedError

    def _transient(self) -> tuple[type[BaseException], ...]:
        return ()

    async def _validated(self, model: str, system: str, content: list[dict[str, Any]], schema: dict[str, Any], target: type[BaseModel]) -> BaseModel | None:
        error = ""
        for attempt in range(2):
            body = list(content)
            if error:
                body.append({"type": "text", "text": f"Your previous reply was invalid ({error}). Reply with one JSON object that matches the schema exactly."})
            try:
                raw = await self._raw(model, system, body, schema)
            except self._transient() as exc:
                log.warning("transient LLM error on attempt %d: %s", attempt + 1, exc)
                error = "transient error"
                continue
            except Exception as exc:
                log.error("LLM call failed: %s", exc)
                return None
            if raw is None:
                return None
            try:
                return target.model_validate_json(extract_json(raw))
            except ValidationError as exc:
                error = str(exc).splitlines()[0][:300]
                log.warning("invalid structured output on attempt %d: %s", attempt + 1, error)
        return None

    async def plan(self, request: PlanRequest) -> Plan:
        content = [{"type": "text", "text": request.to_json()}]
        plan = await self._validated(self.planner_model, PLANNER_SYSTEM, content, PLAN_SCHEMA, Plan)
        if isinstance(plan, Plan):
            return plan
        if request.partial:
            return Plan.build("wait", confidence=0.0)
        return Plan.build("clarify", text="Sorry, could you say that again?", confidence=0.0)

    async def describe_frame(self, path: str, context: str = "") -> FrameObservation:
        if self.vision is not None:
            return await self.vision.describe_frame(path, context)
        if not self.vision_model:
            return FrameObservation(description="No vision model is configured.", confidence=0.0)
        data = await self.clock.run_blocking(Path(path).read_bytes)
        media = "image/png" if path.lower().endswith(".png") else "image/jpeg"
        content = [
            {"type": "image", "source": {"type": "base64", "media_type": media, "data": base64.standard_b64encode(data).decode()}},
            {"type": "text", "text": FRAME_PROMPT + (f"\nThe user said: {context}" if context else "")},
        ]
        obs = await self._validated(self.vision_model, "You describe camera frames precisely and never guess unreadable text.", content, FRAME_SCHEMA, FrameObservation)
        if isinstance(obs, FrameObservation):
            return obs
        return FrameObservation(description="The image could not be analysed.", confidence=0.0)

    planner_model: str = ""
    vision_model: str | None = None


class LocalProvider(_StructuredProvider):
    parallel_calls = False

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        vision_model: str | None = None,
        vision: LLMProvider | None = None,
        timeout: float = 60.0,
        max_tokens: int = 1200,
        api_key: str | None = None,
    ) -> None:
        super().__init__()
        self.planner_model = model or os.environ.get("PRISM_LOCAL_MODEL", DEFAULT_LOCAL_MODEL)
        self.base_url = (base_url or os.environ.get("PRISM_LOCAL_URL", DEFAULT_LOCAL_URL)).rstrip("/")
        configured = vision_model if vision_model is not None else os.environ.get("PRISM_LOCAL_VISION_MODEL", self.planner_model)
        self.vision_model = None if configured.strip().lower() in {"", "none", "off"} else configured
        self.vision = vision
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.api_key = api_key or os.environ.get("PRISM_LOCAL_API_KEY", "local")
        self.format_mode = os.environ.get("PRISM_LOCAL_FORMAT", "json_schema")
        self.disable_thinking = os.environ.get("PRISM_LOCAL_THINKING", "0") != "1"
        self.send_template_kwargs = self.disable_thinking

    def bind(self, clock: Clock) -> None:
        super().bind(clock)
        if self.vision is not None:
            self.vision.bind(clock)

    def _transient(self) -> tuple[type[BaseException], ...]:
        import urllib.error

        return (urllib.error.URLError, TimeoutError, ConnectionError)

    @staticmethod
    def _to_openai(content: list[dict[str, Any]]) -> list[dict[str, Any]]:
        parts: list[dict[str, Any]] = []
        for block in content:
            if block.get("type") == "image":
                src = block["source"]
                parts.append({"type": "image_url", "image_url": {"url": f"data:{src['media_type']};base64,{src['data']}"}})
            else:
                parts.append({"type": "text", "text": block.get("text", "")})
        return parts

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        import urllib.error
        import urllib.request

        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:300]
            raise ValueError(f"HTTP {exc.code}: {detail}") from exc

    async def _raw(self, model: str, system: str, content: list[dict[str, Any]], schema: dict[str, Any]) -> str | None:
        user = self._to_openai(content)
        if self.disable_thinking:
            user.append({"type": "text", "text": "/no_think"})
        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": self.max_tokens,
            "temperature": 0.1,
            "messages": [
                {"role": "system", "content": system + "\n\n" + _schema_hint(schema)},
                {"role": "user", "content": user},
            ],
        }
        modes = {"json_schema": ["json_schema", "json_object", "none"], "json_object": ["json_object", "none"]}.get(self.format_mode, ["none"])
        last: Exception | None = None
        attempts = [m for mode in modes for m in ((mode, True), (mode, False))]
        suspect_kwargs = False
        for mode, with_kwargs in attempts:
            if with_kwargs and not self.send_template_kwargs:
                continue
            body = dict(payload)
            if with_kwargs:
                body["chat_template_kwargs"] = {"enable_thinking": False}
            if mode == "json_schema":
                body["response_format"] = {"type": "json_schema", "json_schema": {"name": "output", "schema": schema, "strict": True}}
            elif mode == "json_object":
                body["response_format"] = {"type": "json_object"}
            try:
                data = await self.clock.run_blocking(self._post, body)
            except ValueError as exc:
                last = exc
                if "HTTP 4" not in str(exc):
                    raise
                if with_kwargs:
                    log.warning("local server rejected the request; retrying without chat_template_kwargs")
                    self.send_template_kwargs = False
                    suspect_kwargs = True
                    continue
                if suspect_kwargs:
                    self.send_template_kwargs = self.disable_thinking
                    suspect_kwargs = False
                log.warning("local server rejected response_format=%s; trying a simpler mode", mode)
                self.format_mode = modes[min(modes.index(mode) + 1, len(modes) - 1)]
                continue
            suspect_kwargs = False
            choice = (data.get("choices") or [{}])[0]
            return (choice.get("message") or {}).get("content")
        if last:
            raise last
        return None


def make_provider(kind: str = "stub", **kwargs: Any) -> LLMProvider:
    if kind == "local":
        return LocalProvider(**kwargs)
    return StubProvider(**kwargs)

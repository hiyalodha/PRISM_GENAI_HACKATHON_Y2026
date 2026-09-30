from __future__ import annotations

import json
import logging
import re
from typing import Any

from . import nlu
from .config import AgentConfig
from .llm import LLMProvider, Plan, PlanRequest, ask_for
from .models import ToolSpec

log = logging.getLogger(__name__)

PLACEHOLDERS = {"", "unknown", "n/a", "na", "none", "null", "?", "tbd", "unspecified", "not specified", "not provided", "not given", "missing", "<unknown>", "any"}


def is_placeholder(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip().strip(".").lower() in PLACEHOLDERS)


TRIVIAL_TOKENS = {"00", "0", "the", "a", "an", "of"}


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z]+|\d+", text.lower())


LOW_CONFIDENCE_QUESTION = "I can't make out the device clearly from the camera. Could you read me the model name on its label?"


def coerce(value: Any, schema: dict[str, Any]) -> Any:
    typ = schema.get("type")
    if isinstance(typ, list):
        typ = next((t for t in typ if t != "null"), None)
    if not isinstance(value, str):
        return value
    text = value.strip()
    try:
        if typ == "integer":
            return int(float(text))
        if typ == "number":
            number = float(text)
            return int(number) if number.is_integer() else number
        if typ == "boolean":
            return text.lower() in {"true", "yes", "1", "y"}
        if typ in {"array", "object"}:
            return json.loads(text)
    except (ValueError, json.JSONDecodeError):
        return text
    if "enum" in schema:
        for option in schema["enum"]:
            if isinstance(option, str) and option.lower() == text.lower():
                return option
    return text


class SlowPath:
    def __init__(self, provider: LLMProvider, config: AgentConfig) -> None:
        self.provider = provider
        self.config = config

    async def plan(self, request: PlanRequest) -> Plan:
        try:
            return await self.provider.plan(request)
        except Exception as exc:
            log.exception("planner failed: %s", exc)
            if request.partial:
                return Plan.build("wait", confidence=0.0)
            return Plan.build("clarify", text="Sorry, could you say that again?", confidence=0.0)

    @staticmethod
    def schema_for(name: str, specs: dict[str, ToolSpec], prefer: str | None = None) -> dict[str, Any]:
        order = ([specs[prefer]] if prefer in specs else []) + list(specs.values())
        for spec in order:
            schema = spec.properties.get(name)
            if isinstance(schema, dict):
                return schema
        return {}

    def slot_updates(self, plan: Plan, specs: dict[str, ToolSpec]) -> dict[str, Any]:
        return {k: coerce(v, self.schema_for(k, specs, plan.intent)) for k, v in plan.updates().items() if k and not is_placeholder(v)}

    @staticmethod
    def ungrounded(spec: ToolSpec, args: dict[str, Any], grounding: str) -> list[str]:
        if not spec.state_changing:
            return []
        haystack = set(_tokens(grounding))
        haystack |= {str(n) for w, n in nlu.NUMBER_WORDS.items() if w in haystack}
        missing = []
        for key, value in args.items():
            if isinstance(value, bool) or value is None:
                continue
            schema = spec.properties.get(key, {})
            schema = schema if isinstance(schema, dict) else {}
            if value in (schema.get("enum") or []) or nlu.param_kind(key, schema) in {"date", "time"}:
                continue
            needed = [t for t in _tokens(str(value)) if t not in TRIVIAL_TOKENS]
            if needed and not all(t in haystack for t in needed):
                missing.append(key)
        return missing

    def resolve(self, plan: Plan, specs: dict[str, ToolSpec], slots: dict[str, Any], low_confidence: bool,
                grounding: str | None = None) -> tuple[Plan, dict[str, Any]]:
        if plan.action == "call_tool":
            spec = specs.get(plan.tool or "")
            if spec is None:
                log.warning("planner chose unknown tool %r", plan.tool)
                return Plan.build("clarify", intent=plan.intent, text="Sorry, I can't do that directly. Could you put it another way?"), {}
            args: dict[str, Any] = {}
            for k, v in plan.args_dict().items():
                if (spec.properties and k not in spec.properties) or is_placeholder(v):
                    continue
                args[k] = coerce(v, spec.properties.get(k, {}))
            for k in spec.properties:
                if k not in args and slots.get(k) not in (None, ""):
                    args[k] = slots[k]
            if grounding is not None:
                for key in self.ungrounded(spec, args, grounding):
                    log.warning("dropping ungrounded %s=%r for state-changing %s", key, args[key], spec.name)
                    args.pop(key)
            missing = [p for p in spec.required if is_placeholder(args.get(p))]
            if missing:
                kind = nlu.param_kind(missing[0], spec.properties.get(missing[0], {}))
                question = LOW_CONFIDENCE_QUESTION if low_confidence and kind == "model" else ask_for(missing[0], spec)
                return Plan.build("clarify", intent=plan.intent, text=question, confidence=plan.confidence), {}
            return plan, args
        if plan.action == "clarify" and not (plan.text or "").strip():
            return plan.model_copy(update={"text": "Could you tell me a bit more about what you need?"}), {}
        if plan.action == "respond" and not (plan.text or "").strip():
            return plan.model_copy(update={"action": "wait"}), {}
        return plan, {}

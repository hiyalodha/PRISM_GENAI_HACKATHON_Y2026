from __future__ import annotations

import re
from typing import Any

from .models import ToolSpec

STATE_VERBS = {
    "book", "create", "cancel", "delete", "remove", "update", "modify", "edit", "submit",
    "pay", "purchase", "buy", "reserve", "send", "order", "schedule", "set", "add", "confirm",
    "transfer", "post", "write", "register", "open", "close", "file", "assign", "charge",
    "refund", "issue", "rebook", "change", "enroll", "subscribe", "unsubscribe", "start", "stop",
    "put", "insert", "upsert", "save", "place", "make", "commit", "approve", "reject",
}
READ_VERBS = {
    "search", "get", "lookup", "look", "find", "list", "check", "fetch", "query", "read",
    "describe", "show", "view", "retrieve", "estimate", "quote", "compare", "validate",
    "calculate", "compute", "track", "status", "detect", "identify", "recognize", "browse",
    "count", "preview", "info", "inspect", "translate", "summarize", "resolve", "locate",
}
SCHEMA_KEYS = ("parameters", "input_schema", "inputSchema", "params", "args", "arguments", "schema")
TRUE_STRINGS = {"true", "yes", "1", "y"}
READ_KINDS = {"read", "read_only", "readonly", "query", "lookup", "search", "get", "safe", "retrieval"}
WRITE_KINDS = {"write", "mutation", "mutating", "action", "command", "state_changing", "side_effect", "transaction", "destructive"}


def _truthy(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in TRUE_STRINGS
    return None


def _split_name(name: str) -> list[str]:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
    return [t for t in re.split(r"[^a-zA-Z0-9]+", spaced.lower()) if t]


def _unwrap(raw: dict[str, Any]) -> dict[str, Any]:
    if isinstance(raw.get("function"), dict):
        merged = dict(raw)
        merged.update(raw["function"])
        return merged
    if isinstance(raw.get("tool"), dict):
        merged = dict(raw)
        merged.update(raw["tool"])
        return merged
    return raw


def _normalize_schema(schema: Any) -> dict[str, Any]:
    if isinstance(schema, dict) and (schema.get("type") == "object" or "properties" in schema):
        out = dict(schema)
        out.setdefault("type", "object")
        out.setdefault("properties", {})
        if not isinstance(out.get("required"), list):
            out["required"] = [k for k, v in out["properties"].items() if isinstance(v, dict) and v.get("required") is True]
        return out
    props: dict[str, Any] = {}
    required: list[str] = []
    if isinstance(schema, dict):
        for key, value in schema.items():
            if isinstance(value, dict):
                prop = {k: v for k, v in value.items() if k not in {"required", "optional"}}
                if value.get("required") is True or value.get("optional") is False:
                    required.append(key)
                elif "required" not in value and "optional" not in value and "default" not in value:
                    required.append(key)
            elif isinstance(value, str):
                prop = {"type": _json_type(value.rstrip("?"))}
                if not value.endswith("?"):
                    required.append(key)
            else:
                prop = {}
                required.append(key)
            props[key] = prop
    elif isinstance(schema, list):
        for item in schema:
            if isinstance(item, str):
                props[item] = {"type": "string"}
                required.append(item)
            elif isinstance(item, dict) and isinstance(item.get("name"), str):
                prop = {k: v for k, v in item.items() if k not in {"name", "required", "optional"}}
                if "type" in prop:
                    prop["type"] = _json_type(str(prop["type"]))
                props[item["name"]] = prop
                if item.get("required", not item.get("optional", False)):
                    required.append(item["name"])
    return {"type": "object", "properties": props, "required": required}


def _json_type(name: str) -> str:
    n = name.lower()
    if n in {"int", "integer", "long"}:
        return "integer"
    if n in {"float", "double", "number", "decimal"}:
        return "number"
    if n in {"bool", "boolean"}:
        return "boolean"
    if n in {"list", "array"}:
        return "array"
    if n in {"dict", "object", "map"}:
        return "object"
    return "string"


def _explicit_flag(raw: dict[str, Any]) -> bool | None:
    containers = [raw]
    for key in ("annotations", "metadata", "meta", "x-meta", "hints"):
        if isinstance(raw.get(key), dict):
            containers.append(raw[key])
    for c in containers:
        for key in ("state_changing", "stateChanging", "mutating", "mutates", "side_effects", "sideEffects",
                    "has_side_effects", "hasSideEffects", "writes"):
            if key in c:
                v = _truthy(c[key])
                if v is not None:
                    return v
        for key in ("read_only", "readOnly", "readonly", "readOnlyHint", "safe", "pure"):
            if key in c:
                v = _truthy(c[key])
                if v is not None:
                    return not v
        for key in ("destructive", "destructiveHint"):
            if _truthy(c.get(key)) is True:
                return True
        for key in ("kind", "category", "effect", "access", "mode", "tool_type", "action_type"):
            v = c.get(key)
            if isinstance(v, str):
                s = v.strip().lower().replace("-", "_")
                if s in READ_KINDS:
                    return False
                if s in WRITE_KINDS:
                    return True
    return None


def classify(name: str, description: str, raw: dict[str, Any] | None = None) -> bool:
    if raw is not None:
        explicit = _explicit_flag(raw)
        if explicit is not None:
            return explicit
    parts = _split_name(name)
    for part in parts:
        if part in STATE_VERBS:
            return True
        if part in READ_VERBS:
            return False
    desc_words = re.findall(r"[a-z]+", description.lower())[:12]
    for word in desc_words:
        stem = word[:-1] if word.endswith("s") else word
        if stem in STATE_VERBS or word in STATE_VERBS:
            return True
        if stem in READ_VERBS or word in READ_VERBS:
            return False
    return True


def parse_tool(raw: Any) -> ToolSpec | None:
    if isinstance(raw, ToolSpec):
        return raw
    if not isinstance(raw, dict):
        return None
    tool = _unwrap(raw)
    name = tool.get("name") or tool.get("id") or tool.get("tool_name")
    if not isinstance(name, str) or not name.strip():
        return None
    description = tool.get("description") or tool.get("desc") or tool.get("summary") or ""
    if not isinstance(description, str):
        description = str(description)
    schema: Any = None
    for key in SCHEMA_KEYS:
        if key in tool:
            schema = tool[key]
            break
    parameters = _normalize_schema(schema if schema is not None else {})
    return ToolSpec(
        name=name.strip(),
        description=description.strip(),
        parameters=parameters,
        state_changing=classify(name, description, tool),
    )


def parse_manifest(tools: Any) -> list[ToolSpec]:
    if isinstance(tools, dict):
        if isinstance(tools.get("tools"), list):
            tools = tools["tools"]
        else:
            tools = [dict(v, name=k) if isinstance(v, dict) and "name" not in v else v for k, v in tools.items()]
    specs: list[ToolSpec] = []
    seen: set[str] = set()
    for raw in tools or []:
        spec = parse_tool(raw)
        if spec and spec.name not in seen:
            seen.add(spec.name)
            specs.append(spec)
    return specs

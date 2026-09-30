from __future__ import annotations

import json
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator

CALL_ID_PATTERN = r"^[A-Za-z0-9_\-:.]{1,128}$"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _ensure_json(value: Any) -> Any:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"value is not JSON-serializable: {exc}") from exc
    return value


class TextChunk(_Model):
    type: Literal["text_chunk"] = "text_chunk"
    text: str
    end_of_turn: bool = False
    ts: float = 0.0


class AudioClip(_Model):
    type: Literal["audio_clip"] = "audio_clip"
    path: str
    ts: float = 0.0


class Frame(_Model):
    type: Literal["frame"] = "frame"
    path: str
    ts: float = 0.0


class Interrupt(_Model):
    type: Literal["interrupt"] = "interrupt"
    ts: float = 0.0


class ToolResult(_Model):
    type: Literal["tool_result"] = "tool_result"
    call_id: str
    ok: bool = True
    result: Any = None
    error: str | None = None
    ts: float = 0.0


class ToolManifest(_Model):
    type: Literal["tool_manifest"] = "tool_manifest"
    tools: list[dict[str, Any]]
    ts: float = 0.0


class SessionEnd(_Model):
    type: Literal["session_end"] = "session_end"
    ts: float = 0.0


InputEvent = Annotated[
    Union[TextChunk, AudioClip, Frame, Interrupt, ToolResult, ToolManifest, SessionEnd],
    Field(discriminator="type"),
]
INPUT_EVENT_ADAPTER: TypeAdapter[Any] = TypeAdapter(InputEvent)


class StateSnapshot(_Model):
    intent: str | None = None
    slots: dict[str, Any] = Field(default_factory=dict)

    @field_validator("slots")
    @classmethod
    def _slots_json(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _ensure_json(value)


class Speak(_Model):
    type: Literal["speak"] = "speak"
    kind: Literal["ack", "filler", "progress"]
    text: str = Field(min_length=1)
    ts: float = 0.0


class ToolCall(_Model):
    type: Literal["tool_call"] = "tool_call"
    call_id: str = Field(pattern=CALL_ID_PATTERN)
    name: str = Field(min_length=1)
    args: dict[str, Any] = Field(default_factory=dict)
    ts: float = 0.0

    @field_validator("args")
    @classmethod
    def _args_json(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _ensure_json(value)


class Cancel(_Model):
    type: Literal["cancel"] = "cancel"
    call_id: str = Field(pattern=CALL_ID_PATTERN)
    reason: str = "superseded"
    ts: float = 0.0


class Clarify(_Model):
    type: Literal["clarify"] = "clarify"
    question: str = Field(min_length=1)
    snapshot: StateSnapshot | None = None
    ts: float = 0.0


class FinalResponse(_Model):
    type: Literal["final_response"] = "final_response"
    text: str = Field(min_length=1)
    snapshot: StateSnapshot
    ts: float = 0.0


class StateUpdate(_Model):
    type: Literal["state_update"] = "state_update"
    snapshot: StateSnapshot
    ts: float = 0.0


Action = Annotated[
    Union[Speak, ToolCall, Cancel, Clarify, FinalResponse, StateUpdate],
    Field(discriminator="type"),
]
ACTION_ADAPTER: TypeAdapter[Any] = TypeAdapter(Action)


class ToolSpec(_Model):
    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})
    state_changing: bool = True

    @property
    def properties(self) -> dict[str, Any]:
        props = self.parameters.get("properties")
        return props if isinstance(props, dict) else {}

    @property
    def required(self) -> list[str]:
        req = self.parameters.get("required")
        return [r for r in req if isinstance(r, str)] if isinstance(req, list) else []


def parse_event(raw: Any) -> Any:
    if isinstance(raw, BaseModel):
        return raw
    return INPUT_EVENT_ADAPTER.validate_python(raw)


def parse_action(raw: Any) -> Any:
    if isinstance(raw, BaseModel):
        raw = raw.model_dump()
    return ACTION_ADAPTER.validate_python(raw)

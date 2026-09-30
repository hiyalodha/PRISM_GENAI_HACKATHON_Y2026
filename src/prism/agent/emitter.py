from __future__ import annotations

import asyncio
import logging
from typing import Any

from pydantic import BaseModel, ValidationError

from .clock import Clock
from .models import ACTION_ADAPTER

log = logging.getLogger(__name__)


class Emitter:
    def __init__(self, out_q: asyncio.Queue, clock: Clock) -> None:
        self.out_q = out_q
        self.clock = clock
        self.history: list[Any] = []
        self.rejected: list[tuple[Any, str]] = []

    def emit(self, action: BaseModel) -> Any | None:
        data = action.model_dump()
        data["ts"] = round(self.clock.now(), 6)
        try:
            validated = ACTION_ADAPTER.validate_python(data)
        except ValidationError as exc:
            self.rejected.append((data, str(exc)))
            log.error("dropping invalid action %s: %s", data.get("type"), exc)
            return None
        self.out_q.put_nowait(validated)
        self.history.append(validated)
        return validated

    def last_spoken_at(self) -> float | None:
        for action in reversed(self.history):
            if action.type in {"speak", "clarify", "final_response"}:
                return action.ts
        return None

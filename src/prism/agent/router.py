from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

from pydantic import ValidationError

from .models import SessionEnd, parse_event

log = logging.getLogger(__name__)

Handler = Callable[[Any], None]


class EventRouter:
    def __init__(self, handlers: dict[str, Handler]) -> None:
        self.handlers = handlers
        self.seen = 0
        self.rejected = 0

    def dispatch(self, raw: Any) -> bool:
        try:
            event = parse_event(raw)
        except ValidationError as exc:
            self.rejected += 1
            log.error("rejected malformed event: %s", exc)
            return True
        self.seen += 1
        if isinstance(event, SessionEnd):
            handler = self.handlers.get(event.type)
            if handler:
                handler(event)
            return False
        handler = self.handlers.get(event.type)
        if handler is None:
            log.warning("no handler for event type %s", event.type)
            return True
        try:
            handler(event)
        except Exception:
            log.exception("handler for %s failed", event.type)
        return True

    async def run(self, in_q: asyncio.Queue) -> None:
        while True:
            raw = await in_q.get()
            if raw is None:
                return
            if not self.dispatch(raw):
                return

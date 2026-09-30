from __future__ import annotations

import asyncio
import re
from typing import Any

from . import nlu
from .call_manager import CallManager, CallRecord
from .clock import Clock
from .config import AgentConfig
from .dedup import normalize_value
from .emitter import Emitter
from .manifest import READ_VERBS, STATE_VERBS
from .models import Speak, ToolSpec

COMPLETION_CLAIM_RE = re.compile(
    r"\b(i've|i have|we've|we have|it's|it is|is|has been|have been|successfully|all)\s+"
    r"(booked|created|reserved|submitted|confirmed|scheduled|placed|completed|filed|sent|cancelled|done|set)\b"
    r"|\byou're (all set|booked|confirmed)\b|\bdone, your\b",
    re.IGNORECASE,
)

ACK_READ = ["Let me check that.", "Sure, let me look that up.", "One moment, checking now."]
ACK_WRITE = ["Sure, I'll take care of that.", "Okay, let me set that up.", "Sure, working on it."]
ACK_GENERIC = ["Okay, one moment.", "Sure, one second.", "Got it, give me a moment."]
ACK_AUDIO = ["Got it, one moment.", "Okay, give me a second."]
PROGRESS = ["Still {what}, one moment.", "Bear with me, still {what}.", "Almost there, still {what}."]


def claims_completion(text: str) -> bool:
    return bool(COMPLETION_CLAIM_RE.search(text))


def _gerund(verb: str) -> str:
    if verb.endswith("e") and not verb.endswith("ee"):
        return verb[:-1] + "ing"
    return verb + "ing"


def activity_phrase(name: str) -> str:
    words = nlu.humanize(name).split()
    if words and (words[0] in STATE_VERBS or words[0] in READ_VERBS) and len(words) > 1:
        return f"{_gerund(words[0])} the {' '.join(words[1:])}"
    return f"working on the {' '.join(words) or 'request'}"


class FastPath:
    def __init__(self, clock: Clock, emitter: Emitter, calls: CallManager, config: AgentConfig) -> None:
        self.clock = clock
        self.emitter = emitter
        self.calls = calls
        self.config = config
        self.turn_index = 0
        self.turn_fillers = 0
        self.progress_counts: dict[str, int] = {}
        self._narrator: asyncio.Task[None] | None = None
        self._rotation = 0

    def new_turn(self) -> None:
        self.turn_index += 1
        self.turn_fillers = 0

    def _pick(self, options: list[str]) -> str:
        self._rotation += 1
        return options[(self._rotation - 1) % len(options)]

    def speak(self, kind: str, text: str, force: bool = False) -> bool:
        if not force and self.turn_fillers >= self.config.max_fillers_per_turn:
            return False
        if claims_completion(text):
            return False
        if self.emitter.emit(Speak(kind=kind, text=text)) is None:  # type: ignore[arg-type]
            return False
        self.turn_fillers += 1
        return True

    def ack(self, text: str, specs: list[ToolSpec], corrections: dict[str, Any] | None = None, audio: bool = False) -> bool:
        if audio:
            return self.speak("ack", self._pick(ACK_AUDIO))
        fixed = next((v for v in (corrections or {}).values() if isinstance(v, str) and v), None)
        if fixed:
            return self.speak("ack", f"Got it, {fixed} instead.")
        if nlu.is_affirmation(text):
            return self.speak("ack", "Okay, on it.")
        guess = nlu.guess_tool(text, specs)
        if guess is None:
            return self.speak("ack", self._pick(ACK_GENERIC))
        return self.speak("ack", self._pick(ACK_WRITE if guess.state_changing else ACK_READ))

    def detect_changes(self, text: str, current: dict[str, Any], specs: list[ToolSpec]) -> dict[str, Any]:
        schemas: dict[str, dict[str, Any]] = {}
        for spec in specs:
            for name, schema in spec.properties.items():
                schemas.setdefault(name, schema if isinstance(schema, dict) else {})
        kinds = {k: nlu.param_kind(k, schemas.get(k, {})) for k in current}
        by_kind: dict[str, list[str]] = {}
        for key, kind in kinds.items():
            by_kind.setdefault(kind, []).append(key)
        cue = nlu.has_correction(text)
        found: dict[str, Any] = {}
        cities = nlu.find_cities(text)
        for role in ("destination", "origin"):
            hit = nlu.last_positive(cities, role)
            if hit:
                for key in by_kind.get(role, []):
                    found[key] = hit.value
        if cue:
            bare = nlu.last_positive([c for c in cities if c.role in (None, "location")])
            if bare:
                negated = {c.value for c in cities if c.negated}
                origin_keys = [k for k in by_kind.get("origin", []) if current.get(k) in negated]
                targets = origin_keys or by_kind.get("destination", []) or by_kind.get("location", [])
                for key in targets:
                    found.setdefault(key, bare.value)
            date = nlu.last_positive(nlu.find_dates(text))
            if date:
                for key in by_kind.get("date", []):
                    found[key] = date.value
            when = nlu.find_time(text)
            if when:
                for key in by_kind.get("time", []):
                    found[key] = when
        name = nlu.find_name(text)
        if name:
            for key in by_kind.get("person", []):
                found[key] = name
        for key, value in current.items():
            if isinstance(value, str) and value and re.search(r"\bnot\s+(?:the\s+)?" + re.escape(value) + r"\b", text, re.IGNORECASE):
                found.setdefault(key, None)
        return {k: v for k, v in found.items() if v is None or normalize_value(v) != normalize_value(current.get(k))}

    def on_call_started(self, record: CallRecord) -> None:
        if self._narrator is None or self._narrator.done():
            self._narrator = asyncio.create_task(self._narrate())

    def stop_narration(self) -> None:
        if self._narrator and not self._narrator.done():
            self._narrator.cancel()
        self._narrator = None

    async def _narrate(self) -> None:
        wait = self.config.progress_after
        while True:
            await self.clock.sleep(wait)
            for _ in range(20):
                await asyncio.sleep(0)
            live = [r for r in self.calls.in_flight(include_speculative=False)
                    if self.progress_counts.get(r.call_id, 0) < self.config.max_progress_per_call]
            if not self.calls.in_flight(include_speculative=False):
                return
            if not live:
                wait = self.config.progress_interval
                continue
            last = self.emitter.last_spoken_at() or 0.0
            since = self.clock.now() - last
            gap = self.config.progress_after if self.progress_counts.get(live[0].call_id, 0) == 0 else self.config.progress_interval
            if since < gap:
                wait = gap - since
                continue
            record = min(live, key=lambda r: r.issued_at)
            text = self._pick(PROGRESS).format(what=activity_phrase(record.name))
            if self.speak("progress", text):
                self.progress_counts[record.call_id] = self.progress_counts.get(record.call_id, 0) + 1
            wait = self.config.progress_interval

    def retry_notice(self, record: CallRecord) -> None:
        self.speak("progress", "That didn't go through on the first try, so I'm trying again.")

    def guard_text(self, text: str, has_success: bool) -> str:
        if claims_completion(text) and not has_success:
            return "I haven't been able to complete that yet. Would you like me to keep trying?"
        return text

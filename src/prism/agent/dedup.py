from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any


def normalize_value(value: Any) -> Any:
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value.strip()).casefold()
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {str(k): normalize_value(v) for k, v in value.items() if v is not None}
    if isinstance(value, (list, tuple)):
        return [normalize_value(v) for v in value]
    return value


def idempotency_key(name: str, args: dict[str, Any]) -> str:
    return name + ":" + json.dumps(normalize_value(args), sort_keys=True, separators=(",", ":"), default=str)


class Verdict(str, Enum):
    ALLOW = "allow"
    IN_FLIGHT = "in_flight"
    SUCCEEDED = "succeeded"
    EXHAUSTED = "exhausted"
    UNCERTAIN = "uncertain"


@dataclass
class _Entry:
    status: str = "idle"
    failures: int = 0
    call_id: str | None = None


class DuplicateGuard:
    def __init__(self, max_retries: int = 2) -> None:
        self.max_retries = max_retries
        self._entries: dict[str, _Entry] = {}

    def check(self, key: str) -> Verdict:
        entry = self._entries.get(key)
        if entry is None or entry.status == "idle":
            return Verdict.ALLOW
        if entry.status == "in_flight":
            return Verdict.IN_FLIGHT
        if entry.status == "succeeded":
            return Verdict.SUCCEEDED
        if entry.status == "uncertain":
            return Verdict.UNCERTAIN
        if entry.status == "failed":
            return Verdict.ALLOW if entry.failures <= self.max_retries else Verdict.EXHAUSTED
        return Verdict.ALLOW

    def failures(self, key: str) -> int:
        entry = self._entries.get(key)
        return entry.failures if entry else 0

    def mark_in_flight(self, key: str, call_id: str) -> None:
        entry = self._entries.setdefault(key, _Entry())
        entry.status = "in_flight"
        entry.call_id = call_id

    def mark_succeeded(self, key: str) -> None:
        self._entries.setdefault(key, _Entry()).status = "succeeded"

    def mark_failed(self, key: str) -> None:
        entry = self._entries.setdefault(key, _Entry())
        entry.status = "failed"
        entry.failures += 1

    def mark_uncertain(self, key: str) -> None:
        self._entries.setdefault(key, _Entry()).status = "uncertain"

    def release(self, key: str) -> None:
        entry = self._entries.get(key)
        if entry and entry.status == "in_flight":
            entry.status = "failed" if entry.failures else "idle"

    def status(self, key: str) -> str:
        entry = self._entries.get(key)
        return entry.status if entry else "idle"

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .dedup import normalize_value
from .models import StateSnapshot


@dataclass
class SlotChange:
    intent_changed: bool = False
    changed: set[str] = field(default_factory=set)
    cleared: set[str] = field(default_factory=set)
    previous: dict[str, Any] = field(default_factory=dict)

    @property
    def any(self) -> bool:
        return self.intent_changed or bool(self.changed) or bool(self.cleared)

    @property
    def touched(self) -> set[str]:
        return self.changed | self.cleared


class SlotStore:
    def __init__(self) -> None:
        self.intent: str | None = None
        self.slots: dict[str, Any] = {}
        self.derived: dict[str, set[str]] = {}
        self.version = 0

    def snapshot(self) -> StateSnapshot:
        return StateSnapshot(intent=self.intent, slots=dict(self.slots))

    def get(self, key: str) -> Any:
        return self.slots.get(key)

    def closure(self, keys: set[str]) -> set[str]:
        out = set(keys)
        frontier = list(keys)
        while frontier:
            k = frontier.pop()
            for dep in self.derived.get(k, set()):
                if dep not in out:
                    out.add(dep)
                    frontier.append(dep)
        return out

    def apply(
        self,
        intent: str | None = None,
        updates: dict[str, Any] | None = None,
        clears: list[str] | None = None,
        derived_from: dict[str, set[str]] | None = None,
    ) -> SlotChange:
        change = SlotChange(previous=dict(self.slots))
        if intent and intent != self.intent:
            self.intent = intent
            change.intent_changed = True
        for key, value in (updates or {}).items():
            if value is None or (isinstance(value, str) and not value.strip()):
                continue
            if isinstance(value, str):
                value = value.strip()
            if key in self.slots and normalize_value(self.slots[key]) == normalize_value(value):
                continue
            self.slots[key] = value
            change.changed.add(key)
            if derived_from and key in derived_from:
                self.derived[key] = set(derived_from[key]) - {key}
            else:
                self.derived.pop(key, None)
        for key in clears or []:
            if key in self.slots and key not in change.changed:
                del self.slots[key]
                self.derived.pop(key, None)
                change.cleared.add(key)
        self._cascade(change)
        if change.any:
            self.version += 1
        return change

    def _cascade(self, change: SlotChange) -> None:
        dirty = set(change.touched)
        while dirty:
            nxt: set[str] = set()
            for key, deps in list(self.derived.items()):
                if key in change.changed or key not in self.slots:
                    continue
                if deps & dirty:
                    del self.slots[key]
                    del self.derived[key]
                    change.cleared.add(key)
                    nxt.add(key)
            dirty = nxt

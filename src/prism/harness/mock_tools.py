from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from prism.agent.clock import Clock

BUILTIN_MANIFEST: list[dict[str, Any]] = [
    {
        "name": "flight_search",
        "description": "Search available flights between two cities on a given date.",
        "parameters": {
            "type": "object",
            "properties": {
                "origin": {"type": "string", "description": "Departure city"},
                "destination": {"type": "string", "description": "Arrival city"},
                "date": {"type": "string", "description": "Travel date"},
            },
            "required": ["origin", "destination", "date"],
        },
        "read_only": True,
    },
    {
        "name": "book_flight",
        "description": "Book a seat on a specific flight for a passenger.",
        "parameters": {
            "type": "object",
            "properties": {
                "flight_id": {"type": "string", "description": "Flight identifier from flight_search"},
                "passenger_name": {"type": "string", "description": "Full name of the passenger"},
            },
            "required": ["flight_id", "passenger_name"],
        },
        "read_only": False,
    },
    {
        "name": "create_ticket",
        "description": "Create a customer support ticket for a problem or complaint.",
        "parameters": {
            "type": "object",
            "properties": {
                "issue": {"type": "string", "description": "Description of the problem"},
                "priority": {"type": "string", "enum": ["low", "medium", "high"]},
                "device_model": {"type": "string"},
            },
            "required": ["issue"],
        },
        "read_only": False,
    },
    {
        "name": "manual_lookup",
        "description": "Look up troubleshooting information in the device manual, for example what a blinking light, error code, or button means.",
        "parameters": {
            "type": "object",
            "properties": {
                "device_model": {"type": "string", "description": "Model identifier visible on the device"},
                "query": {"type": "string", "description": "What the user wants to know"},
            },
            "required": ["device_model", "query"],
        },
        "read_only": True,
    },
]

MANUALS: dict[str, list[tuple[set[str], str, str]]] = {
    "x200": [
        ({"red", "blinking", "flashing"}, "LED status", "A blinking red light means the router has no internet connection. Check the WAN cable, then restart the router."),
        ({"green", "solid"}, "LED status", "A solid green light means the router is online and working normally."),
        ({"reset", "button"}, "Factory reset", "Hold the reset button for 10 seconds until the light flashes amber."),
    ],
    "tv-55q": [
        ({"red", "standby"}, "Power", "A red standby light means the TV is powered but in standby. Press the power button on the remote."),
    ],
}
AIRLINES = ["AF", "BA", "LH", "UA", "DL", "EK", "SQ", "KL"]


def _digest(*parts: Any) -> str:
    raw = json.dumps(parts, sort_keys=True, default=str).lower()
    return hashlib.sha256(raw.encode()).hexdigest()


def flight_search(args: dict[str, Any]) -> dict[str, Any]:
    h = _digest("flight_search", args.get("origin"), args.get("destination"), args.get("date"))
    flights = []
    for i in range(3):
        seg = int(h[i * 6: i * 6 + 6], 16)
        airline = AIRLINES[seg % len(AIRLINES)]
        number = 100 + seg % 900
        hour = 6 + (seg // 7) % 14
        flights.append(
            {
                "flight_id": f"{airline}{number}",
                "depart": f"{hour:02d}:{(seg % 4) * 15:02d}",
                "price": 90 + (seg % 400),
                "origin": args.get("origin"),
                "destination": args.get("destination"),
                "date": args.get("date"),
            }
        )
    flights.sort(key=lambda f: f["depart"])
    return {"flights": flights}


def book_flight(args: dict[str, Any]) -> dict[str, Any]:
    h = _digest("book_flight", args.get("flight_id"), args.get("passenger_name"))
    return {
        "booking_id": f"BK-{h[:6].upper()}",
        "flight_id": args.get("flight_id"),
        "passenger_name": args.get("passenger_name"),
        "status": "confirmed",
    }


def create_ticket(args: dict[str, Any]) -> dict[str, Any]:
    h = _digest("create_ticket", args)
    return {"ticket_id": f"TCK-{h[:6].upper()}", "status": "open", "priority": args.get("priority", "medium")}


def manual_lookup(args: dict[str, Any]) -> dict[str, Any]:
    model = str(args.get("device_model", "")).lower().replace(" ", "")
    words = set(re.findall(r"[a-z]+", str(args.get("query", "")).lower()))
    for entry_model, entries in MANUALS.items():
        if entry_model.replace("-", "") == model.replace("-", ""):
            best = max(entries, key=lambda e: len(e[0] & words))
            if best[0] & words:
                return {"found": True, "device_model": args.get("device_model"), "section": best[1], "text": best[2]}
            return {"found": False, "device_model": args.get("device_model"), "text": "No matching manual section."}
    return {"found": False, "device_model": args.get("device_model"), "text": "Unknown device model."}


BUILTIN_TOOLS: dict[str, Callable[[dict[str, Any]], Any]] = {
    "flight_search": flight_search,
    "book_flight": book_flight,
    "create_ticket": create_ticket,
    "manual_lookup": manual_lookup,
}


@dataclass
class ToolBehavior:
    latency: float = 1.0
    faults: list[str] = field(default_factory=list)
    ignore_cancel: bool = False
    result: Any = None
    state_changing: bool | None = None

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None) -> "ToolBehavior":
        cfg = cfg or {}
        return cls(
            latency=float(cfg.get("latency", 1.0)),
            faults=list(cfg.get("faults", [])),
            ignore_cancel=bool(cfg.get("ignore_cancel", False)),
            result=cfg.get("result"),
            state_changing=cfg.get("state_changing"),
        )


@dataclass
class Invocation:
    call_id: str
    name: str
    args: dict[str, Any]
    started: float
    outcome: str = "pending"
    finished: float | None = None
    result: Any = None
    task: asyncio.Task[None] | None = None


class MockEnvironment:
    def __init__(
        self,
        clock: Clock,
        deliver: Callable[[dict[str, Any]], Awaitable[None]],
        log: Callable[[str, dict[str, Any]], None],
        config: dict[str, Any] | None = None,
        state_changing: set[str] | None = None,
    ) -> None:
        self.clock = clock
        self.state_changing = state_changing if state_changing is not None else {"book_flight", "create_ticket"}
        self.deliver = deliver
        self.log = log
        self.behaviors = {name: ToolBehavior.from_config(cfg) for name, cfg in (config or {}).items()}
        self.invocations: dict[str, Invocation] = {}
        self.counts: dict[str, int] = {}
        self.commits: list[Invocation] = []

    def behavior(self, name: str) -> ToolBehavior:
        return self.behaviors.get(name) or ToolBehavior()

    def start(self, call_id: str, name: str, args: dict[str, Any]) -> None:
        if call_id in self.invocations:
            self.log("env", {"op": "duplicate_call_id", "call_id": call_id, "name": name})
            return
        inv = Invocation(call_id, name, dict(args), self.clock.now())
        self.invocations[call_id] = inv
        n = self.counts.get(name, 0)
        self.counts[name] = n + 1
        beh = self.behavior(name)
        fault = beh.faults[n] if n < len(beh.faults) else "ok"
        self.log("env", {"op": "tool_start", "call_id": call_id, "name": name, "args": args, "fault": fault})
        inv.task = asyncio.create_task(self._run(inv, beh, fault))

    async def _run(self, inv: Invocation, beh: ToolBehavior, fault: str) -> None:
        if fault == "timeout":
            inv.outcome = "hung"
            return
        await self.clock.sleep(beh.latency)
        if inv.outcome == "cancelled" and not beh.ignore_cancel:
            return
        if fault in {"fail", "error"}:
            payload = {"type": "tool_result", "call_id": inv.call_id, "ok": False, "result": None,
                       "error": "service_unavailable", "ts": self.clock.now()}
            if inv.outcome != "cancelled":
                inv.outcome = "failed"
        else:
            try:
                result = self._compute(inv.name, inv.args, beh)
                payload = {"type": "tool_result", "call_id": inv.call_id, "ok": True, "result": result,
                           "error": None, "ts": self.clock.now()}
                inv.result = result
                if inv.outcome != "cancelled":
                    inv.outcome = "succeeded"
                    if beh.state_changing or (beh.state_changing is None and inv.name in self.state_changing):
                        self.commits.append(inv)
            except Exception as exc:
                payload = {"type": "tool_result", "call_id": inv.call_id, "ok": False, "result": None,
                           "error": f"bad_arguments: {exc}", "ts": self.clock.now()}
                inv.outcome = "failed"
        inv.finished = self.clock.now()
        late = inv.outcome == "cancelled"
        self.log("env", {"op": "tool_result_late" if late else "tool_complete", "call_id": inv.call_id,
                         "name": inv.name, "ok": payload["ok"], "result": payload["result"]})
        await self.deliver(payload)

    def _compute(self, name: str, args: dict[str, Any], beh: ToolBehavior) -> Any:
        if beh.result is not None:
            template = json.dumps(beh.result)
            for key, value in args.items():
                template = template.replace("{" + key + "}", str(value))
            template = template.replace("{hash}", _digest(name, args)[:6].upper())
            return json.loads(template)
        fn = BUILTIN_TOOLS.get(name)
        if fn is None:
            return {"ok": True, "echo": args, "ref": _digest(name, args)[:6].upper()}
        return fn(args)

    def cancel(self, call_id: str, reason: str) -> None:
        inv = self.invocations.get(call_id)
        if inv is None:
            self.log("env", {"op": "cancel_unknown", "call_id": call_id})
            return
        if inv.outcome in {"pending", "hung"}:
            inv.outcome = "cancelled"
            inv.finished = self.clock.now()
            beh = self.behavior(inv.name)
            if inv.task and not beh.ignore_cancel:
                inv.task.cancel()
            self.log("env", {"op": "tool_cancelled", "call_id": call_id, "name": inv.name, "reason": reason})
        else:
            self.log("env", {"op": "cancel_after_completion", "call_id": call_id, "name": inv.name, "outcome": inv.outcome})

    def shutdown(self) -> None:
        for inv in self.invocations.values():
            if inv.task and not inv.task.done():
                inv.task.cancel()

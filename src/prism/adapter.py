"""Evaluation-kit adapter (STUB).

This is the only module that knows the kit's wire format. Until the kit is released it
passes internal models through unchanged: inbound items may be internal event models or
dicts in the internal schema, and outbound actions are emitted as internal-schema dicts.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from prism.agent.agent import Agent, create_agent
from prism.agent.clock import Clock, RealClock
from prism.agent.config import AgentConfig
from prism.agent.models import parse_event

log = logging.getLogger(__name__)

KIT_FORMAT = "internal-passthrough-stub"


def to_internal_event(raw: Any) -> Any | None:
    try:
        return parse_event(raw)
    except Exception as exc:
        log.error("adapter could not convert kit event %r: %s", raw, exc)
        return None


def to_kit_action(action: Any) -> Any:
    return action.model_dump(mode="json")


def make_clock(kit_clock: Any = None) -> Clock:
    if isinstance(kit_clock, Clock):
        return kit_clock
    return RealClock()


async def run_session(
    kit_in: asyncio.Queue,
    kit_out: asyncio.Queue,
    *,
    clock: Any = None,
    provider: Any = None,
    transcriber: Any = None,
    config: AgentConfig | None = None,
    agent: Agent | None = None,
) -> None:
    agent = agent or create_agent(config=config, clock=make_clock(clock), provider=provider, transcriber=transcriber)
    agent_in: asyncio.Queue = asyncio.Queue()
    agent_out: asyncio.Queue = asyncio.Queue()

    async def inbound() -> None:
        while True:
            raw = await kit_in.get()
            if raw is None:
                await agent_in.put(None)
                return
            event = to_internal_event(raw)
            if event is not None:
                await agent_in.put(event)

    async def outbound() -> None:
        while True:
            action = await agent_out.get()
            converted = to_kit_action(action)
            if converted is not None:
                await kit_out.put(converted)

    pumps = [asyncio.create_task(inbound()), asyncio.create_task(outbound())]
    try:
        await agent.run(agent_in, agent_out)
        while not agent_out.empty():
            await asyncio.sleep(0)
    finally:
        for task in pumps:
            task.cancel()
        await asyncio.gather(*pumps, return_exceptions=True)


async def warmup(config: AgentConfig | None = None, transcriber: str = "auto", provider: Any = None) -> Agent:
    agent = create_agent(config=config, provider=provider, transcriber=transcriber)
    await agent.warmup()
    return agent

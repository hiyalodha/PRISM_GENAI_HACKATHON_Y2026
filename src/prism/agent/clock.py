from __future__ import annotations

import asyncio
import heapq
import itertools
from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable, TypeVar

T = TypeVar("T")


class Clock(ABC):
    @abstractmethod
    def now(self) -> float: ...

    @abstractmethod
    async def sleep(self, delay: float) -> None: ...

    @abstractmethod
    async def run_blocking(self, fn: Callable[..., T], *args: Any) -> T: ...

    @abstractmethod
    async def track(self, awaitable: Awaitable[T]) -> T: ...


class RealClock(Clock):
    def __init__(self) -> None:
        self._origin: float | None = None
        self._external = 0

    @property
    def external_pending(self) -> int:
        return self._external

    def _loop_time(self) -> float:
        return asyncio.get_event_loop().time()

    def now(self) -> float:
        t = self._loop_time()
        if self._origin is None:
            self._origin = t
        return t - self._origin

    async def sleep(self, delay: float) -> None:
        await asyncio.sleep(max(0.0, delay))

    async def run_blocking(self, fn: Callable[..., T], *args: Any) -> T:
        self._external += 1
        try:
            return await asyncio.to_thread(fn, *args)
        finally:
            self._external -= 1

    async def track(self, awaitable: Awaitable[T]) -> T:
        self._external += 1
        try:
            return await awaitable
        finally:
            self._external -= 1


class VirtualClock(Clock):
    def __init__(self, start: float = 0.0) -> None:
        self._now = start
        self._heap: list[tuple[float, int, asyncio.Future[None]]] = []
        self._seq = itertools.count()
        self._external = 0

    def now(self) -> float:
        return self._now

    async def sleep(self, delay: float) -> None:
        if delay <= 0:
            await asyncio.sleep(0)
            return
        fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        heapq.heappush(self._heap, (self._now + delay, next(self._seq), fut))
        await fut

    async def run_blocking(self, fn: Callable[..., T], *args: Any) -> T:
        self._external += 1
        try:
            return await asyncio.to_thread(fn, *args)
        finally:
            self._external -= 1

    async def track(self, awaitable: Awaitable[T]) -> T:
        self._external += 1
        try:
            return await awaitable
        finally:
            self._external -= 1

    @property
    def external_pending(self) -> int:
        return self._external

    def next_wake(self) -> float | None:
        while self._heap and self._heap[0][2].done():
            heapq.heappop(self._heap)
        return self._heap[0][0] if self._heap else None

    def advance_to(self, t: float) -> None:
        if t > self._now:
            self._now = t
        while self._heap and self._heap[0][0] <= self._now:
            _, _, fut = heapq.heappop(self._heap)
            if not fut.done():
                fut.set_result(None)

    async def settle(self, max_iterations: int = 100000) -> None:
        loop = asyncio.get_running_loop()
        idle = 0
        for _ in range(max_iterations):
            if self._external:
                idle = 0
                await asyncio.sleep(0.001)
                continue
            await asyncio.sleep(0)
            ready = getattr(loop, "_ready", None)
            if ready is None or len(ready) == 0:
                idle += 1
                if idle >= 3:
                    return
            else:
                idle = 0

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .clock import Clock
from .llm import FrameObservation, LLMProvider

log = logging.getLogger(__name__)


@dataclass
class Transcript:
    text: str
    confidence: float


class Transcriber:
    def load(self) -> None:
        return None

    def transcribe(self, path: str) -> Transcript:
        raise NotImplementedError


class SidecarTranscriber(Transcriber):
    def transcribe(self, path: str) -> Transcript:
        for candidate in (Path(path + ".txt"), Path(path).with_suffix(".txt")):
            if candidate.exists():
                return Transcript(candidate.read_text().strip(), 1.0)
        return Transcript("", 0.0)


def load_wav(path: str, rate: int = 16000) -> Any:
    import wave

    import numpy as np

    try:
        with wave.open(path, "rb") as w:
            channels, width, src_rate, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
            raw = w.readframes(n)
    except (wave.Error, EOFError, OSError):
        return None
    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}.get(width)
    if dtype is None:
        return None
    data = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if width == 1:
        data = (data - 128.0) / 128.0
    else:
        data /= float(2 ** (8 * width - 1))
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    if src_rate != rate and len(data):
        target = int(len(data) * rate / src_rate)
        data = np.interp(np.linspace(0, len(data) - 1, target), np.arange(len(data)), data).astype(np.float32)
    return data


class WhisperTranscriber(Transcriber):
    def __init__(self, model_size: str = "base.en", device: str = "cpu", compute_type: str = "int8") -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model: Any = None

    def load(self) -> None:
        if self.model is None:
            from faster_whisper import WhisperModel

            try:
                self.model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type, local_files_only=True)
            except Exception:
                self.model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)

    def transcribe(self, path: str) -> Transcript:
        self.load()
        language = "en" if self.model_size.endswith(".en") else None
        audio = load_wav(path)
        source: Any = audio if audio is not None else path
        segments, _info = self.model.transcribe(source, beam_size=1, language=language, condition_on_previous_text=False)
        segs = list(segments)
        text = " ".join(s.text.strip() for s in segs).strip()
        if not segs:
            return Transcript("", 0.0)
        probs = [math.exp(s.avg_logprob) * (1.0 - s.no_speech_prob) for s in segs]
        return Transcript(text, max(0.0, min(1.0, sum(probs) / len(probs))))


def whisper_available() -> bool:
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        return False
    return True


def make_transcriber(kind: str, config: Any = None) -> Transcriber:
    if kind in {"whisper", "auto"}:
        if whisper_available():
            return WhisperTranscriber(
                getattr(config, "whisper_model", "base.en"),
                getattr(config, "whisper_device", "cpu"),
                getattr(config, "whisper_compute_type", "int8"),
            )
        log.warning("faster-whisper is not installed; falling back to sidecar transcripts")
    return SidecarTranscriber()


@dataclass
class Observation:
    ts: float
    path: str
    description: str
    confidence: float
    entities: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"ts": self.ts, "description": self.description, "confidence": self.confidence, "entities": self.entities}


class Perception:
    def __init__(self, clock: Clock, transcriber: Transcriber, provider: LLMProvider) -> None:
        self.clock = clock
        self.transcriber = transcriber
        self.provider = provider
        self.observations: list[Observation] = []
        self.pending: set[asyncio.Task[Any]] = set()

    async def transcribe(self, path: str) -> Transcript:
        try:
            return await self.clock.run_blocking(self.transcriber.transcribe, path)
        except Exception as exc:
            log.error("transcription failed for %s: %s", path, exc)
            return Transcript("", 0.0)

    def submit_frame(self, path: str, ts: float, context: str = "", on_done: Callable[[Observation], None] | None = None) -> asyncio.Task[Any]:
        async def work() -> None:
            try:
                obs: FrameObservation = await self.provider.describe_frame(path, context)
            except Exception as exc:
                log.error("frame analysis failed for %s: %s", path, exc)
                obs = FrameObservation(description="The image could not be analysed.", confidence=0.0)
            record = Observation(ts, path, obs.description, obs.confidence, obs.entities_dict())
            self.observations.append(record)
            if on_done:
                on_done(record)

        task = asyncio.create_task(work())
        self.pending.add(task)
        task.add_done_callback(self.pending.discard)
        return task

    def track(self, task: asyncio.Task[Any]) -> None:
        self.pending.add(task)
        task.add_done_callback(self.pending.discard)

    async def wait_pending(self, timeout: float) -> bool:
        if not self.pending:
            return True
        timer = asyncio.create_task(self.clock.sleep(timeout))
        try:
            waiting = set(self.pending)
            while waiting and not timer.done():
                done, _ = await asyncio.wait(waiting | {timer}, return_when=asyncio.FIRST_COMPLETED)
                waiting = {t for t in self.pending if not t.done()}
            return not waiting
        finally:
            timer.cancel()

    def recent(self, now: float, window: float) -> list[Observation]:
        return [o for o in self.observations if now - o.ts <= window]

    def shutdown(self) -> None:
        for task in list(self.pending):
            task.cancel()

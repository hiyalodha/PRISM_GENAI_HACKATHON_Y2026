import asyncio
import os
from pathlib import Path

import pytest

from prism.agent.clock import VirtualClock
from prism.agent.llm import StubProvider
from prism.agent.perception import Perception, SidecarTranscriber, WhisperTranscriber, load_wav, make_transcriber, whisper_available
from prism.harness.replay import load_scenario, run_scenario
from prism.harness.scorer import score_trace

ASSETS = Path(__file__).resolve().parent.parent / "scenarios" / "assets"
SCENARIOS = Path(__file__).resolve().parent.parent / "scenarios"
RUN_WHISPER = os.environ.get("PRISM_TEST_WHISPER") == "1"


def test_sidecar_transcriber():
    t = SidecarTranscriber()
    got = t.transcribe(str(ASSETS / "find_flights_tokyo.wav"))
    assert got.text.startswith("Find flights") and got.confidence == 1.0
    assert t.transcribe(str(ASSETS / "missing.wav")).confidence == 0.0


def test_load_wav_resamples_to_16k():
    audio = load_wav(str(ASSETS / "find_flights_tokyo.wav"))
    assert audio is not None and audio.dtype.name == "float32"
    assert 1.0 < len(audio) / 16000 < 10.0
    assert float(abs(audio).max()) <= 1.0
    assert load_wav(str(ASSETS / "router_x200.png")) is None


def test_make_transcriber_falls_back():
    t = make_transcriber("sidecar")
    assert isinstance(t, SidecarTranscriber)
    t = make_transcriber("whisper")
    assert isinstance(t, WhisperTranscriber) == whisper_available()


async def test_frame_observation_from_sidecar_and_wait():
    clock = VirtualClock()
    provider = StubProvider()
    provider.bind(clock)
    perception = Perception(clock, SidecarTranscriber(), provider)
    seen = []
    perception.submit_frame(str(ASSETS / "router_x200.png"), 0.0, "", seen.append)
    assert await perception.wait_pending(5.0)
    assert seen[0].entities["device_model"] == "X200" and seen[0].confidence > 0.9
    perception.submit_frame(str(ASSETS / "router_blurry.png"), 1.0)
    await perception.wait_pending(5.0)
    assert perception.observations[-1].confidence < 0.6
    assert len(perception.recent(10.0, 30.0)) == 2


async def test_wait_pending_times_out_on_virtual_clock():
    clock = VirtualClock()
    provider = StubProvider()
    provider.bind(clock)
    perception = Perception(clock, SidecarTranscriber(), provider)
    perception.track(asyncio.create_task(asyncio.Event().wait()))
    waiter = asyncio.create_task(perception.wait_pending(3.0))
    await clock.settle()
    clock.advance_to(clock.next_wake())
    await clock.settle()
    assert waiter.done() and waiter.result() is False
    perception.shutdown()


async def test_transcription_failure_is_contained():
    class Broken(SidecarTranscriber):
        def transcribe(self, path):
            raise RuntimeError("decoder exploded")

    clock = VirtualClock()
    perception = Perception(clock, Broken(), StubProvider())
    got = await perception.transcribe("x.wav")
    assert got.text == "" and got.confidence == 0.0


async def test_unreadable_audio_triggers_clarify():
    scenario = {
        "name": "unreadable_audio",
        "_dir": str(SCENARIOS),
        "events": [{"t": 0.5, "type": "audio_clip", "path": "assets/missing.wav"}],
        "expected": {"clarify": True, "final_response": False},
    }
    result = await run_scenario(scenario, provider=StubProvider(), transcriber=SidecarTranscriber())
    report = score_trace(result.trace, scenario)
    assert report.check("clarify").passed
    assert report.check("first_response_latency").passed


@pytest.mark.skipif(not (RUN_WHISPER and whisper_available()), reason="set PRISM_TEST_WHISPER=1 to run real faster-whisper (downloads tiny.en)")
async def test_real_whisper_audio_scenario():
    transcriber = WhisperTranscriber("tiny.en")
    transcriber.load()
    scenario = load_scenario(SCENARIOS / "08_audio_correction.json")
    result = await run_scenario(scenario, provider=StubProvider(), transcriber=transcriber)
    report = score_trace(result.trace, scenario)
    assert report.failed == [], [(c.name, c.detail) for c in report.failed]

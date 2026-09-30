from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel


class AgentConfig(BaseModel):
    interrupt_policy: Literal["preempt_readonly", "wait_for_text"] = "preempt_readonly"
    speculation: bool = True
    force_speculation: bool = False
    speculation_min_words: int = 4
    speculation_min_confidence: float = 0.75
    progress_after: float = 2.0
    progress_interval: float = 3.0
    max_progress_per_call: int = 2
    max_fillers_per_turn: int = 3
    call_timeout: float = 20.0
    interrupt_silence: float = 4.0
    max_retries: int = 2
    perception_wait: float = 8.0
    frame_confidence_threshold: float = 0.6
    audio_confidence_threshold: float = 0.35
    max_plan_iterations: int = 6
    frame_window: float = 30.0
    whisper_model: str = os.environ.get("PRISM_WHISPER_MODEL", "base.en")
    whisper_device: str = os.environ.get("PRISM_WHISPER_DEVICE", "cpu")
    whisper_compute_type: str = os.environ.get("PRISM_WHISPER_COMPUTE", "int8")

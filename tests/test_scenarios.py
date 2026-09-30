from pathlib import Path

import pytest

from prism.agent.llm import StubProvider
from prism.agent.perception import SidecarTranscriber
from prism.harness.replay import load_scenario, run_scenario
from prism.harness.scorer import format_report, score_trace

SCENARIO_DIR = Path(__file__).resolve().parent.parent / "scenarios"
SCENARIOS = sorted(SCENARIO_DIR.glob("*.json"))


def test_suite_covers_required_modalities():
    modalities = [load_scenario(p).get("modality") for p in SCENARIOS]
    assert len(SCENARIOS) >= 9
    assert {"text", "audio", "visual"} <= set(modalities)


@pytest.mark.parametrize("path", SCENARIOS, ids=[p.stem for p in SCENARIOS])
async def test_scenario(path, tmp_path):
    scenario = load_scenario(path)
    result = await run_scenario(
        scenario,
        provider=StubProvider(),
        transcriber=SidecarTranscriber(),
        trace_path=tmp_path / f"{path.stem}.jsonl",
    )
    report = score_trace(result.trace, scenario)
    assert report.failed == [], format_report(report)
    for name in ("actions_valid", "call_ids_unique", "no_duplicate_state_changes", "cancelled_results_unused",
                 "no_stale_reruns", "first_response_latency", "no_false_completion_claims"):
        assert report.check(name).passed, name
    assert report.score == 100.0
    assert result.trace_path.exists()

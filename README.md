# PRISM: Interruptible Real-Time Voice Agent

Entry for the Samsung PRISM GenAI Hackathon 2026, **Theme 05: Interruptible Real-Time Agents**.

PRISM is the core of a full-duplex voice assistant. It listens, thinks, calls tools and speaks at the same time. The user can interrupt it, change their mind, or correct a detail halfway through a sentence while tools are still running. It talks to the outside world over two async queues: timestamped events come in, validated actions go out.

It runs **fully offline on a laptop**: [Qwen3-VL-8B](https://huggingface.co/mlx-community/Qwen3-VL-8B-Instruct-4bit) plans and reads camera frames, [faster-whisper](https://github.com/SYSTRAN/faster-whisper) transcribes speech, and everything was measured on an Apple M5 MacBook with 16 GB of memory.

> **Status.** The official evaluation kit (harness, wire format, public scenarios) has not been released yet. Everything here is independent of the kit's format except [`src/prism/adapter.py`](src/prism/adapter.py), which is a passthrough stub to be filled in when the kit arrives. The scenarios and scorer in this repo are our own stand-ins.

## Results at a glance

Best setup: **Qwen3-VL-8B (4-bit, MLX) for planning and vision, faster-whisper `tiny.en` for speech, all on-device.**

| | Stub (rules only) | Final: Qwen3-VL-8B + Whisper |
|---|---|---|
| Mean score, 11 shared scenarios | 100.0 | **90.4** |
| Median time to first spoken response (wall clock) | 1.0 ms | **1.2 ms** |
| Median time to first decision (tool call or question) | instant | 11.9 s |
| Cancellation after an interrupt or correction (wall clock) | 0.5 to 2.0 ms | **0.8 ms** |
| Unrequested bookings | 0 | **0** |
| Duplicate state-changing calls | 0 | **0** |

The stub is a deterministic rule-based planner used for CI. It was written alongside the scenarios, so its 100 is an upper bound, not a fair model score.

## Setups we tried

Model runs used the real clock on an Apple M5 with 16 GB of unified memory. Traces for every run are in [`benchmarks/runs/`](benchmarks/runs/), and the tables below are generated from them with `make compare`.

| # | Setup | Planner | Vision | Speech | Mean score (11 shared) | Unrequested bookings | What changed |
|---|---|---|---|---|---|---|---|
| 1 | Stub, virtual clock | rules | sidecar files | sidecar files | 100.0 | 0 | CI baseline, deterministic |
| 1b | Stub, real clock | rules | sidecar files | sidecar files | 100.0 | 0 | measures wall-clock cancellation latency |
| 2 | Qwen3-8B baseline | Qwen3-8B 4-bit (mlx-lm) | none | sidecar files | 81.0 | 0 | first valid local run |
| 3 | Qwen3-8B + turn carry-forward | Qwen3-8B 4-bit | none | sidecar files | 89.6 | 1 | user speech that arrives while the model is thinking is no longer lost |
| 4 | Qwen3-VL-8B single model | Qwen3-VL-8B 4-bit (mlx-vlm) | same model | sidecar files | 90.5 | **4** | one model for text and images |
| 5 | Qwen3-VL-8B + grounding guard | Qwen3-VL-8B 4-bit | same model | sidecar files | 90.4 | 0 | state-changing arguments must be grounded |
| 6 | **Final**: + Whisper | Qwen3-VL-8B 4-bit | same model | faster-whisper `tiny.en` | **90.4** | **0** | fully on-device, real speech recognition |

Scenario 12 was added after runs 2 to 5, so means are taken over the 11 scenarios every run has. Runs 1, 1b and 6 also include scenario 12 (per-scenario table below).

| Run | Mean score (11 shared scenarios) | Median time to first speech | Median time to first decision | Median time to final answer |
|---|---|---|---|---|
| `1_stub_virtual_clock` | 100.0 | 0.0 ms | 0.00 s | 1.20 s |
| `1b_stub_real_clock` | 100.0 | 1.0 ms | 0.00 s | 1.21 s |
| `2_qwen3-8b_baseline` | 81.0 | 0.7 ms | 8.39 s | 15.66 s |
| `3_qwen3-8b_turn_carry_forward` | 89.6 | 0.6 ms | 10.56 s | 22.58 s |
| `4_qwen3-vl-8b_single_model` | 90.5 | 1.2 ms | 11.80 s | 26.44 s |
| `5_qwen3-vl-8b_grounding_guard` | 90.4 | 0.7 ms | 11.81 s | 26.79 s |
| `6_qwen3-vl-8b_whisper_final` | 90.4 | 1.2 ms | 11.89 s | 23.72 s |

Per-scenario scores (`n/a` means the run predates the scenario):

| Scenario | `1_stub_virtual_clock` | `1b_stub_real_clock` | `2_qwen3-8b_baseline` | `3_qwen3-8b_turn_carry_forward` | `4_qwen3-vl-8b_single_model` | `5_qwen3-vl-8b_grounding_guard` | `6_qwen3-vl-8b_whisper_final` |
|---|---|---|---|---|---|---|---|
| 01_search_then_book | 100.0 | 100.0 | 68.9 | 100.0 | 100.0 | 100.0 | 100.0 |
| 02_destination_change_midflight | 100.0 | 100.0 | 68.3 | 81.7 | 75.0 | 68.3 | 68.3 |
| 03_slot_correction_mid_booking | 100.0 | 100.0 | 79.4 | 83.9 | 88.3 | 88.3 | 88.3 |
| 04_booking_fails_then_retry | 100.0 | 100.0 | 71.4 | 100.0 | 100.0 | 100.0 | 100.0 |
| 05_ambiguous_clarify | 100.0 | 100.0 | 82.9 | 82.9 | 82.9 | 88.6 | 88.6 |
| 06_chained_calls | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 07_unseen_tool | 100.0 | 100.0 | 94.3 | 94.3 | 94.3 | 94.3 | 94.3 |
| 08_audio_correction | 100.0 | 100.0 | 72.3 | 88.3 | 80.3 | 88.3 | 88.3 |
| 09_visual_manual_lookup | 100.0 | 100.0 | 71.4 | 71.4 | 100.0 | 100.0 | 100.0 |
| 10_correction_without_interrupt | 100.0 | 100.0 | 88.3 | 88.3 | 80.3 | 72.3 | 72.3 |
| 11_visual_low_confidence | 100.0 | 100.0 | 94.3 | 94.3 | 94.3 | 94.3 | 94.3 |
| 12_late_correction_slow_planner | 100.0 | 100.0 | n/a | n/a | n/a | n/a | 80.0 |

"Time to final answer" includes mock tool latency (1 to 18 s per call) and chained calls, so it is always longer than one model call.

### What we learned

- **Cancellation does not wait for the model.** In scenario 12 with Qwen3-VL in the loop, the in-flight search was cancelled 0.8 ms after the interrupt, while the model took another 6.6 s to plan the replacement search. On the real clock the stub run cancelled within 0.5 to 2.0 ms in every scenario that needed it.
- **The fast path makes latency model-independent.** Time to first speech stayed around 1 ms whether the planner was an instant stub or a 12 s local model. Only time to task completion depends on the model.
- **Slow planners exposed a real bug.** With Qwen3-8B taking about 9 s per plan, the user's next turn arrived while the previous plan was still running. The stale plan was correctly pre-empted, but the new plan saw only the newest utterance, so details like "London to Paris on Monday" were lost (run 2). Plans now receive all user speech since the last applied plan (run 3, 81.0 to 89.6).
- **One multimodal model beats two models on 16 GB.** Qwen3-VL-8B plans about as well as Qwen3-8B and adds real vision: it read "X200" from the router image with 0.95 confidence and rated the blurry image 0.1. Scenario 09 went from 71.4 to 100. A second model alongside it would not have fit reliably.
- **Small models invent arguments for side effects.** Run 4 booked flights nobody asked for, four times, for "Ann Lee", a name copied from an example in our own prompt. Qwen3-8B once booked for a passenger called "User". We removed invented examples from the prompt and added a grounding guard: any text argument of a state-changing call must appear in the user's speech, a tool result or a camera observation, otherwise the agent asks. Unrequested bookings went to zero at no cost in score (runs 5 and 6).
- **Real speech recognition was free on these scenarios.** Switching from pre-written transcripts to Whisper `tiny.en` (about 0.1 s per clip) left the score unchanged (run 6).

### Remaining failures

- **Search requests labelled as bookings (02, 05, 10, 12).** The model still sometimes treats "find flights" as a booking intent. The grounding guard now stops it from booking, but it asks for a passenger name instead of answering the search. A deterministic check that a state-changing intent was explicitly requested would fix this; it is the next item on the list.
- **Timing artifacts (02, 03, 08, 10).** These scenarios expect a cancellation, but with a 12 s planner the user corrects themselves before the first call is even made, so there is nothing to cancel. The behaviour is fine; the scenario timing assumes a faster planner. Scenario 12 was added with slower timing so that cancellation is exercised with a real model.
- **Format guesses (07).** The model wrote `"7pm"` where our scenario expects `"7:00 pm"`. The real scorer's format is unknown.

### Attempts that did not produce valid runs

- **First Qwen3-8B run.** The replay harness ended each scenario after 3 s without activity, which cut sessions off while the model was still thinking. Fixed by having the real clock track in-flight model and Whisper work.
- **Second Qwen3-8B run.** The Metal GPU ran out of memory when speculative plans on partial speech overlapped with the main plan. Fixed by serving one request at a time and disabling speculation for local providers.
- **Hosted models (Claude Haiku and Sonnet)** were designed for but never run: no API key was available, and we decided to stay fully local.
- **vLLM** was ruled out for this machine because it has no Apple GPU backend. It is the recommended runtime for larger models such as 32B on a Linux GPU server.

## How it works

```
events in ──► adapter ──► EventRouter ──► handlers (never block)
                                             │
    ┌────────────────────────────────────────┼──────────────────────────────────┐
    │ Fast path (rules, same tick)           │ Slow path (LLM planner task)     │
    │  - acknowledge end of turn / audio     │  - intent + slot updates         │
    │  - detect corrections, cancel calls    │  - call_tool / clarify /         │
    │  - capped progress narration           │    respond / wait                │
    │  - block false "done" claims           │  - pre-empted by new user input  │
    └────────────────────────────────────────┴──────────────────────────────────┘
                                             │
    Coordination layer
      SlotStore        session slots, localized corrections, derived-slot cascade
      CallManager      call_id tracking, timeouts, stale results dropped
      DuplicateGuard   idempotency key = tool name + normalized args, bounded retries
      Grounding guard  state-changing args must come from the user, a tool or the camera
      Perception       Whisper and vision run in the background
                                             │
    Emitter (pydantic validation) ──► adapter ──► actions out
```

Key design decisions:

- **Two speeds.** The fast path is rule-based and answers in the same event-loop tick (well under a millisecond), so perceived latency never depends on the LLM. The slow path does the reasoning. A 10 s local model and an instant stub produce the same time to first speech.
- **Cancellation first.** A correction cancels exactly the in-flight calls whose arguments it invalidates, before the LLM has even seen the new words. On a bare interrupt, read-only calls are cancelled at once; state-changing calls wait until the user's words actually contradict them. Results that arrive for cancelled calls are dropped and can never leak into an answer.
- **No duplicate side effects.** Every state-changing call has an idempotency key. The same booking is never sent twice while one is in flight or has succeeded. Retries happen only after an explicit failure and are bounded. A timeout on a state-changing call is treated as an unknown outcome and is never retried automatically.
- **Never act on an invented value.** Every text argument of a state-changing call must be traceable to what the user said, a tool result or the camera. Otherwise the agent asks. This guard exists because an 8B model copied an example name from the prompt into a real booking during testing (see [Setups we tried](#setups-we-tried)).
- **Deterministic time.** The agent never calls `time.time()` or `asyncio.sleep()` directly. A `VirtualClock` replays scenarios deterministically; a `RealClock` runs live.
- **Schema-driven tools.** Tools are parsed from whatever manifest shape arrives (OpenAI, Anthropic, MCP, flat, or list parameters) and classified read-only or state-changing from explicit flags first, then from name and description verbs, defaulting to state-changing when unsure.

## Quick start

Requires Python 3.10 to 3.12. Local-model serving requires Apple silicon.

```bash
make install                 # .venv with pinned runtime and test dependencies
make test                    # 82 tests, no network, deterministic
make replay-stub             # replay all scenarios with the rule-based stub (seconds)
```

Run fully local with Qwen3-VL-8B:

```bash
make install-mlx             # separate .venv-mlx with mlx-lm and mlx-vlm
make serve-model             # serves Qwen3-VL-8B-Instruct-4bit on http://127.0.0.1:8081/v1
make replay-local            # real-time replay with Qwen3-VL + Whisper tiny.en, traces in traces/local
make compare                 # score and timing tables for every run in benchmarks/runs
```

The first `serve-model` downloads about 5.8 GB of weights; the first Whisper use downloads about 75 MB.

Useful one-offs:

```bash
PYTHONPATH=src .venv/bin/python -m prism.harness.replay scenarios/02_destination_change_midflight.json --provider stub
PYTHONPATH=src .venv/bin/python -m prism.harness.scorer traces/local/*.jsonl --scenarios scenarios
PRISM_TEST_WHISPER=1 make test-whisper      # end-to-end audio scenario with real faster-whisper
```

## Scenarios

The official public scenarios are not released yet, so we wrote twelve of our own (`scenarios/`). They mirror the categories named in the brief (interruptions, chained calls, retries, clarifications, unseen tools) and its modality mix. **Because we wrote both the scenarios and the expected answers, treat the scores as a regression signal, not a leaderboard.** For example, the expected intent names and slot formats are our guesses at what the hidden scorer wants.

| # | Scenario | Modality | What it tests |
|---|---|---|---|
| 01 | `search_then_book` | text | streamed partial text, speculative search, then booking the cheapest option |
| 02 | `destination_change_midflight` | text | interrupt plus "make it Rome" while a search runs; the stale result is still delivered and must be ignored |
| 03 | `slot_correction_mid_booking` | text | passenger name corrected while the booking is in flight: cancel, rebook, exactly one commit |
| 04 | `booking_fails_then_retry` | text | explicit failure, one bounded retry, exactly one commit |
| 05 | `ambiguous_clarify` | text | "I need to fly out on Friday": must ask where from and to |
| 06 | `chained_calls` | text | search, pick the cheapest, book it, all from one sentence |
| 07 | `unseen_tool` | text | an MCP-style `reserve_table` tool known only from the manifest |
| 08 | `audio_correction` | audio | spoken request, then a spoken date correction mid-search (WAV input) |
| 09 | `visual_manual_lookup` | visual | camera frame of a router; the model number must be read from the image |
| 10 | `correction_without_interrupt` | text | correction with no interrupt signal |
| 11 | `visual_low_confidence` | visual | blurry frame: must not guess the model; asks, then looks it up |
| 12 | `late_correction_slow_planner` | text | correction 16 s in, while a slow planner's search is in flight; the search must be cancelled |

Audio clips were generated with macOS `say`; images were drawn with Pillow (`scenarios/assets/make_assets.py`). Sidecar `.txt` and `.json` files next to each asset let the deterministic stub run without speech or vision models; real runs ignore them.

Mock tools (`src/prism/harness/mock_tools.py`) match the brief's list: `flight_search`, `book_flight`, `create_ticket`, `manual_lookup`. Each has configurable latency and fault injection (fail, hang, keep running after cancel).

### How runs are scored

`src/prism/harness/scorer.py` scores each trace from 0 to 100 using the brief's weights:

| Category | Weight | Checks |
|---|---|---|
| Task completion | 40% | final state snapshot, expected and forbidden tool calls, commit counts, answer content, clarification, answers grounded in tool results, no false "done" claims |
| Interruption recovery | 35% | expected cancellations within 50 ms of the triggering input, no reuse of cancelled results, no stale re-runs |
| Response latency | 15% | time from end of user turn to first spoken action, filler budget |
| Safety and protocol | 10% | valid actions, unique call ids, cancels reference real calls, no duplicate state-changing calls or commits |

The brief also mentions a 0.8x to 1.2x quality multiplier and 1.5x weighting for multimodal scenarios; our scorer does not model those.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `PRISM_LOCAL_URL` | `http://127.0.0.1:8081/v1` | any OpenAI-compatible `/v1/chat/completions` server |
| `PRISM_LOCAL_MODEL` | `mlx-community/Qwen3-VL-8B-Instruct-4bit` | planner model |
| `PRISM_LOCAL_VISION_MODEL` | same as planner | model for camera frames; `none` disables vision |
| `PRISM_LOCAL_FORMAT` | `json_schema` | `json_schema`, `json_object` or `none`; the provider downgrades automatically if the server rejects a mode |
| `PRISM_LOCAL_THINKING` | `0` | `1` lets Qwen3 text models think before answering (slower) |
| `PRISM_WHISPER_MODEL` | `base.en` | faster-whisper model size (`tiny.en` was used for the benchmarks) |

Agent behaviour is tuned through `prism.agent.config.AgentConfig`: interrupt policy, speculation, filler caps, call timeout, retry budget, perception wait and confidence thresholds. Speculative planning on partial speech is on by default but automatically off for local providers, because on one GPU the extra requests slow down the real plan and can exhaust memory.

Other local runtimes work through the same provider: Ollama, llama.cpp (`llama-server`), LM Studio, or vLLM on a Linux GPU box. vLLM is the better fit for larger models such as 32B, which do not fit in 16 GB; it has no Apple GPU backend.

## Adapting to the evaluation kit

Only `src/prism/adapter.py` should change:

1. `to_internal_event(raw)`: map kit events to `TextChunk` (with `end_of_turn`), `AudioClip`, `Frame`, `Interrupt`, `ToolResult` (`call_id`, `ok`, `error`) and `ToolManifest`. If audio or frames arrive as bytes, write them to a temporary file or extend the models.
2. `to_kit_action(action)`: map `Speak`, `ToolCall`, `Cancel`, `Clarify`, `FinalResponse` and `StateUpdate`. Returning `None` drops an action the kit does not support.
3. `make_clock(kit_clock)`: wrap the kit's virtual clock in a `Clock` subclass if it provides one.
4. `run_session(...)` and `warmup(...)`: match the kit's entry point and its 300 s warm-up hook (load Whisper there, and make sure the model server is up).

## Assumptions the kit might break

- **Events.** Text arrives in chunks with an `end_of_turn` flag. Each audio clip is a complete user turn. Audio and frames are file paths. An interrupt carries no payload.
- **Snapshots.** They ride on `FinalResponse`, `Clarify` and a separate `StateUpdate` action, which the kit may not accept.
- **Names and formats.** Intent names are tool names and slot names are tool parameter names. Slot values are stored as spoken ("Monday", "7pm"), not normalized to dates.
- **Latency.** It is measured from the end of the user's turn, not from the interrupt signal; the agent does not talk over a user who is still speaking.
- **Interrupts.** A bare interrupt cancels in-flight read-only calls at once (`interrupt_policy="preempt_readonly"`). If the kit counts re-issuing an identical search as a stale re-run, switch to `"wait_for_text"`.
- **Tool results.** A result with `ok: false` is an explicit failure and may be retried. A timeout on a state-changing call is never retried.
- **Runtime.** The hidden environment may not have a GPU or allow a model server. An 8B model on CPU alone would likely be too slow, so confirm this when the kit ships.
- **Clock.** On the virtual clock, model and Whisper calls take zero virtual time. The benchmark numbers above are from real-time runs, where they do not.

## Project layout

```
src/prism/
  adapter.py            the only kit-aware module
  agent/                agent core (models, clock, router, fast/slow path, call manager, slots,
                        dedup, manifest parser, perception, LLM providers, config, emitter)
  harness/              replay harness, mock tools, scorer, run comparison
scenarios/              scenario JSON files and generated audio/image assets
benchmarks/runs/        JSONL traces of every benchmark run reported above
tests/                  unit, behaviour and full-scenario tests
scripts/                model server launcher
```

## License

Licensed under the [Apache License, Version 2.0](LICENSE).

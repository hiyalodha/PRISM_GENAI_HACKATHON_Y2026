PYTHON ?= python3.12
VENV ?= .venv
MLX_VENV ?= .venv-mlx
PY := $(VENV)/bin/python
export PYTHONPATH := src

.PHONY: install install-mlx test test-whisper replay-stub serve-model replay-local compare assets

install:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install -r requirements.txt

install-mlx:
	$(PYTHON) -m venv $(MLX_VENV)
	$(MLX_VENV)/bin/pip install -r requirements-mlx.txt

test:
	$(PY) -m pytest -q

test-whisper:
	PRISM_TEST_WHISPER=1 $(PY) -m pytest -q -k whisper

replay-stub:
	$(PY) -m prism.harness.replay scenarios --provider stub --transcriber sidecar --out traces/stub

serve-model:
	./scripts/serve_local_model.sh

replay-local:
	$(PY) -m prism.harness.replay scenarios --provider local --transcriber whisper --whisper-model tiny.en --realtime --out traces/local

compare:
	$(PY) -m prism.harness.compare benchmarks/runs/* --markdown

assets:
	$(PY) scenarios/assets/make_assets.py

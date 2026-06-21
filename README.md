# Hirundo Evals

An evaluation CLI for running LLM benchmarks through [Inspect AI](https://github.com/UKGovernmentBEIS/inspect_ai) and future evaluation framework adapters.

## Features

- Inspect AI task aliases for common benchmarks such as `aime25`, `gpqa`, `ifeval`, `livecodebench`, `mmlu-pro`, and `scicode`.
- Optional managed local vLLM server with OpenAI-compatible routing into Inspect.
- Per-run raw framework logs plus an appended summary CSV.

## Installation

```bash
uv venv .venv
source .venv/bin/activate
uv pip install --python .venv/bin/python -e .
```

Install vLLM support when you need managed local serving:

```bash
uv pip install --python .venv/bin/python -e ".[vllm]"
```

## Usage

```bash
hirundo-evals MODEL inspect-ai TASK[,TASK...] [OPTIONS] [INSPECT_OPTIONS]
```

Prefer the console script after editable install, or use:

```bash
python -m hirundo_evals MODEL inspect-ai TASK[,TASK...] [OPTIONS] [INSPECT_OPTIONS]
```

### Basic Inspect Run

```bash
hirundo-evals ibm-granite/granite-4.1-3b inspect-ai aime25 --limit 1
```

Multiple tasks are comma-separated:

```bash
hirundo-evals ibm-granite/granite-4.1-3b inspect-ai aime25,gpqa --limit 1
```

Bare Hugging Face model IDs are passed to Inspect as `hf/<model>`. Explicit provider prefixes such as `openai/...` are preserved.

### Managed Local vLLM

```bash
hirundo-evals ibm-granite/granite-4.1-3b inspect-ai aime25 \
    --vllm-local \
    --vllm-devices 0 \
    --vllm-args "--tensor-parallel-size 1" \
    --limit 1
```

The managed vLLM path starts a local OpenAI-compatible server, sets a temporary dummy `OPENAI_API_KEY`, and forwards Inspect to the local server URL.

### Outputs

By default, outputs are written under `logs/<model>/<run_timestamp>/`, with a summary CSV at `logs/<model>/results.csv`.

```bash
hirundo-evals ibm-granite/granite-4.1-3b inspect-ai mmlu-pro \
    --output-dir eval_outputs \
    --limit 1
```

The summary CSV is appended across runs and includes fields such as framework, run ID, benchmark, metric, score, and runtime.

## Contributing

See `AGENTS.md` for project guidelines, test suite setup, and PR practices.

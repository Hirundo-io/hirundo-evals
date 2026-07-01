# Hirundo Evals

An evaluation CLI for running LLM benchmarks through pluggable framework adapters.

## Features

- A common CLI for launching model evaluations across supported adapters.
- Optional managed local vLLM server for frameworks that can use an OpenAI-compatible endpoint.
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
hirundo-evals MODEL FRAMEWORK TASK[,TASK...] [OPTIONS] [FRAMEWORK_OPTIONS]
```

Prefer the console script after editable install, or use:

```bash
python -m hirundo_evals MODEL FRAMEWORK TASK[,TASK...] [OPTIONS] [FRAMEWORK_OPTIONS]
```

### Basic Run

```bash
hirundo-evals MODEL FRAMEWORK TASK --framework-option value
```

Multiple tasks are comma-separated:

```bash
hirundo-evals MODEL FRAMEWORK TASK_A,TASK_B --framework-option value
```

Arguments that are not recognized by `hirundo-evals` are forwarded to the selected framework adapter.

### Managed Local vLLM

```bash
hirundo-evals MODEL FRAMEWORK TASK \
    --vllm-local \
    --vllm-devices 0 \
    --vllm-args "--tensor-parallel-size 1"
```

When supported by the selected adapter, the managed vLLM path starts a local OpenAI-compatible server, sets a temporary dummy `OPENAI_API_KEY`, and routes the evaluation through the local server URL.

### Outputs

By default, outputs are written under `logs/<model>/<run_timestamp>/`, with a summary CSV at `logs/<model>/results.csv`.

```bash
hirundo-evals MODEL FRAMEWORK TASK --output-dir eval_outputs
```

The summary CSV is appended across runs and includes fields such as framework, run ID, benchmark, metric, score, and runtime.

Example `results.csv` output:

| Run ID | Framework | Benchmark | Metric | Score | Runtime (sec) |
| --- | --- | --- | --- | --- | --- |
| 20260624_233538 | inspect-ai | ifeval | final_acc (%) ⬆️ | 67.00 | 600 |
| 20260624_233538 | inspect-ai | scicode | percentage_main_problems_solved (%) ⬆️ | 45.00 | 1200 |
| 20260624_233538 | inspect-ai | scicode | percentage_subproblems_solved (%) ⬆️ | 56.67 | 1200 |

## Supported Frameworks

Supported frameworks currently include Inspect AI.

For adapter-specific details, see the relevant framework documentation:

- Inspect AI: `hirundo_evals/frameworks/inspect_ai/README.md`

## Contributing

See `AGENTS.md` for project guidelines, test suite setup, and PR practices.

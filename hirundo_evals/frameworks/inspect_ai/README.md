# Inspect AI Adapter

This adapter runs [Inspect AI](https://github.com/UKGovernmentBEIS/inspect_ai) tasks through `hirundo-evals`.

## Usage

```bash
hirundo-evals MODEL inspect-ai TASK[,TASK...] [HIRUNDO_OPTIONS] [INSPECT_OPTIONS]
```

Example:

```bash
hirundo-evals ibm-granite/granite-4.1-3b inspect-ai inspect_evals/aime2025 --limit 1
```

Multiple tasks are comma-separated:

```bash
hirundo-evals ibm-granite/granite-4.1-3b inspect-ai inspect_evals/aime2025,inspect_evals/gpqa_diamond --limit 1
```

Arguments not recognized by `hirundo-evals` are forwarded to `inspect eval`.

## Task Aliases

The adapter supports these short aliases:

- `aime25` -> `inspect_evals/aime2025`
- `gpqa` -> `inspect_evals/gpqa_diamond`
- `ifeval` -> `inspect_evals/ifeval`
- `livecodebench` -> `inspect_evals/livecodebench_pro`
- `mmlu-pro` -> `inspect_evals/mmlu_pro`
- `scicode` -> `inspect_evals/scicode`

Unknown task names are ignored. If no valid tasks remain, the adapter raises an error.

## Model Names

Bare Hugging Face model IDs are passed to Inspect as `hf/<model>`. Explicit `hf/...` and `openai/...` prefixes are preserved.

For example, `ibm-granite/granite-4.1-3b` is sent to Inspect as `hf/ibm-granite/granite-4.1-3b`.

## Managed Local vLLM

```bash
hirundo-evals ibm-granite/granite-4.1-3b inspect-ai aime25 \
    --vllm-local \
    --vllm-devices 0 \
    --vllm-args "--tensor-parallel-size 1" \
    --limit 1
```

For Inspect AI, `--vllm-local` starts Hirundo Evals' managed OpenAI-compatible vLLM server, sets a temporary dummy `OPENAI_API_KEY`, and forwards Inspect to the local server URL.

## Results

Inspect logs are written to the run log directory configured by `hirundo-evals`. The adapter reads those logs after the run and exports selected metrics into the package summary CSV.

# PinchBench Adapter

This adapter runs the pinned [Inspect AI PinchBench wrapper](https://ukgovernmentbeis.github.io/inspect_evals/evals/pinchbench/).
The wrapper runs the native PinchBench harness in Docker and reports its native
aggregate score through Inspect AI.

## Requirements

- `uv`, `git`, and Docker.
- The `hirundo-evals[vllm]` extra when using `--vllm-local`.
- A PinchBench checkout at the tested commit. Set `PINCHBENCH_ROOT` to reuse
  one; otherwise the adapter clones it into the run log directory.

## Usage

```bash
hirundo-evals MODEL pinchbench MODE_OR_SUITE[,SUITE...] --vllm-local [HIRUNDO_OPTIONS] [INSPECT_OPTIONS]
```

Example:

```bash
hirundo-evals ibm-granite/granite-4.1-3b pinchbench full \
    --vllm-local \
    --vllm-devices 0 \
    --vllm-args "--tensor-parallel-size 1 --enable-auto-tool-choice --tool-call-parser granite4"
```

Multiple suites or tasks are comma-separated:

```bash
hirundo-evals ibm-granite/granite-4.1-3b pinchbench suite_a,suite_b --vllm-local
```

The supported modes are `smoke`, `subset`, and `full`. A comma-separated list
of task names is passed as the native PinchBench suite. `all` is retained as an
alias for `full`.

Arguments not recognized by `hirundo-evals` are forwarded to `inspect eval`.

## Model Endpoint

PinchBench requires an OpenAI-compatible model endpoint. In the current
`hirundo-evals` CLI, use `--vllm-local` to provide that endpoint.

When `--vllm-local` is used, Hirundo Evals starts a local OpenAI-compatible
vLLM server and passes its URL to the Inspect AI adapter. The adapter itself
handles the Dockerized OpenClaw compatibility layer.

## Enabling Tool Use

PinchBench tasks use tools, so vLLM must enable automatic tool selection and
configure a parser for the served model. For IBM Granite 4 models, use:

```bash
--vllm-args "--tensor-parallel-size 1 --enable-auto-tool-choice --tool-call-parser granite4"
```

The `granite4` parser is specific to Granite 4 models. Other models require
their corresponding vLLM tool-call parser; see the
[vLLM tool-calling documentation](https://docs.vllm.ai/en/stable/features/tool_calling/).
Without these flags, the model request fails with an error similar to:
`"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser`.

## Outputs

Inspect AI JSON logs and the adapter's native artifacts are written to the run
log directory. The pinned Inspect wrapper and, when needed, PinchBench checkout
are reused from the persistent cache at `~/.cache/hirundo-evals` (or
`$XDG_CACHE_HOME/hirundo-evals`). Set `HIRUNDO_EVALS_CACHE_DIR` to override it.

After the run, the adapter exports the Inspect AI aggregate score into the
`hirundo-evals` summary CSV.

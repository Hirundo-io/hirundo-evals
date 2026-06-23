# PinchBench Adapter

This adapter runs [PinchBench](https://github.com/pinchbench/skill) through `hirundo-evals`.

PinchBench is currently supported as a local-only run: it must be used with `--vllm-local` so Hirundo Evals can start a local OpenAI-compatible vLLM server and configure OpenClaw to call it.

## System Dependencies

PinchBench runs through the PinchBench skill repository and OpenClaw, so these tools must be available on the system:

- `git`, used to clone the PinchBench skill repository into the run log directory.
- `node` and `npm`, used to install and run OpenClaw.
- `openclaw`, used to configure and execute the PinchBench agent workflow.
- `vllm`, installed through the `hirundo-evals` Python extra for local serving.

Check whether the required command-line tools are installed:

```bash
git --version
node --version
npm --version
openclaw --version
```

On Ubuntu/Debian systems, install `git`, Node.js, and npm with:

```bash
sudo apt-get update
sudo apt-get install -y git nodejs npm
```

Install OpenClaw with npm:

```bash
npm install -g openclaw@latest
```

Install the Python vLLM extra from the `hirundo-evals` repository:

```bash
uv pip install --python .venv/bin/python -e ".[vllm]"
```

## Usage

```bash
hirundo-evals MODEL pinchbench TASK[,TASK...] --vllm-local [HIRUNDO_OPTIONS] [PINCHBENCH_OPTIONS]
```

Example:

```bash
hirundo-evals ibm-granite/granite-4.1-3b pinchbench all \
    --vllm-local \
    --vllm-devices 0 \
    --vllm-args "--tensor-parallel-size 1"
```

Multiple suites or tasks are comma-separated:

```bash
hirundo-evals ibm-granite/granite-4.1-3b pinchbench suite_a,suite_b --vllm-local
```

Arguments not recognized by `hirundo-evals` are forwarded to PinchBench's `scripts/run.sh`.

## Required Local vLLM

PinchBench requires `--vllm-local`. Running this adapter without it raises an error because OpenClaw needs a local model server URL configured before the benchmark starts.

When `--vllm-local` is used, Hirundo Evals:

- starts a local OpenAI-compatible vLLM server for the requested model;
- sets the vLLM API key used by OpenClaw to `vllm-local`;
- configures OpenClaw's `vllm` provider to point at the local server;
- runs PinchBench from a cloned copy of the PinchBench skill repository.

The adapter automatically adds these vLLM server arguments unless you already provide them:

```bash
--enable-auto-tool-choice --tool-call-parser hermes
```

These flags are required so vLLM can handle OpenClaw tool-calling requests.

## Outputs

The PinchBench skill repository is cloned into the run log directory under `pinchbench-skill/`. PinchBench JSON outputs are written to the same run log directory.

After the run, the adapter exports one overall score per task into the `hirundo-evals` summary CSV.

# Hirundo Evals

An evaluation suite wrapping [inspect-ai](https://github.com/UKGovernmentBEIS/inspect_ai) and other open-source evaluation frameworks.

## Features

- Wrapper for `inspect-ai` tasks.
- Integrated support for `vLLM` inference. You can use vLLM either through Inspect AI's native integration or run a local vLLM server managed dynamically using Python's `asyncio`.

## Installation

Using `uv`:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e .
# Note: Add dependencies in pyproject.toml as needed.
uv pip install typer inspect-ai vllm
```

## Usage

You can use the `run.py` script which utilizes `typer` to accept arguments and pass them down to `inspect-ai`.

### Basic Usage

```bash
python run.py path/to/your_task_module.py --model openai/gpt-3.5-turbo
```

### Running with a Native vLLM Model

If `inspect-ai` supports `vLLM` directly, you can pass the model string as expected by their provider:

```bash
python run.py path/to/your_task_module.py --model vllm/facebook/opt-125m
```

### Running with a Managed Local vLLM Server

If you'd like the suite to spin up a local vLLM OpenAI-compatible API server using `asyncio` and point `inspect-ai` to it:

```bash
python run.py path/to/your_task_module.py \
    --model facebook/opt-125m \
    --vllm-local \
    --vllm-args="--tensor-parallel-size 1"
```

## Contributing

See `AGENTS.md` for project guidelines, test suite setup, and PR practices.

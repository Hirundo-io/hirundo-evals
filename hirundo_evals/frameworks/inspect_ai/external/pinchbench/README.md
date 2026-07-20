# PinchBench Adapter

This adapter runs the pinned [Inspect AI PinchBench wrapper](https://ukgovernmentbeis.github.io/inspect_evals/evals/pinchbench/). The wrapper runs the native PinchBench harness in Docker and reports its native aggregate score through Inspect AI.

## Requirements

- `uv`, `git`, and Docker.
- The `hirundo-evals[vllm]` extra when using `--vllm-local`.
- A PinchBench checkout at the tested commit. Set `PINCHBENCH_ROOT` to reuse one; otherwise the adapter clones it into the run log directory.

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

The supported modes are `smoke`, `subset`, and `full`. A comma-separated list of task names is passed as the native PinchBench suite. `all` is retained as an alias for `full`.

Arguments not recognized by `hirundo-evals` are forwarded to `inspect eval`.

## Model Endpoint

PinchBench requires an OpenAI-compatible model endpoint. In the current `hirundo-evals` CLI, use `--vllm-local` to provide that endpoint.

When `--vllm-local` is used, Hirundo Evals starts a local OpenAI-compatible vLLM server and passes its URL to the Inspect AI adapter. The adapter itself handles the Dockerized OpenClaw compatibility layer.

## Enabling Tool Use

PinchBench tasks use tools, so vLLM must enable automatic tool selection and configure a parser for the served model. For IBM Granite 4 models, use:

```bash
--vllm-args "--enable-auto-tool-choice --tool-call-parser PARSER_NAME"
```

Common architectures use these parser settings:

```bash
# Granite 3
--vllm-args "--enable-auto-tool-choice --tool-call-parser granite --chat-template examples/tool_chat_template_granite.jinja"

# Granite 4
--vllm-args "--enable-auto-tool-choice --tool-call-parser granite4"

# DeepSeek-V3
--vllm-args "--enable-auto-tool-choice --tool-call-parser deepseek_v3 --chat-template examples/tool_chat_template_deepseekv3.jinja"

# DeepSeek-R1
--vllm-args "--enable-auto-tool-choice --tool-call-parser deepseek_v3 --chat-template examples/tool_chat_template_deepseekr1.jinja"

# FunctionGemma
--vllm-args "--enable-auto-tool-choice --tool-call-parser functiongemma --chat-template examples/tool_chat_template_functiongemma.jinja"

# Gemma 4
--vllm-args "--enable-auto-tool-choice --tool-call-parser gemma4 --chat-template examples/tool_chat_template_gemma4.jinja"

# GLM-4.5 / GLM-4.6
--vllm-args "--enable-auto-tool-choice --tool-call-parser glm45 --reasoning-parser glm45"

# GLM-4.7 / GLM-5 / GLM-5.1
--vllm-args "--enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45 --chat-template-content-format=string"

# GPT-OSS (vLLM >= 0.10.2)
--vllm-args "--enable-auto-tool-choice --tool-call-parser openai"

# Kimi K2 Instruct
--vllm-args "--trust-remote-code --enable-auto-tool-choice --tool-call-parser kimi_k2"

# Kimi K2 Thinking / K2.5
--vllm-args "--trust-remote-code --enable-auto-tool-choice --tool-call-parser kimi_k2 --reasoning-parser kimi_k2"

# Llama 3.1/3.2/3.3
--vllm-args "--enable-auto-tool-choice --tool-call-parser llama3_json"

# Llama 4
--vllm-args "--enable-auto-tool-choice --tool-call-parser llama4_pythonic --chat-template examples/tool_chat_template_llama4_pythonic.jinja"

# Mistral
--vllm-args "--enable-auto-tool-choice --tool-call-parser mistral"

# Nemotron 3 Super / Ultra
--vllm-args "--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser nemotron_v3"

# Nemotron 3 Nano (requires the parser plugin file)
--vllm-args "--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser-plugin nano_v3_reasoning_parser.py --reasoning-parser nano_v3"

# Qwen 2.5 / QwQ
--vllm-args "--enable-auto-tool-choice --tool-call-parser hermes"

# Qwen3.5 / Qwen3.6
--vllm-args "--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3"
```

Some Llama and Mistral checkpoints also require an explicit tool-compatible `--chat-template`; consult the vLLM documentation for the model-specific template. For Qwen3.5/3.6, consult the [vLLM Qwen usage guide](https://docs.vllm.ai/projects/recipes/en/stable/Qwen/Qwen3.5.html) for model- and version-specific templates and parser updates.
For Nemotron 3 Nano, download `nano_v3_reasoning_parser.py` from the model's [vLLM Nemotron recipe](https://docs.vllm.ai/projects/recipes/en/stable/NVIDIA/Nemotron-3-Nano-30B-A3B.html) and provide its path in `--reasoning-parser-plugin`.

The `granite4` parser is specific to Granite 4 models. Other models require their corresponding vLLM tool-call parser; see the
[vLLM tool-calling documentation](https://docs.vllm.ai/en/stable/features/tool_calling/).
Without these flags, the model request fails with an error similar to: `"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser`.

## Outputs

Inspect AI JSON logs and the adapter's native artifacts are written to the run log directory. The pinned Inspect wrapper and, when needed, PinchBench checkout are reused from the persistent cache at `~/.cache/hirundo-evals` (or `$XDG_CACHE_HOME/hirundo-evals`). Set `HIRUNDO_EVALS_CACHE_DIR` to override it.

After the run, the adapter exports the Inspect AI aggregate score into the `hirundo-evals` summary CSV.

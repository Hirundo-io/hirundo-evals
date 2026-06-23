# LLM Behavior Eval Adapter

This adapter runs Hirundo's `llm-behavior-eval` CLI through `hirundo-evals`.

## Usage

```bash
hirundo-evals MODEL llm-behavior-eval BEHAVIOR[,BEHAVIOR...] [HIRUNDO_OPTIONS] [LLM_BEHAVIOR_EVAL_OPTIONS]
```

Example:

```bash
hirundo-evals ibm-granite/granite-4.1-3b llm-behavior-eval hallu --max-samples 1
```

Multiple behaviors are comma-separated:

```bash
hirundo-evals ibm-granite/granite-4.1-3b llm-behavior-eval hallu,prompt-injection --max-samples 1
```

Arguments not recognized by `hirundo-evals` are forwarded to `llm-behavior-eval`.

## Behavior Presets

The behavior argument is passed to `llm-behavior-eval` as its behavior preset string. Supported preset formats include:

- Hallucination: `hallu` or `hallu-med`
- Prompt injection: `prompt-injection`
- BBQ bias and unbias: `bias:<type>` or `unbias:<type>`
- UNQOVER bias: `unqover:bias:<type>`
- Bloom bias and unbias: `bloom:bias:<type|all>` or `bloom:unbias:<type|all>`

Use comma-separated values to run more than one behavior in the same command.

## Native vLLM

This adapter does not use Hirundo Evals' managed OpenAI-compatible vLLM server. Instead, `llm-behavior-eval` has native vLLM support.

When `--vllm-local` is passed to `hirundo-evals`, the adapter forwards the run to `llm-behavior-eval` with native vLLM enabled:

```bash
hirundo-evals ibm-granite/granite-4.1-3b llm-behavior-eval hallu \
    --vllm-local \
    --max-samples 1
```

If neither `--inference-engine` nor `--model-engine` is already provided, `--vllm-local` adds:

```bash
--model-engine vllm
```

You can still pass `llm-behavior-eval` vLLM options directly, such as `--vllm-max-model-len`, `--vllm-gpu-memory-utilization`, or `--vllm-enforce-eager`.

## Outputs

The adapter passes the run log directory to `llm-behavior-eval` with `--base-output-dir`.

After the run, it reads `summary_full.csv` files under the run log directory and exports these metrics into the `hirundo-evals` summary CSV when present:

- `Accuracy (%) ⬆️`
- `Error (%) ⬇️`
- `Attack success rate (%) ⬇️`

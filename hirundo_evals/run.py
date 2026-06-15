import asyncio
from typing import Any

import typer
from inspect_ai import eval as inspect_eval

from hirundo_evals.vllm_server import serve_vllm

app = typer.Typer(
    help="Hirundo Evals: A wrapper around inspect-ai and other eval frameworks."
)


async def run_with_vllm(
    task: str, model: str, vllm_args: str | None, inspect_args: dict
):
    async with serve_vllm(model, vllm_args) as server_url:
        # Map the model to use the local OpenAI compatible endpoint
        local_model = f"openai/{model}"
        inspect_args["model_base_url"] = server_url

        # Use asyncio.to_thread to run the blocking eval without stopping the event loop
        await asyncio.to_thread(inspect_eval, task, model=local_model, **inspect_args)


@app.command()
def main(
    task: str = typer.Argument(..., help="The inspect-ai task to run"),
    model: str = typer.Option(..., help="The model to evaluate"),
    vllm_local: bool = typer.Option(False, help="Run a local vLLM server via asyncio"),
    vllm_args: str | None = typer.Option(
        None,
        help="Additional arguments for vLLM server (e.g. '--tensor-parallel-size 2')",
    ),
    model_base_url: str | None = typer.Option(None, help="Base URL for the model"),
    limit: int | None = typer.Option(None, help="Limit number of samples to evaluate"),
    epochs: int = typer.Option(1, help="Number of epochs to run"),
):
    """
    Evaluate tasks using inspect-ai, optionally spinning up a local vLLM server.
    """
    inspect_args: dict[str, Any] = {
        "limit": limit,
        "epochs": epochs,
    }

    if vllm_local:
        asyncio.run(run_with_vllm(task, model, vllm_args, inspect_args))
    else:
        if model_base_url:
            inspect_args["model_base_url"] = model_base_url

        inspect_eval(task, model=model, **inspect_args)


if __name__ == "__main__":
    app()

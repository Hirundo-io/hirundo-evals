import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from .frameworks.registry import EvalFramework, get_eval_framework_wrapper
from .vllm_server import serve_vllm

if TYPE_CHECKING:
    from hirundo_evals.frameworks._base import BaseEvalFrameworkWrapper

app = typer.Typer(
    help="Hirundo Evals: A CLI for running LLM evaluations through framework adapters."
)


def _parse_tasks(tasks: str) -> list[str]:
    """
    Parse the tasks from the command line argument.

    Args:
        tasks: The tasks to parse.

    Returns:
        The parsed tasks.
    """
    # Verify that the tasks are provided after the model and framework arguments
    if tasks.startswith("--"):
        raise typer.BadParameter(
            "Tasks must be provided after the model and framework arguments."
        )
    # Split the tasks by commas and strip whitespace
    parsed_tasks = [task.strip() for task in tasks.split(",") if task.strip()]
    # Verify that at least one task was provided
    if not parsed_tasks:
        raise typer.BadParameter("At least one task must be provided.")

    return parsed_tasks


async def run_with_vllm(
    framework_wrapper: "BaseEvalFrameworkWrapper",
    vllm_args: str | None,
    vllm_devices: str | None,
    framework_args: list[str] | None = None,
) -> None:
    """
    Spins up a local vLLM server and executes the provided evaluation callback.

    Args:
        framework_wrapper: The evaluation framework wrapper.
        vllm_args: Additional arguments for the vLLM server.
        vllm_devices: Comma-separated CUDA device IDs for local vLLM server (e.g. '0' or '0,1').
        framework_args: Extra arguments to pass to the framework.
    """
    previous_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    previous_openai_api_key = os.environ.get("OPENAI_API_KEY")

    if vllm_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = vllm_devices
    os.environ.setdefault("OPENAI_API_KEY", "dummy_key")

    try:
        async with serve_vllm(
            framework_wrapper.model, framework_wrapper.get_vllm_args(vllm_args)
        ) as server_url:
            # Map the model to use the local OpenAI compatible endpoint
            local_model = f"openai/{framework_wrapper.model}"
            # Use asyncio.to_thread to run the blocking eval without stopping the event loop
            await asyncio.to_thread(
                framework_wrapper.run, local_model, server_url, framework_args
            )
    finally:
        if previous_openai_api_key is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = previous_openai_api_key

        if vllm_devices:
            if previous_cuda_visible_devices is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = previous_cuda_visible_devices


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def main(
    ctx: typer.Context,
    model: Annotated[str, typer.Argument(help="The model to evaluate")],
    framework: Annotated[
        str,
        typer.Argument(
            help=(
                f"The evaluation framework to use. Currently supports: "
                f"({', '.join(e.value for e in EvalFramework)})"
            ),
        ),
    ],
    tasks: Annotated[
        str,
        typer.Argument(help="Comma-separated evaluation task/benchmark names to run"),
    ],
    output_dir: str = typer.Option(
        "logs",
        "--output-dir",
        "--output_dir",
        help="Directory for framework raw outputs/logs and summary CSV",
    ),
    vllm_local: bool = typer.Option(
        False,
        "--vllm-local/--no-vllm-local",
        "--vllm_local/--no_vllm_local",
        help="Run a local vLLM server via asyncio",
    ),
    vllm_args: str | None = typer.Option(
        None,
        "--vllm-args",
        "--vllm_args",
        help="Additional arguments for vLLM server (e.g. '--tensor-parallel-size 2')",
    ),
    vllm_devices: str | None = typer.Option(
        None,
        "--vllm-devices",
        "--vllm_devices",
        help="Comma-separated CUDA device IDs for local vLLM server (e.g. '0' or '0,1')",
    ),
):
    """
    Evaluate tasks, optionally spinning up a local vLLM server.

    CLI format:
        hirundo-evals [MODEL] [FRAMEWORK] [TASK[,TASK...]] [FRAMEWORK_OPTIONS]
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Validate the framework
    try:
        framework = EvalFramework(framework)
    except Exception as e:
        raise NotImplementedError(f"Framework '{framework}' is not supported.") from e
    parsed_tasks = _parse_tasks(tasks)
    output_dir = str(Path(output_dir) / model)
    log_dir = str(Path(output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    logging.info(
        f"🚀 Starting evaluation suite on {len(parsed_tasks)} tasks: {', '.join(parsed_tasks)}"
    )
    logging.info(f"🧠 Model: {model}")
    logging.info(f"📁 Log Directory: {log_dir}")
    logging.info(f"📊 CSV Output: {Path(output_dir) / 'results.csv'}")
    # Get the evaluation framework wrapper
    framework_wrapper = get_eval_framework_wrapper(
        framework, model, parsed_tasks, log_dir
    )
    # Run the evaluation
    if vllm_local:
        # Run the evaluation with a local vLLM server
        asyncio.run(run_with_vllm(framework_wrapper, vllm_args, vllm_devices, ctx.args))
    else:
        # Run the evaluation without a local vLLM server
        framework_wrapper.run(extra=ctx.args)
    logging.info(f"✅ Evaluation Suite Complete! Logs saved to: {log_dir}")
    # Export the results to a CSV file
    framework_wrapper.export_results(os.path.join(output_dir, "results.csv"))


if __name__ == "__main__":
    app()

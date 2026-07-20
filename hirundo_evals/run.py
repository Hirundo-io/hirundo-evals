import asyncio
import logging
import os
import shlex
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from .frameworks.registry import EvalFramework, get_eval_framework_wrapper
from .utils.vllm import serve_vllm

if TYPE_CHECKING:
    from .frameworks._base import BaseEvalFrameworkWrapper

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
    # Verify that the tasks are provided as an argument (not an option),
    # and after the model and framework arguments
    if tasks.startswith("--tasks"):
        raise typer.BadParameter(
            "Tasks must be provided as an argument, not an option (i.e. without a leading '--tasks')."
        )
    elif tasks.startswith("--"):
        raise typer.BadParameter(
            f"Tasks must be provided after the model and framework arguments. Got {tasks} instead."
        )
    # Split the tasks by commas and strip whitespace
    parsed_tasks = [task.strip() for task in tasks.split(",") if task.strip()]
    # Verify that at least one task was provided
    if not parsed_tasks:
        raise typer.BadParameter("At least one task must be provided.")

    return parsed_tasks


def _parse_output_and_log_dirs(output_dir: Path, model: str) -> tuple[Path, Path]:
    """
    Parse the output and log directories from the command line argument.

    Args:
        output_dir: The directory to save the outputs.
        model: The model to evaluate.

    Returns:
        The output and log directories.
    """
    output_dir = output_dir / Path(
        *Path(os.path.abspath(model) if model.startswith(".") else model).parts[-2:]
    )
    log_dir = output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    return output_dir, log_dir


def _parse_cli_args(cli_args: str | list[str] | None) -> list[str] | None:
    """
    Parse the CLI arguments from the command line argument.

    Args:
        cli_args: The CLI arguments to parse.

    Returns:
        The parsed CLI arguments.
    """
    return shlex.split(cli_args) if isinstance(cli_args, str) else cli_args


async def run_with_vllm(
    framework_wrapper: "BaseEvalFrameworkWrapper",
    vllm_args: list[str] | None,
    vllm_devices: str | None,
    vllm_port: int,
    framework_args: list[str] | None = None,
) -> None:
    """
    Spins up a local vLLM server and executes the provided evaluation callback.

    Args:
        framework_wrapper: The evaluation framework wrapper.
        vllm_args: Additional arguments for the vLLM server.
        vllm_devices: Comma-separated CUDA device IDs for local vLLM server (e.g. '0' or '0,1').
        vllm_port: Port for the local vLLM server.
        framework_args: Extra arguments to pass to the framework.
    """
    previous_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    previous_openai_api_key = os.environ.get("OPENAI_API_KEY")

    if vllm_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = vllm_devices
    os.environ.setdefault("OPENAI_API_KEY", "dummy_key")

    try:
        async with serve_vllm(
            framework_wrapper.model, vllm_args, port=vllm_port
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


def run_evaluation(
    model: str,
    framework: EvalFramework,
    tasks: str | list[str],
    output_dir: Path = Path("logs"),
    framework_cli_args: str | list[str] | None = None,
    model_base_url: str | None = None,
    run_with_local_vllm: bool = False,
    vllm_cli_args: str | list[str] | None = None,
    vllm_devices: str | int | list[int] | None = None,
    vllm_port: int = 8000,
) -> None:
    """
    Run an evaluation.

    Args:
        model: The model to evaluate.
        framework: The evaluation framework to use.
        tasks: The tasks to evaluate.
        output_dir: The directory to save the outputs.
        framework_cli_args: Additional arguments to pass to the framework.
        model_base_url: The base URL of the model.
        run_with_local_vllm: Whether to run the evaluation with a local vLLM server.
        vllm_cli_args: Additional arguments for the vLLM server.
        vllm_devices: CUDA device IDs for local vLLM server (e.g. '0' or '0,1').
        vllm_port: Port for the local vLLM server.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Parse the tasks
    tasks = [tasks] if isinstance(tasks, str) else tasks
    # Create the output and log directories
    output_dir, log_dir = _parse_output_and_log_dirs(output_dir, model)
    logging.info(
        f"🚀 Starting evaluation suite on {len(tasks)} tasks: {', '.join(tasks)}"
    )
    # Parse the framework CLI arguments
    framework_args = _parse_cli_args(framework_cli_args)
    logging.info(f"🧠 Model: {model}")
    logging.info(f"📁 Log Directory: {log_dir}")
    logging.info(f"📊 CSV Output: {output_dir / 'results.csv'}")
    # Get the evaluation framework wrapper
    framework_wrapper = get_eval_framework_wrapper(framework, model, tasks, log_dir)
    # Run the evaluation
    if run_with_local_vllm:
        # Parse the vLLM CLI arguments
        vllm_args = _parse_cli_args(vllm_cli_args)
        # Parse the vLLM devices
        vllm_devices = (
            str(vllm_devices)
            if isinstance(vllm_devices, int)
            else vllm_devices
            if isinstance(vllm_devices, str | None)
            else ",".join(map(str, vllm_devices))
        )
        # Run the evaluation with a local vLLM server
        asyncio.run(
            run_with_vllm(
                framework_wrapper, vllm_args, vllm_devices, vllm_port, framework_args
            )
        )
    else:
        # Run the evaluation without a local vLLM server
        framework_wrapper.run(model_base_url=model_base_url, extra=framework_args)
    logging.info(f"✅ Evaluation complete! Logs saved to: {log_dir}")
    # Export the results to a CSV file
    framework_wrapper.export_results(os.path.join(output_dir, "results.csv"))


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
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            help="Directory for framework raw outputs/logs and summary CSV",
        ),
    ] = Path("logs"),
    model_base_url: Annotated[
        str | None,
        typer.Option(
            "--model-base-url",
            help="The base URL of the model. Required if the model is hosted externally.",
        ),
    ] = None,
    vllm_local: Annotated[
        bool,
        typer.Option(
            "--vllm-local/--no-vllm-local",
            help="Run a local vLLM server via asyncio",
        ),
    ] = False,
    vllm_args: Annotated[
        str | None,
        typer.Option(
            "--vllm-args",
            help="Additional arguments for vLLM server (e.g. '--tensor-parallel-size 2')",
        ),
    ] = None,
    vllm_devices: Annotated[
        str | None,
        typer.Option(
            "--vllm-devices",
            help="Comma-separated CUDA device IDs for local vLLM server (e.g. '0' or '0,1')",
        ),
    ] = None,
    vllm_port: Annotated[
        int,
        typer.Option(
            "--vllm-port",
            min=1,
            max=65535,
            help="Port for the local vLLM OpenAI-compatible server.",
        ),
    ] = 8000,
) -> None:
    """
    Evaluate tasks, optionally spinning up a local vLLM server.

    CLI format:
        hirundo-evals [MODEL] [FRAMEWORK] [TASK[,TASK...]] [FRAMEWORK_OPTIONS]
    """
    # Validate the framework
    try:
        framework = EvalFramework(framework)
    except ValueError:
        raise typer.BadParameter(
            f"Framework '{framework}' is not supported. "
            f"Currently supported frameworks: {', '.join(e.value for e in EvalFramework)}"
        ) from None
    # Parse the tasks
    parsed_tasks = _parse_tasks(tasks)
    # Run the evaluation
    run_evaluation(
        model,
        framework,
        parsed_tasks,
        output_dir,
        framework_cli_args=ctx.args,
        model_base_url=model_base_url,
        run_with_local_vllm=vllm_local,
        vllm_cli_args=vllm_args,
        vllm_devices=vllm_devices,
        vllm_port=vllm_port,
    )


if __name__ == "__main__":
    app()

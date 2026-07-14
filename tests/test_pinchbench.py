from pathlib import Path

from hirundo_evals.frameworks.inspect_ai.external.pinchbench.wrapper import (
    PinchBenchWrapper,
)


def test_pinchbench_command_uses_inspect_wrapper() -> None:
    wrapper = PinchBenchWrapper(
        "ibm-granite/granite-4.1-3b", ["all"], "logs/run"
    )

    cmd = wrapper.get_cli_cmd(
        model="openai/ibm-granite/granite-4.1-3b",
        model_base_url="http://localhost:8000/v1",
        extra=["--limit", "1"],
    )

    assert cmd[:6] == [
        "uv",
        "run",
        "inspect",
        "eval",
        "src/pinchbench/pinchbench.py@pinchbench",
        "--model",
    ]
    assert cmd[cmd.index("--model") + 1] == "openai/ibm-granite/granite-4.1-3b"
    assert cmd[cmd.index("--model-base-url") + 1] == "http://localhost:8000/v1"
    assert cmd[cmd.index("-T") + 1] == "mode=full"
    assert cmd[-2:] == ["--limit", "1"]


def test_pinchbench_command_passes_mode_and_suite_task_args() -> None:
    wrapper = PinchBenchWrapper(
        "ibm-granite/granite-4.1-3b",
        ["subset", "task_calendar", "task_weather"],
        "logs/run",
    )

    cmd = wrapper.get_cli_cmd(model_base_url="http://localhost:8000/v1")

    task_args = cmd[cmd.index("-T") :]
    assert task_args[:6] == [
        "-T",
        "mode=subset",
        "-T",
        "model=ibm-granite/granite-4.1-3b",
        "-T",
        "suite=task_calendar,task_weather",
    ]


def test_pinchbench_environment_configures_external_adapter() -> None:
    wrapper = PinchBenchWrapper("model", ["smoke"], "logs/run")
    benchmark_root = Path("pinchbench")

    environment = wrapper.environment(
        "model-id",
        "http://localhost:8000/v1",
        benchmark_root,
    )

    assert environment == {
        "PINCHBENCH_ROOT": str(benchmark_root),
        "PINCHBENCH_MODEL_BASE_URL": "http://localhost:8000/v1",
        "PINCHBENCH_MODEL": "model-id",
        "PINCHBENCH_API_KEY": "vllm-local",
    }

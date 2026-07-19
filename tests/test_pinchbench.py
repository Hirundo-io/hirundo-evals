import subprocess
from pathlib import Path
from types import SimpleNamespace

from hirundo_evals.frameworks.inspect_ai.external.pinchbench.wrapper import (
    PinchBenchWrapper,
)


def test_pinchbench_command_uses_inspect_wrapper() -> None:
    wrapper = PinchBenchWrapper("ibm-granite/granite-4.1-3b", ["all"], "logs/run")

    cmd = wrapper.get_cli_cmd(
        model="openai/ibm-granite/granite-4.1-3b",
        model_base_url="http://localhost:8000/v1",
        extra=["--limit", "1"],
    )

    assert cmd[:7] == [
        "uv",
        "run",
        "--with",
        "openai",
        "inspect",
        "eval",
        "src/pinchbench/pinchbench.py@pinchbench",
    ]
    assert cmd[cmd.index("inspect") + 1] == "eval"
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
    assert task_args[:8] == [
        "-T",
        "mode=subset",
        "-T",
        "model=ibm-granite/granite-4.1-3b",
        "-T",
        f"output_root={Path('logs/run').resolve()}",
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


def test_clone_at_reuses_clean_checkout_at_pinned_commit(tmp_path, monkeypatch) -> None:
    wrapper = PinchBenchWrapper("model", ["smoke"], "logs/run")
    destination = tmp_path / "repo"
    destination.mkdir()
    calls: list[list[str]] = []

    def run(command: list[str], **kwargs) -> SimpleNamespace:
        calls.append(command)
        if command[-2:] == ["rev-parse", "HEAD"]:
            return SimpleNamespace(stdout="abc123\n")
        if command[-2:] == ["status", "--porcelain"]:
            return SimpleNamespace(stdout="")
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(wrapper, "_require_tool", lambda command: "git")

    wrapper._clone_at("https://example.com/repo.git", "abc123", destination)

    assert calls == [
        ["git", "-C", str(destination), "rev-parse", "HEAD"],
        ["git", "-C", str(destination), "status", "--porcelain"],
    ]


def test_clone_at_refreshes_mismatched_or_dirty_checkout(tmp_path, monkeypatch) -> None:
    wrapper = PinchBenchWrapper("model", ["smoke"], "logs/run")
    destination = tmp_path / "repo"
    destination.mkdir()
    calls: list[list[str]] = []

    def run(command: list[str], **kwargs) -> SimpleNamespace:
        calls.append(command)
        if command[-2:] == ["rev-parse", "HEAD"]:
            return SimpleNamespace(stdout="old-commit\n")
        if command[-2:] == ["status", "--porcelain"]:
            return SimpleNamespace(stdout=" M file.py\n")
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(wrapper, "_require_tool", lambda command: "git")

    wrapper._clone_at("https://example.com/repo.git", "new-commit", destination)

    assert calls[2:] == [
        [
            "git",
            "-C",
            str(destination),
            "fetch",
            "--force",
            "origin",
            "new-commit",
        ],
        [
            "git",
            "-C",
            str(destination),
            "checkout",
            "--detach",
            "--force",
            "new-commit",
        ],
        ["git", "-C", str(destination), "clean", "-fdx"],
    ]

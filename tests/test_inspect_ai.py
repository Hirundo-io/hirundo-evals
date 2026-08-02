from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from inspect_ai.log import EvalLog

from hirundo_evals.frameworks.inspect_ai import wrapper as inspect_wrapper
from hirundo_evals.frameworks.inspect_ai.wrapper import InspectWrapper


def test_inspect_command_maps_aliases_and_preserves_model_name() -> None:
    wrapper = InspectWrapper(
        "ibm-granite/granite-4.1-3b",
        ["aime25", "gpqa"],
        "raw-logs",
    )

    assert wrapper.get_cli_cmd(extra=["--limit", "1"]) == [
        "inspect",
        "eval",
        "inspect_evals/aime2025",
        "inspect_evals/gpqa_diamond",
        "--model",
        "ibm-granite/granite-4.1-3b",
        "--log-dir",
        "raw-logs",
        "--log-format",
        "eval",
        "--limit",
        "1",
    ]


def test_inspect_task_names_deduplicate_expanded_tasks() -> None:
    assert InspectWrapper._inspect_task_names(["nemo-skills", "gpqa"]) == [
        "inspect_evals/aime2025",
        "inspect_evals/gpqa_diamond",
        "inspect_evals/ifeval",
        "inspect_evals/livecodebench_pro",
        "inspect_evals/mmlu_pro",
        "inspect_evals/scicode",
    ]


def test_inspect_command_configures_existing_local_model_path(tmp_path) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    wrapper = InspectWrapper(str(model_path), ["aime25"], "raw-logs")

    cmd = wrapper.get_cli_cmd()

    assert cmd[cmd.index("--model") + 1] == "hf/local"
    assert cmd[cmd.index("-M") + 1] == f"model_path={model_path}"


def test_load_logs_reads_eval_files_and_ignores_json(tmp_path, monkeypatch) -> None:
    eval_path = tmp_path / "run.eval"
    eval_path.touch()
    (tmp_path / "unrelated.json").touch()
    loaded_log = cast("EvalLog", SimpleNamespace())
    read_paths = []

    def fake_read_eval_log(path):
        read_paths.append(path)
        return loaded_log

    monkeypatch.setattr(inspect_wrapper, "read_eval_log", fake_read_eval_log)

    wrapper = InspectWrapper("some/model", ["aime25"], tmp_path)

    assert wrapper._load_logs() == [loaded_log]
    assert read_paths == [eval_path]


def test_export_json_logs_writes_same_stem_as_eval(tmp_path, monkeypatch) -> None:
    eval_path = tmp_path / "run.eval"
    eval_path.touch()
    written = []

    monkeypatch.setattr(inspect_wrapper, "read_eval_log", lambda path: "log")
    monkeypatch.setattr(
        inspect_wrapper,
        "write_eval_log",
        lambda log, path, **kwargs: written.append((log, path, kwargs["format"])),
    )

    wrapper = InspectWrapper("some/model", ["aime25"], tmp_path)
    wrapper._export_json_logs()

    assert written == [("log", tmp_path / "run.json", "json")]


def test_inspect_command_preserves_explicit_openai_model_base_url() -> None:
    wrapper = InspectWrapper("some/model", ["inspect_evals/aime2025"], "raw-logs")

    cmd = wrapper.get_cli_cmd(
        model="openai/some/model",
        model_base_url="http://localhost:8000/v1",
    )

    assert "--model" in cmd
    assert cmd[cmd.index("--model") + 1] == "openai/some/model"
    assert "--model-base-url" in cmd
    assert cmd[cmd.index("--model-base-url") + 1] == "http://localhost:8000/v1"


def test_prepare_results_normalizes_configured_metric() -> None:
    wrapper = InspectWrapper("some/model", ["aime25"], "raw-logs")
    log = cast(
        "EvalLog",
        SimpleNamespace(
            eval=SimpleNamespace(task="inspect_evals/aime2025"),
            status="success",
            stats=None,
            results=SimpleNamespace(
                scores=[
                    SimpleNamespace(
                        name="accuracy",
                        metrics={"accuracy": SimpleNamespace(value=0.75)},
                    )
                ]
            ),
        ),
    )

    results = wrapper._prepare_log_results(log, "run-123")

    assert results == [
        {
            "Run ID": "run-123",
            "Framework": "inspect-ai",
            "Benchmark": "aime25",
            "Metric": "accuracy (%) ⬆️",
            "Score": 75.0,
            "Runtime (sec)": "N/A",
        }
    ]


def test_prepare_results_reports_failed_task() -> None:
    wrapper = InspectWrapper("some/model", ["ifeval"], "raw-logs")
    log = cast(
        "EvalLog",
        SimpleNamespace(
            eval=SimpleNamespace(task="inspect_evals/ifeval"),
            status="error",
            stats=None,
            results=None,
        ),
    )

    results = wrapper._prepare_log_results(log, "run-123")

    assert results[0]["Metric"] == "final_acc (%) ⬆️"
    assert results[0]["Score"] == "Failed (error)"


def test_prepare_results_reports_missing_configured_metric() -> None:
    wrapper = InspectWrapper("some/model", ["aime25"], "raw-logs")
    log = cast(
        "EvalLog",
        SimpleNamespace(
            eval=SimpleNamespace(task="inspect_evals/aime2025"),
            status="success",
            stats=None,
            results=SimpleNamespace(
                scores=[
                    SimpleNamespace(
                        name="accuracy",
                        metrics={"other_metric": SimpleNamespace(value=0.5)},
                    )
                ]
            ),
        ),
    )

    results = wrapper._prepare_log_results(log, "run-123")

    assert results[0]["Metric"] == "accuracy (%) ⬆️"
    assert results[0]["Score"] == "Metric 'accuracy' not found"


def test_prepare_results_reports_missing_scores_for_configured_metric() -> None:
    wrapper = InspectWrapper("some/model", ["aime25"], "raw-logs")
    log = cast(
        "EvalLog",
        SimpleNamespace(
            eval=SimpleNamespace(task="inspect_evals/aime2025"),
            status="success",
            stats=None,
            results=None,
        ),
    )

    results = wrapper._prepare_log_results(log, "run-123")

    assert results[0]["Metric"] == "accuracy (%) ⬆️"
    assert results[0]["Score"] == "Metric 'accuracy' not found"


def test_prepare_results_reports_missing_scores_without_configured_metric() -> None:
    wrapper = InspectWrapper("some/model", ["aime25"], "raw-logs")
    log = cast(
        "EvalLog",
        SimpleNamespace(
            eval=SimpleNamespace(task="custom_task"),
            status="success",
            stats=None,
            results=SimpleNamespace(scores=[]),
        ),
    )

    results = wrapper._prepare_log_results(log, "run-123")

    assert results[0]["Metric"] == "status"
    assert results[0]["Score"] == "No scores available"

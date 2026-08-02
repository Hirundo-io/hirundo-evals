from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from inspect_ai.log import EvalLog

from hirundo_evals.frameworks.inspect_ai import wrapper as inspect_wrapper
from hirundo_evals.frameworks.inspect_ai.wrapper import InspectWrapper


def test_inspect_task_names_deduplicate_expanded_tasks() -> None:
    assert InspectWrapper._inspect_task_names(["nemo-skills", "gpqa"]) == [
        "inspect_evals/aime2025",
        "inspect_evals/gpqa_diamond",
        "inspect_evals/ifeval",
        "inspect_evals/livecodebench_pro",
        "inspect_evals/mmlu_pro",
        "inspect_evals/scicode",
    ]


def test_run_configures_existing_local_model_path(monkeypatch, tmp_path) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    calls = []

    monkeypatch.setattr(
        inspect_wrapper,
        "inspect_eval",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    monkeypatch.setattr(InspectWrapper, "_export_json_logs", lambda self: None)

    wrapper = InspectWrapper(str(model_path), ["aime25"], tmp_path, tmp_path)
    wrapper.run()

    assert calls[0][1]["model"] == "hf/local"
    assert calls[0][1]["model_args"]["model_path"] == str(model_path)


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

    wrapper = InspectWrapper("some/model", ["aime25"], tmp_path, tmp_path)

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

    wrapper = InspectWrapper("some/model", ["aime25"], tmp_path, tmp_path)
    wrapper._export_json_logs()

    assert written == [("log", tmp_path / "run.json", "json")]


def test_run_uses_inspect_python_api(monkeypatch, tmp_path) -> None:
    calls = []

    def fake_inspect_eval(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(inspect_wrapper, "inspect_eval", fake_inspect_eval)
    monkeypatch.setattr(InspectWrapper, "_export_json_logs", lambda self: None)

    wrapper = InspectWrapper("some/model", ["aime25"], tmp_path, tmp_path)
    wrapper.run(extra=["--limit", "1", "--epochs", "2"])

    assert calls == [
        (
            (["inspect_evals/aime2025"],),
            {
                "model": "some/model",
                "model_base_url": None,
                "log_dir": str(tmp_path),
                "log_format": "eval",
                "log_level": "info",
                "limit": 1,
                "epochs": 2,
            },
        )
    ]


def test_prepare_results_normalizes_configured_metric() -> None:
    wrapper = InspectWrapper(
        "some/model", ["aime25"], Path("raw-logs"), Path("raw-logs")
    )
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
    wrapper = InspectWrapper(
        "some/model", ["ifeval"], Path("raw-logs"), Path("raw-logs")
    )
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
    wrapper = InspectWrapper(
        "some/model", ["aime25"], Path("raw-logs"), Path("raw-logs")
    )
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
    wrapper = InspectWrapper(
        "some/model", ["aime25"], Path("raw-logs"), Path("raw-logs")
    )
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
    wrapper = InspectWrapper(
        "some/model", ["aime25"], Path("raw-logs"), Path("raw-logs")
    )
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

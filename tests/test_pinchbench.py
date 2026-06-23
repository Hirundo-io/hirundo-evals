import json
from pathlib import Path

from hirundo_evals.frameworks.pinchbench.wrapper import PinchBenchWrapper


def test_pinchbench_adds_required_vllm_tool_call_args() -> None:
    wrapper = PinchBenchWrapper("ibm-granite/granite-4.1-3b", ["all"], "logs/run")

    args = wrapper.get_vllm_args("--tensor-parallel-size 1")

    assert "--tensor-parallel-size 1" in args
    assert "--enable-auto-tool-choice" in args
    assert "--tool-call-parser hermes" in args


def test_pinchbench_command_uses_absolute_output_dir_and_forwards_extra(
    tmp_path,
) -> None:
    wrapper = PinchBenchWrapper(
        "ibm-granite/granite-4.1-3b",
        ["task_calendar"],
        str(tmp_path / "raw"),
    )

    cmd = wrapper.get_cli_cmd(
        model="openai/ibm-granite/granite-4.1-3b",
        extra=["--no-upload"],
    )

    assert cmd[:4] == [
        "bash",
        "scripts/run.sh",
        "--model",
        "vllm/ibm-granite/granite-4.1-3b",
    ]
    assert cmd[cmd.index("--suite") + 1] == "task_calendar"
    assert Path(cmd[cmd.index("--output-dir") + 1]).is_absolute()
    assert cmd[-1] == "--no-upload"


def test_pinchbench_prepare_results_reads_per_task_scores(tmp_path) -> None:
    output = {
        "efficiency": {"total_execution_time_seconds": 12.5},
        "tasks": [
            {
                "task_id": "task_calendar",
                "execution_time": 6.3,
                "grading": {"mean": 0.8333333333},
            },
            {
                "task_id": "task_without_score",
                "execution_time": 1.0,
                "grading": {},
            },
        ],
    }
    (tmp_path / "0001_vllm-model.json").write_text(json.dumps(output), encoding="utf-8")

    rows = PinchBenchWrapper(
        "model", ["task_calendar"], str(tmp_path)
    ).prepare_results()

    assert rows == [
        {
            "Run ID": tmp_path.name,
            "Framework": "pinchbench",
            "Benchmark": "task_calendar",
            "Metric": "overall_score (%) ⬆️",
            "Score": 83.33333333,
            "Runtime (sec)": 6,
        }
    ]

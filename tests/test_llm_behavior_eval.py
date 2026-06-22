from hirundo_evals.frameworks.llm_behavior_eval import LLMBehaviorEvalWrapper


def test_llm_behavior_eval_command_forwards_behavior_and_output_dir() -> None:
    wrapper = LLMBehaviorEvalWrapper(
        "ibm-granite/granite-4.1-3b",
        ["bias:gender", "hallu"],
        "raw-logs",
    )

    assert wrapper.get_cli_cmd(extra=["--max-samples", "1"]) == [
        "llm-behavior-eval",
        "ibm-granite/granite-4.1-3b",
        "bias:gender,hallu",
        "--base-output-dir",
        "raw-logs",
        "--max-samples",
        "1",
    ]


def test_llm_behavior_eval_uses_native_vllm_args() -> None:
    wrapper = LLMBehaviorEvalWrapper("model", ["hallu"], "raw-logs")

    assert wrapper.supports_managed_vllm() is False
    assert wrapper.get_framework_vllm_args(["--max-samples", "1"]) == [
        "--max-samples",
        "1",
        "--model-engine",
        "vllm",
    ]
    assert wrapper.get_framework_vllm_args(["--inference-engine", "vllm"]) == [
        "--inference-engine",
        "vllm",
    ]


def test_llm_behavior_eval_prepare_results_reads_summary_full(tmp_path) -> None:
    model_dir = tmp_path / "ibm-granite-granite-4-1-3b"
    model_dir.mkdir()
    (model_dir / "summary_full.csv").write_text(
        "\n".join(
            [
                "Model,Dataset,Accuracy (%) ⬆️,Error (%) ⬇️,Attack success rate (%) ⬇️",
                "model,halueval,91.25,,",
                "model,bbq-gender-bias-free-text,,12.5,",
                "model,prompt-injection-purple-llama,,,7.75",
            ]
        ),
        encoding="utf-8",
    )

    rows = LLMBehaviorEvalWrapper("model", ["hallu"], str(tmp_path)).prepare_results()

    assert rows == [
        {
            "Run ID": tmp_path.name,
            "Framework": "llm-behavior-eval",
            "Benchmark": "halueval",
            "Metric": "Accuracy (%) ⬆️",
            "Score": 91.25,
            "Runtime (sec)": "N/A",
        },
        {
            "Run ID": tmp_path.name,
            "Framework": "llm-behavior-eval",
            "Benchmark": "bbq-gender-bias-free-text",
            "Metric": "Error (%) ⬇️",
            "Score": 12.5,
            "Runtime (sec)": "N/A",
        },
        {
            "Run ID": tmp_path.name,
            "Framework": "llm-behavior-eval",
            "Benchmark": "prompt-injection-purple-llama",
            "Metric": "Attack success rate (%) ⬇️",
            "Score": 7.75,
            "Runtime (sec)": "N/A",
        },
    ]

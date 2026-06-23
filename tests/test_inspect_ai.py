from hirundo_evals.frameworks.inspect_ai.wrapper import InspectWrapper


def test_inspect_command_maps_aliases_and_prefixes_hf_models() -> None:
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
        "hf/ibm-granite/granite-4.1-3b",
        "--log-dir",
        "raw-logs",
        "--limit",
        "1",
    ]


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

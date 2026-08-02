from hirundo_evals.utils.cli import clean_cli_args


def test_clean_cli_args_removes_managed_flags() -> None:
    assert clean_cli_args(
        [
            "--model",
            "model-name",
            "--port=9000",
            "--tensor-parallel-size",
            "2",
            "--port",
            "10000",
        ],
        {"--model", "--port"},
    ) == ["--tensor-parallel-size", "2"]


def test_clean_cli_args_preserves_flags_after_missing_value() -> None:
    assert clean_cli_args(
        ["--port", "--tensor-parallel-size", "2"], {"--model", "--port"}
    ) == ["--tensor-parallel-size", "2"]


def test_clean_cli_args_accepts_custom_managed_flags() -> None:
    assert clean_cli_args(
        ["--model", "model-name", "--custom", "value"],
        {"--custom"},
    ) == ["--model", "model-name"]

import pytest
import typer

from hirundo_evals.run import _parse_tasks


def test_parse_tasks_splits_comma_separated_values() -> None:
    assert _parse_tasks("aime25, gpqa,mmlu-pro") == ["aime25", "gpqa", "mmlu-pro"]


def test_parse_tasks_rejects_missing_task_argument() -> None:
    with pytest.raises(typer.BadParameter):
        _parse_tasks("--limit")

import csv

from hirundo_evals.frameworks._base import BaseEvalFrameworkWrapper, OutputEntry


class DummyWrapper(BaseEvalFrameworkWrapper):
    def __init__(self, rows: list[OutputEntry]) -> None:
        super().__init__("model", ["task"], "logs/run")
        self.rows = rows

    def get_cli_cmd(
        self,
        model: str | None = None,
        model_base_url: str | None = None,
        extra: list[str] | None = None,
    ) -> list[str]:
        return ["dummy"]

    def prepare_results(self) -> list[OutputEntry]:
        return self.rows


def test_export_results_appends_without_duplicate_header_and_formats_score(
    tmp_path,
) -> None:
    output_path = tmp_path / "results.csv"
    row = OutputEntry(
        {
            "Run ID": "run-1",
            "Framework": "inspect-ai",
            "Benchmark": "aime25",
            "Metric": "accuracy (%)",
            "Score": 83.33333,
            "Runtime (sec)": 12,
        }
    )

    DummyWrapper([row]).export_results(str(output_path))
    DummyWrapper([row]).export_results(str(output_path))

    with output_path.open(newline="", encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 3
    assert lines[0].startswith("Run ID,Framework,Benchmark,Metric,Score,")

    with output_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [csv_row["Score"] for csv_row in rows] == ["83.33", "83.33"]

import csv
import logging
import os
import subprocess
from abc import ABC, abstractmethod
from typing import TypedDict, get_type_hints

OutputEntry = TypedDict(
    "OutputEntry",
    {
        "Run ID": str,
        "Framework": str,
        "Benchmark": str,
        "Metric": str,
        "Score": float | str,
        "Runtime (sec)": int | str,
    },
)


class BaseEvalFrameworkWrapper(ABC):
    """
    Base class for evaluation framework wrappers.

    Args:
        model: The model to evaluate.
        tasks: The tasks/benchmarks to evaluate.
    """

    def __init__(self, model: str, tasks: list[str], log_dir: str):
        self.model = model
        self.tasks = tasks
        self.log_dir = log_dir

    @abstractmethod
    def get_cli_cmd(
        self,
        model: str | None = None,
        model_base_url: str | None = None,
        extra: list[str] | None = None,
    ) -> list[str]:
        """
        Get the CLI command to run the framework.

        Args:
            model: Optional model override.
            model_base_url: Optional model base URL.
            extra: Extra arguments to pass to the framework.

        Returns:
            The CLI command to run the framework.
        """
        raise NotImplementedError

    def run(
        self,
        model: str | None = None,
        model_base_url: str | None = None,
        extra: list[str] | None = None,
    ) -> None:
        """
        Run the framework.

        Args:
            model: Optional model override.
            model_base_url: Optional model base URL.
            extra: Extra arguments to pass to the framework.
        """
        cmd = self.get_cli_cmd(model, model_base_url, extra)
        subprocess.run(cmd, check=True)  # noqa: S603

    @abstractmethod
    def prepare_results(self) -> list[OutputEntry]:
        """
        Prepare the results of the evaluation for CSV export.

        Returns:
            The results of the evaluation for CSV export.
        """
        raise NotImplementedError

    @staticmethod
    def _format_output_row(row: OutputEntry) -> dict[str, object]:
        formatted_row = dict(row)
        score = formatted_row.get("Score")
        if isinstance(score, int | float):
            formatted_row["Score"] = f"{score:.2f}"

        return formatted_row

    def export_results(self, output_path: str) -> None:
        """
        Export the results of the evaluation to a CSV file.

        Args:
            outputs: The outputs of the evaluation.
            output_path: The path to the output CSV file.
        """
        # Prepare the results for CSV export
        results = [self._format_output_row(row) for row in self.prepare_results()]
        fieldnames = list(get_type_hints(OutputEntry).keys())
        try:
            existing_rows: list[dict[str, str]] = []
            should_write_header = True
            # Get existing rows and fieldnames if the file exists
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                with open(output_path, newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    existing_fieldnames = reader.fieldnames or []
                    existing_rows = list(reader)
                should_write_header = existing_fieldnames != fieldnames
            # Rewrite an existing file if the header has changed
            if should_write_header and existing_rows:
                with open(output_path, mode="w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(
                        {
                            fieldname: row.get(fieldname, "N/A")
                            for fieldname in fieldnames
                        }
                        for row in existing_rows
                    )
                should_write_header = False
            # Append to CSV, writing the header only for new, empty, or migrated files
            with open(output_path, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if should_write_header:
                    writer.writeheader()
                writer.writerows(results)
            logging.info(f"📈 Summary metrics successfully written to {output_path}")
        except Exception as e:
            raise Exception(f"Failed to write CSV file: {e}") from e

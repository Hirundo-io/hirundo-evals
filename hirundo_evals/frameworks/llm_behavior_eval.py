import csv
from pathlib import Path

from ._base import BaseEvalFrameworkWrapper, OutputEntry


class LLMBehaviorEvalWrapper(BaseEvalFrameworkWrapper):
    """
    Wrapper for Hirundo's llm-behavior-eval CLI.
    """

    FRAMEWORK_NAME = "llm-behavior-eval"

    def supports_managed_vllm(self) -> bool:
        return False

    def get_framework_vllm_args(self, extra: list[str] | None = None) -> list[str]:
        args = list(extra or [])
        if "--inference-engine" not in args and "--model-engine" not in args:
            args.extend(["--model-engine", "vllm"])
        return args

    def get_cli_cmd(
        self,
        model: str | None = None,
        model_base_url: str | None = None,
        extra: list[str] | None = None,
    ) -> list[str]:
        cmd = [
            "llm-behavior-eval",
            model or self.model,
            ",".join(self.tasks),
            "--base-output-dir",
            self.log_dir,
        ]
        if extra:
            cmd.extend(extra)
        return cmd

    def prepare_results(self) -> list[OutputEntry]:
        results: list[OutputEntry] = []
        run_id = Path(self.log_dir).name
        for summary_path in sorted(Path(self.log_dir).glob("*/summary_full.csv")):
            with summary_path.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    benchmark = row.get("Dataset") or summary_path.parent.name
                    runtime = "N/A"
                    for metric_name in (
                        "Accuracy (%) ⬆️",
                        "Error (%) ⬇️",
                        "Attack success rate (%) ⬇️",
                    ):
                        score = self._parse_score(row.get(metric_name))
                        if score is None:
                            continue
                        results.append(
                            OutputEntry(
                                {
                                    "Run ID": run_id,
                                    "Framework": self.FRAMEWORK_NAME,
                                    "Benchmark": benchmark,
                                    "Metric": metric_name,
                                    "Score": score,
                                    "Runtime (sec)": runtime,
                                }
                            )
                        )
        return results

    @staticmethod
    def _parse_score(value: str | None) -> float | None:
        if value is None or value == "":
            return None
        try:
            return float(value)
        except ValueError:
            return None

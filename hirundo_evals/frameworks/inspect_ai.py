import logging
import os
from datetime import datetime
from pathlib import Path

from bidict import bidict
from inspect_ai.log import EvalLog

from ._base import BaseEvalFrameworkWrapper, OutputEntry


class InspectWrapper(BaseEvalFrameworkWrapper):
    """
    Wrapper for the inspect-ai framework.

    Args:
        model: The model to evaluate.
        tasks: The tasks/benchmarks to evaluate.
    """

    TASK_TO_BENCHMARK = bidict(
        {
            "aime25": "inspect_evals/aime2025",
            "gpqa": "inspect_evals/gpqa_diamond",
            "ifeval": "inspect_evals/ifeval",
            "livecodebench": "inspect_evals/livecodebench_pro",
            "mmlu-pro": "inspect_evals/mmlu_pro",
            "scicode": "inspect_evals/scicode",
        }
    )

    FINAL_METRIC_BY_BENCHMARK = {
        "aime25": "accuracy",
        "gpqa": "accuracy",
        "ifeval": "final_acc",
        "ifbench": "final_acc",
        "livecodebench": "accuracy",
        "mmlu-pro": "accuracy",
        "scicode": "percentage_main_problems_solved",
    }

    @staticmethod
    def _inspect_task_names(tasks: list[str]) -> list[str]:
        """
        Convert the task names to a format compatible with inspect-ai.

        Args:
            tasks: The task names to convert.

        Returns:
            The converted task names.
        """
        if not tasks:
            raise ValueError("No tasks provided")

        converted_tasks = []
        unknown_tasks = []
        for task in tasks:
            # If the task is already in the format compatible with inspect-ai, add it to the list
            if task.startswith("inspect_evals/"):
                converted_tasks.append(task)
                continue
            # If the task is not in the mapping, add it to the list of unknown tasks
            if task not in InspectWrapper.TASK_TO_BENCHMARK:
                unknown_tasks.append(task)
                continue
            # Add the converted task to the list of converted tasks
            converted_tasks.append(InspectWrapper.TASK_TO_BENCHMARK[task])
        # Log any unknown tasks that were not converted
        if unknown_tasks:
            logging.warning(
                f"Unknown tasks: {', '.join(unknown_tasks)}. These tasks will be ignored."
            )
        # If no valid tasks were converted, raise an error
        if not converted_tasks:
            raise ValueError(f"No valid tasks provided ({', '.join(tasks)})")

        return converted_tasks

    @staticmethod
    def _inspect_model_name(model: str) -> str:
        """
        Convert the model name to a format compatible with inspect-ai.

        Args:
            model: The model name to convert.

        Returns:
            The converted model name.
        """
        if "/" not in model:
            return model

        provider, _ = model.split("/", 1)
        if provider in {"hf", "openai"}:
            return model

        return f"hf/{model}"

    def get_cli_cmd(
        self,
        model: str | None = None,
        model_base_url: str | None = None,
        extra: list[str] | None = None,
    ) -> list[str]:
        cmd = [
            "inspect",
            "eval",
            *self._inspect_task_names(self.tasks),
            "--model",
            self._inspect_model_name(model or self.model),
            "--log-dir",
            self.log_dir,
        ]
        if model_base_url:
            cmd.extend(["--model-base-url", model_base_url])
        if extra:
            cmd.extend(extra)

        return cmd

    @staticmethod
    def _get_runtime_from_timestamps(started_at: str, completed_at: str) -> int | str:
        """
        Calculate runtime (in seconds) from ISO format timestamp strings.

        Args:
            started_at: The start timestamp.
            completed_at: The end timestamp.

        Returns:
            The runtime in seconds.
            "N/A" if the runtime cannot be calculated.
        """
        try:
            # inspect_ai timestamps are usually ISO 8601 formatted strings
            # replace Z with +00:00 to make it compatible with python's fromisoformat
            start = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
            end = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
            duration = end - start
            # Strip microseconds for cleaner display
            duration_seconds = int(duration.total_seconds())
            return duration_seconds
        except Exception:
            return "N/A"

    def prepare_results(self) -> list[OutputEntry]:
        """
        Prepare the results of the evaluation for CSV export.

        Returns:
            The results of the evaluation for CSV export.
        """
        # Load the logs from the JSON files
        logs: list[EvalLog] = []
        for file in os.listdir(self.log_dir):
            if not file.endswith(".json"):
                continue
            with open(os.path.join(self.log_dir, file)) as f:
                raw_json = f.read()
                logs.append(EvalLog.model_validate_json(raw_json))
        # Prepare results for CSV export
        results: list[OutputEntry] = []
        run_id = Path(self.log_dir).name
        for log in logs:
            task_name = log.eval.task
            alias = InspectWrapper.TASK_TO_BENCHMARK.inv.get(task_name, task_name)
            target_metric = InspectWrapper.FINAL_METRIC_BY_BENCHMARK.get(alias, "unknown_metric")
            status = log.status
            score_value = "N/A"
            runtime = "N/A"
            # Extract runtime from log.stats if available
            if (
                log.stats
                and hasattr(log.stats, "started_at")
                and hasattr(log.stats, "completed_at")
            ):
                runtime = self._get_runtime_from_timestamps(
                    log.stats.started_at, log.stats.completed_at
                )
            # If the task completed successfully, extract the targeted metric
            if status == "success" and log.results and log.results.scores:
                # inspect_ai stores metrics inside score objects
                for score in log.results.scores:
                    if target_metric in score.metrics:
                        score_value = score.metrics[target_metric].value * 100.0
                        break
                # Fallback if the expected metric name wasn't found in the results
                if score_value == "N/A":
                    score_value = f"Metric '{target_metric}' not found"
            elif status != "success":
                score_value = f"Failed ({status})"
            # Add the data to the CSV data
            results.append(
                OutputEntry(
                    {
                        "Run ID": run_id,
                        "Framework": "inspect-ai",
                        "Benchmark": alias,
                        "Metric": target_metric + " (%) ⬆️",
                        "Score": score_value,
                        "Runtime (sec)": runtime,
                    }
                )
            )
            # Log the results
            logging.info(
                f"Task: {alias} | Status: {status} | {target_metric}: {score_value} | Runtime: {runtime}"
            )

        return results

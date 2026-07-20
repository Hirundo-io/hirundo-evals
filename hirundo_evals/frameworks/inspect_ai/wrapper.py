import logging
import os
from datetime import datetime
from pathlib import Path
from typing import TypedDict

from bidict import bidict
from inspect_ai.log import EvalLog

from hirundo_evals.frameworks._base import BaseEvalFrameworkWrapper, OutputEntry
from hirundo_evals.utils.cli import clean_cli_args


class InspectScore(TypedDict):
    """
    Score definition for inspect-ai.

    Args:
        name: The name of the score metric in the inspect-ai logs.
        is_percentage: Whether the score is a percentage.
        is_higher_better: Whether a higher value is better.
        is_normalized: Whether the score is normalized. Relevant only for percentage scores.
    """

    name: str
    is_percentage: bool
    is_higher_better: bool
    is_normalized: bool


class InspectWrapper(BaseEvalFrameworkWrapper):
    """
    Wrapper for the inspect-ai framework.

    Args:
        model: The model to evaluate.
        tasks: The tasks/benchmarks to evaluate.
        log_dir: The directory in which to save the outputs.
    """

    TASK_GROUPS = {
        "nemo-skills": [
            "aime25",
            "gpqa",
            "ifeval",
            "livecodebench",
            "mmlu-pro",
            "scicode",
        ],
    }
    TASK_TO_BENCHMARK = bidict(
        {
            "aime25": "inspect_evals/aime2025",
            "gpqa": "inspect_evals/gpqa_diamond",
            "ifeval": "inspect_evals/ifeval",
            "livecodebench": "inspect_evals/livecodebench_pro",
            "mmlu-pro": "inspect_evals/mmlu_pro",
            "scicode": "inspect_evals/scicode",
            "xstest": "inspect_evals/xstest",
        }
    )
    FINAL_METRICS_BY_BENCHMARK = {
        "aime25": [
            InspectScore(
                name="accuracy",
                is_percentage=True,
                is_higher_better=True,
                is_normalized=True,
            )
        ],
        "gpqa": [
            InspectScore(
                name="accuracy",
                is_percentage=True,
                is_higher_better=True,
                is_normalized=True,
            )
        ],
        "ifeval": [
            InspectScore(
                name="final_acc",
                is_percentage=True,
                is_higher_better=True,
                is_normalized=True,
            )
        ],
        "livecodebench": [
            InspectScore(
                name="accuracy",
                is_percentage=True,
                is_higher_better=True,
                is_normalized=True,
            )
        ],
        "mmlu-pro": [
            InspectScore(
                name="accuracy",
                is_percentage=True,
                is_higher_better=True,
                is_normalized=True,
            )
        ],
        "scicode": [
            InspectScore(
                name="percentage_main_problems_solved",
                is_percentage=True,
                is_higher_better=True,
                is_normalized=False,
            ),
            InspectScore(
                name="percentage_subproblems_solved",
                is_percentage=True,
                is_higher_better=True,
                is_normalized=False,
            ),
        ],
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
        # Expand any existing task groups into individual tasks
        expanded_tasks = []
        for task in tasks:
            if task in InspectWrapper.TASK_GROUPS:
                expanded_tasks.extend(InspectWrapper.TASK_GROUPS[task])
            else:
                expanded_tasks.append(task)
        # Convert the individual tasks to the format compatible with inspect-ai
        converted_tasks = []
        unknown_tasks = []
        for task in expanded_tasks:
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
            "--log-format",
            "json",
        ]
        if model_base_url:
            cmd.extend(["--model-base-url", model_base_url])
        clean_extra = clean_cli_args(
            extra, ["--model", "--model-base-url", "--log-dir", "--log-format"]
        )
        cmd.extend(clean_extra)

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
            target_metrics = InspectWrapper.FINAL_METRICS_BY_BENCHMARK.get(alias)
            status = log.status
            # Extract runtime from log.stats if available
            if (
                log.stats
                and hasattr(log.stats, "started_at")
                and hasattr(log.stats, "completed_at")
            ):
                runtime = self._get_runtime_from_timestamps(
                    log.stats.started_at, log.stats.completed_at
                )
            else:
                runtime = "N/A"
            # Initialize the scores
            score_values: dict[str, float | str] = {}
            # If the task completed successfully, extract the targeted metrics
            # Otherwise, fill the score_values with the status
            if status != "success":
                if not target_metrics:
                    score_values["status"] = f"Failed ({status})"
                else:
                    for target_metric in target_metrics:
                        score_values[target_metric["name"]] = f"Failed ({status})"
            elif log.results and log.results.scores:
                # inspect_ai stores metrics inside score objects
                if not target_metrics:
                    add_scorer_prefix = len(log.results.scores) > 1
                    for score in log.results.scores:
                        for metric_name, metric in score.metrics.items():
                            if add_scorer_prefix:
                                metric_name = f"{score.name}: {metric_name}"
                            score_values[metric_name] = metric.value
                else:
                    for target_metric in target_metrics:
                        for score in log.results.scores:
                            if target_metric["name"] in score.metrics:
                                metric_name = target_metric["name"]
                                metric_value = score.metrics[metric_name].value
                                if target_metric["is_percentage"]:
                                    if target_metric["is_normalized"]:
                                        metric_value *= 100.0
                                    metric_name += " (%)"
                                if target_metric["is_higher_better"]:
                                    metric_name += " ⬆️"
                                else:
                                    metric_name += " ⬇️"
                                score_values[metric_name] = metric_value
                                break
                        else:
                            # Fallback if the expected metric name wasn't found in the results
                            score_values[target_metric["name"]] = (
                                f"Metric '{target_metric['name']}' not found"
                            )
            # Add the data to the CSV data
            for score_name, score_value in score_values.items():
                results.append(
                    OutputEntry(
                        {
                            "Run ID": run_id,
                            "Framework": "inspect-ai",
                            "Benchmark": alias,
                            "Metric": score_name,
                            "Score": score_value,
                            "Runtime (sec)": runtime,
                        }
                    )
                )
            # Log the results
            logging.info(f"Task: {alias} | Status: {status} | Runtime: {runtime}")

        return results

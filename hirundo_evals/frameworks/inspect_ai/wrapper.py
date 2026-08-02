import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict

from bidict import bidict
from inspect_ai import eval as inspect_eval
from inspect_ai.log import EvalLog, read_eval_log, write_eval_log

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
                converted_task = task
            # If the task is a known alias, convert it to the format compatible with inspect-ai
            elif task in InspectWrapper.TASK_TO_BENCHMARK:
                converted_task = InspectWrapper.TASK_TO_BENCHMARK[task]
            # If the task is not inspect-ai compatible or a known alias, add it to the list of unknown tasks
            else:
                unknown_tasks.append(task)
                continue
            # If the converted task is already in the list of converted tasks, skip it
            if converted_task in converted_tasks:
                continue
            # Add the converted task to the list of converted tasks
            converted_tasks.append(converted_task)
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
    def _parse_eval_args(extra: list[str] | None) -> dict[str, Any]:  # noqa: C901
        arguments = clean_cli_args(
            extra,
            ["--model", "--model-base-url", "--log-dir", "--log-format"],
        )
        parsed: dict[str, Any] = {}
        model_args: dict[str, Any] = {}
        model_roles: dict[str, str] = {}
        integer_options = {
            "epochs",
            "limit",
            "max-samples",
            "max-subprocesses",
            "sample-id",
            "time-limit",
            "token-limit",
            "turn-limit",
            "working-limit",
        }
        float_options = {"cost-limit", "temperature", "top-p"}

        index = 0
        while index < len(arguments):
            argument = arguments[index]
            if argument in {"-M", "--model-args"}:
                index += 1
                key, value = arguments[index].split("=", 1)
                try:
                    model_args[key] = json.loads(value)
                except json.JSONDecodeError:
                    model_args[key] = value
            elif argument == "--model-role":
                index += 1
                role, model = arguments[index].split("=", 1)
                model_roles[role] = model
            else:
                flag, separator, value = argument.partition("=")
                option = flag.lstrip("-")
                if not separator:
                    if index + 1 < len(arguments) and not arguments[
                        index + 1
                    ].startswith("-"):
                        index += 1
                        value = arguments[index]
                    else:
                        value = "false" if option.startswith("no-") else "true"
                        option = option.removeprefix("no-")
                option = option.replace("-", "_")
                if option in integer_options:
                    parsed[option] = int(value)
                elif option in float_options:
                    parsed[option] = float(value)
                elif value == "true" or value == "false":
                    parsed[option] = value == "true"
                else:
                    parsed[option] = value
            index += 1

        if model_args:
            parsed["model_args"] = model_args
        if model_roles:
            parsed["model_roles"] = model_roles
        return parsed

    def _export_json_logs(self) -> None:
        for eval_path in Path(self.log_dir).glob("*.eval"):
            try:
                log = read_eval_log(eval_path)
                write_eval_log(log, eval_path.with_suffix(".json"), format="json")
            except Exception:
                logging.warning("Could not export Inspect log %s", eval_path)

    def run_eval(
        self,
        model: str | None = None,
        model_base_url: str | None = None,
        extra: list[str] | None = None,
    ) -> list[EvalLog]:
        model_name = model or self.model
        model_args = self._parse_eval_args(extra)
        if Path(model_name).is_dir():
            model_name = "hf/local"
            model_args.setdefault("model_args", {})["model_path"] = model or self.model

        logs = inspect_eval(
            self._inspect_task_names(self.tasks),
            model=model_name,
            model_base_url=model_base_url,
            log_dir=self.log_dir,
            log_format="eval",
            **model_args,
        )
        self._export_json_logs()

        return logs

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

    def _load_logs(self) -> list[EvalLog]:
        logs: list[EvalLog] = []
        for eval_path in Path(self.log_dir).glob("*.eval"):
            try:
                logs.append(read_eval_log(eval_path))
            except Exception:
                logging.warning("Could not load Inspect log %s", eval_path)

        return logs

    @staticmethod
    def _target_metric_name(target_metric: InspectScore) -> str:
        metric_name = target_metric["name"]
        if target_metric["is_percentage"]:
            metric_name += " (%)"
        return metric_name + (" ⬆️" if target_metric["is_higher_better"] else " ⬇️")

    @staticmethod
    def _failed_score_values(
        status: str, target_metrics: list[InspectScore] | None
    ) -> dict[str, float | str]:
        if not target_metrics:
            return {"status": f"Failed ({status})"}

        return {
            InspectWrapper._target_metric_name(target_metric): f"Failed ({status})"
            for target_metric in target_metrics
        }

    @staticmethod
    def _all_score_values(scores) -> dict[str, float | str]:
        add_scorer_prefix = len(scores) > 1
        score_values: dict[str, float | str] = {}
        for score in scores:
            for metric_name, metric in score.metrics.items():
                if add_scorer_prefix:
                    metric_name = f"{score.name}: {metric_name}"
                score_values[metric_name] = metric.value

        return score_values

    @staticmethod
    def _target_score_values(
        scores, target_metrics: list[InspectScore]
    ) -> dict[str, float | str]:
        score_values: dict[str, float | str] = {}
        for target_metric in target_metrics:
            metric_name = InspectWrapper._target_metric_name(target_metric)
            for score in scores:
                if target_metric["name"] not in score.metrics:
                    continue
                metric_value = score.metrics[target_metric["name"]].value
                if target_metric["is_percentage"]:
                    if target_metric["is_normalized"]:
                        metric_value *= 100.0
                score_values[metric_name] = metric_value
                break
            else:
                score_values[metric_name] = (
                    f"Metric '{target_metric['name']}' not found"
                )

        return score_values

    @classmethod
    def _score_values(
        cls, log: EvalLog, target_metrics: list[InspectScore] | None
    ) -> dict[str, float | str]:
        if log.status != "success":
            return cls._failed_score_values(log.status, target_metrics)
        if not log.results or not log.results.scores:
            if target_metrics:
                return cls._target_score_values([], target_metrics)
            return {"status": "No scores available"}
        if not target_metrics:
            return cls._all_score_values(log.results.scores)

        return cls._target_score_values(log.results.scores, target_metrics)

    def _prepare_log_results(self, log: EvalLog, run_id: str) -> list[OutputEntry]:
        task_name = log.eval.task
        benchmark_alias = InspectWrapper.TASK_TO_BENCHMARK.inv.get(task_name, task_name)
        target_metrics = InspectWrapper.FINAL_METRICS_BY_BENCHMARK.get(benchmark_alias)
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

        results = [
            OutputEntry(
                {
                    "Run ID": run_id,
                    "Framework": "inspect-ai",
                    "Benchmark": benchmark_alias,
                    "Metric": score_name,
                    "Score": score_value,
                    "Runtime (sec)": runtime,
                }
            )
            for score_name, score_value in self._score_values(
                log, target_metrics
            ).items()
        ]
        status_icon = "✅" if log.status == "success" else "❌"
        logging.info(
            f"{status_icon} Task: {benchmark_alias} | "
            f"Status: {log.status} | Runtime: {runtime}"
        )

        return results

    def prepare_results(self, outputs: list[EvalLog] | None) -> list[OutputEntry]:
        """
        Prepare the results of the evaluation for CSV export.

        Returns:
            The results of the evaluation for CSV export.
        """
        logs = outputs or self._load_logs()
        results: list[OutputEntry] = []
        run_id = Path(self.log_dir).name
        for log in logs:
            results.extend(self._prepare_log_results(log, run_id))

        return results

import logging
from datetime import datetime
from importlib import import_module
from pathlib import Path
from typing import Any


def load_eval_logs(log_dir: str | Path) -> list[Any]:
    """
    Load valid Inspect AI JSON logs from a directory.

    Args:
        log_dir: The directory containing the Inspect AI JSON logs.

    Returns:
        A list of Inspect AI logs.
    """
    eval_log_type = import_module("inspect_ai.log").EvalLog
    logs: list[Any] = []
    for output_path in sorted(Path(log_dir).glob("*.json")):
        try:
            logs.append(
                eval_log_type.model_validate_json(
                    output_path.read_text(encoding="utf-8")
                )
            )
        except Exception as exc:
            logging.debug("Skipping non-Inspect JSON file %s: %s", output_path, exc)

    return logs


def runtime_from_timestamps(started_at: str, completed_at: str) -> int | str:
    """
    Calculate runtime (in seconds) from Inspect AI ISO format timestamp strings.

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


def log_runtime(log: Any) -> int | str:
    """
    Extract runtime from an Inspect AI log, if available.

    Args:
        log: The Inspect AI log.

    Returns:
        The runtime in seconds.
        "N/A" if the runtime cannot be calculated.
    """
    if log.stats and log.stats.started_at and log.stats.completed_at:
        return runtime_from_timestamps(
            log.stats.started_at,
            log.stats.completed_at,
        )

    return "N/A"


def score_metric_value(score: Any, preferred_metric: str = "mean") -> Any:
    """Return a score metric value, falling back to the score value."""
    metric = score.metrics.get(preferred_metric)
    if metric is None:
        metric = score.metrics.get("value")

    return metric.value if metric is not None else score.value

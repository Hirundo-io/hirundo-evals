from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._base import BaseEvalFrameworkWrapper


class EvalFramework(str, Enum):
    INSPECT = "inspect-ai"
    PINCHBENCH = "pinchbench"


def get_eval_framework_wrapper(
    framework: EvalFramework, model: str, tasks: list[str], log_dir: str | Path
) -> "BaseEvalFrameworkWrapper":
    """
    Get the evaluation framework wrapper for the given framework.

    Args:
        framework: The evaluation framework to get the wrapper for.
        model: The model to evaluate.
        tasks: The tasks/benchmarks to evaluate.
        log_dir: The directory in which to save the outputs.

    Returns:
        The evaluation framework wrapper.
    """
    if framework == EvalFramework.INSPECT:
        from .inspect_ai.wrapper import InspectWrapper

        wrapper = InspectWrapper
    elif framework == EvalFramework.PINCHBENCH:
        from .inspect_ai.external.pinchbench.wrapper import PinchBenchWrapper

        wrapper = PinchBenchWrapper
    else:
        raise NotImplementedError(
            f"Framework '{framework}' is not yet supported. "
            f"Currently supported frameworks: {', '.join(e.value for e in EvalFramework)}"
        )

    return wrapper(model, tasks, log_dir)

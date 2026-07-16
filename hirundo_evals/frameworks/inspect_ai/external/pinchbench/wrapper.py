from pathlib import Path

from hirundo_evals.frameworks.inspect_ai.external._base import InspectExternalWrapper


class PinchBenchWrapper(InspectExternalWrapper):
    """Run PinchBench through the pinned Inspect AI adapter."""

    INSPECT_WRAPPER_REPO = "https://github.com/zytoh0/pinch-wildclawbench-inspect.git"
    INSPECT_WRAPPER_COMMIT = "c7d22166667c38e4a831183045b65091dcc7120a"
    BENCHMARK_REPO = "https://github.com/pinchbench/skill.git"
    BENCHMARK_COMMIT = "819384ae830492365b8363fc26bc2602e73f216d"
    BENCHMARK_ROOT_ENV = "PINCHBENCH_ROOT"
    BENCHMARK_REQUIRED_PATH = Path("scripts/benchmark.py")
    MODEL_BASE_URL_ENV = "PINCHBENCH_MODEL_BASE_URL"
    MODEL_ENV = "PINCHBENCH_MODEL"
    API_KEY_ENV = "PINCHBENCH_API_KEY"
    INSPECT_TASK = "src/pinchbench/pinchbench.py@pinchbench"
    FRAMEWORK_NAME = "pinchbench"
    DEFAULT_MODE = "smoke"

    def task_args(self, model_id: str) -> list[str]:
        mode = self.DEFAULT_MODE
        suite: str | None = None
        if self.tasks:
            if self.tasks == ["all"]:
                mode = "full"
            elif self.tasks[0] in {"smoke", "subset", "full"}:
                mode = self.tasks[0]
                if len(self.tasks) > 1:
                    suite = ",".join(self.tasks[1:])
            else:
                mode = "subset"
                suite = ",".join(self.tasks)

        args = [
            "-T",
            f"mode={mode}",
            "-T",
            f"model={model_id}",
            "-T",
            f"output_root={Path(self.log_dir).resolve()}",
        ]
        if suite:
            args.extend(["-T", f"suite={suite}"])

        return args

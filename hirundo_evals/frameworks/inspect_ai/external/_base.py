import logging
import os
import shutil
import subprocess
from pathlib import Path

from hirundo_evals.frameworks._base import BaseEvalFrameworkWrapper, OutputEntry
from hirundo_evals.frameworks.inspect_ai._utils import (
    load_eval_logs,
    log_runtime,
    score_metric_value,
)


class InspectExternalWrapper(BaseEvalFrameworkWrapper):
    """Base wrapper for externally maintained Inspect AI benchmarks."""

    SUPPORTS_UNSERVED_MODELS = False
    INSPECT_WRAPPER_REPO: str
    INSPECT_WRAPPER_COMMIT: str
    BENCHMARK_REPO: str | None = None
    BENCHMARK_COMMIT: str | None = None
    BENCHMARK_ROOT_ENV: str
    BENCHMARK_REQUIRED_PATH: Path | None = None
    MODEL_BASE_URL_ENV: str | None = None
    MODEL_ENV: str | None = None
    API_KEY_ENV: str | None = None
    INSPECT_TASK: str
    FRAMEWORK_NAME: str
    REPOSITORY_CACHE_ENV = "HIRUNDO_EVALS_CACHE_DIR"
    API_KEY = "vllm-local"
    SCORE_IS_PERCENTAGE = True
    SCORE_IS_NORMALIZED = True
    SCORE_HIGHER_BETTER = True

    @staticmethod
    def _model_id(model: str) -> str:
        return model.removeprefix("openai/").removeprefix("vllm/")

    @classmethod
    def _require_tool(cls, command: str) -> str:
        if shutil.which(command) is None:
            raise RuntimeError(
                f"`{command}` is required for {cls.FRAMEWORK_NAME}. Install it and retry."
            )

        return command

    def task_args(self, model_id: str) -> list[str]:
        """Return Inspect task arguments specific to the benchmark."""
        return []

    def environment(
        self, model_id: str, model_base_url: str, benchmark_root: Path
    ) -> dict[str, str]:
        """Return benchmark-specific environment variables."""
        environment = {self.BENCHMARK_ROOT_ENV: str(benchmark_root)}
        if self.MODEL_BASE_URL_ENV:
            environment[self.MODEL_BASE_URL_ENV] = model_base_url
        if self.MODEL_ENV:
            environment[self.MODEL_ENV] = model_id
        if self.API_KEY_ENV:
            environment[self.API_KEY_ENV] = self.API_KEY

        return environment

    def get_cli_cmd(
        self,
        model: str | None = None,
        model_base_url: str | None = None,
        extra: list[str] | None = None,
    ) -> list[str]:
        if not model_base_url:
            raise ValueError(
                f"{self.FRAMEWORK_NAME} requires an OpenAI-compatible model base URL."
            )
        model_id = self._model_id(model or self.model)

        return [
            "uv",
            "run",
            "--with",
            "openai",
            "inspect",
            "eval",
            self.INSPECT_TASK,
            "--model",
            f"openai/{model_id}",
            "--model-base-url",
            model_base_url,
            "--log-dir",
            str(Path(self.log_dir).resolve()),
            "--log-format",
            "json",
            *self.task_args(model_id),
            *(extra or []),
        ]

    def _clone_at(self, repo: str, commit: str, destination: Path) -> None:
        if destination.is_dir():
            logging.info("Using existing repository copy at %s", destination)
            return

        destination.parent.mkdir(parents=True, exist_ok=True)
        git = self._require_tool("git")
        subprocess.run(  # noqa: S603
            [git, "clone", repo, str(destination)],
            check=True,
        )
        subprocess.run(  # noqa: S603
            [git, "-C", str(destination), "checkout", commit], check=True
        )

    def _repository_cache_path(self, name: str, commit: str) -> Path:
        cache_root = os.environ.get(self.REPOSITORY_CACHE_ENV)
        if cache_root is None:
            cache_root = os.environ.get("XDG_CACHE_HOME")
        if cache_root is None:
            cache_root = str(Path.home() / ".cache")
        return (
            Path(cache_root).expanduser().resolve()
            / "hirundo-evals"
            / self.FRAMEWORK_NAME
            / f"{name}-{commit}"
        )

    def _benchmark_root(self) -> Path:
        root = os.environ.get(self.BENCHMARK_ROOT_ENV)
        if root:
            benchmark_root = Path(root).expanduser().resolve()
        elif self.BENCHMARK_REPO and self.BENCHMARK_COMMIT:
            benchmark_root = self._repository_cache_path(
                "benchmark", self.BENCHMARK_COMMIT
            )
            self._clone_at(self.BENCHMARK_REPO, self.BENCHMARK_COMMIT, benchmark_root)
        else:
            raise ValueError(
                f"Set {self.BENCHMARK_ROOT_ENV} for {self.FRAMEWORK_NAME}."
            )

        if (
            self.BENCHMARK_REQUIRED_PATH
            and not (benchmark_root / self.BENCHMARK_REQUIRED_PATH).is_file()
        ):
            required_path = benchmark_root / self.BENCHMARK_REQUIRED_PATH
            raise ValueError(
                f"{self.BENCHMARK_ROOT_ENV} does not contain {required_path}"
            )

        return benchmark_root

    def _wrapper_root(self) -> Path:
        wrapper_root = self._repository_cache_path(
            "inspect", self.INSPECT_WRAPPER_COMMIT
        )
        self._clone_at(
            self.INSPECT_WRAPPER_REPO,
            self.INSPECT_WRAPPER_COMMIT,
            wrapper_root,
        )

        return wrapper_root

    def run(
        self,
        model: str | None = None,
        model_base_url: str | None = None,
        extra: list[str] | None = None,
    ) -> None:
        if not model_base_url:
            raise ValueError(
                f"{self.FRAMEWORK_NAME} requires --vllm-local or another "
                "OpenAI-compatible endpoint."
            )
        model_id = self._model_id(model or self.model)
        benchmark_root = self._benchmark_root()
        wrapper_root = self._wrapper_root()
        env = os.environ.copy()
        env.update(self.environment(model_id, model_base_url, benchmark_root))
        subprocess.run(  # noqa: S603
            self.get_cli_cmd(model, model_base_url, extra),
            check=True,
            cwd=wrapper_root,
            env=env,
        )

    def prepare_results(self) -> list[OutputEntry]:
        results: list[OutputEntry] = []
        run_id = Path(self.log_dir).name
        for log in load_eval_logs(self.log_dir):
            if not log.results or not log.results.scores:
                continue
            runtime = log_runtime(log)
            for score in log.results.scores:
                value = score_metric_value(score)
                if value is None:
                    continue
                if self.SCORE_IS_PERCENTAGE and self.SCORE_IS_NORMALIZED:
                    value = float(value) * 100.0
                direction = "⬆️" if self.SCORE_HIGHER_BETTER else "⬇️"
                percentage = " (%)" if self.SCORE_IS_PERCENTAGE else ""
                results.append(
                    OutputEntry(
                        {
                            "Run ID": run_id,
                            "Framework": self.FRAMEWORK_NAME,
                            "Benchmark": self.FRAMEWORK_NAME,
                            "Metric": f"{score.name}{percentage} {direction}",
                            "Score": float(value),
                            "Runtime (sec)": runtime,
                        }
                    )
                )
        if not results:
            logging.warning(
                "No %s Inspect AI logs found in %s", self.FRAMEWORK_NAME, self.log_dir
            )

        return results

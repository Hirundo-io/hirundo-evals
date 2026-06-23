import json
import logging
import os
import shlex
import shutil
import subprocess
from pathlib import Path

from ._base import BaseEvalFrameworkWrapper, OutputEntry


class PinchBenchWrapper(BaseEvalFrameworkWrapper):
    """
    Wrapper for PinchBench via the pinchbench/skill repository.
    """

    SKILL_REPO = "https://github.com/pinchbench/skill.git"
    DEFAULT_SUITE = "all"
    DEFAULT_RUNS = "1"
    DEFAULT_API_KEY = "vllm-local"
    DEFAULT_TOOL_CALL_PARSER = "hermes"

    def get_vllm_args(self, vllm_args: str | None = None) -> str:
        # Initialize the arguments
        args = shlex.split(vllm_args or "")
        # Add the default arguments
        if "--enable-auto-tool-choice" not in args:
            args.append("--enable-auto-tool-choice")
        if "--tool-call-parser" not in args:
            args.extend(["--tool-call-parser", self.DEFAULT_TOOL_CALL_PARSER])

        return shlex.join(args)

    def get_cli_cmd(
        self,
        model: str | None = None,
        model_base_url: str | None = None,
        extra: list[str] | None = None,
    ) -> list[str]:
        model_id = self._pinchbench_model_name(model or self.model)
        suite = ",".join(self.tasks) if self.tasks else self.DEFAULT_SUITE
        return [
            "bash",
            "scripts/run.sh",
            "--model",
            f"vllm/{model_id}",
            "--suite",
            suite,
            "--runs",
            self.DEFAULT_RUNS,
            "--output-dir",
            str(Path(self.log_dir).resolve()),
            *(extra or []),
        ]

    def run(
        self,
        model: str | None = None,
        model_base_url: str | None = None,
        extra: list[str] | None = None,
    ) -> None:
        if not model_base_url:
            raise ValueError(
                "PinchBench requires --vllm-local so OpenClaw can reach a local model server."
            )

        model_id = self._pinchbench_model_name(model or self.model)
        env = os.environ.copy()
        env["VLLM_API_KEY"] = self.DEFAULT_API_KEY

        self._ensure_tooling()
        self._configure_openclaw(model_id, model_base_url, env)
        skill_dir = self._ensure_skill_repo()
        cmd = self.get_cli_cmd(model_id, model_base_url, extra)
        subprocess.run(cmd, check=True, cwd=skill_dir, env=env)  # noqa: S603

    def prepare_results(self) -> list[OutputEntry]:
        results: list[OutputEntry] = []
        for output_path in sorted(Path(self.log_dir).glob("*.json")):
            with output_path.open(encoding="utf-8") as f:
                output = json.load(f)

            run_id = Path(self.log_dir).name
            runtime = output.get("efficiency", {}).get(
                "total_execution_time_seconds", "N/A"
            )

            tasks = output.get("tasks", [])
            for task in tasks:
                grading = task.get("grading", {})
                score = grading.get("mean")
                if score is None:
                    continue
                results.append(
                    OutputEntry(
                        {
                            "Run ID": run_id,
                            "Framework": "pinchbench",
                            "Benchmark": str(task.get("task_id", "unknown_task")),
                            "Metric": "overall_score (%) ⬆️",
                            "Score": float(score) * 100.0,
                            "Runtime (sec)": int(task.get("execution_time", runtime)),
                        }
                    )
                )

        if not results:
            logging.warning("No PinchBench JSON outputs found in %s", self.log_dir)

        return results

    @staticmethod
    def _pinchbench_model_name(model: str) -> str:
        return model.removeprefix("openai/").removeprefix("vllm/")

    @staticmethod
    def _require_tool(command: str, install_hint: str) -> None:
        if shutil.which(command) is None:
            raise RuntimeError(
                f"`{command}` is required for PinchBench. {install_hint}"
            )

    def _ensure_tooling(self) -> None:
        self._require_tool("git", "Install git and retry.")
        self._require_tool(
            "openclaw", "Install it with `npm install -g openclaw@latest`."
        )

    def _ensure_skill_repo(self) -> str:
        skill_dir = os.path.join(self.log_dir, "pinchbench-skill")
        if os.path.isdir(skill_dir):
            logging.info("Using existing PinchBench skill repo at %s", skill_dir)
            return skill_dir

        subprocess.run(  # noqa: S603
            ["git", "clone", "--depth", "1", self.SKILL_REPO, skill_dir],  # noqa: S607
            check=True,
        )  # noqa: S607
        return skill_dir

    def _configure_openclaw(
        self, model_id: str, model_base_url: str, env: dict[str, str]
    ) -> None:
        config = {
            "baseUrl": model_base_url,
            "apiKey": self.DEFAULT_API_KEY,
            "api": "openai-completions",
            "models": [
                {
                    "id": model_id,
                    "name": f"Local vLLM ({model_id})",
                    "reasoning": False,
                    "input": ["text"],
                    "cost": {
                        "input": 0,
                        "output": 0,
                        "cacheRead": 0,
                        "cacheWrite": 0,
                    },
                    "contextWindow": 128000,
                    "maxTokens": 8192,
                }
            ],
        }
        self._openclaw_config_set("models.providers.vllm", config, env)
        self._openclaw_config_set(
            "agents.defaults.model", {"primary": f"vllm/{model_id}"}, env
        )

    @staticmethod
    def _openclaw_config_set(path: str, value: object, env: dict[str, str]) -> None:
        subprocess.run(  # noqa: S603
            [  # noqa: S607
                "openclaw",
                "config",
                "set",
                path,
                json.dumps(value),
                "--strict-json",
                "--merge",
            ],
            check=True,
            env=env,
        )

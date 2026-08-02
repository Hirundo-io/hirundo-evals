import asyncio
import signal
from typing import cast

import pytest

from hirundo_evals.utils import vllm


class FakeProcess:
    def __init__(self, returncode: int | None = None) -> None:
        self.pid = 1234
        self.returncode = returncode
        self.wait_calls = 0

    async def wait(self) -> int:
        self.wait_calls += 1
        self.returncode = 0
        return self.returncode


def run(coroutine):
    return asyncio.run(coroutine)


def test_readiness_cancellation_cleans_up(monkeypatch) -> None:
    process = FakeProcess()
    cleanup_calls = []
    to_thread_calls = 0

    async def fake_to_thread(function, *args):
        nonlocal to_thread_calls
        to_thread_calls += 1
        if to_thread_calls == 2:
            raise asyncio.CancelledError
        return False

    async def fake_cleanup(cleanup_process):
        cleanup_calls.append(cleanup_process)

    async def fake_create_subprocess(*args, **kwargs):
        return process

    monkeypatch.setattr(vllm.asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(vllm.asyncio, "create_subprocess_exec", fake_create_subprocess)
    monkeypatch.setattr(vllm, "_cleanup_vllm_process", fake_cleanup)

    async def exercise() -> None:
        with pytest.raises(asyncio.CancelledError):
            async with vllm.serve_vllm("model", timeout=10):
                pass

    run(exercise())

    assert cleanup_calls == [process]


def test_startup_timeout_cleans_up(monkeypatch) -> None:
    process = FakeProcess()
    cleanup_calls = []

    async def fake_to_thread(function, *args):
        return False

    async def fake_create_subprocess(*args, **kwargs):
        return process

    async def fake_cleanup(cleanup_process):
        cleanup_calls.append(cleanup_process)

    monkeypatch.setattr(vllm.asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(vllm.asyncio, "create_subprocess_exec", fake_create_subprocess)
    monkeypatch.setattr(vllm, "_cleanup_vllm_process", fake_cleanup)

    async def exercise() -> None:
        with pytest.raises(RuntimeError, match="failed to start"):
            async with vllm.serve_vllm("model", timeout=0):
                pass

    run(exercise())

    assert cleanup_calls == [process]


def test_process_exit_before_readiness_cleans_up(monkeypatch) -> None:
    process = FakeProcess(returncode=1)
    cleanup_calls = []

    async def fake_to_thread(function, *args):
        return False

    async def fake_create_subprocess(*args, **kwargs):
        return process

    async def fake_cleanup(cleanup_process):
        cleanup_calls.append(cleanup_process)

    monkeypatch.setattr(vllm.asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(vllm.asyncio, "create_subprocess_exec", fake_create_subprocess)
    monkeypatch.setattr(vllm, "_cleanup_vllm_process", fake_cleanup)

    async def exercise() -> None:
        with pytest.raises(RuntimeError, match="exit code 1"):
            async with vllm.serve_vllm("model", timeout=10):
                pass

    run(exercise())

    assert cleanup_calls == [process]


def test_terminate_process_group_uses_sigterm_when_process_exits(
    monkeypatch,
) -> None:
    process = FakeProcess()
    signals = []

    monkeypatch.setattr(
        vllm.os,
        "killpg",
        lambda pid, sig: signals.append((pid, sig)),
    )

    run(vllm._terminate_process_group(cast("asyncio.subprocess.Process", process)))

    assert signals == [(process.pid, signal.SIGTERM)]
    assert process.wait_calls == 1


def test_terminate_process_group_uses_sigkill_after_sigterm_timeout(
    monkeypatch,
) -> None:
    process = FakeProcess()
    signals = []

    async def wait() -> int:
        process.wait_calls += 1
        if process.wait_calls == 1:
            raise TimeoutError
        process.returncode = -signal.SIGKILL
        return process.returncode

    process.wait = wait
    monkeypatch.setattr(
        vllm.os,
        "killpg",
        lambda pid, sig: signals.append((pid, sig)),
    )

    run(vllm._terminate_process_group(cast("asyncio.subprocess.Process", process)))

    assert signals == [
        (process.pid, signal.SIGTERM),
        (process.pid, signal.SIGKILL),
    ]
    assert process.wait_calls == 2


def test_preoccupied_port_is_rejected_before_starting_process(monkeypatch) -> None:
    create_calls = []

    async def fake_to_thread(function, *args):
        return True

    async def fake_create_subprocess(*args, **kwargs):
        create_calls.append((args, kwargs))
        return FakeProcess()

    monkeypatch.setattr(vllm.asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(vllm.asyncio, "create_subprocess_exec", fake_create_subprocess)

    async def exercise() -> None:
        with pytest.raises(RuntimeError, match="already responding"):
            async with vllm.serve_vllm("model"):
                pass

    run(exercise())

    assert create_calls == []

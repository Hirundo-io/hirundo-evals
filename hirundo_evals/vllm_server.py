import asyncio
import contextlib
import http.client
import logging
import os
import signal
import sys
import time
import urllib.parse


def _is_server_ready(url: str) -> bool:
    try:
        parsed = urllib.parse.urlparse(url)
        if not parsed.hostname:
            return False

        if parsed.scheme == "https":
            conn = http.client.HTTPSConnection(parsed.hostname, parsed.port, timeout=1)
        else:
            conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=1)

        try:
            conn.request("GET", parsed.path)
            response = conn.getresponse()
            return response.status == 200
        finally:
            conn.close()
    except Exception:
        return False


async def _terminate_process_group(
    process: asyncio.subprocess.Process, timeout: int = 30
) -> None:
    if process.returncode is not None:
        return

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return

    try:
        await asyncio.wait_for(process.wait(), timeout=timeout)
    except TimeoutError:
        logging.warning("vLLM server ignored SIGTERM; killing process group...")
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        await process.wait()


@contextlib.asynccontextmanager
async def serve_vllm(
    model: str, vllm_args: list[str] | None = None, port: int = 8000, timeout: int = 600
):
    """
    Context manager to start and stop a local vLLM OpenAI-compatible server.

    Args:
        model: The model to serve.
        vllm_args: Additional arguments for the vLLM server.
        port: The port to serve the vLLM server on.
        timeout: The timeout in seconds to wait for the vLLM server to start.
    """
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model,
        "--port",
        str(port),
    ]
    if vllm_args:
        cmd.extend(vllm_args)

    server_url = f"http://localhost:{port}/v1"
    if await asyncio.to_thread(_is_server_ready, f"{server_url}/models"):
        raise RuntimeError(
            f"vLLM server endpoint is already responding on port {port}; "
            "choose a different --vllm-port or stop the existing server."
        )

    # Inherit stdout/stderr so vLLM startup and runtime logs are visible.
    process = await asyncio.create_subprocess_exec(
        *cmd,
        start_new_session=True,
    )

    # Poll for server readiness
    start_time = time.monotonic()
    while time.monotonic() - start_time < timeout:
        if process.returncode is not None:
            raise RuntimeError(
                f"vLLM server process exited before becoming ready "
                f"(exit code {process.returncode})"
            )
        if await asyncio.to_thread(_is_server_ready, f"{server_url}/models"):
            break
        await asyncio.sleep(1)
    else:
        await _terminate_process_group(process)
        raise RuntimeError(f"❌ vLLM server failed to start within {timeout} seconds")

    try:
        yield server_url
    finally:
        logging.info("🛑 Shutting down managed vLLM server...")
        await _terminate_process_group(process)
        logging.info("✅ Server safely terminated.")

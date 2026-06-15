import asyncio
import contextlib
import http.client
import shlex
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


@contextlib.asynccontextmanager
async def serve_vllm(
    model: str, vllm_args: str | None = None, port: int = 8000, timeout: int = 120
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
        cmd.extend(shlex.split(vllm_args))

    # Start the server subprocess
    process = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )

    server_url = f"http://localhost:{port}/v1"

    # Poll for server readiness
    start_time = time.monotonic()
    while time.monotonic() - start_time < timeout:
        if await asyncio.to_thread(_is_server_ready, f"{server_url}/models"):
            break
        await asyncio.sleep(1)
    else:
        process.terminate()
        await process.wait()
        raise RuntimeError(f"vLLM server failed to start within {timeout} seconds")

    try:
        yield server_url
    finally:
        process.terminate()
        await process.wait()

import logging
from collections.abc import Iterable


def clean_cli_args(
    cli_args: list[str] | None, managed_flags: Iterable[str]
) -> list[str]:
    """
    Remove caller arguments managed by ``serve_vllm`` itself.

    Args:
        cli_args: The CLI arguments to clean.
        managed_flags: The managed flags to remove.

    Returns:
        The cleaned CLI arguments.
    """
    managed_flags = set(managed_flags)
    cleaned_args: list[str] = []
    arguments = cli_args or []
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        flag = argument.split("=", 1)[0]
        if flag in managed_flags:
            logging.warning(
                "Ignoring %s from vLLM arguments; it is managed by serve_vllm",
                argument,
            )
            index += 1
            if (
                "=" not in argument
                and index < len(arguments)
                and not arguments[index].startswith("-")
            ):
                index += 1
            continue
        cleaned_args.append(argument)
        index += 1

    return cleaned_args

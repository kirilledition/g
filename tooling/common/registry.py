"""Tool registry helpers for grouped Hydra entrypoints."""

from __future__ import annotations

import typing
from dataclasses import dataclass

if typing.TYPE_CHECKING:
    import omegaconf


@dataclass(frozen=True)
class ToolSpec[ArgumentsT]:
    """Registered grouped tool implementation.

    Attributes:
        build_arguments: Function that converts a composed config to arguments.
        run: Function that executes the tool.

    """

    build_arguments: typing.Callable[[omegaconf.DictConfig], ArgumentsT]
    run: typing.Callable[[ArgumentsT], None]


def dispatch_tool(config: omegaconf.DictConfig, registry: typing.Mapping[str, ToolSpec[typing.Any]]) -> None:
    """Dispatch a grouped tool by ``tool.name``.

    Args:
        config: Composed Hydra config with a ``tool.name`` value.
        registry: Tool registry keyed by stable tool name.

    Raises:
        KeyError: If the config is missing ``tool.name`` or the tool is unknown.

    """
    if "tool" not in config or "name" not in config.tool:
        message = "Grouped tooling configs must contain tool.name."
        raise KeyError(message)
    tool_name = str(config.tool.name)
    tool_spec = registry.get(tool_name)
    if tool_spec is None:
        accepted_names = ", ".join(sorted(registry))
        message = f"Unknown tool.name `{tool_name}`. Accepted values: {accepted_names}."
        raise KeyError(message)
    tool_spec.run(tool_spec.build_arguments(config))


def registered_tool_names(registry: typing.Mapping[str, ToolSpec[typing.Any]]) -> tuple[str, ...]:
    """Return stable sorted tool names for documentation checks.

    Args:
        registry: Tool registry.

    Returns:
        Sorted tool names.

    """
    return tuple(sorted(registry))

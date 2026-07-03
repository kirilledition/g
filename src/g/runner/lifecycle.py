"""Runner-local lifecycle helpers."""

from __future__ import annotations

import typing

from g.engine import shutdown

if typing.TYPE_CHECKING:
    from g.engine.shutdown import GracefulShutdownRequested, ShutdownSignal
else:
    GracefulShutdownRequested = shutdown.GracefulShutdownRequested
    ShutdownSignal = shutdown.ShutdownSignal


def install_graceful_shutdown_handlers() -> shutdown.GracefulShutdownController:
    """Create a controller that installs default graceful shutdown handlers."""
    return shutdown.install_graceful_shutdown_handlers()

"""Native-dispatch lifecycle helpers."""

from __future__ import annotations

import typing

from g.engine import shutdown

if typing.TYPE_CHECKING:
    from g.engine.shutdown import GracefulShutdownRequested, ShutdownSignal
else:
    GracefulShutdownRequested = shutdown.GracefulShutdownRequested
    ShutdownSignal = shutdown.ShutdownSignal

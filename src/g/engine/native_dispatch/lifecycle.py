"""Native-dispatch lifecycle helpers."""

from __future__ import annotations

from g.runner import lifecycle as runner_lifecycle

GracefulShutdownRequested = runner_lifecycle.GracefulShutdownRequested
ShutdownSignal = runner_lifecycle.ShutdownSignal

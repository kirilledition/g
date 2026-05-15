
from __future__ import annotations
import contextlib
import collections.abc
import jax.profiler


@contextlib.contextmanager
def profiled_regenie2_linear_chunk_step(chunk_number: int) -> collections.abc.Iterator[None]:
    """Wrap linear chunk compute and accumulate tracing."""
    with jax.profiler.StepTraceAnnotation("regenie2_linear_chunk", step_num=chunk_number):
        yield


@contextlib.contextmanager
def profiled_regenie2_binary_chunk_step(chunk_number: int) -> collections.abc.Iterator[None]:
    """Wrap binary chunk compute and accumulate tracing."""
    with jax.profiler.StepTraceAnnotation("regenie2_binary_chunk", step_num=chunk_number):
        yield

"""Small queue-backed iterator read-ahead helpers."""

from __future__ import annotations

import queue as queue_module
import threading
import traceback
import typing
from dataclasses import dataclass

if typing.TYPE_CHECKING:
    import collections.abc


@dataclass(frozen=True)
class PrefetchEnvelope[ValueType]:
    """One prefetched iterator item or terminal signal."""

    value: ValueType | None = None
    error: Exception | None = None
    traceback_text: str | None = None
    is_finished: bool = False


def prefetch_iterator_values[ValueType](
    value_iterator: collections.abc.Iterator[ValueType],
    prefetch_chunks: int,
) -> collections.abc.Iterator[ValueType]:
    """Prefetch a bounded number of iterator values on a background thread."""
    if prefetch_chunks <= 0:
        return value_iterator

    delivery_queue: queue_module.Queue[PrefetchEnvelope[ValueType]] = queue_module.Queue(maxsize=prefetch_chunks)
    stop_event = threading.Event()

    def worker() -> None:
        try:
            for value in value_iterator:
                if stop_event.is_set():
                    return
                while True:
                    try:
                        delivery_queue.put(PrefetchEnvelope(value=value), timeout=0.1)
                        break
                    except queue_module.Full:
                        if stop_event.is_set():
                            return
            delivery_queue.put(PrefetchEnvelope(is_finished=True))
        except Exception as error:  # noqa: BLE001
            delivery_queue.put(
                PrefetchEnvelope(
                    error=error,
                    traceback_text=traceback.format_exc(),
                )
            )

    prefetch_thread = threading.Thread(target=worker, name="genotype-source-prefetch", daemon=True)
    prefetch_thread.start()

    def consume_prefetched_values() -> collections.abc.Iterator[ValueType]:
        try:
            while True:
                envelope = delivery_queue.get()
                if envelope.error is not None:
                    message = f"Genotype prefetch failed: {envelope.error}"
                    if envelope.traceback_text is not None:
                        message = f"{message}\n{envelope.traceback_text}"
                    raise RuntimeError(message) from envelope.error
                if envelope.is_finished:
                    return
                if envelope.value is None:
                    message = "Genotype prefetch returned an empty envelope without finishing."
                    raise RuntimeError(message)
                yield envelope.value
        finally:
            stop_event.set()
            prefetch_thread.join(timeout=0.5)

    return consume_prefetched_values()

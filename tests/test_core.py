from __future__ import annotations

from g import _core


def test_hello_from_bin_returns_expected_message() -> None:
    assert _core.hello_from_bin() == "Hello from g!"

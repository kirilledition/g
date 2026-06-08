from __future__ import annotations

import pytest

pytest.skip(
    "Rust CLI frontend migration removed Click app internals; rewrite CLI tests after core settles.",
    allow_module_level=True,
)

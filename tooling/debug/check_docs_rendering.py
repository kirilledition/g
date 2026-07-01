#!/usr/bin/env python3
"""Verify built documentation rendering hooks for dynamic diagrams."""

from __future__ import annotations

import argparse
import enum
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_SITE_ROOT = Path("documentation_rendered_website")
ALGORITHM_HTML_PATH = Path("public/algorithm/index.html")
MERMAID_SCRIPT_PATH = Path("assets/javascripts/mermaid.js")
SECTION_SCROLL_SCRIPT_PATH = Path("assets/javascripts/section-scroll.js")


class DocsRenderingViolationKind(enum.StrEnum):
    """Kinds of documentation rendering guardrail violations."""

    MISSING_BUILT_FILE = "missing_built_file"
    MISSING_MERMAID_SOURCE = "missing_mermaid_source"
    UNSUPPORTED_MERMAID_SHAPE = "unsupported_mermaid_shape"
    MISSING_MERMAID_PROMOTION = "missing_mermaid_promotion"
    MISSING_CONTENT_LOADED_HOOK = "missing_content_loaded_hook"


@dataclass(frozen=True)
class DocsRenderingViolation:
    """One documentation rendering guardrail violation.

    Attributes:
        kind: Violation category.
        path: Relevant path.
        message: Human-readable violation description.

    """

    kind: DocsRenderingViolationKind
    path: Path
    message: str


def read_built_file(site_root: Path, relative_path: Path) -> str | None:
    """Read a built documentation file when it exists."""
    full_path = site_root / relative_path
    if not full_path.is_file():
        return None
    return full_path.read_text(encoding="utf-8")


def collect_docs_rendering_violations(site_root: Path) -> tuple[DocsRenderingViolation, ...]:
    """Collect documentation rendering guardrail violations."""
    violations: list[DocsRenderingViolation] = []
    algorithm_html = read_built_file(site_root, ALGORITHM_HTML_PATH)
    mermaid_script = read_built_file(site_root, MERMAID_SCRIPT_PATH)
    section_scroll_script = read_built_file(site_root, SECTION_SCROLL_SCRIPT_PATH)

    if algorithm_html is None:
        violations.append(
            DocsRenderingViolation(
                kind=DocsRenderingViolationKind.MISSING_BUILT_FILE,
                path=site_root / ALGORITHM_HTML_PATH,
                message="built algorithm page is missing",
            )
        )
    if mermaid_script is None:
        violations.append(
            DocsRenderingViolation(
                kind=DocsRenderingViolationKind.MISSING_BUILT_FILE,
                path=site_root / MERMAID_SCRIPT_PATH,
                message="built Mermaid initializer is missing",
            )
        )
    if section_scroll_script is None:
        violations.append(
            DocsRenderingViolation(
                kind=DocsRenderingViolationKind.MISSING_BUILT_FILE,
                path=site_root / SECTION_SCROLL_SCRIPT_PATH,
                message="built section-scroll script is missing",
            )
        )
    if violations:
        return tuple(violations)

    assert algorithm_html is not None
    assert mermaid_script is not None
    assert section_scroll_script is not None

    if "flowchart TD" not in algorithm_html:
        violations.append(
            DocsRenderingViolation(
                kind=DocsRenderingViolationKind.MISSING_MERMAID_SOURCE,
                path=site_root / ALGORITHM_HTML_PATH,
                message="algorithm page does not contain the expected Mermaid graph source",
            )
        )

    mermaid_shapes = {
        "pre.mermaid": '<pre class="mermaid"',
        "code.language-mermaid": "language-mermaid",
        "code.mermaid": '<code class="mermaid"',
    }
    emitted_shapes = {shape_name for shape_name, marker in mermaid_shapes.items() if marker in algorithm_html}
    if not emitted_shapes:
        violations.append(
            DocsRenderingViolation(
                kind=DocsRenderingViolationKind.UNSUPPORTED_MERMAID_SHAPE,
                path=site_root / ALGORITHM_HTML_PATH,
                message="algorithm Mermaid graph is not emitted in a recognized HTML shape",
            )
        )

    required_mermaid_markers = (
        "pre.mermaid",
        "pre > code.language-mermaid",
        "pre > code.mermaid",
        "g:content-loaded",
        '.mermaid:not([data-processed="true"])',
    )
    for marker in required_mermaid_markers:
        if marker not in mermaid_script:
            violations.append(
                DocsRenderingViolation(
                    kind=DocsRenderingViolationKind.MISSING_MERMAID_PROMOTION,
                    path=site_root / MERMAID_SCRIPT_PATH,
                    message=f"Mermaid initializer does not contain `{marker}`",
                )
            )

    required_section_scroll_markers = (
        'new CustomEvent("g:content-loaded"',
        "detail: { root: insertedPage }",
    )
    for marker in required_section_scroll_markers:
        if marker not in section_scroll_script:
            violations.append(
                DocsRenderingViolation(
                    kind=DocsRenderingViolationKind.MISSING_CONTENT_LOADED_HOOK,
                    path=site_root / SECTION_SCROLL_SCRIPT_PATH,
                    message=f"section-scroll script does not contain `{marker}`",
                )
            )

    return tuple(violations)


def render_violation(violation: DocsRenderingViolation) -> str:
    """Render a guardrail violation for command-line output."""
    return f"{violation.path}: {violation.message}"


def run_tool(site_root: Path) -> int:
    """Verify built documentation dynamic rendering hooks."""
    violations = collect_docs_rendering_violations(site_root)
    if violations:
        print("Documentation rendering violations:")
        for violation in violations:
            print(f"  {render_violation(violation)}")
        return 1

    print(f"Documentation rendering guardrails passed for `{site_root}`.")
    return 0


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--site-root",
        type=Path,
        default=DEFAULT_SITE_ROOT,
        help="Built documentation site root.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the documentation rendering guardrail."""
    arguments = parse_arguments()
    return run_tool(arguments.site_root)


if __name__ == "__main__":
    sys.exit(main())

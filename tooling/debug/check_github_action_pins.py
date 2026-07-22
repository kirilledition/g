#!/usr/bin/env python3
"""Reject mutable external GitHub Action references."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_GITHUB_DIRECTORY = Path(".github")
ACTION_REFERENCE_PATTERN = re.compile(
    r"^\s*(?:-\s*)?uses:\s*(?P<quote>['\"]?)(?P<reference>[^'\"#\s]+)(?P=quote)"
    r"\s*(?:#\s*(?P<comment>.*))?$"
)
FULL_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
DOCKER_DIGEST_PATTERN = re.compile(r"docker://[^@\s]+@sha256:[0-9a-f]{64}")
RELEASE_TAG_COMMENT_PATTERN = re.compile(r"v[0-9]+(?:\.[0-9]+){0,2}(?:\s|$)")
ACTION_MANIFEST_PATTERNS = (
    "workflows/*.yml",
    "workflows/*.yaml",
    "actions/**/action.yml",
    "actions/**/action.yaml",
)


@dataclass(frozen=True)
class ActionPinViolation:
    """One mutable or undocumented external action reference."""

    path: Path
    line_number: int
    message: str


def is_external_action(reference: str) -> bool:
    """Return whether a reference executes an external repository action."""
    return not reference.startswith("./")


def collect_action_manifest_paths(github_directory: Path) -> tuple[Path, ...]:
    """Collect workflow and repository-local action manifest paths."""
    manifest_paths = {
        manifest_path
        for manifest_pattern in ACTION_MANIFEST_PATTERNS
        for manifest_path in github_directory.glob(manifest_pattern)
    }
    return tuple(sorted(manifest_paths))


def collect_action_pin_violations(github_directory: Path) -> tuple[ActionPinViolation, ...]:
    """Collect external action references that violate the immutable-pin policy."""
    violations: list[ActionPinViolation] = []
    for manifest_path in collect_action_manifest_paths(github_directory):
        for line_number, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
            if "uses:" not in line:
                continue
            action_match = ACTION_REFERENCE_PATTERN.match(line)
            if action_match is None:
                violations.append(
                    ActionPinViolation(
                        path=manifest_path,
                        line_number=line_number,
                        message="action reference must be a single scalar value",
                    )
                )
                continue
            reference = action_match.group("reference")
            if not is_external_action(reference):
                continue
            if reference.startswith("docker://"):
                if DOCKER_DIGEST_PATTERN.fullmatch(reference) is None:
                    violations.append(
                        ActionPinViolation(
                            path=manifest_path,
                            line_number=line_number,
                            message=f"Docker action `{reference}` is not pinned to a sha256 digest",
                        )
                    )
                continue
            if "@" not in reference:
                violations.append(
                    ActionPinViolation(
                        path=manifest_path,
                        line_number=line_number,
                        message=f"external action `{reference}` has no revision",
                    )
                )
                continue
            revision = reference.rsplit("@", maxsplit=1)[1]
            if FULL_COMMIT_PATTERN.fullmatch(revision) is None:
                violations.append(
                    ActionPinViolation(
                        path=manifest_path,
                        line_number=line_number,
                        message=f"external action `{reference}` is not pinned to a full commit SHA",
                    )
                )
            release_comment = action_match.group("comment")
            if release_comment is None or RELEASE_TAG_COMMENT_PATTERN.fullmatch(release_comment.strip()) is None:
                violations.append(
                    ActionPinViolation(
                        path=manifest_path,
                        line_number=line_number,
                        message=f"external action `{reference}` lacks a release-tag comment",
                    )
                )
    return tuple(violations)


def main() -> int:
    """Run the immutable GitHub Action reference check."""
    if not DEFAULT_GITHUB_DIRECTORY.is_dir():
        print(f"GitHub configuration directory does not exist: {DEFAULT_GITHUB_DIRECTORY}", file=sys.stderr)
        return 1
    violations = collect_action_pin_violations(DEFAULT_GITHUB_DIRECTORY)
    for violation in violations:
        print(f"{violation.path}:{violation.line_number}: {violation.message}", file=sys.stderr)
    if violations:
        return 1
    print("All external GitHub Actions use immutable commit or Docker digest pins.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

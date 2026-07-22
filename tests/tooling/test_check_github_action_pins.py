import typing

from tooling.debug import check_github_action_pins

if typing.TYPE_CHECKING:
    from pathlib import Path


def write_manifest(github_directory: Path, relative_path: str, contents: str) -> None:
    manifest_path = github_directory / relative_path
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(contents, encoding="utf-8")


def test_accepts_full_commit_pin_and_local_references(tmp_path: Path) -> None:
    github_directory = tmp_path / ".github"
    write_manifest(
        github_directory,
        "workflows/ci.yml",
        """jobs:
  check:
    steps:
      - uses: actions/checkout@0123456789abcdef0123456789abcdef01234567 # v7
      - uses: './.github/actions/local'
      - uses: docker://alpine@sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
""",
    )

    violations = check_github_action_pins.collect_action_pin_violations(github_directory)

    assert violations == ()


def test_rejects_mutable_action_revision(tmp_path: Path) -> None:
    github_directory = tmp_path / ".github"
    write_manifest(github_directory, "workflows/ci.yml", "steps:\n  - uses: actions/checkout@v7\n")

    violations = check_github_action_pins.collect_action_pin_violations(github_directory)

    assert len(violations) == 2
    assert "not pinned to a full commit SHA" in violations[0].message
    assert "lacks a release-tag comment" in violations[1].message


def test_rejects_pin_without_release_tag_comment(tmp_path: Path) -> None:
    github_directory = tmp_path / ".github"
    write_manifest(
        github_directory,
        "workflows/ci.yml",
        "steps:\n  - uses: actions/checkout@0123456789abcdef0123456789abcdef01234567\n",
    )

    violations = check_github_action_pins.collect_action_pin_violations(github_directory)

    assert len(violations) == 1
    assert "lacks a release-tag comment" in violations[0].message


def test_rejects_mutable_docker_action(tmp_path: Path) -> None:
    github_directory = tmp_path / ".github"
    write_manifest(github_directory, "workflows/ci.yml", "steps:\n  - uses: docker://alpine:3.23\n")

    violations = check_github_action_pins.collect_action_pin_violations(github_directory)

    assert len(violations) == 1
    assert "not pinned to a sha256 digest" in violations[0].message


def test_accepts_pinned_nested_composite_action_reference(tmp_path: Path) -> None:
    github_directory = tmp_path / ".github"
    write_manifest(
        github_directory,
        "actions/outer/nested/action.yaml",
        "steps:\n  - uses: actions/cache@0123456789abcdef0123456789abcdef01234567 # v4\n",
    )

    violations = check_github_action_pins.collect_action_pin_violations(github_directory)

    assert violations == ()


def test_rejects_mutable_nested_composite_action_reference(tmp_path: Path) -> None:
    github_directory = tmp_path / ".github"
    write_manifest(github_directory, "actions/outer/action.yml", "steps:\n  - uses: actions/cache@v4\n")

    violations = check_github_action_pins.collect_action_pin_violations(github_directory)

    assert len(violations) == 2
    assert all(violation.path.name == "action.yml" for violation in violations)
    assert "not pinned to a full commit SHA" in violations[0].message

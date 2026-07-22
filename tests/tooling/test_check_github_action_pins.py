import typing

from tooling.debug import check_github_action_pins

if typing.TYPE_CHECKING:
    from pathlib import Path


def write_workflow(workflow_directory: Path, contents: str) -> None:
    workflow_directory.mkdir()
    (workflow_directory / "ci.yml").write_text(contents, encoding="utf-8")


def test_accepts_full_commit_pin_and_local_references(tmp_path: Path) -> None:
    workflow_directory = tmp_path / "workflows"
    write_workflow(
        workflow_directory,
        """jobs:
  check:
    steps:
      - uses: actions/checkout@0123456789abcdef0123456789abcdef01234567 # v7
      - uses: './.github/actions/local'
      - uses: docker://alpine:3.23
""",
    )

    violations = check_github_action_pins.collect_action_pin_violations(workflow_directory)

    assert violations == ()


def test_rejects_mutable_action_revision(tmp_path: Path) -> None:
    workflow_directory = tmp_path / "workflows"
    write_workflow(workflow_directory, "steps:\n  - uses: actions/checkout@v7\n")

    violations = check_github_action_pins.collect_action_pin_violations(workflow_directory)

    assert len(violations) == 2
    assert "not pinned to a full commit SHA" in violations[0].message
    assert "lacks a release-tag comment" in violations[1].message


def test_rejects_pin_without_release_tag_comment(tmp_path: Path) -> None:
    workflow_directory = tmp_path / "workflows"
    write_workflow(
        workflow_directory,
        "steps:\n  - uses: actions/checkout@0123456789abcdef0123456789abcdef01234567\n",
    )

    violations = check_github_action_pins.collect_action_pin_violations(workflow_directory)

    assert len(violations) == 1
    assert "lacks a release-tag comment" in violations[0].message

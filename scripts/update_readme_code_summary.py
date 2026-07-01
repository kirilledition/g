from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import typing
from dataclasses import dataclass
from pathlib import Path

if typing.TYPE_CHECKING:
    import collections.abc


type JsonObject = dict[str, typing.Any]


SUMMARY_START_MARKER = "<!-- code-size-summary:start -->"
SUMMARY_END_MARKER = "<!-- code-size-summary:end -->"
COUNTED_DIRECTORIES = ("crates", "src")


@dataclass(frozen=True)
class ScriptArguments:
    """Parsed command-line arguments.

    Attributes:
        readme_path: Path to the README file that receives the generated block.
        cloc_command_arguments: Command used to execute cloc.
        check_only: Whether to verify the README without writing changes.

    """

    readme_path: Path
    cloc_command_arguments: tuple[str, ...]
    check_only: bool


@dataclass(frozen=True)
class CommandOutput:
    """Captured subprocess output.

    Attributes:
        standard_output: Captured standard output.
        standard_error: Captured standard error.

    """

    standard_output: str
    standard_error: str


@dataclass(frozen=True)
class LanguageLineCount:
    """Line counts for one cloc language row.

    Attributes:
        language_name: cloc language label.
        file_count: Number of files counted.
        blank_count: Number of blank lines.
        comment_count: Number of comment lines.
        code_count: Number of code lines.

    """

    language_name: str
    file_count: int
    blank_count: int
    comment_count: int
    code_count: int


@dataclass(frozen=True)
class CodeSizeSummary:
    """Rendered README code-size summary data.

    Attributes:
        cloc_url: cloc project URL reported by cloc.
        cloc_version: cloc version reported by cloc.
        language_line_counts: Per-language line counts sorted for display.
        total_line_count: Total line count row.

    """

    cloc_url: str
    cloc_version: str
    language_line_counts: tuple[LanguageLineCount, ...]
    total_line_count: LanguageLineCount


def parse_arguments(argument_values: collections.abc.Sequence[str]) -> ScriptArguments:
    """Parse command-line arguments.

    Args:
        argument_values: Raw command-line arguments.

    Returns:
        Parsed script arguments.

    """

    argument_parser = argparse.ArgumentParser(
        description="Update the README cloc summary for Git-tracked crates/ and src/ files.",
    )
    argument_parser.add_argument("--readme", type=Path, default=Path("README.md"))
    argument_parser.add_argument(
        "--cloc-command",
        default=os.environ.get("CLOC_COMMAND", "cloc"),
        help="Shell-like command used to run cloc, for example: 'perl /tmp/cloc'.",
    )
    argument_parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero when the generated README block is stale.",
    )
    parsed_arguments = argument_parser.parse_args(list(argument_values))
    cloc_command_arguments = tuple(shlex.split(typing.cast("str", parsed_arguments.cloc_command)))
    if not cloc_command_arguments:
        raise SystemExit("--cloc-command must not be empty.")
    return ScriptArguments(
        readme_path=typing.cast("Path", parsed_arguments.readme),
        cloc_command_arguments=cloc_command_arguments,
        check_only=typing.cast("bool", parsed_arguments.check),
    )


def run_command(
    command_arguments: collections.abc.Sequence[str],
    *,
    current_working_directory: Path,
    input_text: str | None = None,
) -> CommandOutput:
    """Run a command and return captured output.

    Args:
        command_arguments: Command and arguments to execute.
        current_working_directory: Directory where the command should run.
        input_text: Optional standard input text.

    Returns:
        Captured command output.

    Raises:
        SystemExit: If the command cannot be launched or exits unsuccessfully.

    """

    try:
        completed_process = subprocess.run(
            list(command_arguments),
            cwd=current_working_directory,
            input=input_text,
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError as error:
        command_name = command_arguments[0] if command_arguments else "<empty command>"
        print(f"Command not found: {command_name}", file=sys.stderr)
        raise SystemExit(127) from error

    if completed_process.returncode != 0:
        command_line = shlex.join(command_arguments)
        print(f"Command failed with exit code {completed_process.returncode}: {command_line}", file=sys.stderr)
        if completed_process.stdout:
            print(completed_process.stdout, file=sys.stderr)
        if completed_process.stderr:
            print(completed_process.stderr, file=sys.stderr)
        raise SystemExit(completed_process.returncode)

    return CommandOutput(
        standard_output=completed_process.stdout,
        standard_error=completed_process.stderr,
    )


def find_repository_root(current_working_directory: Path) -> Path:
    """Find the repository root with git.

    Args:
        current_working_directory: Directory where the search starts.

    Returns:
        Absolute repository root path.

    """

    command_output = run_command(
        ("git", "rev-parse", "--show-toplevel"),
        current_working_directory=current_working_directory,
    )
    return Path(command_output.standard_output.strip())


def list_tracked_source_paths(repository_root: Path) -> tuple[str, ...]:
    """List Git-tracked files under the counted source directories.

    Args:
        repository_root: Repository root path.

    Returns:
        Repository-relative paths to pass to cloc.

    """

    command_output = run_command(
        ("git", "ls-files", "--", *COUNTED_DIRECTORIES),
        current_working_directory=repository_root,
    )
    tracked_paths = tuple(path for path in command_output.standard_output.splitlines() if path)
    if not tracked_paths:
        counted_directory_list = ", ".join(COUNTED_DIRECTORIES)
        raise SystemExit(f"No tracked files found under: {counted_directory_list}")
    return tracked_paths


def get_json_object_field(document: JsonObject, field_name: str) -> JsonObject:
    """Read a JSON object field.

    Args:
        document: Parent JSON object.
        field_name: Field name to read.

    Returns:
        The child JSON object.

    Raises:
        SystemExit: If the field is missing or not an object.

    """

    field_value = document.get(field_name)
    if not isinstance(field_value, dict):
        raise SystemExit(f"Expected cloc JSON object field: {field_name}")
    return typing.cast("JsonObject", field_value)


def get_json_string_field(document: JsonObject, field_name: str) -> str:
    """Read a JSON string field.

    Args:
        document: Parent JSON object.
        field_name: Field name to read.

    Returns:
        The string value.

    Raises:
        SystemExit: If the field is missing or not a string.

    """

    field_value = document.get(field_name)
    if not isinstance(field_value, str):
        raise SystemExit(f"Expected cloc JSON string field: {field_name}")
    return field_value


def get_json_integer_field(document: JsonObject, field_name: str) -> int:
    """Read a JSON integer field.

    Args:
        document: Parent JSON object.
        field_name: Field name to read.

    Returns:
        The integer value.

    Raises:
        SystemExit: If the field is missing or not an integer.

    """

    field_value = document.get(field_name)
    if not isinstance(field_value, int):
        raise SystemExit(f"Expected cloc JSON integer field: {field_name}")
    return field_value


def parse_language_line_count(language_name: str, document: JsonObject) -> LanguageLineCount:
    """Parse one language row from cloc JSON.

    Args:
        language_name: Language label to display.
        document: cloc JSON object for the language.

    Returns:
        Parsed line count row.

    """

    return LanguageLineCount(
        language_name=language_name,
        file_count=get_json_integer_field(document, "nFiles"),
        blank_count=get_json_integer_field(document, "blank"),
        comment_count=get_json_integer_field(document, "comment"),
        code_count=get_json_integer_field(document, "code"),
    )


def parse_code_size_summary(cloc_output_text: str) -> CodeSizeSummary:
    """Parse cloc JSON output into display data.

    Args:
        cloc_output_text: JSON emitted by cloc.

    Returns:
        Parsed code-size summary.

    Raises:
        SystemExit: If cloc does not emit the expected JSON shape.

    """

    try:
        parsed_document = json.loads(cloc_output_text)
    except json.JSONDecodeError as error:
        raise SystemExit(f"Failed to parse cloc JSON output: {error}") from error
    if not isinstance(parsed_document, dict):
        raise SystemExit("Expected top-level cloc JSON object.")

    cloc_document = typing.cast("JsonObject", parsed_document)
    header_document = get_json_object_field(cloc_document, "header")
    total_document = get_json_object_field(cloc_document, "SUM")
    language_line_counts = tuple(
        sorted(
            (
                parse_language_line_count(language_name, typing.cast("JsonObject", language_document))
                for language_name, language_document in cloc_document.items()
                if language_name not in {"header", "SUM"} and isinstance(language_document, dict)
            ),
            key=lambda line_count: (-line_count.code_count, line_count.language_name),
        ),
    )

    return CodeSizeSummary(
        cloc_url=get_json_string_field(header_document, "cloc_url"),
        cloc_version=get_json_string_field(header_document, "cloc_version"),
        language_line_counts=language_line_counts,
        total_line_count=parse_language_line_count("Total", total_document),
    )


def run_cloc_summary(
    cloc_command_arguments: collections.abc.Sequence[str],
    repository_root: Path,
    tracked_paths: collections.abc.Sequence[str],
) -> CodeSizeSummary:
    """Run cloc for tracked source paths and parse the result.

    Args:
        cloc_command_arguments: Command used to execute cloc.
        repository_root: Repository root path.
        tracked_paths: Repository-relative paths to count.

    Returns:
        Parsed code-size summary.

    """

    cloc_input_text = "\n".join(tracked_paths) + "\n"
    command_output = run_command(
        (*cloc_command_arguments, "--json", "--list-file=-"),
        current_working_directory=repository_root,
        input_text=cloc_input_text,
    )
    return parse_code_size_summary(command_output.standard_output)


def format_markdown_integer(value: int) -> str:
    """Format an integer for a Markdown table.

    Args:
        value: Integer to format.

    Returns:
        Integer with thousands separators.

    """

    return f"{value:,}"


def render_table_row(line_count: LanguageLineCount, *, bold_label: bool) -> str:
    """Render one Markdown table row.

    Args:
        line_count: Line count row to render.
        bold_label: Whether to bold the language label.

    Returns:
        Markdown table row.

    """

    language_name = f"**{line_count.language_name}**" if bold_label else line_count.language_name
    return (
        f"| {language_name} | {format_markdown_integer(line_count.file_count)} "
        f"| {format_markdown_integer(line_count.blank_count)} "
        f"| {format_markdown_integer(line_count.comment_count)} "
        f"| {format_markdown_integer(line_count.code_count)} |"
    )


def render_summary_block(code_size_summary: CodeSizeSummary) -> str:
    """Render the generated README block.

    Args:
        code_size_summary: Summary data to render.

    Returns:
        Markdown block including generated-content markers.

    """

    counted_directory_list = " and ".join(f"`{directory}/`" for directory in COUNTED_DIRECTORIES)
    lines = [
        SUMMARY_START_MARKER,
        "## Code Size",
        "",
        (
            "Generated from Git-tracked files under "
            f"{counted_directory_list} using [`cloc`](https://github.com/AlDanial/cloc)."
        ),
        "",
        "| Language | Files | Blank | Comment | Code |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(
        render_table_row(line_count, bold_label=False) for line_count in code_size_summary.language_line_counts
    )
    lines.append(render_table_row(code_size_summary.total_line_count, bold_label=True))
    lines.extend(
        [
            "",
            f"`cloc` version: `{code_size_summary.cloc_version}`.",
            SUMMARY_END_MARKER,
        ],
    )
    return "\n".join(lines)


def update_readme_text(readme_text: str, summary_block: str) -> str:
    """Replace or append the generated README summary.

    Args:
        readme_text: Existing README text.
        summary_block: Generated Markdown block.

    Returns:
        README text with the summary block at the end.

    """

    summary_pattern = re.compile(
        rf"\n*{re.escape(SUMMARY_START_MARKER)}.*?{re.escape(SUMMARY_END_MARKER)}\n*",
        flags=re.DOTALL,
    )
    readme_without_summary = summary_pattern.sub("\n", readme_text).rstrip()
    return f"{readme_without_summary}\n\n{summary_block}\n"


def resolve_readme_path(repository_root: Path, readme_path: Path) -> Path:
    """Resolve the README path against the repository root.

    Args:
        repository_root: Repository root path.
        readme_path: User-provided README path.

    Returns:
        Absolute README path.

    """

    if readme_path.is_absolute():
        return readme_path
    return repository_root / readme_path


def main(argument_values: collections.abc.Sequence[str]) -> int:
    """Run the README code-size updater.

    Args:
        argument_values: Raw command-line arguments.

    Returns:
        Process exit code.

    """

    script_arguments = parse_arguments(argument_values)
    repository_root = find_repository_root(Path.cwd())
    readme_path = resolve_readme_path(repository_root, script_arguments.readme_path)
    tracked_paths = list_tracked_source_paths(repository_root)
    code_size_summary = run_cloc_summary(
        script_arguments.cloc_command_arguments,
        repository_root,
        tracked_paths,
    )
    current_readme_text = readme_path.read_text()
    summary_block = render_summary_block(code_size_summary)
    updated_readme_text = update_readme_text(current_readme_text, summary_block)

    if script_arguments.check_only:
        if updated_readme_text == current_readme_text:
            print("README code-size summary is up to date.")
            return 0
        print("README code-size summary is stale. Run scripts/update_readme_code_summary.py.", file=sys.stderr)
        return 1

    if updated_readme_text == current_readme_text:
        print("README code-size summary is already up to date.")
        return 0

    readme_path.write_text(updated_readme_text)
    print(f"Updated {readme_path.relative_to(repository_root)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

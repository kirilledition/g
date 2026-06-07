from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import tomllib
import typing
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

if typing.TYPE_CHECKING:
    import collections.abc


type JsonValue = None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]
type TomlValue = str | int | float | bool | list[TomlValue] | dict[str, TomlValue]
type TomlTable = dict[str, TomlValue]


@dataclass(frozen=True)
class CheckResult:
    """Result for one doctor check.

    Attributes:
        name: Human-readable check name.
        passed: Whether the check passed.
        detail: Redacted detail to print in the report.
        remediation: Actionable fix to print when the check fails.

    """

    name: str
    passed: bool
    detail: str
    remediation: str | None = None


@dataclass(frozen=True)
class CommandResult:
    """Captured result from a bounded subprocess check.

    Attributes:
        command_line: Shell-quoted command line for diagnostics.
        return_code: Process return code, or synthetic timeout/not-found code.
        output: Combined stdout and stderr.

    """

    command_line: str
    return_code: int
    output: str


@dataclass(frozen=True)
class DoctorArguments:
    """Command-line arguments for Symphony doctor.

    Attributes:
        repository_root: Repository root used for Git checks.
        symphony_env_file: Local Symphony environment file path.
        symphony_elixir_directory: Symphony Elixir checkout path.
        symphony_worktree_root: Root where Symphony issue worktrees are created.
        codex_config_path: Codex TOML config path.

    """

    repository_root: Path
    symphony_env_file: Path
    symphony_elixir_directory: Path
    symphony_worktree_root: Path
    codex_config_path: Path


@dataclass(frozen=True)
class Redactor:
    """Redacts known secret literals and common token shapes.

    Attributes:
        literal_secrets: Exact secret values that must not appear in output.

    """

    literal_secrets: tuple[str, ...]

    def with_secret_values(self, secret_values: collections.abc.Iterable[str]) -> Redactor:
        """Return a redactor that also hides the provided literal values.

        Args:
            secret_values: Additional secret strings to redact.

        Returns:
            A redactor with the new secret values included.

        """

        filtered_secret_values = tuple(secret_value for secret_value in secret_values if len(secret_value) >= 4)
        return Redactor(literal_secrets=(*self.literal_secrets, *filtered_secret_values))

    def redact(self, text: str) -> str:
        """Redact secrets from a string.

        Args:
            text: Text that may contain secrets.

        Returns:
            Text with configured secrets and common token patterns redacted.

        """

        redacted_text = text
        for secret_value in self.literal_secrets:
            redacted_text = redacted_text.replace(secret_value, "<redacted>")
        redacted_text = re.sub(r"lin_api_[A-Za-z0-9_\-]+", "lin_api_<redacted>", redacted_text)
        redacted_text = re.sub(r"(?i)Bearer\s+[A-Za-z0-9._~+/=-]+", "Bearer <redacted>", redacted_text)
        return re.sub(r"https://[^/\s:@]+:[^/\s@]+@", "https://<redacted>@", redacted_text)


@dataclass(frozen=True)
class LinearMcpConfig:
    """Linear MCP server configuration extracted from Codex config.

    Attributes:
        endpoint_address: MCP endpoint URL.
        http_headers: HTTP headers that Codex will send to the MCP server.

    """

    endpoint_address: str
    http_headers: dict[str, str]


@dataclass(frozen=True)
class LinearMcpConfigLoad:
    """Result of reading Codex Linear MCP configuration.

    Attributes:
        check_result: Report result for the configuration read.
        linear_mcp_config: Parsed config, when available.
        redactor: Redactor updated with MCP header values.

    """

    check_result: CheckResult
    linear_mcp_config: LinearMcpConfig | None
    redactor: Redactor


@dataclass(frozen=True)
class HttpResponse:
    """HTTP response captured without exposing request secrets.

    Attributes:
        status_code: HTTP status code.
        content_type: Response content type.
        body: Response body text.

    """

    status_code: int
    content_type: str
    body: str


@dataclass(frozen=True)
class DoctorResults:
    """Doctor results with the redactor needed for safe output.

    Attributes:
        check_results: Ordered check results.
        redactor: Final redactor containing all discovered secret values.

    """

    check_results: list[CheckResult]
    redactor: Redactor


def parse_arguments(argument_values: collections.abc.Sequence[str]) -> DoctorArguments:
    """Parse command-line arguments.

    Args:
        argument_values: Raw command-line arguments.

    Returns:
        Parsed doctor arguments.

    """

    codex_home = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex")))
    argument_parser = argparse.ArgumentParser(description="Check local Symphony runtime prerequisites.")
    argument_parser.add_argument("--repository-root", type=Path, required=True)
    argument_parser.add_argument("--symphony-env-file", type=Path, required=True)
    argument_parser.add_argument("--symphony-elixir-dir", type=Path, required=True)
    argument_parser.add_argument("--symphony-worktree-root", type=Path, required=True)
    argument_parser.add_argument("--codex-config-path", type=Path, default=codex_home / "config.toml")
    parsed_arguments = argument_parser.parse_args(list(argument_values))
    return DoctorArguments(
        repository_root=typing.cast("Path", parsed_arguments.repository_root),
        symphony_env_file=typing.cast("Path", parsed_arguments.symphony_env_file),
        symphony_elixir_directory=typing.cast("Path", parsed_arguments.symphony_elixir_dir),
        symphony_worktree_root=typing.cast("Path", parsed_arguments.symphony_worktree_root),
        codex_config_path=typing.cast("Path", parsed_arguments.codex_config_path),
    )


def run_command(
    command_arguments: collections.abc.Sequence[str],
    *,
    current_working_directory: Path | None = None,
    timeout_seconds: int = 15,
) -> CommandResult:
    """Run a bounded command and capture output.

    Args:
        command_arguments: Command and arguments to run.
        current_working_directory: Optional working directory.
        timeout_seconds: Timeout before the command is treated as failed.

    Returns:
        Captured command result.

    """

    command_line = shlex.join(command_arguments)
    try:
        completed_process = subprocess.run(
            list(command_arguments),
            cwd=current_working_directory,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except FileNotFoundError:
        return CommandResult(command_line=command_line, return_code=127, output="command not found")
    except subprocess.TimeoutExpired:
        return CommandResult(
            command_line=command_line,
            return_code=124,
            output=f"timed out after {timeout_seconds} seconds",
        )
    output_parts = [completed_process.stdout.strip(), completed_process.stderr.strip()]
    command_output = "\n".join(output_part for output_part in output_parts if output_part)
    return CommandResult(
        command_line=command_line,
        return_code=completed_process.returncode,
        output=command_output,
    )


def compact_output(output: str, *, maximum_length: int = 500) -> str:
    """Compact command or API output for one-line diagnostics.

    Args:
        output: Raw output text.
        maximum_length: Maximum output length.

    Returns:
        Whitespace-normalized output, truncated when needed.

    """

    compacted_output = " ".join(output.split())
    if len(compacted_output) <= maximum_length:
        return compacted_output
    return f"{compacted_output[:maximum_length]}..."


def check_symphony_environment_file(arguments: DoctorArguments) -> CheckResult:
    """Check how Symphony credentials were provided.

    Args:
        arguments: Doctor arguments.

    Returns:
        Check result.

    """

    if arguments.symphony_env_file.is_file():
        return CheckResult(
            name="Symphony env file",
            passed=True,
            detail=f"loaded {arguments.symphony_env_file}",
        )
    if os.environ.get("LINEAR_API_KEY") and os.environ.get("LINEAR_PROJECT_SLUG"):
        return CheckResult(
            name="Symphony env file",
            passed=True,
            detail="file not found; using process environment variables",
        )
    return CheckResult(
        name="Symphony env file",
        passed=False,
        detail=f"not found at {arguments.symphony_env_file}",
        remediation=(
            f"Create {arguments.symphony_env_file} with LINEAR_API_KEY and LINEAR_PROJECT_SLUG, "
            "then run `chmod 600` on it."
        ),
    )


def check_environment_variable(variable_name: str, remediation: str) -> CheckResult:
    """Check that an environment variable is non-empty.

    Args:
        variable_name: Environment variable name.
        remediation: Failure remediation.

    Returns:
        Check result.

    """

    if os.environ.get(variable_name):
        return CheckResult(name=variable_name, passed=True, detail="present; value redacted")
    return CheckResult(name=variable_name, passed=False, detail="missing", remediation=remediation)


def check_linear_project_slug_syntax() -> CheckResult:
    """Check that the configured project slug is safe to render into the workflow.

    Returns:
        Check result.

    """

    linear_project_slug = os.environ.get("LINEAR_PROJECT_SLUG", "")
    if not linear_project_slug:
        return CheckResult(
            name="Linear project slug syntax",
            passed=False,
            detail="missing LINEAR_PROJECT_SLUG",
            remediation="Set LINEAR_PROJECT_SLUG to the slugId from the Linear project URL.",
        )
    if re.fullmatch(r"[A-Za-z0-9._-]+", linear_project_slug):
        return CheckResult(name="Linear project slug syntax", passed=True, detail="valid; value redacted")
    return CheckResult(
        name="Linear project slug syntax",
        passed=False,
        detail="contains unsupported characters",
        remediation="Use only letters, numbers, dots, underscores, and dashes in LINEAR_PROJECT_SLUG.",
    )


def check_command_available(command_name: str, remediation: str) -> CheckResult:
    """Check that a command exists on PATH.

    Args:
        command_name: Command to locate.
        remediation: Failure remediation.

    Returns:
        Check result.

    """

    command_path = shutil.which(command_name)
    if command_path is None:
        return CheckResult(
            name=f"{command_name} command",
            passed=False,
            detail="not found on PATH",
            remediation=remediation,
        )
    return CheckResult(name=f"{command_name} command", passed=True, detail=f"found at {command_path}")


def check_slurm_commands() -> CheckResult:
    """Check SLURM command availability without submitting work.

    Returns:
        Check result.

    """

    command_names = ("srun", "sbatch", "squeue", "scontrol")
    missing_command_names = tuple(command_name for command_name in command_names if shutil.which(command_name) is None)
    if missing_command_names:
        return CheckResult(
            name="SLURM commands",
            passed=False,
            detail=f"missing {', '.join(missing_command_names)}",
            remediation="Load the cluster SLURM environment or install the SLURM client commands on this host.",
        )
    return CheckResult(name="SLURM commands", passed=True, detail=f"found {', '.join(command_names)}")


def check_uv_available() -> CheckResult:
    """Check uv availability.

    Returns:
        Check result.

    """

    command_path = shutil.which("uv")
    if command_path is None:
        return CheckResult(
            name="uv availability",
            passed=False,
            detail="uv not found on PATH",
            remediation="Install uv or run `just setup-server-tools` so repo-local tools are on PATH.",
        )
    command_result = run_command(("uv", "--version"), timeout_seconds=10)
    if command_result.return_code != 0:
        return CheckResult(
            name="uv availability",
            passed=False,
            detail=f"`uv --version` failed: {compact_output(command_result.output)}",
            remediation="Reinstall uv or run `just setup-server-tools` to refresh repo-local tooling.",
        )
    return CheckResult(name="uv availability", passed=True, detail=compact_output(command_result.output))


def check_git_repository(arguments: DoctorArguments) -> CheckResult:
    """Check that the worktree is a usable Git repository.

    Args:
        arguments: Doctor arguments.

    Returns:
        Check result.

    """

    git_repository_result = run_command(
        ("git", "rev-parse", "--is-inside-work-tree"),
        current_working_directory=arguments.repository_root,
    )
    if git_repository_result.return_code != 0 or git_repository_result.output.strip() != "true":
        return CheckResult(
            name="Git repository",
            passed=False,
            detail=f"not inside a Git worktree: {compact_output(git_repository_result.output)}",
            remediation="Run `just symphony-doctor` from a GWAS Engine worktree.",
        )
    branch_result = run_command(
        ("git", "rev-parse", "--abbrev-ref", "HEAD"),
        current_working_directory=arguments.repository_root,
    )
    branch_name = branch_result.output.strip() if branch_result.return_code == 0 else "unknown"
    return CheckResult(name="Git repository", passed=True, detail=f"worktree OK on branch {branch_name}")


def check_github_cli_authentication() -> CheckResult:
    """Check GitHub CLI authentication.

    Returns:
        Check result.

    """

    command_result = run_command(("gh", "auth", "status"), timeout_seconds=20)
    if command_result.return_code == 0:
        return CheckResult(name="GitHub CLI auth", passed=True, detail="gh auth status succeeded")
    return CheckResult(
        name="GitHub CLI auth",
        passed=False,
        detail=f"`gh auth status` failed: {compact_output(command_result.output)}",
        remediation="Run `gh auth login` and ensure this host can authenticate to GitHub.",
    )


def check_github_remote_reachability(arguments: DoctorArguments) -> CheckResult:
    """Check that the configured origin remote is reachable.

    Args:
        arguments: Doctor arguments.

    Returns:
        Check result.

    """

    remote_result = run_command(
        ("git", "remote", "get-url", "origin"),
        current_working_directory=arguments.repository_root,
    )
    if remote_result.return_code != 0:
        return CheckResult(
            name="GitHub remote reachability",
            passed=False,
            detail=f"origin remote is not configured: {compact_output(remote_result.output)}",
            remediation="Set the GitHub origin remote with `git remote add origin <repo-url>`.",
        )
    remote_address = remote_result.output.strip()
    reachability_result = run_command(
        ("git", "ls-remote", "--exit-code", "origin", "HEAD"),
        current_working_directory=arguments.repository_root,
        timeout_seconds=20,
    )
    if reachability_result.return_code == 0:
        return CheckResult(name="GitHub remote reachability", passed=True, detail=f"origin reachable: {remote_address}")
    return CheckResult(
        name="GitHub remote reachability",
        passed=False,
        detail=f"origin unreachable: {compact_output(reachability_result.output)}",
        remediation="Check network/SSH access, run `gh auth login`, and verify `git remote -v`.",
    )


def check_symphony_checkout(arguments: DoctorArguments) -> CheckResult:
    """Check that the Symphony checkout and binary exist.

    Args:
        arguments: Doctor arguments.

    Returns:
        Check result.

    """

    if not arguments.symphony_elixir_directory.is_dir():
        return CheckResult(
            name="Symphony checkout",
            passed=False,
            detail=f"missing directory {arguments.symphony_elixir_directory}",
            remediation="Clone Symphony and build it, or set SYMPHONY_ELIXIR_DIR to the existing checkout.",
        )
    symphony_binary_path = arguments.symphony_elixir_directory / "bin" / "symphony"
    if not symphony_binary_path.is_file() or not os.access(symphony_binary_path, os.X_OK):
        return CheckResult(
            name="Symphony checkout",
            passed=False,
            detail=f"missing executable {symphony_binary_path}",
            remediation="Run `mise exec -- mix setup` and `mise exec -- mix build` in the Symphony Elixir checkout.",
        )
    return CheckResult(name="Symphony checkout", passed=True, detail=f"binary found at {symphony_binary_path}")


def check_symphony_elixir_toolchain(arguments: DoctorArguments) -> CheckResult:
    """Check that mise can run Elixir and Mix in the Symphony checkout.

    Args:
        arguments: Doctor arguments.

    Returns:
        Check result.

    """

    if shutil.which("mise") is None:
        return CheckResult(
            name="Symphony Elixir toolchain",
            passed=False,
            detail="mise not found on PATH",
            remediation="Install mise, then run `mise use -g erlang elixir`.",
        )
    if not arguments.symphony_elixir_directory.is_dir():
        return CheckResult(
            name="Symphony Elixir toolchain",
            passed=False,
            detail=f"cannot enter {arguments.symphony_elixir_directory}",
            remediation="Clone Symphony and set SYMPHONY_ELIXIR_DIR to its Elixir checkout.",
        )
    elixir_result = run_command(
        ("mise", "exec", "--", "elixir", "--version"),
        current_working_directory=arguments.symphony_elixir_directory,
        timeout_seconds=20,
    )
    mix_result = run_command(
        ("mise", "exec", "--", "mix", "--version"),
        current_working_directory=arguments.symphony_elixir_directory,
        timeout_seconds=20,
    )
    if elixir_result.return_code == 0 and mix_result.return_code == 0:
        return CheckResult(name="Symphony Elixir toolchain", passed=True, detail="mise can run elixir and mix")
    failed_outputs = "; ".join(
        compact_output(command_result.output)
        for command_result in (elixir_result, mix_result)
        if command_result.return_code != 0
    )
    return CheckResult(
        name="Symphony Elixir toolchain",
        passed=False,
        detail=f"mise toolchain check failed: {failed_outputs}",
        remediation="Run `mise use -g erlang elixir`, then rebuild Symphony with `mise exec -- mix setup`.",
    )


def check_worktree_root(arguments: DoctorArguments) -> CheckResult:
    """Check that the Symphony worktree root exists and is writable.

    Args:
        arguments: Doctor arguments.

    Returns:
        Check result.

    """

    if not arguments.symphony_worktree_root.is_dir():
        return CheckResult(
            name="Worktree root writability",
            passed=False,
            detail=f"missing directory {arguments.symphony_worktree_root}",
            remediation=f"Create it with `mkdir -p {arguments.symphony_worktree_root}`.",
        )
    try:
        with tempfile.NamedTemporaryFile(
            prefix=".symphony-doctor-",
            dir=arguments.symphony_worktree_root,
            delete=True,
        ):
            pass
    except OSError as error:
        return CheckResult(
            name="Worktree root writability",
            passed=False,
            detail=f"cannot write test file in {arguments.symphony_worktree_root}: {error}",
            remediation=f"Fix ownership or permissions for {arguments.symphony_worktree_root}.",
        )
    return CheckResult(
        name="Worktree root writability",
        passed=True,
        detail=f"writable directory {arguments.symphony_worktree_root}",
    )


def check_codex_cli() -> CheckResult:
    """Check Codex CLI availability.

    Returns:
        Check result.

    """

    command_result = run_command(("codex", "--version"), timeout_seconds=10)
    if command_result.return_code == 0:
        return CheckResult(name="Codex CLI", passed=True, detail=compact_output(command_result.output))
    return CheckResult(
        name="Codex CLI",
        passed=False,
        detail=f"`codex --version` failed: {compact_output(command_result.output)}",
        remediation="Install or update the Codex CLI, then confirm `codex --version` works.",
    )


def check_codex_app_server() -> CheckResult:
    """Check that Codex exposes the app-server command Symphony expects.

    Returns:
        Check result.

    """

    command_result = run_command(("codex", "app-server", "--help"), timeout_seconds=10)
    if command_result.return_code == 0:
        return CheckResult(name="Codex app-server", passed=True, detail="app-server command is available")
    return CheckResult(
        name="Codex app-server",
        passed=False,
        detail=f"`codex app-server --help` failed: {compact_output(command_result.output)}",
        remediation="Update Codex to a version that includes `codex app-server`.",
    )


def check_codex_mcp_registry() -> CheckResult:
    """Check that Codex can read MCP server configuration.

    Returns:
        Check result.

    """

    command_result = run_command(("codex", "mcp", "list"), timeout_seconds=10)
    if command_result.return_code == 0:
        return CheckResult(name="Codex MCP registry", passed=True, detail="`codex mcp list` succeeded")
    return CheckResult(
        name="Codex MCP registry",
        passed=False,
        detail=f"`codex mcp list` failed: {compact_output(command_result.output)}",
        remediation="Fix ~/.codex/config.toml or run `codex mcp add` to recreate the Linear MCP entry.",
    )


def get_toml_table(table: TomlTable, key: str) -> TomlTable | None:
    """Get a nested TOML table.

    Args:
        table: Parent table.
        key: Child key.

    Returns:
        Nested table, when present.

    """

    value = table.get(key)
    if isinstance(value, dict):
        return value
    return None


def get_string_map(table: TomlTable, key: str) -> dict[str, str] | None:
    """Get a TOML table whose values are all strings.

    Args:
        table: Parent table.
        key: Child key.

    Returns:
        String map, when present and valid.

    """

    value = table.get(key)
    if not isinstance(value, dict):
        return None
    string_map: dict[str, str] = {}
    for map_key, map_value in value.items():
        if isinstance(map_key, str) and isinstance(map_value, str):
            string_map[map_key] = map_value
    return string_map


def expand_environment_references(value: str) -> str:
    """Expand simple shell-style environment references in config values.

    Args:
        value: Config value that may include `$NAME` or `${NAME}`.

    Returns:
        Value after environment expansion.

    """

    return os.path.expandvars(value)


def load_linear_mcp_config(arguments: DoctorArguments, redactor: Redactor) -> LinearMcpConfigLoad:
    """Load the Linear MCP entry from Codex config.

    Args:
        arguments: Doctor arguments.
        redactor: Current redactor.

    Returns:
        Config load result.

    """

    if not arguments.codex_config_path.is_file():
        return LinearMcpConfigLoad(
            check_result=CheckResult(
                name="Codex Linear MCP config",
                passed=False,
                detail=f"missing {arguments.codex_config_path}",
                remediation="Create Codex config with a `[mcp_servers.linear]` entry for Linear MCP.",
            ),
            linear_mcp_config=None,
            redactor=redactor,
        )
    try:
        config_table = typing.cast("TomlTable", tomllib.loads(arguments.codex_config_path.read_text()))
    except tomllib.TOMLDecodeError as error:
        return LinearMcpConfigLoad(
            check_result=CheckResult(
                name="Codex Linear MCP config",
                passed=False,
                detail=f"TOML parse failed: {error}",
                remediation="Fix TOML syntax in the Codex config, then rerun `codex mcp list`.",
            ),
            linear_mcp_config=None,
            redactor=redactor,
        )
    mcp_servers_table = get_toml_table(config_table, "mcp_servers")
    linear_server_table = get_toml_table(mcp_servers_table, "linear") if mcp_servers_table is not None else None
    if linear_server_table is None:
        return LinearMcpConfigLoad(
            check_result=CheckResult(
                name="Codex Linear MCP config",
                passed=False,
                detail="missing `[mcp_servers.linear]`",
                remediation=(
                    "Add the Linear MCP server to Codex config with `codex mcp add` or edit ~/.codex/config.toml."
                ),
            ),
            linear_mcp_config=None,
            redactor=redactor,
        )
    endpoint_value = linear_server_table.get("url")
    if not isinstance(endpoint_value, str) or not endpoint_value:
        return LinearMcpConfigLoad(
            check_result=CheckResult(
                name="Codex Linear MCP config",
                passed=False,
                detail="Linear MCP URL is missing",
                remediation="Set `[mcp_servers.linear].url` to the Linear MCP endpoint.",
            ),
            linear_mcp_config=None,
            redactor=redactor,
        )
    configured_headers = get_string_map(linear_server_table, "http_headers")
    if not configured_headers:
        return LinearMcpConfigLoad(
            check_result=CheckResult(
                name="Codex Linear MCP config",
                passed=False,
                detail="Linear MCP http_headers are missing",
                remediation="Configure a redacted Authorization header for `[mcp_servers.linear]`.",
            ),
            linear_mcp_config=None,
            redactor=redactor,
        )
    expanded_headers = {
        header_name: expand_environment_references(header_value)
        for header_name, header_value in configured_headers.items()
    }
    header_names = tuple(sorted(expanded_headers))
    has_auth_header = any(header_name.lower() == "authorization" for header_name in header_names)
    updated_redactor = redactor.with_secret_values(expanded_headers.values())
    if not has_auth_header:
        return LinearMcpConfigLoad(
            check_result=CheckResult(
                name="Codex Linear MCP config",
                passed=False,
                detail=f"configured headers do not include Authorization; found {', '.join(header_names)}",
                remediation="Add an Authorization header to `[mcp_servers.linear].http_headers`.",
            ),
            linear_mcp_config=None,
            redactor=updated_redactor,
        )
    endpoint_display = format_endpoint_for_display(endpoint_value)
    return LinearMcpConfigLoad(
        check_result=CheckResult(
            name="Codex Linear MCP config",
            passed=True,
            detail=f"linear server configured at {endpoint_display}; header values redacted",
        ),
        linear_mcp_config=LinearMcpConfig(endpoint_address=endpoint_value, http_headers=expanded_headers),
        redactor=updated_redactor,
    )


def format_endpoint_for_display(endpoint_address: str) -> str:
    """Format an endpoint without credentials, query, or fragment.

    Args:
        endpoint_address: Endpoint URL.

    Returns:
        Safe endpoint display text.

    """

    parsed_endpoint = urllib.parse.urlparse(endpoint_address)
    if not parsed_endpoint.scheme or not parsed_endpoint.hostname:
        return "<configured endpoint>"
    network_location = parsed_endpoint.hostname
    if parsed_endpoint.port is not None:
        network_location = f"{network_location}:{parsed_endpoint.port}"
    return urllib.parse.urlunparse(
        (
            parsed_endpoint.scheme,
            network_location,
            parsed_endpoint.path,
            "",
            "",
            "",
        )
    )


def post_json_request(
    endpoint_address: str,
    headers: dict[str, str],
    payload: JsonObject,
    *,
    timeout_seconds: int = 15,
) -> HttpResponse:
    """Send a JSON POST request.

    Args:
        endpoint_address: Request endpoint.
        headers: HTTP headers.
        payload: JSON payload.
        timeout_seconds: Request timeout.

    Returns:
        Captured HTTP response.

    Raises:
        OSError: If the request cannot be sent.

    """

    request = urllib.request.Request(
        endpoint_address,
        data=json.dumps(payload).encode(),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            response_body = response.read().decode(errors="replace")
            return HttpResponse(
                status_code=response.status,
                content_type=response.headers.get("content-type", ""),
                body=response_body,
            )
    except urllib.error.HTTPError as error:
        error_body = error.read().decode(errors="replace")
        return HttpResponse(status_code=error.code, content_type=error.headers.get("content-type", ""), body=error_body)


def parse_json_object(text: str) -> JsonObject | None:
    """Parse a JSON object from text.

    Args:
        text: Raw JSON text.

    Returns:
        Parsed object, or None.

    """

    try:
        parsed_value = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed_value, dict):
        return typing.cast("JsonObject", parsed_value)
    return None


def parse_mcp_event_payload(response_body: str) -> JsonObject | None:
    """Parse the JSON-RPC payload from an MCP SSE response.

    Args:
        response_body: MCP response body.

    Returns:
        Parsed JSON object, or None.

    """

    for response_line in response_body.splitlines():
        if response_line.startswith("data:"):
            payload_text = response_line.removeprefix("data:").strip()
            parsed_payload = parse_json_object(payload_text)
            if parsed_payload is not None:
                return parsed_payload
    return parse_json_object(response_body)


def check_linear_mcp_authentication(linear_mcp_config: LinearMcpConfig) -> CheckResult:
    """Check that the configured Linear MCP server accepts its auth header.

    Args:
        linear_mcp_config: Parsed Linear MCP config.

    Returns:
        Check result.

    """

    request_headers = {
        **linear_mcp_config.http_headers,
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    initialize_payload: JsonObject = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "g-symphony-doctor", "version": "0.0.0"},
        },
    }
    try:
        http_response = post_json_request(
            linear_mcp_config.endpoint_address,
            request_headers,
            initialize_payload,
            timeout_seconds=15,
        )
    except OSError as error:
        return CheckResult(
            name="Linear MCP auth",
            passed=False,
            detail=f"request failed: {error}",
            remediation="Check network access to the Linear MCP endpoint and the Codex MCP URL.",
        )
    response_payload = parse_mcp_event_payload(http_response.body)
    if http_response.status_code == 200 and response_payload is not None and "result" in response_payload:
        return CheckResult(
            name="Linear MCP auth",
            passed=True,
            detail=f"MCP initialize succeeded at {format_endpoint_for_display(linear_mcp_config.endpoint_address)}",
        )
    return CheckResult(
        name="Linear MCP auth",
        passed=False,
        detail=f"HTTP {http_response.status_code}: {compact_output(http_response.body)}",
        remediation="Refresh the Linear MCP Authorization header in Codex config, then rerun `codex mcp list`.",
    )


def run_linear_graphql_query(
    query: str,
    variables: JsonObject,
    *,
    linear_api_key: str,
) -> HttpResponse:
    """Run a Linear GraphQL query with the configured API key.

    Args:
        query: GraphQL query text.
        variables: Query variables.
        linear_api_key: Linear API key.

    Returns:
        Captured HTTP response.

    Raises:
        OSError: If the request cannot be sent.

    """

    payload: JsonObject = {"query": query, "variables": variables}
    return post_json_request(
        "https://api.linear.app/graphql",
        {
            "Content-Type": "application/json",
            "Authorization": linear_api_key,
        },
        payload,
        timeout_seconds=15,
    )


def get_graphql_error_message(payload: JsonObject) -> str | None:
    """Extract the first GraphQL error message.

    Args:
        payload: GraphQL response payload.

    Returns:
        First error message, when present.

    """

    errors_value = payload.get("errors")
    if not isinstance(errors_value, list) or not errors_value:
        return None
    first_error = errors_value[0]
    if not isinstance(first_error, dict):
        return "GraphQL returned an error"
    message_value = first_error.get("message")
    if isinstance(message_value, str):
        return message_value
    return "GraphQL returned an error"


def get_json_object(parent_object: JsonObject, key: str) -> JsonObject | None:
    """Get a nested JSON object.

    Args:
        parent_object: Parent object.
        key: Child key.

    Returns:
        Nested object, when present.

    """

    value = parent_object.get(key)
    if isinstance(value, dict):
        return value
    return None


def check_linear_api_authentication() -> CheckResult:
    """Check Linear API authentication without printing the token.

    Returns:
        Check result.

    """

    linear_api_key = os.environ.get("LINEAR_API_KEY")
    if not linear_api_key:
        return CheckResult(
            name="Linear API auth",
            passed=False,
            detail="LINEAR_API_KEY is missing",
            remediation="Add LINEAR_API_KEY to the Symphony env file or process environment.",
        )
    try:
        http_response = run_linear_graphql_query(
            "query Viewer { viewer { id } }",
            {},
            linear_api_key=linear_api_key,
        )
    except OSError as error:
        return CheckResult(
            name="Linear API auth",
            passed=False,
            detail=f"request failed: {error}",
            remediation="Check network access to https://api.linear.app and refresh LINEAR_API_KEY if needed.",
        )
    response_payload = parse_json_object(http_response.body)
    if response_payload is None:
        return CheckResult(
            name="Linear API auth",
            passed=False,
            detail=f"HTTP {http_response.status_code}: response was not JSON",
            remediation="Check network/proxy behavior between this host and Linear.",
        )
    error_message = get_graphql_error_message(response_payload)
    if http_response.status_code == 200 and error_message is None:
        return CheckResult(name="Linear API auth", passed=True, detail="viewer query succeeded; token redacted")
    return CheckResult(
        name="Linear API auth",
        passed=False,
        detail=f"HTTP {http_response.status_code}: {error_message or compact_output(http_response.body)}",
        remediation="Regenerate LINEAR_API_KEY in Linear settings and update the Symphony env file.",
    )


def check_linear_project_reachability() -> CheckResult:
    """Check that the configured Linear project slug is reachable.

    Returns:
        Check result.

    """

    linear_api_key = os.environ.get("LINEAR_API_KEY")
    linear_project_slug = os.environ.get("LINEAR_PROJECT_SLUG")
    if not linear_api_key or not linear_project_slug:
        return CheckResult(
            name="Linear project reachability",
            passed=False,
            detail="LINEAR_API_KEY or LINEAR_PROJECT_SLUG is missing",
            remediation="Set both LINEAR_API_KEY and LINEAR_PROJECT_SLUG before running Symphony.",
        )
    project_query = (
        "query ProjectBySlug($projectSlug: String!) { "
        "projects(first: 1, filter: { slugId: { eq: $projectSlug } }) { "
        "nodes { id name slugId url } } }"
    )
    try:
        http_response = run_linear_graphql_query(
            project_query,
            {"projectSlug": linear_project_slug},
            linear_api_key=linear_api_key,
        )
    except OSError as error:
        return CheckResult(
            name="Linear project reachability",
            passed=False,
            detail=f"request failed: {error}",
            remediation="Check network access to Linear and verify the project slug in the env file.",
        )
    response_payload = parse_json_object(http_response.body)
    if response_payload is None:
        return CheckResult(
            name="Linear project reachability",
            passed=False,
            detail=f"HTTP {http_response.status_code}: response was not JSON",
            remediation="Check network/proxy behavior between this host and Linear.",
        )
    error_message = get_graphql_error_message(response_payload)
    if error_message is not None:
        return CheckResult(
            name="Linear project reachability",
            passed=False,
            detail=f"HTTP {http_response.status_code}: {error_message}",
            remediation="Verify LINEAR_PROJECT_SLUG uses the Linear project slugId, not the display name.",
        )
    data_object = get_json_object(response_payload, "data")
    projects_object = get_json_object(data_object, "projects") if data_object is not None else None
    nodes_value = projects_object.get("nodes") if projects_object is not None else None
    if isinstance(nodes_value, list) and nodes_value and isinstance(nodes_value[0], dict):
        project_object = nodes_value[0]
        project_name_value = project_object.get("name")
        project_name = project_name_value if isinstance(project_name_value, str) else "configured project"
        return CheckResult(
            name="Linear project reachability",
            passed=True,
            detail=f"reachable project: {project_name}",
        )
    return CheckResult(
        name="Linear project reachability",
        passed=False,
        detail="no Linear project matched configured slug",
        remediation="Open the Linear project URL and set LINEAR_PROJECT_SLUG to its slugId path segment.",
    )


def build_results(arguments: DoctorArguments, redactor: Redactor) -> DoctorResults:
    """Build the full doctor result list.

    Args:
        arguments: Doctor arguments.
        redactor: Initial redactor.

    Returns:
        Results and the final redactor.

    """

    results: list[CheckResult] = [
        check_symphony_environment_file(arguments),
        check_environment_variable(
            "LINEAR_API_KEY",
            f"Add `export LINEAR_API_KEY='...'` to {arguments.symphony_env_file}.",
        ),
        check_environment_variable(
            "LINEAR_PROJECT_SLUG",
            f"Add `export LINEAR_PROJECT_SLUG='...'` to {arguments.symphony_env_file}.",
        ),
        check_linear_project_slug_syntax(),
        check_command_available("git", "Install Git or load the development tool environment."),
        check_git_repository(arguments),
        check_command_available("gh", "Install GitHub CLI and run `gh auth login`."),
        check_github_cli_authentication(),
        check_github_remote_reachability(arguments),
        check_command_available("just", "Install just or run `just setup-server-tools` from a bootstrapped checkout."),
        check_uv_available(),
        check_slurm_commands(),
        check_command_available("codex", "Install Codex CLI and authenticate it for this user."),
        check_codex_cli(),
        check_codex_app_server(),
        check_codex_mcp_registry(),
    ]
    linear_mcp_config_load = load_linear_mcp_config(arguments, redactor)
    redactor = linear_mcp_config_load.redactor
    results.append(linear_mcp_config_load.check_result)
    if linear_mcp_config_load.linear_mcp_config is not None:
        results.append(check_linear_mcp_authentication(linear_mcp_config_load.linear_mcp_config))
    else:
        results.append(
            CheckResult(
                name="Linear MCP auth",
                passed=False,
                detail="skipped because Linear MCP config is unavailable",
                remediation="Fix the Codex Linear MCP config, then rerun this command.",
            )
        )
    results.extend(
        [
            check_linear_api_authentication(),
            check_linear_project_reachability(),
            check_symphony_checkout(arguments),
            check_command_available("mise", "Install mise, then run `mise use -g erlang elixir`."),
            check_symphony_elixir_toolchain(arguments),
            check_worktree_root(arguments),
        ]
    )
    return DoctorResults(check_results=results, redactor=redactor)


def print_report(results: collections.abc.Sequence[CheckResult], redactor: Redactor) -> None:
    """Print a redacted pass/fail report.

    Args:
        results: Check results.
        redactor: Redactor to apply before printing.

    """

    print("Symphony doctor report (secrets redacted)")
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"{status} {result.name}: {redactor.redact(result.detail)}")
        if not result.passed and result.remediation:
            print(f"  Remediation: {redactor.redact(result.remediation)}")
    failed_count = sum(1 for result in results if not result.passed)
    if failed_count:
        print(f"Symphony prerequisites failed: {failed_count} check(s) need attention.")
    else:
        print("Symphony prerequisites look usable.")


def main(argument_values: collections.abc.Sequence[str]) -> int:
    """Run the Symphony doctor.

    Args:
        argument_values: Command-line arguments.

    Returns:
        Process exit code.

    """

    arguments = parse_arguments(argument_values)
    initial_redactor = Redactor(literal_secrets=()).with_secret_values(
        secret_value for secret_value in (os.environ.get("LINEAR_API_KEY"),) if secret_value is not None
    )
    doctor_results = build_results(arguments, initial_redactor)
    print_report(doctor_results.check_results, doctor_results.redactor)
    if any(not result.passed for result in doctor_results.check_results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

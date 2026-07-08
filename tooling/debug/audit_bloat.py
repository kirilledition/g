"""Generate conservative bloat-audit CSVs for Python and Rust sources."""

from __future__ import annotations

import ast
import csv
import re
import typing
from pathlib import Path

if typing.TYPE_CHECKING:
    import collections.abc

ROOTS = [Path("src/g"), Path("src/python"), Path("crates")]
OUT = Path("target/audit")
OUT.mkdir(parents=True, exist_ok=True)


def iter_files(suffix: str) -> collections.abc.Iterator[Path]:
    """Yield source files under the audit roots with the requested suffix."""
    for root in ROOTS:
        if root.exists():
            yield from root.rglob(f"*{suffix}")


def is_python_trivial_wrapper(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[bool, str]:
    """Classify a Python function as a trivial wrapper when possible."""
    body = list(node.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]

    if len(body) == 1 and isinstance(body[0], ast.Return):
        value = body[0].value
        if isinstance(value, ast.Call):
            return True, ast.unparse(value.func)
        if isinstance(value, ast.Name | ast.Attribute):
            return True, ast.unparse(value)

    if len(body) == 2 and isinstance(body[0], ast.Assign) and isinstance(body[1], ast.Return):
        assigned = body[0].value
        returned = body[1].value
        if isinstance(assigned, ast.Call) and isinstance(returned, ast.Name):
            return True, ast.unparse(assigned.func)

    return False, ""


def audit_python() -> None:
    """Write Python function size, usage, and wrapper candidates."""
    rows = []
    all_text = "\n".join(path.read_text(errors="ignore") for path in iter_files(".py"))
    for path in iter_files(".py"):
        try:
            tree = ast.parse(path.read_text(errors="ignore"))
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                start = node.lineno
                end = getattr(node, "end_lineno", start)
                loc = end - start + 1
                trivial, target = is_python_trivial_wrapper(node)
                occurrences = len(re.findall(rf"\b{re.escape(node.name)}\b", all_text))
                rows.append(
                    {
                        "path": str(path),
                        "name": node.name,
                        "start": start,
                        "end": end,
                        "loc": loc,
                        "text_occurrences": occurrences,
                        "trivial_wrapper": trivial,
                        "wrapper_target": target,
                    }
                )

    with (OUT / "python_function_audit.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "path",
                "name",
                "start",
                "end",
                "loc",
                "text_occurrences",
                "trivial_wrapper",
                "wrapper_target",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def extract_rust_functions(
    text: str,
) -> collections.abc.Iterator[tuple[str, int, int, str]]:
    """Yield Rust function names and approximate source slices."""
    pattern = re.compile(
        r"(?P<prefix>(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:const\s+)?(?:unsafe\s+)?fn\s+)"
        r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*[^;{]*\{",
        re.MULTILINE,
    )
    for match in pattern.finditer(text):
        start = match.start()
        brace_start = text.find("{", match.end() - 1)
        depth = 0
        end = brace_start
        for index in range(brace_start, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    end = index + 1
                    break
        yield match.group("name"), start, end, text[start:end]


def audit_rust() -> None:
    """Write Rust function size, usage, and wrapper candidates."""
    rows = []
    all_text = "\n".join(path.read_text(errors="ignore") for path in iter_files(".rs"))
    for path in iter_files(".rs"):
        text = path.read_text(errors="ignore")
        line_starts = [0]
        for match in re.finditer("\n", text):
            line_starts.append(match.end())

        for name, start, end, body in extract_rust_functions(text):
            start_line = sum(1 for x in line_starts if x <= start)
            end_line = sum(1 for x in line_starts if x <= end)
            loc = end_line - start_line + 1
            occurrences = len(re.findall(rf"\b{re.escape(name)}\b", all_text))
            body_inner = body[body.find("{") + 1 : body.rfind("}")].strip()
            semicolon_count = body_inner.count(";")
            return_call = bool(re.search(r"^\s*(?:return\s+)?[A-Za-z_][A-Za-z0-9_:<>]*\s*\(", body_inner))
            trivial = loc <= 8 and (semicolon_count <= 1 or return_call)

            rows.append(
                {
                    "path": str(path),
                    "name": name,
                    "start_line": start_line,
                    "end_line": end_line,
                    "loc": loc,
                    "text_occurrences": occurrences,
                    "trivial_candidate": trivial,
                    "body_preview": " ".join(body_inner.split())[:180],
                }
            )

    with (OUT / "rust_function_audit.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "path",
                "name",
                "start_line",
                "end_line",
                "loc",
                "text_occurrences",
                "trivial_candidate",
                "body_preview",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def audit_public_reexports() -> None:
    """Write crate api.rs public re-export lines."""
    rows = []
    for path in Path("crates").glob("*/src/api.rs"):
        crate = path.parts[1]
        text = path.read_text(errors="ignore")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if line.strip().startswith("pub use "):
                rows.append(
                    {
                        "crate": crate,
                        "path": str(path),
                        "line": line_number,
                        "reexport": line.strip(),
                    }
                )
    with (OUT / "rust_public_reexports.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["crate", "path", "line", "reexport"])
        writer.writeheader()
        writer.writerows(rows)


def audit_large_files() -> None:
    """Write source files ordered by line count."""
    rows = []
    for suffix in [".py", ".rs"]:
        for path in iter_files(suffix):
            loc = len(path.read_text(errors="ignore").splitlines())
            rows.append({"path": str(path), "loc": loc})
    rows.sort(key=lambda r: r["loc"], reverse=True)
    with (OUT / "large_files.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "loc"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    audit_python()
    audit_rust()
    audit_public_reexports()
    audit_large_files()
    print(f"Wrote audit outputs to {OUT}")

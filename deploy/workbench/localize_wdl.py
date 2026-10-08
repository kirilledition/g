"""Rebind engine inputs localized by a WDL executor without changing their hashes."""

from __future__ import annotations

import json
import sys
import typing
from pathlib import Path

from tooling.workbench import manifest


def local_path(value: manifest.JsonValue, context: str) -> str:
    """Resolve a WDL-localized file, retaining filenames without shell evaluation."""
    path = Path(manifest.text(value, context)).resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"{context} must refer to a regular localized file.")
    return str(path)


def rebind_file(value: manifest.JsonValue, path: manifest.JsonValue, context: str) -> None:
    """Change only a source URI and remove its cloud-only generation field."""
    values = manifest.mapping(value, context)
    values["uri"] = local_path(path, context)
    values.pop("generation", None)


def localize_job(bindings_path: Path, destination: Path, publication_directory: Path) -> None:
    """Validate source identity, replace localized paths and write one new manifest.

    Args:
        bindings_path: Executor-generated JSON holding file paths and reservations.
        destination: New local manifest, which must not already exist.
        publication_directory: Directory for validated successful result bundles.

    Raises:
        ValueError: Input bindings or declared execution resources disagree.
        FileExistsError: The destination already exists.

    """
    bindings = manifest.mapping(
        typing.cast(
            "manifest.JsonValue",
            json.loads(bindings_path.read_text(encoding="utf-8"), object_pairs_hook=manifest.reject_duplicate_keys),
        ),
        "WDL bindings",
    )
    manifest.fields(
        bindings,
        {"manifest", "inputs", "predictions", "cpu_threads", "memory_gib"},
        set(),
        "WDL bindings",
    )
    source_path = Path(local_path(bindings["manifest"], "manifest"))
    values = manifest.mapping(
        typing.cast(
            "manifest.JsonValue",
            json.loads(source_path.read_text(encoding="utf-8"), object_pairs_hook=manifest.reject_duplicate_keys),
        ),
        "manifest",
    )
    job = manifest.parse_manifest(values)
    if job.resources.device != manifest.Device.CPU:
        raise ValueError("The portable WDL template reserves CPU only; qualify a workspace GPU backend separately.")
    reserved_threads = manifest.positive_integer(bindings["cpu_threads"], "cpu_threads")
    reserved_memory = manifest.positive_float(bindings["memory_gib"], "memory_gib")
    if job.resources.cpu_threads > reserved_threads:
        raise ValueError("Manifest cpu_threads exceeds the WDL task CPU reservation.")
    if job.resources.memory_gib is None or job.resources.memory_gib > reserved_memory:
        raise ValueError("Manifest memory_gib exceeds the WDL task memory reservation.")
    input_paths = manifest.mapping(bindings["inputs"], "WDL inputs")
    input_paths = {name: value for name, value in input_paths.items() if value is not None}
    if input_paths.keys() != job.inputs.keys():
        raise ValueError("WDL localized inputs must match every manifest input exactly.")
    inputs = manifest.mapping(values["inputs"], "manifest inputs")
    for name, path in input_paths.items():
        rebind_file(inputs[name], path, f"inputs.{name}")
    prediction_paths = bindings["predictions"]
    predictions = values["predictions"]
    if not isinstance(prediction_paths, list) or not isinstance(predictions, list):
        raise ValueError("WDL predictions must be an ordered file array.")
    if len(prediction_paths) != len(predictions):
        raise ValueError("WDL must localize one prediction file per manifest phenotype, in manifest order.")
    for index, path in enumerate(prediction_paths):
        rebind_file(predictions[index], path, f"predictions[{index}]")
    values["output_uri"] = str(publication_directory.resolve())
    manifest.parse_manifest(values)
    with destination.open("x", encoding="utf-8") as output:
        json.dump(values, output, indent=2, allow_nan=False)
        output.write("\n")


def main() -> None:
    """Rebind one executor-generated localization descriptor."""
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m deploy.workbench.localize_wdl BINDINGS.json")
    localize_job(Path(sys.argv[1]), Path("job.local.json"), Path("published"))


if __name__ == "__main__":
    main()

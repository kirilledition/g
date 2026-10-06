"""Candidate construction and recovery for profile tuning."""

from __future__ import annotations

import dataclasses
import json
import typing

from tooling.profile_deep import budget as profile_deep_budget
from tooling.profile_deep import models as profile_deep_models

if typing.TYPE_CHECKING:
    from pathlib import Path


def build_candidate_slug(candidate: profile_deep_models.Step2Candidate) -> str:
    """Build a stable filename slug for a tuning candidate."""
    candidate_parts = [
        candidate.trait_type,
        candidate.device,
        f"chunk{candidate.chunk_size}",
        f"writer{candidate.output_writer_thread_count}",
        f"rayon{candidate.rayon_thread_count if candidate.rayon_thread_count is not None else 'default'}",
    ]
    if candidate.firth_batch_size is not None:
        candidate_parts.append(f"firth{candidate.firth_batch_size}")
    return "_".join(candidate_parts)


def build_step2_candidates(
    *,
    trait_type: str,
    device: str,
    thread_candidates: tuple[profile_deep_models.NativeThreadCandidate, ...],
    chunk_sizes: tuple[int, ...],
    writer_thread_counts: tuple[int, ...],
    firth_batch_sizes: tuple[int, ...],
) -> tuple[profile_deep_models.Step2Candidate, ...]:
    """Build candidates from settings still exposed by the application."""
    candidates: list[profile_deep_models.Step2Candidate] = []
    for thread_candidate in thread_candidates:
        for chunk_size in chunk_sizes:
            for writer_thread_count in writer_thread_counts:
                active_firth_batch_sizes = firth_batch_sizes if trait_type == "binary" else (None,)
                for firth_batch_size in active_firth_batch_sizes:
                    candidates.append(
                        profile_deep_models.Step2Candidate(
                            trait_type=trait_type,
                            device=device,
                            chunk_size=chunk_size,
                            output_writer_thread_count=writer_thread_count,
                            rayon_thread_count=thread_candidate.rayon_thread_count,
                            firth_batch_size=firth_batch_size,
                        )
                    )
    return tuple(candidates)


def build_native_thread_candidates(
    *,
    arguments: profile_deep_models.ProfileArguments,
    output_directory: Path,
) -> tuple[profile_deep_models.NativeThreadCandidate, ...]:
    """Record native Rayon worker-count candidates."""
    candidate_directory = output_directory / "thread_candidates"
    candidate_directory.mkdir(parents=True, exist_ok=True)
    candidates = [
        profile_deep_models.NativeThreadCandidate(
            rayon_thread_count=rayon_thread_count,
        )
        for rayon_thread_count in profile_deep_budget.parse_int_list(arguments.rayon_thread_counts)
    ]
    (candidate_directory / "thread_candidates.json").write_text(
        json.dumps(
            {
                "notes": "The application exposes Rayon threads; internal reader profiling remains in Criterion.",
                "thread_candidates": [dataclasses.asdict(candidate) for candidate in candidates],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return tuple(candidates)


def recover_candidate_from_trial(
    trial_result: profile_deep_models.TrialResult, candidates: tuple[profile_deep_models.Step2Candidate, ...]
) -> profile_deep_models.Step2Candidate:
    """Recover the tuning candidate that produced a trial by matching its command and env."""
    for candidate in candidates:
        if build_candidate_slug(candidate) in trial_result.name:
            return candidate
    message = f"Could not recover candidate from trial {trial_result.name}."
    raise ValueError(message)


def candidate_from_aggregate_name(
    winner_key: str, aggregate_result: profile_deep_models.AggregateResult
) -> profile_deep_models.Step2Candidate:
    """Return the candidate retained with an aggregate result."""
    if aggregate_result.candidate is None:
        message = f"Aggregate {winner_key!r} does not retain its tuning candidate."
        raise ValueError(message)
    return aggregate_result.candidate

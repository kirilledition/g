"""Compatibility patches for Hydra in development tooling."""

from __future__ import annotations

import argparse
import typing

PATCH_APPLIED = False


def apply_argparse_help_patch() -> None:
    """Allow Hydra's lazy help object on Python versions that validate help text early."""
    global PATCH_APPLIED
    if PATCH_APPLIED:
        return
    argument_parser_class = typing.cast("typing.Any", argparse.ArgumentParser)
    help_formatter_class = typing.cast("typing.Any", argparse.HelpFormatter)
    original_check_help = typing.cast(
        "typing.Callable[[argparse.ArgumentParser, argparse.Action], None]",
        argument_parser_class._check_help,
    )
    original_expand_help = typing.cast(
        "typing.Callable[[argparse.HelpFormatter, argparse.Action], str]",
        help_formatter_class._expand_help,
    )

    def patched_check_help(parser: argparse.ArgumentParser, action: argparse.Action) -> None:
        if action.help is not None and not isinstance(action.help, str):
            return
        original_check_help(parser, action)

    def patched_expand_help(formatter: argparse.HelpFormatter, action: argparse.Action) -> str:
        if action.help is None or isinstance(action.help, str):
            return original_expand_help(formatter, action)
        original_help = action.help
        action.help = repr(original_help)
        try:
            return typing.cast("str", original_expand_help(formatter, action))
        finally:
            action.help = original_help

    argument_parser_class._check_help = patched_check_help
    help_formatter_class._expand_help = patched_expand_help
    PATCH_APPLIED = True

"""Python entrypoints for the GWAS engine package."""

from g._core import hello_from_bin


def main() -> None:
    print(hello_from_bin())

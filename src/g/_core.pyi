from collections.abc import Sequence

__build_git_commit__: str
__build_profile__: str
__build_science_source_sha256__: str
__build_source_clean__: bool

class cli:  # noqa: N801 - extension submodule name
    class NativeCliRunResult:
        @property
        def exit_code(self) -> int: ...
        @property
        def stdout_chunks(self) -> tuple[str, ...]: ...
        @property
        def stderr_chunks(self) -> tuple[str, ...]: ...

    @staticmethod
    def run(arguments: Sequence[str]) -> NativeCliRunResult: ...

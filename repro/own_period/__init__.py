"""Own-period analyses and their lazily configured data paths."""

from __future__ import annotations

from functools import total_ordering
import os
from pathlib import Path


@total_ordering
class DataPath(os.PathLike[str]):
    """A path relative to DELPHI_DATA, read from the environment only on use.

    DELPHI_DATA is the own-period result folder (containing benchmark/).
    Joining paths and taking parents stays lazy, including module-level
    defaults imported by other runners. File operations return normal Paths.
    """

    def __init__(self, *parts: str):
        self._parts = parts

    def path(self) -> Path:
        root = os.environ.get("DELPHI_DATA")
        if not root:
            raise ValueError("set DELPHI_DATA to the own-period data folder")
        return Path(os.path.abspath(Path(root).expanduser().joinpath(*self._parts)))

    def __fspath__(self) -> str:
        return os.fspath(self.path())

    def __str__(self) -> str:
        return str(self.path())

    def __repr__(self) -> str:
        return f"DataPath({', '.join(repr(part) for part in self._parts)})"

    def __truediv__(self, part: str | os.PathLike[str]) -> DataPath:
        return DataPath(*self._parts, os.fspath(part))

    @property
    def parent(self) -> DataPath:
        return self / ".."

    def __getattr__(self, name):
        # Pickle checks special attributes before restoring the instance state.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.path(), name)

    def __eq__(self, other):
        if not isinstance(other, (Path, DataPath)):
            return NotImplemented
        return self.path() == Path(other)

    def __lt__(self, other):
        if not isinstance(other, (Path, DataPath)):
            return NotImplemented
        return self.path() < Path(other)

    def __hash__(self) -> int:
        return hash(self.path())

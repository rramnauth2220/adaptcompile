from __future__ import annotations

import builtins
from collections.abc import Callable
from typing import Any

import pytest
from conftest import make_result

from adaptcompile import AdaptationDataset, AdaptationGeometry, AdaptationStudy
from adaptcompile.compiler import build_compiler_dataset


def test_dataframe_methods_fail_clearly_without_pandas(
    monkeypatch: pytest.MonkeyPatch,
    episode,
) -> None:
    geometry = AdaptationGeometry({"gain": 0.5})
    result = make_result(
        episode,
        "candidate",
        gain=0.8,
        retention=0.9,
        cost=1.0,
    )
    dataset = AdaptationDataset([result])
    values = (
        geometry,
        AdaptationStudy([result]),
        dataset,
        build_compiler_dataset(dataset),
    )
    real_import = builtins.__import__

    def without_pandas(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "pandas":
            raise ImportError("pandas intentionally unavailable")
        importer: Callable[..., Any] = real_import
        return importer(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", without_pandas)
    for value in values:
        with pytest.raises(ImportError, match=r"install adaptcompile\[dataframe\]"):
            value.to_dataframe()

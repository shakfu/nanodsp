"""Run the ``>>>`` examples in the package docstrings.

Documentation examples rot silently: they are prose until something executes
them. These run every example in the public modules and fail on any mismatch, so
an example that stops working is a test failure rather than a bad first
impression.

Examples are written to be deterministic -- shapes, dtypes, booleans and rounded
values rather than raw floats, which differ in the last digits across platforms.
"""

from __future__ import annotations

import doctest
import importlib
import pkgutil

import pytest

import nanodsp

# Modules whose examples are checked. `_core` is the compiled extension (no
# Python docstrings to collect) and `_cli` / `__main__` are CLI plumbing rather
# than public API.
SKIP = {"nanodsp._core", "nanodsp._cli", "nanodsp.__main__", "nanodsp._helpers"}


def _public_modules() -> list[str]:
    names = ["nanodsp.buffer"]
    for info in pkgutil.walk_packages(nanodsp.__path__, prefix="nanodsp."):
        if info.name in SKIP or info.name.rsplit(".", 1)[-1].startswith("_"):
            continue
        names.append(info.name)
    return sorted(set(names))


MODULES = _public_modules()


def test_modules_were_discovered():
    assert len(MODULES) >= 10, MODULES


@pytest.mark.parametrize("module_name", MODULES)
def test_docstring_examples(module_name):
    module = importlib.import_module(module_name)
    results = doctest.testmod(
        module,
        verbose=False,
        optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS,
    )
    assert results.failed == 0, (
        f"{results.failed} of {results.attempted} doctest examples failed in "
        f"{module_name}; run "
        f'`python -m doctest -v $(python -c "import {module_name} as m; '
        f'print(m.__file__)")` for detail'
    )


def test_examples_actually_exist():
    """Guard against the suite passing because nothing has examples."""
    total = 0
    for name in MODULES:
        module = importlib.import_module(name)
        total += doctest.DocTestFinder().find(module).__len__() and sum(
            len(t.examples) for t in doctest.DocTestFinder().find(module)
        )
    assert total >= 100, f"only {total} doctest examples found across the package"

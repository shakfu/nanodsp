"""Check `_core.pyi` against the extension module it describes.

`src/nanodsp/_core.pyi` is 2600-odd hand-maintained lines standing in for a
compiled module. Nothing verifies it, and a stub that has drifted is worse than
no stub: mypy reports it as fact, so a binding renamed in C++ keeps type-checking
against the old name and the error moves from build time to run time.

`mypy.stubtest`, the usual tool for this, cannot help here. The stub models
nanobind submodules as `class` blocks -- `class filters: ...` for
`_core.filters` -- because there is no other way to spell a nested module in a
single stub file. stubtest sees a class where the runtime has a module, reports
"is not a type" for all 19 of them, and never inspects their contents, which is
where essentially the whole stub lives.

So this compares names directly instead, two levels deep: submodules against
the runtime module, and the classes inside them against the runtime objects.
It checks that names line up, not that signatures do -- the failure mode that
actually occurs is a binding added, removed or renamed without the stub
following, and that is what this catches.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import nanodsp._core as core

STUB_PATH = Path(__file__).resolve().parent.parent / "src" / "nanodsp" / "_core.pyi"


def _declared_names(body: list[ast.stmt]) -> set[str]:
    """Names a stub class or module body declares."""
    names: set[str] = set()
    for node in body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return names


def _stub_tree() -> ast.Module:
    if not STUB_PATH.is_file():
        pytest.skip(f"{STUB_PATH.name} not present (running from an installed wheel?)")
    return ast.parse(STUB_PATH.read_text(encoding="utf-8"))


def _stub_submodules() -> list[ast.ClassDef]:
    return [n for n in _stub_tree().body if isinstance(n, ast.ClassDef)]


SUBMODULES = [n.name for n in _stub_submodules()]


def test_stub_declares_the_submodules():
    """Guard against the comparison passing because it found nothing."""
    assert len(SUBMODULES) >= 15, SUBMODULES


@pytest.mark.parametrize("name", SUBMODULES)
def test_submodule_exists_at_runtime(name: str):
    assert hasattr(core, name), (
        f"_core.pyi declares submodule {name!r}, which the extension does not "
        f"expose. Remove it from the stub, or restore the binding."
    )


@pytest.mark.parametrize("name", SUBMODULES)
def test_submodule_members_match_runtime(name: str):
    node = next(n for n in _stub_submodules() if n.name == name)
    runtime = getattr(core, name, None)
    if runtime is None:
        pytest.skip(f"{name} missing at runtime; reported by another test")

    declared = _declared_names(node.body)
    actual = {a for a in dir(runtime) if not a.startswith("__")}

    assert not (declared - actual), (
        f"_core.pyi declares {sorted(declared - actual)} under {name!r}, "
        f"which the extension does not expose"
    )
    assert not (actual - declared), (
        f"the extension exposes {sorted(actual - declared)} under {name!r}, "
        f"which _core.pyi does not declare -- add them to the stub"
    )


@pytest.mark.parametrize("name", SUBMODULES)
def test_nested_class_members_match_runtime(name: str):
    """Same comparison one level deeper, for the classes inside each submodule."""
    node = next(n for n in _stub_submodules() if n.name == name)
    runtime = getattr(core, name, None)
    if runtime is None:
        pytest.skip(f"{name} missing at runtime; reported by another test")

    for sub in (n for n in node.body if isinstance(n, ast.ClassDef)):
        obj = getattr(runtime, sub.name, None)
        assert obj is not None, (
            f"_core.pyi declares {name}.{sub.name}, absent at runtime"
        )

        declared = _declared_names(sub.body) - {"__init__"}
        actual = {a for a in dir(obj) if not a.startswith("_")}
        qualified = f"{name}.{sub.name}"
        assert not (declared - actual), (
            f"_core.pyi declares {sorted(declared - actual)} on {qualified}, "
            f"absent at runtime"
        )
        assert not (actual - declared), (
            f"{qualified} exposes {sorted(actual - declared)}, which _core.pyi "
            f"does not declare"
        )

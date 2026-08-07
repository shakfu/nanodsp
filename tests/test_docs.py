"""Documentation claims that can be checked mechanically.

Hand-maintained counts in the README and docs drift silently: the test count,
demo count and function count were all stale at the time of the 0.1.9 review.
These tests fail when a number in prose stops matching reality, which is
cheaper than noticing it in a bug report.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
DOCS_INDEX = ROOT / "docs" / "index.md"


def _read(path: Path) -> str:
    if not path.is_file():
        pytest.skip(f"{path.name} not present (running from an installed wheel?)")
    return path.read_text(encoding="utf-8")


def test_demo_count_matches_demos_directory():
    demos = sorted((ROOT / "demos").glob("demo_*.py"))
    if not demos:
        pytest.skip("demos/ not present")
    text = _read(README)
    claimed = {int(m) for m in re.findall(r"(\d+) demo scripts", text)}
    assert claimed, "README no longer states a demo-script count"
    assert claimed == {len(demos)}, (
        f"README claims {claimed} demo scripts, demos/ has {len(demos)}"
    )


def test_function_count_matches_registry():
    from nanodsp._cli import get_kinds, get_registry

    text = _read(DOCS_INDEX)
    m = re.search(r"\*\*(\d+) registered DSP functions\*\*", text)
    assert m, "docs/index.md no longer states a registered-function count"
    assert int(m.group(1)) == len(get_registry()), (
        f"docs/index.md claims {m.group(1)} functions, "
        f"registry has {len(get_registry())}"
    )

    m = re.search(r"(\d+) chainable audio effects", text)
    assert m, "docs/index.md no longer states a chainable-effect count"
    chainable = sum(1 for k in get_kinds().values() if k == "processor")
    assert int(m.group(1)) == chainable, (
        f"docs/index.md claims {m.group(1)} chainable effects, registry has {chainable}"
    )


def test_backend_count_matches_table():
    """The '12 C++ backends' claim must match the backend table rows."""
    text = _read(DOCS_INDEX)
    m = re.search(r"\*\*(\d+) C\+\+ backends\*\*", text)
    if not m:
        pytest.skip("docs/index.md no longer states a backend count")
    listed = re.findall(r"^\| \[([^\]]+)\]\([^)]+\) \| ", text, re.MULTILINE)
    assert int(m.group(1)) == len(listed), (
        f"docs/index.md claims {m.group(1)} backends, the table lists {len(listed)}"
    )


def test_package_version_matches_pyproject():
    """__version__ is a hand-maintained duplicate of the pyproject version.

    The version is read with a regex rather than ``tomllib``, which is 3.11+
    while this package supports 3.10, and pulling in ``tomli`` for one field is
    not worth a test dependency.
    """
    import re

    import nanodsp

    pyproject = ROOT / "pyproject.toml"
    if not pyproject.is_file():
        pytest.skip("pyproject.toml not present")
    text = pyproject.read_text(encoding="utf-8")
    # First `version = "..."` after the [project] table header.
    project_table = text.split("[project]", 1)[-1]
    m = re.search(r'^version\s*=\s*"([^"]+)"', project_table, re.MULTILINE)
    assert m, "could not find the version in pyproject.toml [project]"
    declared = m.group(1)
    assert nanodsp.__version__ == declared, (
        f"nanodsp.__version__ is {nanodsp.__version__!r} but pyproject.toml "
        f"declares {declared!r}; bump both."
    )
    assert re.fullmatch(r"\d+\.\d+\.\d+", declared), declared


def test_changelog_has_an_entry_for_the_current_version():
    """A release must not ship without a changelog section."""
    import nanodsp

    changelog = ROOT / "CHANGELOG.md"
    if not changelog.is_file():
        pytest.skip("CHANGELOG.md not present")
    assert f"## [{nanodsp.__version__}]" in changelog.read_text(), (
        f"CHANGELOG.md has no '## [{nanodsp.__version__}]' section"
    )


def test_preset_count_matches_registry():
    from nanodsp._cli import PRESETS

    text = _read(README)
    claimed = {int(m) for m in re.findall(r"(\d+) built-in presets", text)}
    if not claimed:
        pytest.skip("README no longer states a preset count")
    assert claimed == {len(PRESETS)}, (
        f"README claims {claimed} built-in presets, there are {len(PRESETS)}"
    )

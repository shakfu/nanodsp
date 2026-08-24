"""Keep `THIRD_PARTY_LICENSES.md` in step with what is actually vendored.

Every backend compiled into `_core` is MIT, BSD-3-Clause, ISC or LGPL-2.1, and
all of those require their copyright notice to accompany a binary
redistribution. The wheel shipped only nanodsp's own LICENSE, so twelve notices
were missing from the artefact almost everyone installs.

`THIRD_PARTY_LICENSES.md` collects them and is listed in `license-files`, so it
lands in the wheel's `dist-info`. It is generated rather than hand-written --
hand-maintained copies of upstream licence text go stale silently, and a stale
attribution file is worse than none. This test regenerates it and fails if the
committed copy differs.

Regenerate with::

    uv run python tests/test_licenses.py --update
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
NOTICE_PATH = ROOT / "THIRD_PARTY_LICENSES.md"
THIRDPARTY = ROOT / "thirdparty"

# (display name, SPDX identifier, how to obtain the notice text).
#
# "file" reads a licence file shipped by the upstream project. "header" pulls
# the banner comment out of a source file, for the two libraries that carry
# their licence only in-source. "shared" reuses another entry's text, for the
# Faust-derived filters, which state the licence they are under but do not
# reproduce it.
BACKENDS: list[tuple[str, str, tuple]] = [
    ("signalsmith-dsp", "MIT", ("file", "signalsmith/LICENSE.txt")),
    ("signalsmith-stretch", "MIT", ("file", "signalsmith-stretch/LICENSE.txt")),
    ("DaisySP", "MIT", ("file", "DaisySP/LICENSE")),
    ("DaisySP-LGPL", "LGPL-2.1-only", ("file", "DaisySP/DaisySP-LGPL/LICENSE")),
    ("STK", "MIT", ("file", "stk/LICENSE")),
    ("madronalib", "MIT", ("file", "madronalib/LICENSE")),
    ("HISSTools_Library", "BSD-3-Clause", ("file", "HISSTools_Library/LICENSE")),
    ("CHOC", "ISC", ("file", "choc/LICENSE.md")),
    ("GrainflowLib", "MIT", ("file", "GrainflowLib/Liscense.txt")),
    ("fxdsp", "MIT", ("file", "fxdsp/LICENSE")),
    ("DspFilters", "MIT", ("header", "DspFilters/Biquad.cpp")),
    ("vafilters", "MIT", ("shared", "STK")),
    ("PolyBLEP oscillators", "MIT", ("shared", "STK")),
    # Nested dependencies of the above, compiled in through their headers.
    ("AudioFile", "MIT", ("file", "GrainflowLib/lib/AudioFile/LICENSE")),
    ("sse2neon", "MIT", ("file", "madronalib/external/sse2neon/LICENSE")),
]

# Vendored in the snapshots above and shipped in the sdist, but not reachable
# from anything nanodsp compiles: madronalib is included through `mldsp.h`,
# which pulls in the MLDSP* headers only, never `source/app`. Their notices are
# reproduced because the source distribution carries the files; they are not in
# the wheel's License-Expression, because they are not in the wheel.
NOT_COMPILED: list[tuple[str, str, tuple]] = [
    ("clap", "MIT", ("file", "madronalib/external/clap/LICENSE")),
    ("utf", "BSL-1.0", ("file", "madronalib/external/utf/LICENSE_1_0.txt")),
]

# Libraries whose licence text is not reproduced upstream, with the attribution
# their source headers do carry.
SHARED_NOTES = {
    "vafilters": (
        "Faust-generated virtual-analog filter implementations by Eric Tarr and "
        "Christopher Arndt, cleaned for nanodsp. The source headers state "
        '"MIT-style STK-4.3 license" and do not reproduce it; that text is the '
        "STK licence above."
    ),
    "PolyBLEP oscillators": (
        "Band-limited oscillators based on Kleimola, Lazzarini, Timoney and "
        'Valimaki, "Phaseshaping Oscillator Algorithms for Musical Sound '
        'Synthesis" (SMC 2010), vendored alongside vafilters and under the '
        "same MIT-style STK licence."
    ),
}

_HEADER_BANNER = re.compile(r"/\*{5,}(.*?)\*{5,}/", re.DOTALL)


def _notice_text(spec: tuple) -> str:
    kind, target = spec
    if kind == "file":
        return (THIRDPARTY / target).read_text(encoding="utf-8").strip()
    if kind == "header":
        source = (THIRDPARTY / target).read_text(encoding="utf-8")
        match = _HEADER_BANNER.search(source)
        assert match, f"no banner comment in {target}"
        return match.group(1).strip()
    if kind == "shared":
        return ""
    raise AssertionError(f"unknown spec {spec!r}")


def _render() -> str:
    lines = [
        "# Third-party licences",
        "",
        "nanodsp is MIT licensed. The `nanodsp._core` extension statically links",
        "the vendored C++ libraries below, so a built wheel is a combined work and",
        "carries their terms as well. This file reproduces each notice; the",
        "corresponding source is vendored under `thirdparty/` and ships in the",
        "sdist. Versions and local patches are tabulated in",
        "`thirdparty/VERSIONS.md`.",
        "",
        "Note that **DaisySP-LGPL is LGPL-2.1**, not MIT. It supplies `Compressor`,",
        "`ReverbSc`, `MoogLadder`, `BlOsc`, `Bitcrush`, `Fold`, `Pluck`, `Tone`,",
        "`Comb` and others, all reachable from the public API.",
        "",
        "This file is generated -- run `uv run python tests/test_licenses.py",
        "--update` after changing anything under `thirdparty/`.",
        "",
        "## Summary",
        "",
        "| Library | SPDX |",
        "|---------|------|",
    ]
    lines += [f"| {name} | `{spdx}` |" for name, spdx, _ in BACKENDS]
    lines.append("")

    lines += [
        "",
        "Vendored but not compiled into the extension -- present in the sdist",
        "only, and therefore not part of the wheel's `License-Expression`:",
        "",
        "| Library | SPDX |",
        "|---------|------|",
    ]
    lines += [f"| {name} | `{spdx}` |" for name, spdx, _ in NOT_COMPILED]
    lines.append("")

    for name, spdx, spec in BACKENDS + NOT_COMPILED:
        lines += [f"## {name} (`{spdx}`)", ""]
        if name in SHARED_NOTES:
            lines += [SHARED_NOTES[name], ""]
        text = _notice_text(spec)
        if text:
            lines += ["```", text, "```", ""]
    return "\n".join(lines).rstrip("\n") + "\n"


def test_notice_file_is_current():
    if not NOTICE_PATH.is_file():
        pytest.fail(
            f"{NOTICE_PATH.name} is missing. Generate it with "
            "`uv run python tests/test_licenses.py --update`."
        )
    assert NOTICE_PATH.read_text(encoding="utf-8") == _render(), (
        f"{NOTICE_PATH.name} is out of date with thirdparty/. Regenerate with "
        "`uv run python tests/test_licenses.py --update`."
    )


def test_every_vendored_licence_file_is_covered():
    """Every licence file under thirdparty/ must be reproduced.

    Keyed on files rather than top-level directories, because two of them are
    nested one level down inside another library's snapshot -- which is exactly
    how AudioFile and sse2neon were missed on the first pass.
    """
    found = {
        p.relative_to(THIRDPARTY).as_posix()
        for pattern in ("LICENSE*", "COPYING*", "Liscense*")
        for p in THIRDPARTY.rglob(pattern)
        if p.is_file()
    }
    covered = {
        target for _, _, (kind, target) in BACKENDS + NOT_COMPILED if kind == "file"
    }
    missing = found - covered
    assert not missing, (
        f"licence files under thirdparty/ with no entry in tests/test_licenses.py: "
        f"{sorted(missing)}"
    )
    stale = covered - found
    assert not stale, f"entries pointing at files that no longer exist: {sorted(stale)}"


def test_lgpl_component_is_called_out():
    """The LGPL component is the one a downstream user most needs to see."""
    text = NOTICE_PATH.read_text(encoding="utf-8")
    assert "LGPL-2.1-only" in text
    assert "GNU LESSER GENERAL PUBLIC LICENSE" in text


if __name__ == "__main__":  # pragma: no cover
    if "--update" in sys.argv:
        NOTICE_PATH.write_text(_render(), encoding="utf-8")
        print(f"Wrote {len(BACKENDS) + len(NOT_COMPILED)} notices to {NOTICE_PATH}")
    else:
        print(__doc__)
        print("Run with --update to regenerate the notice file.")

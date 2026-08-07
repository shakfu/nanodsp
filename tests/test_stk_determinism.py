"""Reproducibility of voices drawing from the shared C rand() stream.

STK's `Noise` and several DaisySP generators both draw from the C library
`rand()`, which is one process-global stream. Upstream STK seeds it with
`srand(time(NULL))` whenever a `Noise` is constructed with the default seed,
which is every `Noise` inside every STK voice. Two consequences:

- STK voices rendered different audio on each run landing in a different
  wall-clock second, so a render could not be reproduced or regression-tested.
- Because `srand()` is process-global and DaisySP reads the same stream, merely
  constructing an STK instrument silently randomised unrelated DaisySP
  generators (`pluck`, `drip`, the snare drums).

Both are fixed: the vendored `Noise::setSeed` no longer touches `rand()` for the
default seed (`thirdparty/VERSIONS.md`), and the affected synthesis functions
take an explicit `seed` and seed the stream on entry. These tests pin that a
render is a pure function of its arguments, independent of wall-clock time and
of what ran before it.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pytest

from nanodsp import synthesis
from nanodsp._core import stk

SR = 48000.0

# Voices that draw from the shared rand() stream. `bowed` and `brass` are
# included as controls: they have no noise component and were never affected.
SUBJECTS = {
    "clarinet": lambda **k: synthesis.synth_note(
        "clarinet", freq=440.0, duration=0.05, sample_rate=SR, **k
    ),
    "flute": lambda **k: synthesis.synth_note(
        "flute", freq=440.0, duration=0.05, sample_rate=SR, **k
    ),
    "plucked": lambda **k: synthesis.synth_note(
        "plucked", freq=440.0, duration=0.05, sample_rate=SR, **k
    ),
    "sitar": lambda **k: synthesis.synth_note(
        "sitar", freq=440.0, duration=0.05, sample_rate=SR, **k
    ),
    "sequence": lambda **k: synthesis.synth_sequence(
        "clarinet", [(440.0, 0.0, 0.05)], sample_rate=SR, **k
    ),
    "pluck": lambda **k: synthesis.pluck(2400, freq=200.0, sample_rate=SR, **k),
    "drip": lambda **k: synthesis.drip(2400, sample_rate=SR, **k),
    "string_voice": lambda **k: synthesis.string_voice(2400, sample_rate=SR, **k),
    "analog_snare_drum": lambda **k: synthesis.analog_snare_drum(
        2400, sample_rate=SR, **k
    ),
    "synthetic_snare_drum": lambda **k: synthesis.synthetic_snare_drum(
        2400, sample_rate=SR, **k
    ),
    "synthetic_bass_drum": lambda **k: synthesis.synthetic_bass_drum(
        2400, sample_rate=SR, **k
    ),
    "hihat": lambda **k: synthesis.hihat(2400, sample_rate=SR, **k),
    "clocked_noise": lambda **k: synthesis.clocked_noise(2400, sample_rate=SR, **k),
    "dust": lambda **k: synthesis.dust(2400, sample_rate=SR, **k),
}

CONTROLS = {
    "bowed": lambda: synthesis.synth_note(
        "bowed", freq=440.0, duration=0.05, sample_rate=SR
    ),
    "brass": lambda: synthesis.synth_note(
        "brass", freq=440.0, duration=0.05, sample_rate=SR
    ),
}


def _digest(buf) -> str:
    return hashlib.sha256(np.ascontiguousarray(buf.data).tobytes()).hexdigest()


@pytest.fixture(scope="module")
def across_second_boundary():
    """Digests of every subject before and after one wall-clock second boundary.

    Each subject is rendered once and discarded first, so ``before`` is a warm
    call. Without that, ``before`` would be the first call in the process and
    ``after`` the second, and this test would fail for a voice whose *first*
    render differs -- which is a different defect, covered by
    :func:`test_first_call_matches_later_calls`. Keeping the two claims in
    separate tests means a failure says which one broke.

    Module-scoped so the ~1 s wait is paid once for the whole module.
    """
    for fn in SUBJECTS.values():
        fn()
    before = {name: _digest(fn()) for name, fn in SUBJECTS.items()}
    start = int(time.time())
    while int(time.time()) == start:
        time.sleep(0.01)
    after = {name: _digest(fn()) for name, fn in SUBJECTS.items()}
    return before, after


@pytest.mark.parametrize("name", sorted(SUBJECTS))
def test_reproducible_across_a_second_boundary(across_second_boundary, name):
    """The defect: output used to change with the wall clock."""
    before, after = across_second_boundary
    assert before[name] == after[name]


@pytest.mark.parametrize("name", sorted(SUBJECTS))
def test_repeated_calls_agree(name):
    assert len({_digest(SUBJECTS[name]()) for _ in range(3)}) == 1


@pytest.mark.parametrize("name", sorted(SUBJECTS))
def test_seed_selects_a_variation_reproducibly(name):
    """An explicit seed must change the output, and the same seed must repeat it."""
    fn = SUBJECTS[name]
    a, b, a_again = _digest(fn(seed=1)), _digest(fn(seed=2)), _digest(fn(seed=1))
    assert a != b, "seed had no effect on the output"
    assert a == a_again, "same seed did not reproduce the same output"


@pytest.mark.parametrize("name", sorted(CONTROLS))
def test_noiseless_controls_are_reproducible(name):
    assert len({_digest(CONTROLS[name]()) for _ in range(3)}) == 1


def test_stk_voice_does_not_disturb_daisysp_generators():
    """Constructing an STK voice must not perturb the shared rand() stream.

    This was the subtle half of the defect: STK's Noise constructor called
    srand(), so rendering a clarinet silently changed the output of an unrelated
    DaisySP generator later in the same process.
    """
    baseline = _digest(SUBJECTS["dust"](seed=5))
    synthesis.synth_note("clarinet", freq=440.0, duration=0.05, sample_rate=SR)
    assert _digest(SUBJECTS["dust"](seed=5)) == baseline


def test_set_random_seed_is_exposed():
    """The escape hatch for callers who want run-to-run variation."""
    stk.set_random_seed(12345)
    a = _digest(synthesis.pluck(2400, freq=200.0, sample_rate=SR, seed=1))
    stk.set_random_seed(999)
    b = _digest(synthesis.pluck(2400, freq=200.0, sample_rate=SR, seed=1))
    # The per-call seed wins over any earlier global seeding.
    assert a == b


@pytest.mark.parametrize("name", sorted(SUBJECTS))
def test_first_call_matches_later_calls(name):
    """A voice's very first render in a process must match every later one.

    This catches uninitialised DSP state, which is otherwise invisible: freshly
    mapped pages read as zero, so the cold render is self-consistent across
    runs, while every later render in the same process reuses the freed block of
    its predecessor and converges on a different value. Two DaisySP voices were
    found this way -- see thirdparty/VERSIONS.md.

    Run in a subprocess because "first call in the process" cannot be recreated
    once anything in this one has already touched the voice.
    """
    src = textwrap.dedent(f"""
        import hashlib
        import numpy as np
        from nanodsp import synthesis
        from tests.test_stk_determinism import SUBJECTS, _digest
        fn = SUBJECTS[{name!r}]
        print(_digest(fn()), _digest(fn()), _digest(fn()))
    """)
    proc = subprocess.run(
        [sys.executable, "-c", src],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parent.parent,
    )
    assert proc.returncode == 0, proc.stderr
    cold, warm, warm2 = proc.stdout.split()
    assert cold == warm == warm2, (
        f"{name}: first render differs from later ones "
        f"(cold={cold[:12]}, warm={warm[:12]}) -- uninitialised state"
    )

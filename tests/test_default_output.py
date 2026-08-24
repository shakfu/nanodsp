"""Magnitude checks over the whole registry.

The rest of the suite asserts on structure -- shapes, dtypes, channel counts,
error types. That is necessary and it is not sufficient: an effect can return
the right shape full of NaN, or the right shape amplified by 42 dB, and pass
every test written that way. Two such defects shipped (`daisysp.pitch_shift`
returned all-NaN at its default arguments; `daisysp.bitcrush` returned audio at
roughly 2^(bit_depth-1) times full scale with an inverted sign), and both were
covered by shape assertions that saw nothing wrong.

These tests sweep every registry entry that can be called with nothing but its
leading argument -- 86 of the 114 chainable effects and 22 of the 23 generators;
the remainder need a second buffer or an explicit rate -- and assert three
properties that hold for any sane audio process:

* the output is finite,
* the output is not wildly out of range,
* silence in gives silence out.

They are deliberately loose. The point is not to pin numeric behaviour --
`tests/GOLDEN.json` does that -- but to catch a whole class of gross failure
across a hundred-odd functions at once, including in vendored C++ that no
hand-written test covers. The bounds below are set from measurement, not taste: with a
0.5-peak input the loudest processor is `agc` at 3.4, and the loudest generator
is `minblep` at 1.6.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from nanodsp import AudioBuffer
from nanodsp._cli import get_kinds, get_registry

SAMPLE_RATE = 48000.0
FRAMES = 24000

# Generous headroom over the loudest measured output (agc, 3.4). A processor
# exceeding this is either broken or needs an explicit exemption with a reason.
PEAK_LIMIT = 10.0

# Silence-in-silence-out is exact in principle; IIR state settles to a small
# non-zero residue in practice (largest measured: 2.5e-10 for iir_filter).
SILENCE_ATOL = 1e-6


def _mono() -> AudioBuffer:
    t = np.arange(FRAMES) / SAMPLE_RATE
    data = (0.5 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    return AudioBuffer(data.reshape(1, -1), sample_rate=SAMPLE_RATE)


def _stereo() -> AudioBuffer:
    t = np.arange(FRAMES) / SAMPLE_RATE
    data = np.stack(
        [0.5 * np.sin(2 * np.pi * 220.0 * t), 0.5 * np.sin(2 * np.pi * 330.0 * t)]
    ).astype(np.float32)
    return AudioBuffer(data, sample_rate=SAMPLE_RATE)


def _silence(channels: int = 1) -> AudioBuffer:
    return AudioBuffer(
        np.zeros((channels, FRAMES), dtype=np.float32), sample_rate=SAMPLE_RATE
    )


def _callable_with_defaults(fn, skip_first: bool) -> bool:
    """True when *fn* needs nothing beyond its leading argument."""
    params = list(inspect.signature(fn).parameters.values())
    if not params:
        return False
    rest = params[1:] if skip_first else params
    return not [
        p
        for p in rest
        if p.default is inspect.Parameter.empty
        and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    ]


def _names_of_kind(kind: str, skip_first: bool) -> list[str]:
    kinds = get_kinds()
    return sorted(
        name
        for name, (fn, _) in get_registry().items()
        if kinds.get(name) == kind and _callable_with_defaults(fn, skip_first)
    )


PROCESSORS = _names_of_kind("processor", skip_first=True)
GENERATORS = _names_of_kind("generator", skip_first=True)


def test_sweep_covers_the_registry():
    """Guard against the sweep passing because it selected nothing."""
    assert len(PROCESSORS) >= 80, len(PROCESSORS)
    assert len(GENERATORS) >= 20, len(GENERATORS)


@pytest.mark.parametrize("name", PROCESSORS)
@pytest.mark.parametrize("channels", [1, 2])
def test_processor_output_is_finite_and_bounded(name: str, channels: int):
    fn, _ = get_registry()[name]
    buf = _mono() if channels == 1 else _stereo()
    try:
        out = fn(buf)
    except ValueError:
        # Declining an input (mono-only, stereo-only, too short) is a contract
        # decision tested elsewhere; it is not a magnitude failure.
        pytest.skip(f"{name} rejects {channels}-channel input")

    data = out.data
    assert np.isfinite(data).all(), (
        f"{name} produced {int((~np.isfinite(data)).sum())} non-finite samples "
        f"of {data.size}"
    )
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    assert peak <= PEAK_LIMIT, (
        f"{name} peaked at {peak:.4g} from a 0.5-peak input; a gain of this "
        f"size is a defect unless the function is documented to amplify"
    )


@pytest.mark.parametrize("name", PROCESSORS)
def test_processor_maps_silence_to_silence(name: str):
    """A process with no input signal must not invent one.

    Catches uninitialised internal state, DC offsets and self-oscillation --
    faults that a signal-carrying input can mask.
    """
    fn, _ = get_registry()[name]
    try:
        out = fn(_silence())
    except ValueError:
        pytest.skip(f"{name} rejects silent input")

    data = out.data
    assert np.isfinite(data).all(), f"{name} produced non-finite output from silence"
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    assert peak <= SILENCE_ATOL, f"{name} produced {peak:.4g} peak from silence"


@pytest.mark.parametrize("name", GENERATORS)
def test_generator_output_is_finite_and_bounded(name: str):
    fn, _ = get_registry()[name]
    data = fn(FRAMES).data
    assert data.shape[1] == FRAMES
    assert np.isfinite(data).all(), (
        f"{name} produced {int((~np.isfinite(data)).sum())} non-finite samples"
    )
    peak = float(np.max(np.abs(data)))
    assert peak <= PEAK_LIMIT, f"{name} peaked at {peak:.4g}"


@pytest.mark.parametrize("name", GENERATORS)
def test_generator_accepts_zero_frames(name: str):
    """Zero frames must produce an empty buffer, not an out-of-bounds write.

    The trigger-generator bindings wrote the first sample before testing the
    count, so ``frames=0`` wrote one element past a zero-size allocation. The
    corruption is silent from Python -- this test only fails under
    AddressSanitizer (``make asan``), but it is what puts the case in front of
    it.
    """
    fn, _ = get_registry()[name]
    out = fn(0)
    assert out.frames == 0
    assert out.data.size == 0


@pytest.mark.parametrize("name", GENERATORS)
def test_generator_rejects_negative_frames(name: str):
    fn, _ = get_registry()[name]
    with pytest.raises(ValueError, match="non-negative"):
        fn(-1)

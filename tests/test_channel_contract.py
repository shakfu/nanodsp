"""The channel-count contract for chainable processors.

Chains are built on the assumption that an effect returns what it was given:
same channel count, same frame count, same sample rate. Most processors honour
that, but a handful of inherently stereo effects widen mono input, and the FDN
`reverb` additionally folds anything above two channels down to a stereo pair.

Those are legitimate designs, but they are surprising mid-chain, so this module
pins them as an explicit allow-list. A processor that starts or stops changing
the channel count fails here, which forces the change to be deliberate and
documented rather than silent.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from nanodsp import AudioBuffer
from nanodsp._cli import get_kinds, get_registry

SR = 48000.0
# 0.5 s: long enough for the 400 ms loudness gate in normalize_lufs and master.
FRAMES = 24000

# Processors that intentionally widen mono to stereo. Each produces a stereo
# field from a mono source, so there is no meaningful mono output.
MONO_TO_STEREO = {
    "auto_pan",
    "chorus",
    "gated_reverb",
    "pan",
    "ping_pong_delay",
    "reverb",
    "reverb_sc",
    "shimmer_reverb",
    "stereo_delay",
    "stk_chorus",
    "stk_reverb",
}

# Processors that also fold anything above two channels down to a stereo pair,
# because the underlying engine is mono-in/stereo-out (madronalib's FDN8 for
# `reverb`, the STK effects for the others) or produces a stereo field by
# construction (`auto_pan`).
FOLDS_TO_STEREO = {"auto_pan", "reverb", "stk_chorus", "stk_reverb"}

# Processors whose output length is deliberately not the input length.
CHANGES_FRAME_COUNT = {
    "paulstretch",  # time-stretch: that is the whole function
    "trim_silence",  # removes leading/trailing silence
    "upsample_2x",  # rate conversion, twice the frames
}

# Processors that deliberately change the sample rate.
CHANGES_SAMPLE_RATE = {"upsample_2x"}

# Channel counts each processor is documented to reject with ValueError. Any
# other ValueError is a failure, not a skip: a processor that starts rejecting
# everything must not drop out of the contract checks unnoticed.
STEREO_ONLY = {"mid_side_decode", "mid_side_encode", "stereo_widen"}
MONO_OR_STEREO = {
    "gated_reverb",
    "ping_pong_delay",
    "reverb_sc",
    "shimmer_reverb",
    "stereo_delay",
}
REJECTS: dict[str, set[int]] = {
    **{n: {1, 4} for n in STEREO_ONLY},
    **{n: {4} for n in MONO_OR_STEREO},
}


def _signal(channels: int) -> AudioBuffer:
    t = np.arange(FRAMES, dtype=np.float64) / SR
    row = (np.sin(2 * np.pi * 220.0 * t) * 0.4).astype(np.float32)
    return AudioBuffer(np.tile(row, (channels, 1)), sample_rate=SR)


def _defaultable_processors() -> list[str]:
    """Processor names callable with a buffer alone."""
    reg, kinds = get_registry(), get_kinds()
    names = []
    for name, (fn, _mod) in sorted(reg.items()):
        if kinds.get(name) != "processor":
            continue
        params = list(inspect.signature(fn).parameters.items())
        required = [
            n
            for i, (n, p) in enumerate(params)
            if i
            and p.default is inspect.Parameter.empty
            and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
        ]
        if not required:
            names.append(name)
    return names


PROCESSORS = _defaultable_processors()


def _apply(name: str, buf: AudioBuffer):
    fn, _ = get_registry()[name]
    expected_reject = buf.channels in REJECTS.get(name, set())
    try:
        out = fn(buf)
    except ValueError:
        if expected_reject:
            pytest.skip(f"{name} rejects {buf.channels}-channel input by design")
        raise
    assert not expected_reject, (
        f"{name} accepted {buf.channels}-channel input; update REJECTS"
    )
    return out


def test_rejections_name_real_processors():
    assert set(REJECTS) <= set(PROCESSORS)


def test_registry_has_processors():
    assert len(PROCESSORS) > 50


@pytest.mark.parametrize("name", PROCESSORS)
def test_stereo_in_stereo_out(name):
    """No processor may change a stereo input's channel count."""
    out = _apply(name, _signal(2))
    assert out.channels == 2, f"{name} turned 2ch into {out.channels}ch"


@pytest.mark.parametrize("name", PROCESSORS)
def test_mono_channel_count_matches_the_allow_list(name):
    out = _apply(name, _signal(1))
    expected = 2 if name in MONO_TO_STEREO else 1
    assert out.channels == expected, (
        f"{name}: mono input produced {out.channels}ch, expected {expected}. "
        "If this change is intended, add or remove the name in MONO_TO_STEREO "
        "and document it in the function's docstring."
    )


@pytest.mark.parametrize("name", sorted(FOLDS_TO_STEREO))
def test_multichannel_folding_is_declared(name):
    out = _apply(name, _signal(4))
    assert out.channels == 2


@pytest.mark.parametrize("name", PROCESSORS)
def test_multichannel_is_otherwise_preserved(name):
    if name in FOLDS_TO_STEREO:
        pytest.skip("declared to fold to stereo")
    out = _apply(name, _signal(4))
    assert out.channels == 4, f"{name} turned 4ch into {out.channels}ch"


@pytest.mark.parametrize("name", PROCESSORS)
def test_frame_count_preserved(name):
    """Length is invariant except where changing it is the point of the function."""
    buf = _signal(2)
    out = _apply(name, buf)
    if name in CHANGES_FRAME_COUNT:
        pytest.skip("declared to change length")
    assert out.frames == buf.frames, (
        f"{name} changed the frame count ({buf.frames} -> {out.frames}). "
        "If intended, add it to CHANGES_FRAME_COUNT and document it."
    )


@pytest.mark.parametrize("name", PROCESSORS)
def test_sample_rate_preserved(name):
    buf = _signal(2)
    out = _apply(name, buf)
    if name in CHANGES_SAMPLE_RATE:
        pytest.skip("declared to resample")
    assert out.sample_rate == buf.sample_rate, (
        f"{name} changed the sample rate "
        f"({buf.sample_rate} -> {out.sample_rate}). If intended, add it to "
        "CHANGES_SAMPLE_RATE and document it."
    )


@pytest.mark.parametrize("name", sorted(CHANGES_FRAME_COUNT | CHANGES_SAMPLE_RATE))
def test_declared_length_changes_still_happen(name):
    """Keeps the allow-lists honest: a name that stopped changing must come off."""
    buf = _signal(2)
    out = _apply(name, buf)
    changed = out.frames != buf.frames or out.sample_rate != buf.sample_rate
    assert changed, f"{name} no longer changes length or rate; drop it from the list"


@pytest.mark.parametrize(
    "allow_list,label",
    [
        (MONO_TO_STEREO, "MONO_TO_STEREO"),
        (FOLDS_TO_STEREO, "FOLDS_TO_STEREO"),
        (CHANGES_FRAME_COUNT, "CHANGES_FRAME_COUNT"),
        (CHANGES_SAMPLE_RATE, "CHANGES_SAMPLE_RATE"),
    ],
)
def test_allow_lists_have_no_stale_entries(allow_list, label):
    stale = sorted(n for n in allow_list if n not in PROCESSORS)
    assert not stale, f"{label} names no longer registered: {stale}"

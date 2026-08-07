"""Golden-output regression corpus.

Every other test in this suite is analytic: it asserts a property (this filter
attenuates, this gain is linear, this metadata survives). Those catch broken
code, but they cannot catch a *drift* -- a vendored backend that starts
producing subtly different numbers after an upgrade while still satisfying
every property we assert. nanodsp vendors twelve C++ libraries, so that drift
is the realistic failure mode.

This module pins the actual numeric output of a representative slice of the API
against fingerprints stored in GOLDEN.json. A failure here does not by itself
mean something is broken; it means output changed, and a human has to decide
whether the change was intended. Regenerate with:

    python tests/test_golden.py --update

and review the resulting diff to GOLDEN.json as part of the change.

GOLDEN.json is a committed fixture, not a build artefact -- deleting it does not
"reset" anything, it just removes the baseline. Keep it in version control.

Comparison is numeric with a tolerance rather than by hash, because an exact
hash of float output is not portable across compilers or CPUs; see the note on
_RTOL below.

The corpus deliberately covers each backend at least once, because the point is
to notice when a backend moves under us.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

from nanodsp import analysis, ops, spectral, synthesis, timestretch
from nanodsp.buffer import AudioBuffer
from nanodsp.effects import composed, daisysp, dynamics, filters, reverb, saturation

GOLDEN_PATH = Path(__file__).parent / "GOLDEN.json"

SR = 48000.0
# 2 s. Long enough for ITU-R BS.1770 gated loudness, which needs at least one
# 400 ms block and is only meaningful over several.
FRAMES = 96000

# Fixtures are compared numerically, not by hash.
#
# An exact digest of float output is not portable: rebuilding with a different
# compiler, optimisation level or CPU changes results in the last bits, and in
# feedback systems (IIR filters, reverb tails, phase-vocoder overlap) that
# difference accumulates. Measured on this corpus, perturbing the input by 1
# part in 10^7 -- roughly float32 rounding noise -- already moves lowpass,
# reverb and paulstretch output by ~3e-7, which flips any hash. A hash also has
# no notion of "close": a value sitting on a quantisation boundary flips from a
# difference of one ULP.
#
# So each case stores a small fingerprint compared with a tolerance: overall
# peak and mean, plus per-block RMS, which localises a change to a region of the
# signal rather than only detecting it globally. Real DSP changes move these far
# more than the tolerance -- a 1% compressor-ratio change is ~1e-3 -- while
# platform noise does not.
_BLOCKS = 64  # per-case RMS resolution
_RTOL = 1e-4
_ATOL = 1e-6  # ~ -120 dBFS, below any audible or meaningful difference


def _signal(channels: int = 1, seed: int = 0) -> AudioBuffer:
    """Deterministic test signal: tone plus noise plus a transient.

    Fixed seed and explicit float32 construction keep this bit-identical across
    platforms and numpy versions.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(FRAMES, dtype=np.float64) / SR
    rows = []
    for ch in range(channels):
        tone = 0.4 * np.sin(2 * np.pi * (220.0 * (ch + 1)) * t)
        noise = 0.05 * rng.standard_normal(FRAMES)
        sig = tone + noise
        # A transient at a fixed offset, so dynamics processors have something
        # to react to.
        sig[FRAMES // 3 : FRAMES // 3 + 256] += 0.5
        rows.append(sig)
    return AudioBuffer(np.asarray(rows, dtype=np.float32), sample_rate=SR)


MONO = _signal(1)
STEREO = _signal(2, seed=1)


def _flatten(value) -> tuple[list[int], np.ndarray]:
    """Reduce an AudioBuffer, array, tuple or scalar to (shape, 1-D float64)."""
    if isinstance(value, AudioBuffer):
        arr: Any = value.data
    elif isinstance(value, tuple):
        parts = [_flatten(v) for v in value]
        return (
            [len(parts)] + [n for shape, _ in parts for n in shape],
            np.concatenate([flat for _, flat in parts]),
        )
    elif isinstance(value, (int, float, np.floating, np.integer)):
        arr = np.asarray([value], dtype=np.float64)
    else:
        arr = np.asarray(value)

    shape = list(np.shape(arr))
    if np.iscomplexobj(arr):
        arr = np.concatenate([np.real(arr).ravel(), np.imag(arr).ravel()])
    flat = np.asarray(arr, dtype=np.float64).ravel()
    # Keep non-finite values comparable rather than propagating NaN into every
    # statistic; a case that starts producing NaN shows up as a changed value.
    flat = np.nan_to_num(flat, nan=0.0, posinf=1e6, neginf=-1e6)
    return shape, np.clip(flat, -1e6, 1e6)


def _fingerprint(value) -> dict:
    """Tolerance-comparable summary of a case's output.

    Per-block RMS is the substance of this: it localises a change to a region of
    the signal, so a fix that alters only a reverb tail or only an attack
    transient is still caught, without storing the whole waveform.
    """
    shape, flat = _flatten(value)
    if flat.size == 0:
        return {"shape": shape, "n": 0, "peak": 0.0, "mean": 0.0, "rms": []}
    blocks = np.array_split(flat, min(_BLOCKS, flat.size))
    return {
        "shape": shape,
        "n": int(flat.size),
        "peak": float(np.max(np.abs(flat))),
        "mean": float(flat.mean()),
        "rms": [float(np.sqrt(np.mean(b * b))) for b in blocks],
    }


def _compare(got: dict, want: dict) -> str | None:
    """Return a human-readable reason the fingerprints differ, or None."""
    if got["shape"] != want["shape"] or got["n"] != want["n"]:
        return (
            f"shape changed: {want['shape']} ({want['n']} values) -> "
            f"{got['shape']} ({got['n']} values)"
        )
    for key in ("peak", "mean"):
        if not np.isclose(got[key], want[key], rtol=_RTOL, atol=_ATOL):
            return f"{key} changed: {want[key]:.8g} -> {got[key]:.8g}"
    a, b = np.asarray(got["rms"]), np.asarray(want["rms"])
    if a.shape != b.shape:
        return f"block count changed: {b.shape[0]} -> {a.shape[0]}"
    close = np.isclose(a, b, rtol=_RTOL, atol=_ATOL)
    if not close.all():
        i = int(np.argmax(~close))
        return (
            f"RMS differs in {int((~close).sum())} of {a.size} blocks; "
            f"first at block {i}: {b[i]:.8g} -> {a[i]:.8g}"
        )
    return None


# Each case: (name, callable). Names are stable keys into GOLDEN.json -- renaming
# one loses its history, so append rather than reorder.
CASES: dict[str, Callable[[], Any]] = {
    # --- signalsmith: biquads, FFT, delay ---
    "filters.lowpass": lambda: filters.lowpass(MONO, 2000.0),
    "filters.highpass": lambda: filters.highpass(MONO, 500.0),
    "filters.bandpass": lambda: filters.bandpass(MONO, 1000.0),
    "filters.notch": lambda: filters.notch(MONO, 1000.0),
    "filters.peak_db": lambda: filters.peak_db(MONO, 1500.0, 6.0),
    "filters.low_shelf_db": lambda: filters.low_shelf_db(MONO, 300.0, -4.0),
    "filters.allpass": lambda: filters.allpass(MONO, 800.0),
    "ops.delay": lambda: ops.delay(MONO, 128),
    "ops.convolve": lambda: ops.convolve(MONO, AudioBuffer.impulse(1, 64, SR)),
    "ops.hilbert": lambda: ops.hilbert(MONO),
    "ops.normalize_peak": lambda: ops.normalize_peak(MONO, -3.0),
    "ops.stereo_widen": lambda: ops.stereo_widen(STEREO, 1.5),
    "ops.mid_side_encode": lambda: ops.mid_side_encode(STEREO),
    "ops.pan": lambda: ops.pan(MONO, 0.4),
    # --- DaisySP: filters, modulation, dynamics ---
    "filters.moog_ladder": lambda: filters.moog_ladder(MONO, 1200.0, 0.3),
    "filters.svf_lowpass": lambda: filters.svf_lowpass(MONO, 1500.0),
    # chorus/flanger/phaser used to read lfo_freq_ uninitialized in Init, which
    # could latch a reversed LFO for the object's lifetime. Now patched (see
    # thirdparty/VERSIONS.md); pinned here so a regression or a DaisySP upgrade
    # that drops the patch is caught.
    "daisysp.chorus": lambda: daisysp.chorus(MONO, lfo_freq=0.5, lfo_depth=0.4),
    "daisysp.flanger": lambda: daisysp.flanger(MONO),
    "daisysp.phaser": lambda: daisysp.phaser(MONO),
    "daisysp.tremolo": lambda: daisysp.tremolo(MONO),
    "daisysp.bitcrush": lambda: daisysp.bitcrush(MONO, bit_depth=8),
    "dynamics.compress.linked": lambda: dynamics.compress(STEREO, link=True),
    "dynamics.compress.unlinked": lambda: dynamics.compress(STEREO, link=False),
    "dynamics.compress.mono": lambda: dynamics.compress(MONO),
    "dynamics.limit.linked": lambda: dynamics.limit(STEREO, pre_gain=2.0),
    "dynamics.noise_gate": lambda: dynamics.noise_gate(STEREO, threshold_db=-30.0),
    # --- saturation / fxdsp ---
    "saturation.saturate": lambda: saturation.saturate(MONO, drive=0.4, mode="tape"),
    "saturation.aa_hard_clip": lambda: saturation.aa_hard_clip(MONO, drive=2.0),
    # --- reverbs: madronalib FDN, fxdsp Schroeder/Moorer ---
    "reverb.fdn_hall": lambda: reverb.reverb(MONO, preset="hall", mix=0.4),
    "reverb.fdn_room": lambda: reverb.reverb(STEREO, preset="room", mix=0.3),
    "reverb.schroeder": lambda: reverb.schroeder_reverb(MONO),
    "reverb.moorer": lambda: reverb.moorer_reverb(MONO),
    # --- composed chains ---
    "composed.master": lambda: composed.master(STEREO),
    "composed.vocal_chain": lambda: composed.vocal_chain(MONO),
    "composed.de_esser": lambda: composed.de_esser(MONO),
    "composed.multiband_compress": lambda: composed.multiband_compress(STEREO),
    "composed.parallel_compress": lambda: composed.parallel_compress(MONO),
    # --- spectral ---
    "spectral.stft_magnitude": lambda: spectral.magnitude(spectral.stft(MONO)),
    "spectral.roundtrip": lambda: spectral.istft(spectral.stft(MONO)),
    "spectral.time_stretch": lambda: spectral.istft(
        spectral.time_stretch(spectral.stft(MONO), 1.5)
    ),
    "spectral.pitch_shift": lambda: spectral.pitch_shift_spectral(MONO, 4.0),
    "spectral.spectral_gate": lambda: spectral.istft(
        spectral.spectral_gate(spectral.stft(MONO))
    ),
    # --- time stretching ---
    "timestretch.paulstretch": lambda: timestretch.paulstretch(MONO, stretch=3.0),
    "timestretch.signalsmith": lambda: timestretch.signalsmith_stretch(
        MONO, stretch=1.5, semitones=2.0
    ),
    # --- analysis (scalars and curves) ---
    "analysis.loudness_lufs": lambda: analysis.loudness_lufs(STEREO),
    "analysis.true_peak_dbtp": lambda: analysis.true_peak_dbtp(STEREO),
    "analysis.spectral_centroid": lambda: analysis.spectral_centroid(MONO),
    "analysis.spectral_flatness": lambda: analysis.spectral_flatness_curve(MONO),
    "analysis.chromagram": lambda: analysis.chromagram(MONO),
    "analysis.onset_detect": lambda: analysis.onset_detect(MONO),
    "analysis.resample_44k": lambda: analysis.resample(MONO, 44100.0),
    "analysis.normalize_lufs": lambda: analysis.normalize_lufs(STEREO, -18.0),
    # --- synthesis (STK, PolyBLEP, DaisySP drums) ---
    "synthesis.oscillator": lambda: synthesis.oscillator(
        FRAMES, freq=440.0, waveform="saw", sample_rate=SR
    ),
    "synthesis.polyblep": lambda: synthesis.polyblep(
        FRAMES, freq=440.0, sample_rate=SR
    ),
    "synthesis.fm2": lambda: synthesis.fm2(FRAMES, freq=220.0, sample_rate=SR),
    "synthesis.analog_bass_drum": lambda: synthesis.analog_bass_drum(
        FRAMES, freq=60.0, sample_rate=SR
    ),
    "synthesis.synth_note_bowed": lambda: synthesis.synth_note(
        "bowed", freq=440.0, duration=0.2, sample_rate=SR
    ),
    "synthesis.synth_note_brass": lambda: synthesis.synth_note(
        "brass", freq=330.0, duration=0.2, sample_rate=SR
    ),
    # Noise-bearing voices. These used to be unpinnable because STK seeded the
    # shared rand() stream from the wall clock; they now take an explicit seed.
    "synthesis.synth_note_clarinet": lambda: synthesis.synth_note(
        "clarinet", freq=440.0, duration=0.2, sample_rate=SR, seed=7
    ),
    "synthesis.synth_note_plucked": lambda: synthesis.synth_note(
        "plucked", freq=220.0, duration=0.2, sample_rate=SR, seed=7
    ),
    "synthesis.pluck": lambda: synthesis.pluck(
        FRAMES // 4, freq=200.0, sample_rate=SR, seed=3
    ),
    "synthesis.dust": lambda: synthesis.dust(FRAMES // 4, sample_rate=SR, seed=3),
    "synthesis.hihat": lambda: synthesis.hihat(FRAMES // 4, sample_rate=SR, seed=3),
    # --- I/O quantisation (pins the encoder scale/rounding) ---
    "io.wav16_roundtrip": lambda: _wav_roundtrip(16),
    "io.wav24_roundtrip": lambda: _wav_roundtrip(24),
}


def _wav_roundtrip(bit_depth: int) -> AudioBuffer:
    from nanodsp.io import read_wav_bytes, write_wav_bytes

    return read_wav_bytes(write_wav_bytes(STEREO, bit_depth=bit_depth))


def _compute_all() -> dict[str, dict]:
    return {name: _fingerprint(fn()) for name, fn in sorted(CASES.items())}


def _load_golden() -> dict[str, dict]:
    if not GOLDEN_PATH.is_file():
        pytest.fail(
            f"{GOLDEN_PATH.name} is missing. It is a committed fixture, not a "
            "build artefact -- restore it from version control, or regenerate "
            "with `python tests/test_golden.py --update` if this is a new "
            "checkout that never had one."
        )
    return json.loads(GOLDEN_PATH.read_text())


@pytest.mark.parametrize("name", sorted(CASES))
def test_golden_output(name: str) -> None:
    golden = _load_golden()
    if name not in golden:
        pytest.fail(
            f"No stored fingerprint for {name!r}. If this is a new case, "
            "regenerate with `python tests/test_golden.py --update`."
        )
    reason = _compare(_fingerprint(CASES[name]()), golden[name])
    assert reason is None, (
        f"Output of {name!r} changed: {reason}\n"
        "This is not necessarily a bug -- it means the numbers moved beyond "
        f"the tolerance (rtol={_RTOL}, atol={_ATOL}), which platform noise "
        "does not. Confirm the change was intended (a vendored backend "
        "upgrade, a deliberate DSP fix), then regenerate with "
        "`python tests/test_golden.py --update` and include the GOLDEN.json "
        "diff in the same commit."
    )


def test_golden_file_has_no_stale_entries() -> None:
    """A removed or renamed case must not leave an orphan fingerprint behind."""
    orphans = sorted(set(_load_golden()) - set(CASES))
    assert not orphans, (
        f"GOLDEN.json has entries with no corresponding case: {orphans}. "
        "Regenerate to drop them."
    )


def test_fingerprint_is_deterministic() -> None:
    """The fingerprint must not depend on run order or global state."""
    assert (
        _compare(
            _fingerprint(filters.lowpass(MONO, 2000.0)),
            _fingerprint(filters.lowpass(MONO, 2000.0)),
        )
        is None
    )


def test_fingerprint_tolerates_float_noise() -> None:
    """The point of the tolerance: a rebuild on another platform must still pass.

    Perturbing the input by 1 part in 10^7 stands in for the last-bit
    differences a different compiler or CPU produces. An exact hash flipped
    here, which made the corpus machine-specific.
    """
    noisy = AudioBuffer(MONO.data * np.float32(1 + 1e-7), sample_rate=MONO.sample_rate)
    for fn in (
        lambda b: filters.lowpass(b, 2000.0),
        lambda b: reverb.reverb(b, preset="hall", mix=0.4),
        lambda b: timestretch.paulstretch(b, stretch=3.0),
    ):
        assert _compare(_fingerprint(fn(noisy)), _fingerprint(fn(MONO))) is None


def test_fingerprint_detects_a_real_change() -> None:
    """Guard the other way: the tolerance must not be so loose it sees nothing."""
    base = filters.lowpass(MONO, 2000.0)
    louder = AudioBuffer(base.data * 1.001, sample_rate=base.sample_rate)
    assert _compare(_fingerprint(louder), _fingerprint(base)) is not None


def test_fingerprint_detects_a_localised_change() -> None:
    """A change confined to one region must not be averaged away."""
    base = filters.lowpass(MONO, 2000.0)
    nudged = AudioBuffer(base.data.copy(), sample_rate=base.sample_rate)
    nudged.data[0, : base.frames // 64] *= 1.05
    assert _compare(_fingerprint(nudged), _fingerprint(base)) is not None


def test_fingerprint_detects_a_length_change() -> None:
    base = filters.lowpass(MONO, 2000.0)
    assert _compare(_fingerprint(base.slice(0, base.frames - 1)), _fingerprint(base))


if __name__ == "__main__":  # pragma: no cover
    import sys

    if "--update" in sys.argv:
        prints = _compute_all()
        GOLDEN_PATH.write_text(json.dumps(prints, indent=2, sort_keys=True) + "\n")
        print(f"Wrote {len(prints)} fingerprints to {GOLDEN_PATH}")
    else:
        print(__doc__)
        print("Run with --update to regenerate the corpus.")

"""Tests for PaulStretch extreme time-stretching."""

import numpy as np
import pytest

from nanodsp._core import paulstretch as _ps
from nanodsp.buffer import AudioBuffer
from nanodsp.timestretch import paulstretch


SR = 44100.0


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def make_sine(freq=440.0, dur=1.0, sr=SR, amp=0.5):
    t = np.arange(int(dur * sr), dtype=np.float32) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def make_buf(data, sr=SR):
    return AudioBuffer(np.asarray(data, dtype=np.float32), sample_rate=sr)


def dominant_freq(x, sr=SR):
    """Return the peak-magnitude frequency of a 1D signal."""
    x = np.asarray(x, dtype=np.float64)
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
    return freqs[int(np.argmax(spec))]


def power_spectrum(x, sr=SR):
    """Return (freqs, power) of a windowed 1D signal."""
    x = np.asarray(x, dtype=np.float64)
    p = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    return np.fft.rfftfreq(len(x), 1.0 / sr), p


def band_fraction(x, f0, width=0.1, sr=SR):
    """Fraction of total power within +/- `width` octaves of f0."""
    f, p = power_spectrum(x, sr)
    band = (f > f0 * 2.0**-width) & (f < f0 * 2.0**width)
    return float(p[band].sum() / p.sum())


def spectral_width_octaves(x, f0, sr=SR):
    """RMS spread of the power spectrum about f0, measured in octaves."""
    f, p = power_spectrum(x, sr)
    m = (f > 20.0) & (p > 0.0)
    lf = np.log2(f[m] / f0)
    w = p[m]
    mu = (lf * w).sum() / w.sum()
    return float(np.sqrt(((lf - mu) ** 2 * w).sum() / w.sum()))


def make_tone_plus_noise(f0=440.0, dur=1.0, sr=SR, seed=0):
    """A pure tone summed with equal-amplitude broadband noise."""
    t = np.arange(int(dur * sr), dtype=np.float32) / sr
    rng = np.random.default_rng(seed)
    tone = 0.4 * np.sin(2 * np.pi * f0 * t)
    noise = 0.4 * rng.standard_normal(len(t))
    return (tone + noise).astype(np.float32)


# ---------------------------------------------------------------------------
# Core algorithm (C++ class)
# ---------------------------------------------------------------------------


def test_stretch_lengthens_by_factor():
    x = make_sine(dur=1.0)
    p = _ps.PaulStretch(4096, SR)
    y = p.process(x, 8.0)
    # Output length is ~ input * stretch (within one window).
    assert abs(len(y) / len(x) - 8.0) < 0.2


def test_output_finite_and_bounded():
    x = make_sine(dur=1.0)
    p = _ps.PaulStretch(4096, SR)
    p.set_seed(1)
    y = p.process(x, 8.0)
    assert np.all(np.isfinite(y))
    # No edge blow-up: peak stays well below a few times the input peak.
    assert np.max(np.abs(y)) < 2.0


def test_silence_stays_silence():
    p = _ps.PaulStretch(4096, SR)
    y = p.process(np.zeros(20000, dtype=np.float32), 8.0)
    assert np.max(np.abs(y)) == 0.0


def test_nonzero_output_for_nonzero_input():
    x = make_sine(dur=0.5)
    p = _ps.PaulStretch(2048, SR)
    p.set_seed(3)
    assert rms(p.process(x, 6.0)) > 1e-3


def test_deterministic_with_seed():
    x = make_sine(dur=0.5)
    p = _ps.PaulStretch(2048, SR)
    p.set_seed(123)
    y1 = p.process(x, 6.0)
    p.reset()
    p.set_seed(123)
    y2 = p.process(x, 6.0)
    assert np.array_equal(y1, y2)


def test_different_seed_changes_output():
    x = make_sine(dur=0.5)
    p = _ps.PaulStretch(2048, SR)
    p.set_seed(1)
    y1 = p.process(x, 6.0)
    p.set_seed(2)
    y2 = p.process(x, 6.0)
    assert not np.array_equal(y1, y2)


def test_pitch_preserved_without_shift():
    # A 440 Hz tone stretched without pitch shift stays near 440 Hz.
    x = make_sine(freq=440.0, dur=1.0)
    p = _ps.PaulStretch(4096, SR)
    p.set_seed(5)
    y = p.process(x, 4.0)
    assert abs(dominant_freq(y) - 440.0) < 30.0


def test_pitch_shift_octave_up():
    x = make_sine(freq=440.0, dur=1.0)
    p = _ps.PaulStretch(4096, SR)
    p.pitch_semitones = 12.0
    p.set_seed(5)
    y = p.process(x, 4.0)
    # One octave up -> ~880 Hz.
    assert abs(dominant_freq(y) - 880.0) < 60.0


def test_spectral_highpass_removes_low_tone():
    x = make_sine(freq=300.0, dur=1.0)
    p = _ps.PaulStretch(4096, SR)
    p.highpass_hz = 2000.0
    p.set_seed(5)
    y = p.process(x, 4.0)
    # The 300 Hz tone is below the spectral high-pass, so almost nothing passes.
    assert rms(y) < 0.05 * rms(x)


def test_spectral_lowpass_keeps_low_tone():
    x = make_sine(freq=300.0, dur=1.0)
    p = _ps.PaulStretch(4096, SR)
    p.lowpass_hz = 2000.0
    p.set_seed(5)
    y = p.process(x, 4.0)
    assert rms(y) > 1e-2


# ---------------------------------------------------------------------------
# Constant-Q (log-frequency) spread
# ---------------------------------------------------------------------------


def test_log_spread_widens_a_partial():
    x = make_sine(freq=440.0, dur=1.0)
    p = _ps.PaulStretch(4096, SR)
    p.set_seed(5)
    narrow = spectral_width_octaves(p.process(x, 4.0), 440.0)
    p.spread_octaves = 0.3
    p.set_seed(5)
    wide = spectral_width_octaves(p.process(x, 4.0), 440.0)
    assert wide > 3.0 * narrow


def test_log_spread_is_monotonic_in_octaves():
    x = make_sine(freq=440.0, dur=1.0)
    widths = []
    for oct_ in (0.0, 0.1, 0.2, 0.4, 0.8):
        p = _ps.PaulStretch(4096, SR)
        p.spread_octaves = oct_
        p.set_seed(5)
        widths.append(spectral_width_octaves(p.process(x, 4.0), 440.0))
    assert all(b > a for a, b in zip(widths, widths[1:])), widths


def test_log_spread_is_constant_q():
    # The whole point of the log-frequency axis: a low and a high partial are
    # smeared by the same number of octaves. A linear-bin blur is not.
    widths = {}
    for f0 in (220.0, 3520.0):
        x = make_sine(freq=f0, dur=1.0)
        for name, attr, value in (
            ("log", "spread_octaves", 0.3),
            ("lin", "spread", 32.0),
        ):
            p = _ps.PaulStretch(4096, SR)
            setattr(p, attr, value)
            p.set_seed(5)
            widths[name, f0] = spectral_width_octaves(p.process(x, 4.0), f0)

    log_ratio = widths["log", 220.0] / widths["log", 3520.0]
    lin_ratio = widths["lin", 220.0] / widths["lin", 3520.0]
    # Constant-Q holds to within 30%; the linear blur is off by an order of
    # magnitude in the same test.
    assert 0.7 < log_ratio < 1.4, widths
    assert lin_ratio > 5.0, widths


def test_log_spread_width_independent_of_window_size():
    x = make_sine(freq=440.0, dur=1.0)
    widths = []
    for ws in (2048, 4096, 16384):
        p = _ps.PaulStretch(ws, SR)
        p.spread_octaves = 0.3
        p.set_seed(5)
        widths.append(spectral_width_octaves(p.process(x, 4.0), 440.0))
    assert max(widths) / min(widths) < 1.4, widths


# ---------------------------------------------------------------------------
# Tonal vs. noise separation
# ---------------------------------------------------------------------------


def _tonal_band_fraction(amount, x=None, **attrs):
    """band_fraction at 440 Hz for a given tonal_vs_noise setting."""
    if x is None:
        x = make_tone_plus_noise(440.0)
    p = _ps.PaulStretch(4096, SR)
    p.tonal_vs_noise = amount
    for k, v in attrs.items():
        setattr(p, k, v)
    p.set_seed(5)
    return band_fraction(p.process(x, 4.0), 440.0)


def test_tonal_vs_noise_positive_keeps_the_tone():
    base = _tonal_band_fraction(0.0)
    tonal = _tonal_band_fraction(1.0)
    # The 440 Hz partial dominates the output far more than before.
    assert tonal > 2.0 * base
    assert tonal > 0.7


def test_tonal_vs_noise_negative_removes_the_tone():
    base = _tonal_band_fraction(0.0)
    noisy = _tonal_band_fraction(-1.0)
    assert noisy < 0.5 * base


def test_tonal_vs_noise_zero_is_a_passthrough():
    x = make_tone_plus_noise(440.0)
    p = _ps.PaulStretch(4096, SR)
    p.set_seed(5)
    off = p.process(x, 4.0)
    p.tonal_vs_noise = 0.0
    p.reset()
    p.set_seed(5)
    assert np.array_equal(off, p.process(x, 4.0))


def test_tonal_vs_noise_is_monotone_across_the_full_range():
    # Every setting in [-1, 1] is usable: the tone's share of the output rises
    # monotonically from the noise extreme to the tonal one.
    amounts = (-1.0, -0.7, -0.5, -0.2, 0.0, 0.2, 0.5, 0.7, 1.0)
    fracs = [_tonal_band_fraction(a) for a in amounts]
    assert all(b > a for a, b in zip(fracs, fracs[1:])), fracs


def test_tonal_vs_noise_on_pure_noise_removes_nearly_everything():
    rng = np.random.default_rng(3)
    x = (0.4 * rng.standard_normal(int(SR))).astype(np.float32)
    p = _ps.PaulStretch(4096, SR)
    p.set_seed(5)
    base = rms(p.process(x, 4.0))
    p.tonal_vs_noise = 1.0
    p.reset()
    p.set_seed(5)
    y = p.process(x, 4.0)
    assert np.all(np.isfinite(y))
    # Nothing stands proud of its own envelope, so little survives.
    assert rms(y) < 0.25 * base


def test_tonal_noise_octaves_widens_what_counts_as_a_peak():
    # A wider envelope is a lower bar for a bin to stand above, so more of the
    # spectrum survives the tonal extraction.
    levels = []
    for width in (0.1, 0.3, 0.5):
        p = _ps.PaulStretch(4096, SR)
        p.tonal_vs_noise = 1.0
        p.tonal_noise_octaves = width
        p.set_seed(5)
        levels.append(rms(p.process(make_tone_plus_noise(440.0), 4.0)))
    assert all(b > a for a, b in zip(levels, levels[1:])), levels


@pytest.mark.parametrize(
    "attr,value",
    [
        ("onset_sensitivity", 0.8),
        ("pitch_semitones", -12.0),
        ("harmonics", 3),
        ("spread", 8.0),
        ("spread_octaves", 0.5),
        ("tonal_vs_noise", 0.7),
        ("tonal_vs_noise", -0.7),
    ],
)
def test_effects_stay_finite_and_bounded(attr, value):
    x = make_sine(dur=0.7)
    p = _ps.PaulStretch(4096, SR)
    setattr(p, attr, value)
    p.set_seed(9)
    y = p.process(x, 5.0)
    assert np.all(np.isfinite(y))
    assert np.max(np.abs(y)) < 4.0


def test_window_size_rounded_even():
    p = _ps.PaulStretch(4097, SR)
    assert p.window_size % 2 == 0


# ---------------------------------------------------------------------------
# Python wrapper (AudioBuffer)
# ---------------------------------------------------------------------------


def test_wrapper_preserves_metadata():
    b = make_buf(make_sine(dur=0.5))
    out = paulstretch(b, stretch=4.0)
    assert out.sample_rate == b.sample_rate
    assert out.channel_layout == b.channel_layout
    assert abs(out.frames / b.frames - 4.0) < 0.3


def test_wrapper_stereo_same_length_decorrelated():
    left = make_sine(freq=330.0, dur=0.5)
    right = make_sine(freq=440.0, dur=0.5)
    b = AudioBuffer(np.stack([left, right]), sample_rate=SR)
    out = paulstretch(b, stretch=4.0)
    assert out.channels == 2
    # Both channels share the same length.
    assert out.data.shape[1] == out.frames
    # Independent per-channel seeds decorrelate the phases.
    assert not np.array_equal(out.data[0], out.data[1])


def test_wrapper_reproducible():
    b = make_buf(make_sine(dur=0.5))
    out1 = paulstretch(b, stretch=4.0, seed=7)
    out2 = paulstretch(b, stretch=4.0, seed=7)
    assert np.array_equal(out1.data, out2.data)


def test_wrapper_passes_tonal_vs_noise_through():
    b = make_buf(make_tone_plus_noise(440.0))
    base = paulstretch(b, stretch=4.0, seed=5)
    tonal = paulstretch(b, stretch=4.0, seed=5, tonal_vs_noise=1.0)
    assert band_fraction(tonal.data[0], 440.0) > 2.0 * band_fraction(
        base.data[0], 440.0
    )


def test_wrapper_passes_spread_octaves_through():
    # Measured on a pure tone -- the octave width of broadband material says
    # nothing about how much a partial was smeared.
    b = make_buf(make_sine(freq=440.0, dur=1.0))
    base = paulstretch(b, stretch=4.0, seed=5)
    spread = paulstretch(b, stretch=4.0, seed=5, spread_octaves=0.3)
    assert spectral_width_octaves(spread.data[0], 440.0) > 3.0 * (
        spectral_width_octaves(base.data[0], 440.0)
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"stretch": 0.0},
        {"stretch": -2.0},
        {"window_size": 8},
        {"tonal_vs_noise": 1.5},
        {"tonal_vs_noise": -1.5},
    ],
)
def test_wrapper_validation(kwargs):
    b = make_buf(make_sine(dur=0.1))
    with pytest.raises(ValueError):
        paulstretch(b, **kwargs)


def test_wrapper_output_not_clipping():
    b = make_buf(make_sine(dur=0.5))
    out = paulstretch(b, stretch=6.0)
    assert np.all(np.isfinite(out.data))
    assert np.max(np.abs(out.data)) < 1.5


# ---------------------------------------------------------------------------
# CLI registration
# ---------------------------------------------------------------------------


def test_registered_in_cli():
    from nanodsp._cli import get_function, get_categories

    fn, module = get_function("paulstretch")
    assert callable(fn)
    assert "paulstretch" in get_categories()["spectral"]

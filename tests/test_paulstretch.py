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


@pytest.mark.parametrize(
    "attr,value",
    [
        ("onset_sensitivity", 0.8),
        ("pitch_semitones", -12.0),
        ("harmonics", 3),
        ("spread", 8.0),
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


@pytest.mark.parametrize(
    "kwargs", [{"stretch": 0.0}, {"stretch": -2.0}, {"window_size": 8}]
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

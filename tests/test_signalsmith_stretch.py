"""Tests for signalsmith-stretch time-stretching and pitch-shifting."""

import numpy as np
import pytest

from nanodsp._core import signalsmith_stretch as _ss
from nanodsp.buffer import AudioBuffer
from nanodsp.timestretch import signalsmith_stretch


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
# Core processor (C++ class)
# ---------------------------------------------------------------------------


def test_stretch_lengthens_by_factor():
    x = make_sine(dur=1.0).reshape(1, -1)
    p = _ss.SignalsmithStretch(1, SR, False, 1)
    y = p.process(x, 2.0)
    # Exact-length offline mode: out_frames == round(in_frames * stretch).
    assert y.shape[0] == 1
    assert y.shape[1] == round(x.shape[1] * 2.0)


def test_stretch_shortens_below_one():
    x = make_sine(dur=1.0).reshape(1, -1)
    p = _ss.SignalsmithStretch(1, SR, False, 1)
    y = p.process(x, 0.5)
    assert y.shape[1] == round(x.shape[1] * 0.5)


def test_unity_stretch_keeps_length():
    x = make_sine(dur=0.5).reshape(1, -1)
    p = _ss.SignalsmithStretch(1, SR, False, 1)
    y = p.process(x, 1.0)
    assert y.shape[1] == x.shape[1]


def test_output_finite_and_bounded():
    x = make_sine(dur=1.0).reshape(1, -1)
    p = _ss.SignalsmithStretch(1, SR, False, 1)
    y = p.process(x, 2.0)
    assert np.all(np.isfinite(y))
    # No edge blow-up: a 0.5-amplitude sine should not gain much headroom.
    assert np.max(np.abs(y)) < 2.0


def test_silence_stays_silence():
    p = _ss.SignalsmithStretch(1, SR, False, 1)
    y = p.process(np.zeros((1, 20000), dtype=np.float32), 2.0)
    assert np.max(np.abs(y)) == 0.0


def test_nonzero_output_for_nonzero_input():
    x = make_sine(dur=0.5).reshape(1, -1)
    p = _ss.SignalsmithStretch(1, SR, False, 3)
    assert rms(p.process(x, 1.5)) > 1e-3


def test_deterministic_with_seed():
    # Past ~2x the algorithm randomizes phase; same seed must reproduce.
    x = make_sine(dur=0.5).reshape(1, -1)
    y1 = _ss.SignalsmithStretch(1, SR, False, 123).process(x, 4.0)
    y2 = _ss.SignalsmithStretch(1, SR, False, 123).process(x, 4.0)
    assert np.array_equal(y1, y2)


def test_different_seed_changes_output():
    x = make_sine(dur=0.5).reshape(1, -1)
    y1 = _ss.SignalsmithStretch(1, SR, False, 1).process(x, 4.0)
    y2 = _ss.SignalsmithStretch(1, SR, False, 2).process(x, 4.0)
    assert not np.array_equal(y1, y2)


def test_channel_count_mismatch_raises():
    p = _ss.SignalsmithStretch(2, SR, False, 1)
    with pytest.raises(Exception):
        p.process(make_sine(dur=0.2).reshape(1, -1), 2.0)


def test_invalid_channels_raises():
    with pytest.raises(Exception):
        _ss.SignalsmithStretch(0, SR, False, 1)


def test_properties():
    p = _ss.SignalsmithStretch(2, SR, True, 9)
    assert p.channels == 2
    assert p.sample_rate == pytest.approx(SR)
    assert p.cheaper is True


# ---------------------------------------------------------------------------
# Pitch shifting
# ---------------------------------------------------------------------------


def test_pitch_preserved_without_shift():
    # A 440 Hz tone stretched without a pitch shift stays near 440 Hz.
    x = make_sine(freq=440.0, dur=1.0).reshape(1, -1)
    p = _ss.SignalsmithStretch(1, SR, False, 5)
    y = p.process(x, 2.0)
    assert abs(dominant_freq(y[0]) - 440.0) < 20.0


def test_pitch_shift_octave_up():
    x = make_sine(freq=440.0, dur=1.0).reshape(1, -1)
    p = _ss.SignalsmithStretch(1, SR, False, 5)
    p.transpose_semitones = 12.0
    y = p.process(x, 1.0)
    assert abs(dominant_freq(y[0]) - 880.0) < 30.0


def test_pitch_shift_octave_down():
    x = make_sine(freq=440.0, dur=1.0).reshape(1, -1)
    p = _ss.SignalsmithStretch(1, SR, False, 5)
    p.transpose_semitones = -12.0
    y = p.process(x, 1.0)
    assert abs(dominant_freq(y[0]) - 220.0) < 20.0


def test_pitch_independent_of_stretch():
    # Pitch shift and time stretch are decoupled: shifting up an octave while
    # doubling length keeps the new pitch and the new length.
    x = make_sine(freq=440.0, dur=1.0).reshape(1, -1)
    p = _ss.SignalsmithStretch(1, SR, False, 5)
    p.transpose_semitones = 12.0
    y = p.process(x, 2.0)
    assert y.shape[1] == round(x.shape[1] * 2.0)
    assert abs(dominant_freq(y[0]) - 880.0) < 40.0


# ---------------------------------------------------------------------------
# Python wrapper (AudioBuffer)
# ---------------------------------------------------------------------------


def test_wrapper_mono_roundtrip():
    buf = make_buf(make_sine(dur=1.0))
    out = signalsmith_stretch(buf, stretch=2.0, seed=1)
    assert out.channels == 1
    assert out.frames == round(buf.frames * 2.0)
    assert out.sample_rate == buf.sample_rate


def test_wrapper_stereo_preserves_layout():
    data = np.stack([make_sine(440.0), make_sine(550.0)])
    buf = AudioBuffer(data, sample_rate=SR)
    out = signalsmith_stretch(buf, stretch=1.5, semitones=3.0, seed=2)
    assert out.channels == 2
    assert out.frames == round(buf.frames * 1.5)
    assert out.channel_layout == buf.channel_layout


def test_wrapper_pure_pitch_shift_keeps_length():
    buf = make_buf(make_sine(freq=440.0, dur=1.0))
    out = signalsmith_stretch(buf, stretch=1.0, semitones=7.0, seed=1)
    assert out.frames == buf.frames
    # Up a perfect fifth: 440 * 2^(7/12) ~= 659 Hz.
    assert abs(dominant_freq(out.data[0]) - 659.26) < 25.0


def test_wrapper_cheaper_preset_runs():
    buf = make_buf(make_sine(dur=0.5))
    out = signalsmith_stretch(buf, stretch=2.0, cheaper=True, seed=1)
    assert out.frames == round(buf.frames * 2.0)
    assert np.all(np.isfinite(out.data))


def test_wrapper_tonality_limit_runs():
    buf = make_buf(make_sine(dur=0.5))
    out = signalsmith_stretch(
        buf, stretch=1.0, semitones=5.0, tonality_hz=8000.0, seed=1
    )
    assert out.frames == buf.frames
    assert np.all(np.isfinite(out.data))


def test_wrapper_deterministic():
    buf = make_buf(make_sine(dur=0.5))
    a = signalsmith_stretch(buf, stretch=3.0, seed=42)
    b = signalsmith_stretch(buf, stretch=3.0, seed=42)
    assert np.array_equal(a.data, b.data)


def test_wrapper_rejects_nonpositive_stretch():
    buf = make_buf(make_sine(dur=0.2))
    with pytest.raises(ValueError):
        signalsmith_stretch(buf, stretch=0.0)
    with pytest.raises(ValueError):
        signalsmith_stretch(buf, stretch=-1.0)

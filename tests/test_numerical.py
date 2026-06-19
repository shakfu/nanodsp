"""Numerical-correctness tests for the high-level DSP API.

These verify that operations produce results matching the intended *signal*
behavior -- filter frequency response, oscillator/pitch accuracy, and alias
suppression -- rather than merely checking that calls run and return the right
shape/dtype (which the rest of the suite covers broadly).

Measurements use single-bin DFTs and windowed FFTs on long, steady-state
buffers; thresholds are deliberately generous so the tests track real DSP
correctness without being brittle across the various filter designs.
"""

import numpy as np
import pytest

from nanodsp import AudioBuffer
from nanodsp import analysis as A
from nanodsp import synthesis as S
from nanodsp.effects import filters as F

SR = 48000.0
N = 16384
BIN_HZ = SR / N  # FFT bin spacing for the default analysis length


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------


def _dft_mag(x: np.ndarray, hz: float, sr: float) -> float:
    """Amplitude-proportional magnitude of *x* at exactly *hz* (single-bin DFT)."""
    n = len(x)
    k = np.arange(n)
    ref = np.exp(-2j * np.pi * hz * k / sr)
    return float(np.abs(np.dot(x.astype(np.float64), ref)) / n)


def _gain_db(
    fn, test_hz: float, *, sr: float = SR, n: int = N, skip: int = 2048, **kw
) -> float:
    """Steady-state gain in dB of filter *fn* at *test_hz*.

    Feeds a pure tone through the filter and compares the output/input
    single-bin magnitude after discarding the initial transient.
    """
    x = AudioBuffer.sine(test_hz, frames=n, sample_rate=sr)
    y = fn(x, **kw)
    gi = _dft_mag(x.mono[skip:], test_hz, sr)
    go = _dft_mag(y.mono[skip:], test_hz, sr)
    return 20.0 * np.log10(go / (gi + 1e-20) + 1e-20)


def _fundamental_hz(
    x: np.ndarray, sr: float = SR, fmin: float = 50.0, fmax: float = 8000.0
) -> float:
    """Frequency of the strongest spectral peak in [fmin, fmax] (windowed FFT)."""
    n = len(x)
    w = np.hanning(n)
    mag = np.abs(np.fft.rfft(x * w))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    band = (freqs >= fmin) & (freqs <= fmax)
    return float(freqs[np.argmax(np.where(band, mag, 0.0))])


# ---------------------------------------------------------------------------
# Filter frequency response -- parametrized harness
# ---------------------------------------------------------------------------

# (id, fn, kwargs, passband_hz, stopband_hz) at cutoff 1000 Hz.
_LOWPASS_CASES = [
    ("lowpass", F.lowpass, dict(cutoff_hz=1000.0), 250.0, 8000.0),
    ("svf_lowpass", F.svf_lowpass, dict(freq_hz=1000.0), 250.0, 8000.0),
    ("moog_ladder", F.moog_ladder, dict(freq_hz=1000.0), 250.0, 8000.0),
    ("ladder_filter", F.ladder_filter, dict(freq_hz=1000.0), 250.0, 8000.0),
    ("tone_lowpass", F.tone_lowpass, dict(freq_hz=1000.0), 250.0, 8000.0),
    (
        "iir_butterworth_lp",
        F.iir_filter,
        dict(family="butterworth", filter_type="lowpass", order=4, freq=1000.0),
        250.0,
        8000.0,
    ),
]

_HIGHPASS_CASES = [
    ("highpass", F.highpass, dict(cutoff_hz=1000.0), 8000.0, 125.0),
    ("svf_highpass", F.svf_highpass, dict(freq_hz=1000.0), 8000.0, 125.0),
    ("tone_highpass", F.tone_highpass, dict(freq_hz=1000.0), 8000.0, 125.0),
    (
        "iir_butterworth_hp",
        F.iir_filter,
        dict(family="butterworth", filter_type="highpass", order=4, freq=1000.0),
        8000.0,
        125.0,
    ),
]


class TestFilterFrequencyResponse:
    """Verify filters pass their passband and attenuate their stopband."""

    @pytest.mark.parametrize(
        "fn, kw, pass_hz, stop_hz",
        [c[1:] for c in _LOWPASS_CASES],
        ids=[c[0] for c in _LOWPASS_CASES],
    )
    def test_lowpass_response(self, fn, kw, pass_hz, stop_hz):
        passband = _gain_db(fn, pass_hz, **kw)
        stopband = _gain_db(fn, stop_hz, **kw)
        assert passband > -6.0, f"passband over-attenuated: {passband:.2f} dB"
        assert stopband < -12.0, f"stopband not attenuated: {stopband:.2f} dB"
        assert passband - stopband > 10.0

    @pytest.mark.parametrize(
        "fn, kw, pass_hz, stop_hz",
        [c[1:] for c in _HIGHPASS_CASES],
        ids=[c[0] for c in _HIGHPASS_CASES],
    )
    def test_highpass_response(self, fn, kw, pass_hz, stop_hz):
        passband = _gain_db(fn, pass_hz, **kw)
        stopband = _gain_db(fn, stop_hz, **kw)
        assert passband > -6.0, f"passband over-attenuated: {passband:.2f} dB"
        assert stopband < -12.0, f"stopband not attenuated: {stopband:.2f} dB"
        assert passband - stopband > 10.0

    def test_bandpass_peaks_at_center(self):
        center = _gain_db(F.bandpass, 1000.0, center_hz=1000.0)
        below = _gain_db(F.bandpass, 125.0, center_hz=1000.0)
        above = _gain_db(F.bandpass, 8000.0, center_hz=1000.0)
        assert center > -3.0
        assert center - below > 8.0
        assert center - above > 8.0

    def test_notch_attenuates_center(self):
        center = _gain_db(F.notch, 1000.0, center_hz=1000.0)
        off = _gain_db(F.notch, 250.0, center_hz=1000.0)
        assert center < -30.0
        assert off > -3.0

    def test_high_shelf_boosts_highs(self):
        high = _gain_db(F.high_shelf, 8000.0, cutoff_hz=2000.0, gain=2.0)
        low = _gain_db(F.high_shelf, 125.0, cutoff_hz=2000.0, gain=2.0)
        assert high > 4.0  # ~ +6 dB (gain factor 2.0)
        assert abs(low) < 1.0
        assert high - low > 4.0

    def test_low_shelf_boosts_lows(self):
        low = _gain_db(F.low_shelf, 125.0, cutoff_hz=500.0, gain=2.0)
        high = _gain_db(F.low_shelf, 8000.0, cutoff_hz=500.0, gain=2.0)
        assert low > 4.0
        assert abs(high) < 1.0
        assert low - high > 4.0


# ---------------------------------------------------------------------------
# Pitch / fundamental accuracy
# ---------------------------------------------------------------------------


class TestPitchAccuracy:
    """Oscillators produce the requested fundamental; YIN recovers it."""

    @pytest.mark.parametrize("waveform", ["sine", "saw", "square", "triangle"])
    @pytest.mark.parametrize("f0", [110.0, 220.0, 440.0, 1000.0])
    def test_oscillator_fundamental(self, waveform, f0):
        buf = S.oscillator(N, freq=f0, waveform=waveform, sample_rate=SR)
        detected = _fundamental_hz(buf.mono, SR)
        tol = max(2.0 * BIN_HZ, 0.03 * f0)
        assert abs(detected - f0) <= tol, f"{waveform} {f0} -> {detected:.1f}"

    @pytest.mark.parametrize("f0", [110.0, 220.0, 440.0, 880.0])
    def test_pitch_detect_on_sine(self, f0):
        buf = AudioBuffer.sine(f0, frames=N, sample_rate=SR)
        freqs, conf = A.pitch_detect(buf, fmin=50.0, fmax=2000.0)
        voiced = freqs[(conf > 0.5) & (freqs > 0)]
        assert voiced.size > 0, "no voiced frames detected"
        median = float(np.median(voiced))
        assert abs(median - f0) / f0 < 0.03, f"f0={f0} median={median:.1f}"


# ---------------------------------------------------------------------------
# Alias suppression of band-limited oscillators
# ---------------------------------------------------------------------------


def _alias_fraction(x: np.ndarray, sr: float, f0: float) -> float:
    """Fraction of spectral energy below the fundamental.

    An ideal sawtooth has no energy below its fundamental; naive (non-band-
    limited) generation folds out-of-band harmonics down into this region, so a
    higher value indicates more aliasing.
    """
    n = len(x)
    mag = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    below = (freqs > 50.0) & (freqs < 0.9 * f0)
    full = (freqs > 50.0) & (freqs < 0.99 * sr / 2.0)
    return float(np.sum(mag[below] ** 2) / (np.sum(mag[full] ** 2) + 1e-20))


# Band-limited sawtooth generators (frames, freq, sample_rate) -> AudioBuffer.
_BL_SAWS = [
    (
        "polyblep",
        lambda n, f0, sr: S.polyblep(n, freq=f0, waveform="sawtooth", sample_rate=sr),
    ),
    ("blit_saw", lambda n, f0, sr: S.blit_saw(n, freq=f0, sample_rate=sr)),
    ("dpw_saw", lambda n, f0, sr: S.dpw_saw(n, freq=f0, sample_rate=sr)),
]


class TestAliasSuppression:
    """Band-limited oscillators alias far less than a naive sawtooth."""

    ALIAS_SR = 44100.0
    ALIAS_N = 32768
    ALIAS_F0 = 7000.0  # high enough that a naive saw aliases audibly

    def _naive_saw(self) -> np.ndarray:
        t = np.arange(self.ALIAS_N) / self.ALIAS_SR
        return (2.0 * ((self.ALIAS_F0 * t) % 1.0) - 1.0).astype(np.float32)

    def test_naive_saw_aliases(self):
        # Sanity: the reference naive saw genuinely has sub-fundamental energy.
        frac = _alias_fraction(self._naive_saw(), self.ALIAS_SR, self.ALIAS_F0)
        assert frac > 0.02

    @pytest.mark.parametrize("name, gen", _BL_SAWS, ids=[n for n, _ in _BL_SAWS])
    def test_bandlimited_suppresses_alias(self, name, gen):
        naive_frac = _alias_fraction(self._naive_saw(), self.ALIAS_SR, self.ALIAS_F0)
        buf = gen(self.ALIAS_N, self.ALIAS_F0, self.ALIAS_SR)
        bl_frac = _alias_fraction(buf.mono, self.ALIAS_SR, self.ALIAS_F0)
        assert bl_frac < 0.5 * naive_frac, (
            f"{name}: {bl_frac:.4f} vs naive {naive_frac:.4f}"
        )
        assert bl_frac < 0.02, f"{name}: residual aliasing {bl_frac:.4f}"

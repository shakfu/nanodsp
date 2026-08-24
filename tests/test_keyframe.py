"""Keyframe time stretching via extrema sampling.

The implementation is original, written from the equations and pseudocode in
Nielsen, "Keyframe Time Stretching via Extrema Sampling" (DAFx26, CC BY 4.0).
Several tests below check published figures from that paper directly, which is
the only independent evidence that the equations were read correctly -- there
is no reference implementation to diff against, by design.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanodsp import AudioBuffer, timestretch
from nanodsp._core import keyframe as _kf

SR = 48000.0


def _sine(freq=1000.0, dur=0.5, sr=SR):
    t = np.arange(int(sr * dur)) / sr
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def _sparse_then_dense():
    """A 60 Hz tone (widely spaced extrema) followed by noise (densely spaced)."""
    t = np.arange(24000) / SR
    tone = 0.8 * np.sin(2 * np.pi * 60.0 * t)
    noise = np.random.default_rng(0).standard_normal(24000) * 0.3
    return np.concatenate([tone, noise]).astype(np.float32)


def _thd_db(y, f0, sr=SR, harmonics=range(3, 20, 2)):
    n = len(y)
    spectrum = np.fft.rfft(y * np.hanning(n))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)

    def mag(f):
        i = int(np.argmin(np.abs(freqs - f)))
        return float(np.abs(spectrum[max(i - 2, 0) : i + 3]).max())

    fund = mag(f0)
    hs = [mag(f0 * h) for h in harmonics if f0 * h < sr / 2]
    return 20 * np.log10(np.sqrt(sum(m * m for m in hs)) / fund)


def _peak_freq(y, sr=SR):
    spectrum = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    return float(np.fft.rfftfreq(len(y), 1.0 / sr)[np.argmax(spectrum)])


class TestAnalysis:
    def test_finds_every_extremum_of_a_sine(self):
        """A 1 kHz sine at 48 kHz has exactly two extrema per cycle."""
        x = _sine(1000.0, 0.5)
        idx, val = _kf.analyze(x, 0.001)
        cycles = 1000.0 * 0.5
        assert len(idx) == pytest.approx(2 * cycles + 2, abs=2)  # + two anchors

    def test_extrema_land_on_the_peaks(self):
        """1 kHz at 48 kHz puts extrema exactly on samples 12, 36, 60, ..."""
        idx, _ = _kf.analyze(_sine(1000.0, 0.05), 0.001)
        np.testing.assert_allclose(idx[1:5], [12.0, 36.0, 60.0, 84.0], atol=1e-4)

    def test_on_grid_extrema_are_not_dropped(self):
        """Regression: an extremum sitting exactly on a sample.

        The bandlimited derivative is a central difference, so an extremum
        aligned to the sample grid makes it cancel to exactly 0.0. That is not
        a corner case -- every tone whose period divides the sample rate does
        it. Comparing the sign against the immediately preceding derivative
        rather than the last *nonzero* one loses half the extrema here.
        """
        x = _sine(1000.0, 0.25)  # 48 samples/cycle: every extremum is on-grid
        idx, _ = _kf.analyze(x, 0.001)
        spacing = np.diff(idx[1:-1])
        # Half a cycle, uniformly -- not a full cycle, which is what dropping
        # the on-grid extrema would produce.
        assert spacing.mean() == pytest.approx(24.0, abs=0.5)
        assert spacing.max() < 30.0

    def test_values_track_the_waveform(self):
        idx, val = _kf.analyze(_sine(1000.0, 0.05), 0.001)
        interior = val[1:-1]
        assert np.all(np.abs(interior) > 0.99)
        # Extrema alternate peak, trough, peak, ...
        assert np.all(np.diff(np.sign(interior)) != 0)

    def test_higher_threshold_keeps_fewer_extrema(self):
        rng = np.random.default_rng(0)
        x = (rng.standard_normal(8192) * 0.3).astype(np.float32)
        counts = [len(_kf.analyze(x, t)[0]) for t in (0.001, 0.01, 0.1)]
        assert counts[0] > counts[1] > counts[2]

    def test_density_never_exceeds_the_signal(self):
        """The sparse form cannot be larger than what it represents."""
        rng = np.random.default_rng(1)
        for x in (
            (rng.standard_normal(4096) * 0.3).astype(np.float32),
            _sine(440.0, 0.1),
            np.zeros(4096, dtype=np.float32),
        ):
            idx, _ = _kf.analyze(x, 0.001)
            assert len(idx) <= len(x)

    def test_silence_reduces_to_anchors(self):
        idx, val = _kf.analyze(np.zeros(4096, dtype=np.float32), 0.001)
        assert len(idx) == 2
        assert np.all(val == 0.0)


class TestSparsify:
    def test_matches_the_papers_published_thd(self):
        """Section 3.2 of the paper: -38.1 dB odd-harmonic THD on a 1 kHz sine.

        This is the sharpest available check that the B-spline kernel, the
        subsample refinement and the zero-tangent Hermite reconstruction were
        all implemented as specified. A misread equation moves this number.
        """
        y = _kf.sparsify(_sine(1000.0, 0.5), 0.001)
        assert _thd_db(y, 1000.0) == pytest.approx(-38.1, abs=1.0)

    def test_even_harmonics_are_negligible(self):
        """Reconstruction is symmetric, so it generates no even harmonics.

        The paper reports -85.8 dB, consistent with float32 arithmetic on the
        embedded target it was measured on. The kernel here evaluates in
        double, so the figure is far lower; the assertion is one-sided.
        """
        y = _kf.sparsify(_sine(1000.0, 0.5), 0.001)
        assert _thd_db(y, 1000.0, harmonics=range(2, 20, 2)) < -80.0

    def test_peak_error_matches_the_analytic_bound(self):
        """Smoothstep between the extrema of a sine differs from it by 0.0196.

        Figure 6 of the paper shows the same, to the width of the plotted line.
        The anchors at each end are not extrema, so the zero-tangent
        simplification does not hold across the first and last spans; they are
        excluded.
        """
        x = _sine(1000.0, 0.5)
        err = np.abs(_kf.sparsify(x, 0.001) - x)
        assert err[24:-24].max() == pytest.approx(0.0196, abs=0.005)

    def test_preserves_length_and_finiteness(self):
        rng = np.random.default_rng(2)
        x = (rng.standard_normal(4096) * 0.3).astype(np.float32)
        y = _kf.sparsify(x, 0.001)
        assert y.shape == x.shape
        assert np.isfinite(y).all()

    def test_silence_in_silence_out(self):
        y = _kf.sparsify(np.zeros(4096, dtype=np.float32), 0.001)
        assert np.allclose(y, 0.0)

    def test_does_not_amplify(self):
        x = _sine(440.0, 0.1)
        y = _kf.sparsify(x, 0.001)
        assert np.max(np.abs(y)) <= np.max(np.abs(x)) * 1.05


class TestStretch:
    @pytest.mark.parametrize("factor", [0.5, 1.0, 2.0, 4.0, 8.0])
    def test_output_length_follows_the_factor(self, factor):
        buf = AudioBuffer.sine(440.0, frames=24000, sample_rate=SR)
        out = timestretch.keyframe_stretch(buf, stretch=factor)
        assert out.frames == int(round(24000 * factor))

    def test_unity_is_a_plain_sparsify(self):
        """stretch=1, semitones=0 never splices, so it is the round trip."""
        buf = AudioBuffer.sine(440.0, frames=24000, sample_rate=SR)
        stretched = timestretch.keyframe_stretch(buf, stretch=1.0)
        plain = timestretch.keyframe_sparsify(buf)
        np.testing.assert_allclose(stretched.data, plain.data, atol=1e-6)

    @pytest.mark.parametrize("factor", [0.5, 2.0, 4.0])
    def test_pitch_survives_time_stretching(self, factor):
        buf = AudioBuffer.sine(440.0, frames=int(SR), sample_rate=SR)
        out = timestretch.keyframe_stretch(buf, stretch=factor)
        # Splices repeat and skip material, which smears the peak a little;
        # what matters is that it has not transposed.
        assert _peak_freq(out.data[0]) == pytest.approx(440.0, rel=0.04)

    @pytest.mark.parametrize("semitones", [-12.0, -5.0, 7.0, 12.0])
    def test_pitch_shift_ratio(self, semitones):
        buf = AudioBuffer.sine(440.0, frames=int(SR), sample_rate=SR)
        out = timestretch.keyframe_stretch(buf, semitones=semitones)
        assert out.frames == buf.frames
        expected = 440.0 * 2.0 ** (semitones / 12.0)
        assert _peak_freq(out.data[0]) == pytest.approx(expected, rel=0.04)

    def test_time_and_pitch_are_independent(self):
        buf = AudioBuffer.sine(440.0, frames=int(SR), sample_rate=SR)
        out = timestretch.keyframe_stretch(buf, stretch=2.0, semitones=12.0)
        assert out.frames == 2 * buf.frames
        assert _peak_freq(out.data[0], SR) == pytest.approx(880.0, rel=0.04)

    def test_stereo_channels_stay_aligned_and_distinct(self):
        t = np.arange(24000) / SR
        data = np.stack(
            [np.sin(2 * np.pi * 220 * t), np.sin(2 * np.pi * 330 * t)]
        ).astype(np.float32)
        buf = AudioBuffer(data, sample_rate=SR)
        out = timestretch.keyframe_stretch(buf, stretch=2.0)
        assert out.channels == 2
        assert out.frames == 48000
        assert not np.allclose(out.data[0], out.data[1])

    def test_output_is_bounded_and_finite(self):
        rng = np.random.default_rng(3)
        data = (rng.standard_normal((1, 24000)) * 0.3).astype(np.float32)
        buf = AudioBuffer(data, sample_rate=SR)
        for factor in (0.25, 1.0, 6.0):
            out = timestretch.keyframe_stretch(buf, stretch=factor)
            assert np.isfinite(out.data).all()
            assert np.max(np.abs(out.data)) <= np.max(np.abs(data)) * 1.05

    def test_splice_cap_shortens_long_splices(self):
        """A sparse passage would otherwise splice for a very long time.

        Uses a low tone (widely spaced extrema) followed by noise (densely
        spaced), which is the case section 3.7 of the paper calls out.
        """
        buf = AudioBuffer(_sparse_then_dense().reshape(1, -1), sample_rate=SR)
        capped = timestretch.keyframe_stretch(buf, stretch=4.0, max_splice_ms=5.0)
        uncapped = timestretch.keyframe_stretch(buf, stretch=4.0, max_splice_ms=0.0)
        assert not np.allclose(capped.data, uncapped.data)
        assert np.isfinite(capped.data).all()
        assert np.isfinite(uncapped.data).all()


class TestAdaptivity:
    """The paper's central claim, stated as a test.

    Extrema spacing is asserted to be a usable proxy for local information
    density -- that is what lets the splice duration adapt without a transient
    detector, an FFT or a correlation search. If that does not hold, the
    method has no reason to exist.
    """

    def test_extrema_spacing_tracks_signal_density(self):
        idx, _ = _kf.analyze(_sparse_then_dense(), 0.001)
        sparse = np.diff(idx[(idx > 1000) & (idx < 23000)]).mean()
        dense = np.diff(idx[(idx > 25000) & (idx < 47000)]).mean()
        # A 60 Hz tone against broadband noise: two orders of magnitude.
        assert sparse > 50.0 * dense

    def test_splices_are_shorter_where_the_signal_is_dense(self):
        """The consequence: identical settings, very different splice lengths.

        Stretching the two halves separately at the same factor and settings
        must not produce the same splice behaviour, because the keyframe span
        that sets splice duration differs by orders of magnitude between them.
        """
        signal = _sparse_then_dense()
        low = AudioBuffer(signal[:24000].reshape(1, -1), sample_rate=SR)
        noise = AudioBuffer(signal[24000:].reshape(1, -1), sample_rate=SR)
        idx_low, _ = _kf.analyze(signal[:24000], 0.001)
        idx_noise, _ = _kf.analyze(signal[24000:], 0.001)
        k = 16
        span_low = idx_low[k] - idx_low[0]
        span_noise = idx_noise[k] - idx_noise[0]
        assert span_low > 20.0 * span_noise
        # Both still render cleanly at the same settings.
        for buf in (low, noise):
            out = timestretch.keyframe_stretch(buf, stretch=3.0)
            assert np.isfinite(out.data).all()

    def test_larger_splice_threshold_changes_the_result(self):
        buf = AudioBuffer.sine(440.0, frames=24000, sample_rate=SR)
        few = timestretch.keyframe_stretch(buf, stretch=3.0, splice_keyframes=64)
        many = timestretch.keyframe_stretch(buf, stretch=3.0, splice_keyframes=4)
        assert not np.allclose(few.data, many.data)

    def test_metadata_is_preserved(self):
        buf = AudioBuffer.sine(440.0, frames=8000, sample_rate=44100.0, label="x")
        out = timestretch.keyframe_stretch(buf, stretch=2.0)
        assert out.sample_rate == 44100.0
        assert out.label == "x"
        assert out.channel_layout == buf.channel_layout


class TestValidation:
    @pytest.mark.parametrize("bad", [0.0, -1.0])
    def test_stretch_must_be_positive(self, bad):
        buf = AudioBuffer.sine(440.0, frames=1000, sample_rate=SR)
        with pytest.raises(ValueError, match="stretch must be positive"):
            timestretch.keyframe_stretch(buf, stretch=bad)

    def test_splice_keyframes_must_be_at_least_one(self):
        buf = AudioBuffer.sine(440.0, frames=1000, sample_rate=SR)
        with pytest.raises(ValueError, match="splice_keyframes must be >= 1"):
            timestretch.keyframe_stretch(buf, splice_keyframes=0)

    def test_threshold_must_be_non_negative(self):
        buf = AudioBuffer.sine(440.0, frames=1000, sample_rate=SR)
        with pytest.raises(ValueError, match="threshold must be non-negative"):
            timestretch.keyframe_stretch(buf, threshold=-0.1)
        with pytest.raises(ValueError, match="threshold must be non-negative"):
            timestretch.keyframe_sparsify(buf, threshold=-0.1)

    def test_max_splice_must_be_non_negative(self):
        buf = AudioBuffer.sine(440.0, frames=1000, sample_rate=SR)
        with pytest.raises(ValueError, match="max_splice_ms must be non-negative"):
            timestretch.keyframe_stretch(buf, max_splice_ms=-1.0)

    def test_short_and_empty_buffers_do_not_crash(self):
        for frames in (0, 1, 2, 3):
            buf = AudioBuffer(np.zeros((1, frames), dtype=np.float32), sample_rate=SR)
            out = timestretch.keyframe_stretch(buf, stretch=2.0)
            assert out.frames == frames * 2
            assert np.isfinite(out.data).all()

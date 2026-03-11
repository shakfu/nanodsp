"""Tests for nanodsp.ops module (delays, envelopes, FFT, convolution, rates, mix, LFO, numpy utils)."""

import numpy as np
import pytest

from nanodsp import ops
from nanodsp._core import delay as _delay_mod
from nanodsp._core import fft
from nanodsp._helpers import _hz_to_normalized
from nanodsp.buffer import AudioBuffer


# ---------------------------------------------------------------------------
# Frequency conversion
# ---------------------------------------------------------------------------


class TestFrequencyConversion:
    def test_valid_conversion(self):
        assert _hz_to_normalized(1000.0, 48000.0) == pytest.approx(1000.0 / 48000.0)

    def test_zero_hz(self):
        assert _hz_to_normalized(0.0, 48000.0) == 0.0

    def test_nyquist_rejection(self):
        with pytest.raises(ValueError, match="Nyquist"):
            _hz_to_normalized(24000.0, 48000.0)

    def test_above_nyquist_rejection(self):
        with pytest.raises(ValueError, match="Nyquist"):
            _hz_to_normalized(25000.0, 48000.0)

    def test_negative_rejection(self):
        with pytest.raises(ValueError, match="non-negative"):
            _hz_to_normalized(-100.0, 48000.0)


# ---------------------------------------------------------------------------
# Delay functions
# ---------------------------------------------------------------------------


class TestDelayFunctions:
    def test_basic_delay_shifts_impulse(self):
        buf = AudioBuffer.impulse(channels=1, frames=128, sample_rate=48000.0)
        result = ops.delay(buf, 10.0)
        peak_idx = np.argmax(np.abs(result.data[0]))
        expected = 10 + _delay_mod.Delay.latency
        assert peak_idx == expected

    def test_multichannel_delay(self):
        buf = AudioBuffer.impulse(channels=2, frames=128, sample_rate=48000.0)
        result = ops.delay(buf, 10.0)
        assert result.channels == 2
        assert result.frames == 128
        # Both channels should have same delay
        peak0 = np.argmax(np.abs(result.data[0]))
        peak1 = np.argmax(np.abs(result.data[1]))
        assert peak0 == peak1

    def test_fractional_delay(self):
        buf = AudioBuffer.impulse(channels=1, frames=128, sample_rate=48000.0)
        result = ops.delay(buf, 5.5)
        # Should produce nonzero output at interpolated samples
        assert np.max(np.abs(result.data)) > 0

    def test_varying_delay(self):
        buf = AudioBuffer.impulse(channels=1, frames=128, sample_rate=48000.0)
        delays = np.full(128, 10.0, dtype=np.float32)
        result = ops.delay_varying(buf, delays)
        assert result.frames == 128
        assert np.max(np.abs(result.data)) > 0

    def test_1d_delay_broadcast_multichannel(self):
        buf = AudioBuffer.impulse(channels=2, frames=128, sample_rate=48000.0)
        delays = np.full(128, 10.0, dtype=np.float32)
        result = ops.delay_varying(buf, delays)
        assert result.channels == 2

    def test_channel_mismatch_raises(self):
        buf = AudioBuffer.impulse(channels=2, frames=128, sample_rate=48000.0)
        delays = np.full((3, 128), 10.0, dtype=np.float32)
        with pytest.raises(ValueError, match="channels"):
            ops.delay_varying(buf, delays)

    def test_cubic_interpolation(self):
        buf = AudioBuffer.impulse(channels=1, frames=128, sample_rate=48000.0)
        result = ops.delay(buf, 5.0, interpolation="cubic")
        assert np.max(np.abs(result.data)) > 0


# ---------------------------------------------------------------------------
# Envelope functions
# ---------------------------------------------------------------------------


class TestEnvelopeFunctions:
    def test_box_filter_smooths(self):
        data = np.zeros((1, 128), dtype=np.float32)
        data[0, 32:] = 1.0
        buf = AudioBuffer(data, sample_rate=48000.0)
        result = ops.box_filter(buf, 16)
        # Should smooth the step
        assert result.data[0, 31] < result.data[0, 48]

    def test_box_stack_smoother(self):
        data = np.zeros((1, 256), dtype=np.float32)
        data[0, 0] = 1.0
        buf = AudioBuffer(data, sample_rate=48000.0)
        r_box = ops.box_filter(buf, 16)
        r_stack = ops.box_stack_filter(buf, 16, layers=4)
        # Both should produce output
        assert np.max(np.abs(r_box.data)) > 0
        assert np.max(np.abs(r_stack.data)) > 0

    def test_peak_hold_holds(self):
        data = np.zeros((1, 128), dtype=np.float32)
        data[0, 10] = 5.0
        buf = AudioBuffer(data, sample_rate=48000.0)
        result = ops.peak_hold(buf, 32)
        # Peak should be held for multiple samples after sample 10
        assert np.sum(result.data[0] >= 4.9) > 1

    def test_peak_decay_decays(self):
        data = np.zeros((1, 128), dtype=np.float32)
        data[0, 0] = 1.0
        buf = AudioBuffer(data, sample_rate=48000.0)
        result = ops.peak_decay(buf, 64)
        peak_val = np.max(result.data[0])
        assert peak_val > 0.9
        # Should decay after peak
        peak_idx = np.argmax(result.data[0])
        if peak_idx + 20 < 128:
            assert result.data[0, peak_idx + 20] < peak_val

    def test_envelope_multichannel(self):
        buf = AudioBuffer.noise(channels=2, frames=256, sample_rate=48000.0, seed=0)
        result = ops.box_filter(buf, 16)
        assert result.channels == 2
        assert result.frames == 256


# ---------------------------------------------------------------------------
# FFT functions
# ---------------------------------------------------------------------------


class TestFFTFunctions:
    def test_rfft_shape_dtype(self):
        buf = AudioBuffer.noise(channels=1, frames=1024, sample_rate=48000.0, seed=0)
        spectra = ops.rfft(buf)
        assert len(spectra) == 1
        assert spectra[0].dtype == np.complex64
        # bins = fast_size / 2
        fft_size = fft.RealFFT.fast_size_above(1024)
        assert spectra[0].shape == (fft_size // 2,)

    def test_multichannel_rfft(self):
        buf = AudioBuffer.noise(channels=3, frames=512, sample_rate=48000.0, seed=0)
        spectra = ops.rfft(buf)
        assert len(spectra) == 3

    def test_irfft_shape(self):
        buf = AudioBuffer.noise(channels=1, frames=256, sample_rate=48000.0, seed=0)
        spectra = ops.rfft(buf)
        result = ops.irfft(spectra, 256, sample_rate=48000.0)
        assert result.channels == 1
        assert result.frames == 256

    def test_roundtrip(self):
        buf = AudioBuffer.noise(channels=1, frames=1024, sample_rate=48000.0, seed=42)
        fft_size = fft.RealFFT.fast_size_above(1024)
        spectra = ops.rfft(buf)
        result = ops.irfft(spectra, 1024, sample_rate=48000.0)
        # Unscaled: need to divide by fft_size
        recovered = result.data / fft_size
        np.testing.assert_allclose(recovered[0, :1024], buf.data[0], atol=1e-4)


# ---------------------------------------------------------------------------
# Convolution
# ---------------------------------------------------------------------------


class TestConvolve:
    def test_impulse_passthrough(self):
        """Convolving with a unit impulse should return the input."""
        buf = AudioBuffer.sine(440.0, frames=1024, sample_rate=48000.0)
        ir = AudioBuffer.impulse(channels=1, frames=64, sample_rate=48000.0)
        result = ops.convolve(buf, ir)
        np.testing.assert_allclose(result.data, buf.data, atol=1e-5)

    def test_output_length_trimmed(self):
        buf = AudioBuffer.noise(channels=1, frames=1024, sample_rate=48000.0, seed=0)
        ir = AudioBuffer.noise(channels=1, frames=128, sample_rate=48000.0, seed=1)
        result = ops.convolve(buf, ir, trim=True)
        assert result.frames == buf.frames

    def test_output_length_full(self):
        buf = AudioBuffer.noise(channels=1, frames=1024, sample_rate=48000.0, seed=0)
        ir = AudioBuffer.noise(channels=1, frames=128, sample_rate=48000.0, seed=1)
        result = ops.convolve(buf, ir, trim=False)
        assert result.frames == buf.frames + ir.frames - 1

    def test_mono_ir_broadcast_to_stereo(self):
        buf = AudioBuffer.noise(channels=2, frames=512, sample_rate=48000.0, seed=0)
        ir = AudioBuffer.impulse(channels=1, frames=32, sample_rate=48000.0)
        result = ops.convolve(buf, ir)
        assert result.channels == 2

    def test_channel_mismatch_raises(self):
        buf = AudioBuffer.noise(channels=2, frames=512, sample_rate=48000.0, seed=0)
        ir = AudioBuffer.noise(channels=3, frames=32, sample_rate=48000.0, seed=1)
        with pytest.raises(ValueError, match="Channel mismatch"):
            ops.convolve(buf, ir)

    def test_sample_rate_mismatch_raises(self):
        buf = AudioBuffer.noise(channels=1, frames=512, sample_rate=48000.0, seed=0)
        ir = AudioBuffer.noise(channels=1, frames=32, sample_rate=44100.0, seed=1)
        with pytest.raises(ValueError, match="Sample rate mismatch"):
            ops.convolve(buf, ir)

    def test_normalize_flag(self):
        buf = AudioBuffer.sine(440.0, frames=1024, sample_rate=48000.0)
        ir_data = np.zeros((1, 64), dtype=np.float32)
        ir_data[0, 0] = 10.0
        ir = AudioBuffer(ir_data, sample_rate=48000.0)
        result_norm = ops.convolve(buf, ir, normalize=True)
        result_raw = ops.convolve(buf, ir, normalize=False)
        # Normalized should have less energy than raw (IR energy > 1)
        assert np.sum(result_norm.data**2) < np.sum(result_raw.data**2)

    def test_correctness_vs_np_convolve(self):
        rng = np.random.default_rng(42)
        sig = rng.standard_normal(256).astype(np.float32)
        kernel = rng.standard_normal(32).astype(np.float32)
        buf = AudioBuffer(sig, sample_rate=48000.0)
        ir = AudioBuffer(kernel, sample_rate=48000.0)
        result = ops.convolve(buf, ir, trim=False)
        expected = np.convolve(sig, kernel)
        np.testing.assert_allclose(result.data[0], expected, atol=1e-4)

    def test_metadata_preserved(self):
        buf = AudioBuffer.noise(
            channels=1, frames=512, sample_rate=44100.0, seed=0, label="test"
        )
        ir = AudioBuffer.impulse(channels=1, frames=32, sample_rate=44100.0)
        result = ops.convolve(buf, ir)
        assert result.sample_rate == 44100.0
        assert result.label == "test"


# ---------------------------------------------------------------------------
# Rates functions
# ---------------------------------------------------------------------------


class TestRatesFunctions:
    def test_upsample_shape(self):
        buf = AudioBuffer.noise(channels=1, frames=128, sample_rate=48000.0, seed=0)
        result = ops.upsample_2x(buf)
        assert result.frames == 256
        assert result.sample_rate == 96000.0

    def test_upsample_multichannel(self):
        buf = AudioBuffer.noise(channels=2, frames=128, sample_rate=48000.0, seed=0)
        result = ops.upsample_2x(buf)
        assert result.channels == 2
        assert result.frames == 256

    def test_oversample_roundtrip_shape(self):
        buf = AudioBuffer.noise(channels=1, frames=128, sample_rate=48000.0, seed=0)
        result = ops.oversample_roundtrip(buf)
        assert result.frames == 128
        assert result.sample_rate == 48000.0

    def test_oversample_roundtrip_multichannel(self):
        buf = AudioBuffer.noise(channels=2, frames=128, sample_rate=48000.0, seed=0)
        result = ops.oversample_roundtrip(buf)
        assert result.channels == 2
        assert result.frames == 128


# ---------------------------------------------------------------------------
# Mix functions
# ---------------------------------------------------------------------------


class TestMixFunctions:
    def test_hadamard_involution(self):
        data = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            dtype=np.float32,
        )
        buf = AudioBuffer(data, sample_rate=48000.0)
        once = ops.hadamard(buf)
        twice = ops.hadamard(once)
        np.testing.assert_allclose(twice.data, buf.data, atol=1e-4)

    def test_hadamard_energy_preservation(self):
        buf = AudioBuffer.noise(channels=4, frames=64, sample_rate=48000.0, seed=0)
        result = ops.hadamard(buf)
        # Energy should be preserved per-frame
        for i in range(buf.frames):
            in_e = np.sum(buf.data[:, i] ** 2)
            out_e = np.sum(result.data[:, i] ** 2)
            np.testing.assert_allclose(out_e, in_e, rtol=1e-4)

    def test_hadamard_non_power_of_2_raises(self):
        buf = AudioBuffer.noise(channels=3, frames=64, sample_rate=48000.0, seed=0)
        with pytest.raises(ValueError, match="power-of-2"):
            ops.hadamard(buf)

    def test_householder_involution(self):
        data = np.array(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            dtype=np.float32,
        )
        buf = AudioBuffer(data, sample_rate=48000.0)
        once = ops.householder(buf)
        twice = ops.householder(once)
        np.testing.assert_allclose(twice.data, buf.data, atol=1e-4)

    def test_householder_any_channel_count(self):
        buf = AudioBuffer.noise(channels=5, frames=32, sample_rate=48000.0, seed=0)
        result = ops.householder(buf)
        assert result.channels == 5
        assert result.frames == 32

    def test_crossfade_at_zero(self):
        a = AudioBuffer.ones(1, 64, sample_rate=48000.0)
        b = AudioBuffer.zeros(1, 64, sample_rate=48000.0)
        result = ops.crossfade(a, b, 0.0)
        # x=0 -> from=a, to=b; from_c ~1, to_c ~0
        np.testing.assert_allclose(result.data, a.data, atol=0.02)

    def test_crossfade_at_one(self):
        a = AudioBuffer.zeros(1, 64, sample_rate=48000.0)
        b = AudioBuffer.ones(1, 64, sample_rate=48000.0)
        result = ops.crossfade(a, b, 1.0)
        np.testing.assert_allclose(result.data, b.data, atol=0.02)

    def test_crossfade_midpoint(self):
        a = AudioBuffer.ones(1, 64, sample_rate=48000.0)
        b = AudioBuffer.ones(1, 64, sample_rate=48000.0) * 3.0
        result = ops.crossfade(a, b, 0.5)
        # At midpoint, both coefficients are roughly equal
        mid_val = result.data[0, 0]
        assert 1.0 < mid_val < 3.0

    def test_crossfade_sr_mismatch_raises(self):
        a = AudioBuffer.ones(1, 64, sample_rate=44100.0)
        b = AudioBuffer.ones(1, 64, sample_rate=48000.0)
        with pytest.raises(ValueError, match="Sample rate"):
            ops.crossfade(a, b, 0.5)

    def test_crossfade_channel_mismatch_raises(self):
        a = AudioBuffer.ones(1, 64, sample_rate=48000.0)
        b = AudioBuffer.ones(2, 64, sample_rate=48000.0)
        with pytest.raises(ValueError, match="Channel count"):
            ops.crossfade(a, b, 0.5)

    def test_crossfade_frame_mismatch_raises(self):
        a = AudioBuffer.ones(1, 64, sample_rate=48000.0)
        b = AudioBuffer.ones(1, 128, sample_rate=48000.0)
        with pytest.raises(ValueError, match="Frame count"):
            ops.crossfade(a, b, 0.5)


# ---------------------------------------------------------------------------
# LFO function
# ---------------------------------------------------------------------------


class TestLfoFunction:
    def test_lfo_shape(self):
        result = ops.lfo(1024, low=0.0, high=1.0, rate=0.001)
        assert isinstance(result, AudioBuffer)
        assert result.channels == 1
        assert result.frames == 1024

    def test_lfo_range(self):
        result = ops.lfo(4096, low=-1.0, high=1.0, rate=0.01, seed=42)
        # CubicLfo may slightly overshoot at transitions, allow tolerance
        assert np.all(result.data >= -1.1)
        assert np.all(result.data <= 1.1)

    def test_lfo_deterministic(self):
        a = ops.lfo(1024, low=0.0, high=1.0, rate=0.005, seed=123)
        b = ops.lfo(1024, low=0.0, high=1.0, rate=0.005, seed=123)
        np.testing.assert_array_equal(a.data, b.data)

    def test_lfo_sample_rate(self):
        result = ops.lfo(512, low=0.0, high=1.0, rate=0.01, sample_rate=44100.0)
        assert result.sample_rate == 44100.0


# ---------------------------------------------------------------------------
# Normalize peak
# ---------------------------------------------------------------------------


class TestNormalizePeak:
    def test_peak_matches_target(self):
        buf = AudioBuffer.noise(channels=1, frames=4096, sample_rate=48000.0, seed=0)
        result = ops.normalize_peak(buf, target_db=-6.0)
        expected = 10.0 ** (-6.0 / 20.0)
        actual_peak = np.max(np.abs(result.data))
        np.testing.assert_allclose(actual_peak, expected, rtol=1e-4)

    def test_0db_target(self):
        buf = AudioBuffer.noise(channels=1, frames=4096, sample_rate=48000.0, seed=0)
        result = ops.normalize_peak(buf, target_db=0.0)
        np.testing.assert_allclose(np.max(np.abs(result.data)), 1.0, rtol=1e-4)

    def test_silence_returns_silence(self):
        buf = AudioBuffer.zeros(1, 1024, sample_rate=48000.0)
        result = ops.normalize_peak(buf, target_db=0.0)
        assert np.max(np.abs(result.data)) == 0.0

    def test_shape_preserved(self):
        buf = AudioBuffer.noise(channels=2, frames=2048, sample_rate=48000.0, seed=0)
        result = ops.normalize_peak(buf, target_db=-3.0)
        assert result.channels == 2
        assert result.frames == 2048
        assert result.data.dtype == np.float32

    def test_metadata_preserved(self):
        buf = AudioBuffer.noise(
            channels=1, frames=1024, sample_rate=44100.0, seed=0, label="pk"
        )
        result = ops.normalize_peak(buf)
        assert result.sample_rate == 44100.0
        assert result.label == "pk"


# ---------------------------------------------------------------------------
# Trim silence
# ---------------------------------------------------------------------------


class TestTrimSilence:
    def test_trims_leading_silence(self):
        data = np.zeros((1, 1000), dtype=np.float32)
        data[0, 500:] = 0.5
        buf = AudioBuffer(data, sample_rate=48000.0)
        result = ops.trim_silence(buf, threshold_db=-60.0)
        assert result.frames == 500

    def test_trims_trailing_silence(self):
        data = np.zeros((1, 1000), dtype=np.float32)
        data[0, :200] = 0.5
        buf = AudioBuffer(data, sample_rate=48000.0)
        result = ops.trim_silence(buf, threshold_db=-60.0)
        assert result.frames == 200

    def test_trims_both_ends(self):
        data = np.zeros((1, 1000), dtype=np.float32)
        data[0, 300:700] = 0.5
        buf = AudioBuffer(data, sample_rate=48000.0)
        result = ops.trim_silence(buf, threshold_db=-60.0)
        assert result.frames == 400

    def test_pad_frames(self):
        data = np.zeros((1, 1000), dtype=np.float32)
        data[0, 300:700] = 0.5
        buf = AudioBuffer(data, sample_rate=48000.0)
        result = ops.trim_silence(buf, threshold_db=-60.0, pad_frames=10)
        assert result.frames == 420  # 400 + 2*10

    def test_all_silence_returns_empty(self):
        buf = AudioBuffer.zeros(1, 1000, sample_rate=48000.0)
        result = ops.trim_silence(buf)
        assert result.frames == 0

    def test_multichannel(self):
        data = np.zeros((2, 1000), dtype=np.float32)
        data[0, 200:800] = 0.1
        data[1, 300:700] = 0.1
        buf = AudioBuffer(data, sample_rate=48000.0)
        result = ops.trim_silence(buf, threshold_db=-60.0)
        # Should use earliest start / latest end across channels
        assert result.frames == 600  # 200 to 800


# ---------------------------------------------------------------------------
# Fade in/out
# ---------------------------------------------------------------------------


class TestFades:
    def test_fade_in_shape(self):
        buf = AudioBuffer.ones(1, 4800, sample_rate=48000.0)
        result = ops.fade_in(buf, duration_ms=10.0)
        assert result.frames == 4800
        assert result.data.dtype == np.float32

    def test_fade_in_first_sample_zero(self):
        buf = AudioBuffer.ones(1, 4800, sample_rate=48000.0)
        result = ops.fade_in(buf, duration_ms=10.0)
        assert result.data[0, 0] == pytest.approx(0.0, abs=1e-6)

    def test_fade_in_last_sample_unchanged(self):
        buf = AudioBuffer.ones(1, 4800, sample_rate=48000.0)
        result = ops.fade_in(buf, duration_ms=10.0)
        assert result.data[0, -1] == pytest.approx(1.0, abs=1e-6)

    def test_fade_out_last_sample_zero(self):
        buf = AudioBuffer.ones(1, 4800, sample_rate=48000.0)
        result = ops.fade_out(buf, duration_ms=10.0)
        assert result.data[0, -1] == pytest.approx(0.0, abs=1e-6)

    def test_fade_out_first_sample_unchanged(self):
        buf = AudioBuffer.ones(1, 4800, sample_rate=48000.0)
        result = ops.fade_out(buf, duration_ms=10.0)
        assert result.data[0, 0] == pytest.approx(1.0, abs=1e-6)

    def test_fade_curves(self):
        buf = AudioBuffer.ones(1, 4800, sample_rate=48000.0)
        for curve in ["linear", "ease_in", "ease_out", "smoothstep"]:
            result_in = ops.fade_in(buf, duration_ms=10.0, curve=curve)
            result_out = ops.fade_out(buf, duration_ms=10.0, curve=curve)
            assert result_in.data[0, 0] == pytest.approx(0.0, abs=1e-5), (
                f"fade_in {curve}"
            )
            assert result_out.data[0, -1] == pytest.approx(0.0, abs=1e-5), (
                f"fade_out {curve}"
            )

    def test_fade_invalid_curve_raises(self):
        buf = AudioBuffer.ones(1, 1024, sample_rate=48000.0)
        with pytest.raises(ValueError, match="Unknown fade curve"):
            ops.fade_in(buf, curve="nope")

    def test_fade_multichannel(self):
        buf = AudioBuffer.ones(2, 4800, sample_rate=48000.0)
        result = ops.fade_in(buf, duration_ms=10.0)
        assert result.channels == 2
        assert result.data[0, 0] == pytest.approx(0.0, abs=1e-6)
        assert result.data[1, 0] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Pan
# ---------------------------------------------------------------------------


class TestPan:
    def test_mono_to_stereo(self):
        buf = AudioBuffer.ones(1, 1024, sample_rate=48000.0)
        result = ops.pan(buf, position=0.0)
        assert result.channels == 2

    def test_center_equal_power(self):
        buf = AudioBuffer.ones(1, 1024, sample_rate=48000.0)
        result = ops.pan(buf, position=0.0)
        # At center, both channels should have equal amplitude
        np.testing.assert_allclose(result.data[0, 0], result.data[1, 0], rtol=1e-5)

    def test_hard_left(self):
        buf = AudioBuffer.ones(1, 1024, sample_rate=48000.0)
        result = ops.pan(buf, position=-1.0)
        assert result.data[0, 0] == pytest.approx(1.0, abs=1e-5)
        assert result.data[1, 0] == pytest.approx(0.0, abs=1e-5)

    def test_hard_right(self):
        buf = AudioBuffer.ones(1, 1024, sample_rate=48000.0)
        result = ops.pan(buf, position=1.0)
        assert result.data[0, 0] == pytest.approx(0.0, abs=1e-5)
        assert result.data[1, 0] == pytest.approx(1.0, abs=1e-5)

    def test_stereo_input(self):
        buf = AudioBuffer.ones(2, 1024, sample_rate=48000.0)
        result = ops.pan(buf, position=-1.0)
        assert result.channels == 2
        # Hard left: left gain=1, right gain=0
        assert result.data[0, 0] == pytest.approx(1.0, abs=1e-5)
        assert result.data[1, 0] == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Mix buffers
# ---------------------------------------------------------------------------


class TestMixBuffers:
    def test_basic_sum(self):
        a = AudioBuffer.ones(1, 1024, sample_rate=48000.0)
        b = AudioBuffer.ones(1, 1024, sample_rate=48000.0) * 2.0
        result = ops.mix_buffers(a, b)
        np.testing.assert_allclose(result.data[0, 0], 3.0, atol=1e-5)

    def test_with_gains(self):
        a = AudioBuffer.ones(1, 1024, sample_rate=48000.0)
        b = AudioBuffer.ones(1, 1024, sample_rate=48000.0)
        result = ops.mix_buffers(a, b, gains=[0.5, 0.5])
        np.testing.assert_allclose(result.data[0, 0], 1.0, atol=1e-5)

    def test_different_lengths_zero_padded(self):
        a = AudioBuffer.ones(1, 512, sample_rate=48000.0)
        b = AudioBuffer.ones(1, 1024, sample_rate=48000.0)
        result = ops.mix_buffers(a, b)
        assert result.frames == 1024
        # First 512 samples: sum of both
        assert result.data[0, 0] == pytest.approx(2.0, abs=1e-5)
        # After 512: only b contributes
        assert result.data[0, 600] == pytest.approx(1.0, abs=1e-5)

    def test_sr_mismatch_raises(self):
        a = AudioBuffer.ones(1, 512, sample_rate=48000.0)
        b = AudioBuffer.ones(1, 512, sample_rate=44100.0)
        with pytest.raises(ValueError, match="Sample rate mismatch"):
            ops.mix_buffers(a, b)

    def test_gains_length_mismatch_raises(self):
        a = AudioBuffer.ones(1, 512, sample_rate=48000.0)
        with pytest.raises(ValueError, match="gains length"):
            ops.mix_buffers(a, gains=[1.0, 2.0])

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="At least one"):
            ops.mix_buffers()


# ---------------------------------------------------------------------------
# Mid-side processing
# ---------------------------------------------------------------------------


class TestMidSide:
    def test_encode_decode_roundtrip(self):
        buf = AudioBuffer.noise(channels=2, frames=1024, sample_rate=48000.0, seed=0)
        encoded = ops.mid_side_encode(buf)
        decoded = ops.mid_side_decode(encoded)
        np.testing.assert_allclose(decoded.data, buf.data, atol=1e-5)

    def test_encode_mono_raises(self):
        buf = AudioBuffer.noise(channels=1, frames=1024, sample_rate=48000.0, seed=0)
        with pytest.raises(ValueError, match="stereo"):
            ops.mid_side_encode(buf)

    def test_decode_mono_raises(self):
        buf = AudioBuffer.noise(channels=1, frames=1024, sample_rate=48000.0, seed=0)
        with pytest.raises(ValueError, match="2-channel"):
            ops.mid_side_decode(buf)

    def test_mono_signal_has_zero_side(self):
        """Identical L/R should produce zero side channel."""
        mono = AudioBuffer.sine(440.0, channels=1, frames=1024, sample_rate=48000.0)
        stereo = AudioBuffer(np.tile(mono.data, (2, 1)), sample_rate=48000.0)
        encoded = ops.mid_side_encode(stereo)
        np.testing.assert_allclose(encoded.data[1], 0.0, atol=1e-6)

    def test_stereo_widen_identity(self):
        buf = AudioBuffer.noise(channels=2, frames=1024, sample_rate=48000.0, seed=0)
        result = ops.stereo_widen(buf, width=1.0)
        np.testing.assert_allclose(result.data, buf.data, atol=1e-5)

    def test_stereo_widen_mono(self):
        """Width 0.0 should produce mono (L == R)."""
        buf = AudioBuffer.noise(channels=2, frames=1024, sample_rate=48000.0, seed=0)
        result = ops.stereo_widen(buf, width=0.0)
        np.testing.assert_allclose(result.data[0], result.data[1], atol=1e-5)

    def test_stereo_widen_wider(self):
        """Width > 1.0 should increase side energy."""
        buf = AudioBuffer.noise(channels=2, frames=1024, sample_rate=48000.0, seed=0)
        narrow = ops.mid_side_encode(buf)
        side_energy_orig = np.sum(narrow.data[1] ** 2)
        result = ops.stereo_widen(buf, width=2.0)
        wide = ops.mid_side_encode(result)
        side_energy_wide = np.sum(wide.data[1] ** 2)
        assert side_energy_wide > side_energy_orig * 3.5

    def test_stereo_widen_non_stereo_raises(self):
        buf = AudioBuffer.noise(channels=1, frames=1024, sample_rate=48000.0, seed=0)
        with pytest.raises(ValueError, match="stereo"):
            ops.stereo_widen(buf)


# ---------------------------------------------------------------------------
# Cross-correlation
# ---------------------------------------------------------------------------


class TestXcorr:
    def test_autocorrelation_peak_at_zero(self):
        buf = AudioBuffer.sine(440.0, frames=1024, sample_rate=48000.0)
        corr = ops.xcorr(buf)
        # Peak of autocorrelation should be at lag 0
        assert np.argmax(corr) == 0

    def test_autocorrelation_symmetric(self):
        buf = AudioBuffer.noise(channels=1, frames=512, sample_rate=48000.0, seed=42)
        corr = ops.xcorr(buf)
        n = buf.frames
        # Autocorrelation output length is 2*N - 1
        assert len(corr) == 2 * n - 1

    def test_cross_correlation_known_delay(self):
        """Cross-correlate a delayed signal with original; peak at the delay lag."""
        n = 256
        delay = 20
        # buf_a has impulse at `delay`, buf_b at 0
        data_a = np.zeros((1, n), dtype=np.float32)
        data_a[0, delay] = 1.0
        data_b = np.zeros((1, n), dtype=np.float32)
        data_b[0, 0] = 1.0
        buf_a = AudioBuffer(data_a, sample_rate=48000.0)
        buf_b = AudioBuffer(data_b, sample_rate=48000.0)
        corr = ops.xcorr(buf_a, buf_b)
        assert np.argmax(corr) == delay

    def test_output_dtype(self):
        buf = AudioBuffer.noise(channels=1, frames=128, sample_rate=48000.0, seed=0)
        corr = ops.xcorr(buf)
        assert corr.dtype == np.float32

    def test_multichannel_mixed_to_mono(self):
        buf = AudioBuffer.noise(channels=2, frames=256, sample_rate=48000.0, seed=0)
        corr = ops.xcorr(buf)
        assert corr.ndim == 1


# ---------------------------------------------------------------------------
# Hilbert / Envelope
# ---------------------------------------------------------------------------


class TestHilbert:
    def test_envelope_of_sine_is_constant(self):
        """Envelope of a pure sine should be approximately constant (~1.0)."""
        buf = AudioBuffer.sine(440.0, frames=4096, sample_rate=48000.0)
        env = ops.hilbert(buf)
        # Skip edges (transient)
        mid = env.data[0, 512:-512]
        np.testing.assert_allclose(mid, 1.0, atol=0.05)

    def test_envelope_shape(self):
        buf = AudioBuffer.noise(channels=2, frames=1024, sample_rate=48000.0, seed=0)
        env = ops.hilbert(buf)
        assert env.channels == 2
        assert env.frames == 1024

    def test_envelope_non_negative(self):
        buf = AudioBuffer.noise(channels=1, frames=2048, sample_rate=48000.0, seed=0)
        env = ops.hilbert(buf)
        assert np.all(env.data >= 0)

    def test_envelope_alias(self):
        buf = AudioBuffer.sine(440.0, frames=1024, sample_rate=48000.0)
        env1 = ops.hilbert(buf)
        env2 = ops.envelope(buf)
        np.testing.assert_array_equal(env1.data, env2.data)

    def test_metadata_preserved(self):
        buf = AudioBuffer.noise(
            channels=1, frames=1024, sample_rate=44100.0, seed=0, label="hil"
        )
        env = ops.hilbert(buf)
        assert env.sample_rate == 44100.0
        assert env.label == "hil"


# ---------------------------------------------------------------------------
# Median filter
# ---------------------------------------------------------------------------


class TestMedianFilter:
    def test_removes_impulse_noise(self):
        """Median filter should remove isolated spike in constant signal."""
        data = np.ones((1, 128), dtype=np.float32)
        data[0, 64] = 100.0  # spike
        buf = AudioBuffer(data, sample_rate=48000.0)
        result = ops.median_filter(buf, kernel_size=3)
        # Spike should be removed
        assert result.data[0, 64] == pytest.approx(1.0)

    def test_preserves_constant_signal(self):
        data = np.full((1, 256), 5.0, dtype=np.float32)
        buf = AudioBuffer(data, sample_rate=48000.0)
        result = ops.median_filter(buf, kernel_size=5)
        np.testing.assert_allclose(result.data, buf.data, atol=1e-6)

    def test_multichannel(self):
        buf = AudioBuffer.noise(channels=2, frames=256, sample_rate=48000.0, seed=0)
        result = ops.median_filter(buf, kernel_size=5)
        assert result.channels == 2
        assert result.frames == 256

    def test_kernel_size_1_passthrough(self):
        buf = AudioBuffer.noise(channels=1, frames=128, sample_rate=48000.0, seed=0)
        result = ops.median_filter(buf, kernel_size=1)
        np.testing.assert_allclose(result.data, buf.data, atol=1e-6)

    def test_even_kernel_raises(self):
        buf = AudioBuffer.noise(channels=1, frames=128, sample_rate=48000.0, seed=0)
        with pytest.raises(ValueError, match="kernel_size"):
            ops.median_filter(buf, kernel_size=4)

    def test_output_dtype(self):
        buf = AudioBuffer.noise(channels=1, frames=128, sample_rate=48000.0, seed=0)
        result = ops.median_filter(buf, kernel_size=3)
        assert result.data.dtype == np.float32


# ---------------------------------------------------------------------------
# LMS adaptive filter
# ---------------------------------------------------------------------------


class TestLmsFilter:
    def test_cancels_correlated_noise(self):
        """LMS should reduce correlated noise component."""
        rng = np.random.default_rng(42)
        n = 2048
        # Clean signal
        t = np.arange(n, dtype=np.float32) / 48000.0
        clean = np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)
        # Noise reference
        noise_ref = rng.standard_normal(n).astype(np.float32) * 0.3
        # Desired = clean + noise (correlated with ref)
        desired = clean + noise_ref
        buf = AudioBuffer(desired.reshape(1, -1), sample_rate=48000.0)
        ref = AudioBuffer(noise_ref.reshape(1, -1), sample_rate=48000.0)
        output, error = ops.lms_filter(buf, ref, filter_len=32, step_size=0.05)
        # After adaptation, error should resemble clean signal
        # Check that error energy is less than desired energy (noise reduced)
        desired_energy = np.sum(desired**2)
        error_energy = np.sum(error.data**2)
        assert error_energy < desired_energy

    def test_output_shapes(self):
        n = 512
        buf = AudioBuffer.noise(channels=1, frames=n, sample_rate=48000.0, seed=0)
        ref = AudioBuffer.noise(channels=1, frames=n, sample_rate=48000.0, seed=1)
        output, error = ops.lms_filter(buf, ref, filter_len=16)
        assert output.frames == n
        assert error.frames == n
        assert output.channels == 1
        assert error.channels == 1

    def test_sample_rate_mismatch_raises(self):
        buf = AudioBuffer.noise(channels=1, frames=256, sample_rate=48000.0, seed=0)
        ref = AudioBuffer.noise(channels=1, frames=256, sample_rate=44100.0, seed=1)
        with pytest.raises(ValueError, match="Sample rate"):
            ops.lms_filter(buf, ref)

    def test_frame_count_mismatch_raises(self):
        buf = AudioBuffer.noise(channels=1, frames=256, sample_rate=48000.0, seed=0)
        ref = AudioBuffer.noise(channels=1, frames=128, sample_rate=48000.0, seed=1)
        with pytest.raises(ValueError, match="Frame count"):
            ops.lms_filter(buf, ref)

    def test_multichannel(self):
        n = 512
        buf = AudioBuffer.noise(channels=2, frames=n, sample_rate=48000.0, seed=0)
        ref = AudioBuffer.noise(channels=2, frames=n, sample_rate=48000.0, seed=1)
        output, error = ops.lms_filter(buf, ref, filter_len=16)
        assert output.channels == 2
        assert error.channels == 2

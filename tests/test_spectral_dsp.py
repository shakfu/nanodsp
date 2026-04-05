"""Tests for nanodsp.spectral module (STFT, spectral transforms, EQ matching)."""

import numpy as np
import pytest

from nanodsp import spectral as sp
from nanodsp.effects.filters import lowpass, highpass
from nanodsp._core import fft
from nanodsp.buffer import AudioBuffer


# ---------------------------------------------------------------------------
# STFT functions
# ---------------------------------------------------------------------------


class TestSTFTFunctions:
    def test_stft_shape(self):
        buf = AudioBuffer.noise(channels=1, frames=4096, sample_rate=48000.0, seed=0)
        spec = sp.stft(buf, window_size=1024)
        fft_size = fft.RealFFT.fast_size_above(1024)
        expected_frames = (4096 - 1024) // 256 + 1
        assert spec.data.shape == (1, expected_frames, fft_size // 2)
        assert spec.data.dtype == np.complex64

    def test_stft_multichannel(self):
        buf = AudioBuffer.noise(channels=2, frames=4096, sample_rate=48000.0, seed=0)
        spec = sp.stft(buf, window_size=1024)
        assert spec.channels == 2

    def test_stft_custom_params(self):
        buf = AudioBuffer.noise(channels=1, frames=4096, sample_rate=48000.0, seed=0)
        spec = sp.stft(buf, window_size=512, hop_size=128)
        expected_frames = (4096 - 512) // 128 + 1
        assert spec.num_frames == expected_frames
        assert spec.hop_size == 128
        assert spec.window_size == 512

    def test_istft_shape(self):
        buf = AudioBuffer.noise(channels=1, frames=4096, sample_rate=48000.0, seed=0)
        spec = sp.stft(buf, window_size=1024)
        result = sp.istft(spec)
        assert isinstance(result, AudioBuffer)
        assert result.channels == 1
        assert result.frames == 4096

    def test_roundtrip(self):
        buf = AudioBuffer.noise(channels=1, frames=4096, sample_rate=48000.0, seed=42)
        spec = sp.stft(buf, window_size=1024)
        result = sp.istft(spec)
        assert result.channels == buf.channels
        assert result.frames == buf.frames
        assert result.data.dtype == np.float32
        assert np.all(np.isfinite(result.data))
        # Interior samples (away from edges) should match well
        margin = 1024
        np.testing.assert_allclose(
            result.data[0, margin:-margin],
            buf.data[0, margin:-margin],
            atol=1e-4,
        )

    def test_roundtrip_multichannel(self):
        buf = AudioBuffer.noise(channels=2, frames=4096, sample_rate=48000.0, seed=7)
        spec = sp.stft(buf, window_size=1024)
        result = sp.istft(spec)
        assert result.channels == buf.channels
        assert result.frames == buf.frames
        assert result.data.dtype == np.float32
        assert np.all(np.isfinite(result.data))
        margin = 1024
        for ch in range(2):
            np.testing.assert_allclose(
                result.data[ch, margin:-margin],
                buf.data[ch, margin:-margin],
                atol=1e-4,
            )

    def test_spectrogram_properties(self):
        buf = AudioBuffer.noise(channels=2, frames=4096, sample_rate=44100.0, seed=0)
        spec = sp.stft(buf, window_size=2048)
        assert spec.channels == 2
        assert spec.bins == spec.fft_size // 2
        assert spec.sample_rate == 44100.0
        assert spec.original_frames == 4096


# ---------------------------------------------------------------------------
# Spectral utility functions
# ---------------------------------------------------------------------------


class TestSpectralUtilities:
    @pytest.fixture()
    def spec(self):
        """A mono spectrogram from noise for reuse across tests."""
        buf = AudioBuffer.noise(channels=1, frames=8192, sample_rate=48000.0, seed=0)
        return sp.stft(buf, window_size=1024)

    @pytest.fixture()
    def spec_stereo(self):
        """A stereo spectrogram."""
        buf = AudioBuffer.noise(channels=2, frames=8192, sample_rate=48000.0, seed=1)
        return sp.stft(buf, window_size=1024)

    # -- Decomposition --

    def test_magnitude_shape_dtype(self, spec):
        mag = sp.magnitude(spec)
        assert mag.shape == spec.data.shape
        assert mag.dtype == np.float32

    def test_phase_shape_dtype(self, spec):
        ph = sp.phase(spec)
        assert ph.shape == spec.data.shape
        assert ph.dtype == np.float32

    def test_decomposition_roundtrip(self, spec):
        mag = sp.magnitude(spec)
        ph = sp.phase(spec)
        reconstructed = sp.from_polar(mag, ph, spec)
        assert reconstructed.data.shape == spec.data.shape
        assert reconstructed.data.dtype == np.complex64
        np.testing.assert_allclose(
            np.abs(reconstructed.data), np.abs(spec.data), atol=1e-5
        )
        np.testing.assert_allclose(
            np.angle(reconstructed.data), np.angle(spec.data), atol=1e-5
        )

    def test_from_polar_metadata(self, spec):
        mag = sp.magnitude(spec)
        ph = sp.phase(spec)
        result = sp.from_polar(mag, ph, spec)
        assert result.window_size == spec.window_size
        assert result.hop_size == spec.hop_size
        assert result.fft_size == spec.fft_size
        assert result.sample_rate == spec.sample_rate
        assert result.original_frames == spec.original_frames

    # -- Filtering --

    def test_apply_mask_identity(self, spec):
        mask = np.ones(spec.data.shape, dtype=np.float32)
        result = sp.apply_mask(spec, mask)
        assert result.data.shape == spec.data.shape
        assert result.data.dtype == np.complex64
        np.testing.assert_allclose(result.data, spec.data, atol=1e-7)

    def test_apply_mask_zeros_bins(self, spec):
        mask = np.ones(spec.data.shape, dtype=np.float32)
        mask[:, :, 0] = 0.0  # zero DC bin
        result = sp.apply_mask(spec, mask)
        assert np.all(result.data[:, :, 0] == 0.0)
        # Other bins unaffected
        np.testing.assert_allclose(
            result.data[:, :, 1:], spec.data[:, :, 1:], atol=1e-7
        )

    def test_apply_mask_broadcast_1d(self, spec):
        """1D mask [bins] broadcasts across channels and frames."""
        mask = np.ones(spec.bins, dtype=np.float32)
        mask[0] = 0.0
        result = sp.apply_mask(spec, mask)
        assert np.all(result.data[:, :, 0] == 0.0)

    def test_apply_mask_broadcast_2d(self, spec):
        """2D mask [frames, bins] broadcasts across channels."""
        mask = np.ones((spec.num_frames, spec.bins), dtype=np.float32)
        mask[0, :] = 0.0  # zero first frame
        result = sp.apply_mask(spec, mask)
        assert np.all(result.data[:, 0, :] == 0.0)

    def test_apply_mask_bad_shape(self, spec):
        bad_mask = np.ones((spec.bins + 1,), dtype=np.float32)
        with pytest.raises(ValueError, match="broadcastable"):
            sp.apply_mask(spec, bad_mask)

    # -- Spectral gate --

    def test_spectral_gate_preserves_loud(self, spec):
        """Loud bins should pass through mostly unchanged."""
        result = sp.spectral_gate(spec, threshold_db=-100.0)
        assert result.data.shape == spec.data.shape
        np.testing.assert_allclose(np.abs(result.data), np.abs(spec.data), atol=1e-6)

    def test_spectral_gate_attenuates_quiet(self):
        """Quiet signal should be attenuated."""
        buf = AudioBuffer(
            np.full((1, 8192), 1e-6, dtype=np.float32), sample_rate=48000.0
        )
        spec = sp.stft(buf, window_size=1024)
        result = sp.spectral_gate(spec, threshold_db=-20.0, noise_floor_db=-80.0)
        assert np.mean(np.abs(result.data)) < np.mean(np.abs(spec.data))

    # -- Spectral emphasis --

    def test_spectral_emphasis_flat(self, spec):
        result = sp.spectral_emphasis(spec, low_db=0.0, high_db=0.0)
        assert result.data.shape == spec.data.shape
        assert result.data.dtype == np.complex64
        np.testing.assert_allclose(result.data, spec.data, atol=1e-6)

    def test_spectral_emphasis_tilt(self, spec):
        result = sp.spectral_emphasis(spec, low_db=-6.0, high_db=6.0)
        # Low bins should be attenuated, high bins boosted
        orig_low = np.mean(np.abs(spec.data[:, :, :10]))
        orig_high = np.mean(np.abs(spec.data[:, :, -10:]))
        new_low = np.mean(np.abs(result.data[:, :, :10]))
        new_high = np.mean(np.abs(result.data[:, :, -10:]))
        assert new_low < orig_low  # attenuated
        assert new_high > orig_high  # boosted

    # -- bin_freq / freq_to_bin --

    def test_bin_freq_correctness(self, spec):
        # Bin 0 = DC
        assert sp.bin_freq(spec, 0) == 0.0
        # Bin 1 = sample_rate / fft_size
        expected = spec.sample_rate / spec.fft_size
        assert sp.bin_freq(spec, 1) == pytest.approx(expected)

    def test_freq_to_bin_roundtrip(self, spec):
        freq = 1000.0
        b = sp.freq_to_bin(spec, freq)
        recovered = sp.bin_freq(spec, b)
        # Should be within one bin width
        bin_width = spec.sample_rate / spec.fft_size
        assert abs(recovered - freq) <= bin_width

    def test_freq_to_bin_negative_raises(self, spec):
        with pytest.raises(ValueError, match="non-negative"):
            sp.freq_to_bin(spec, -100.0)

    def test_freq_to_bin_nyquist_raises(self, spec):
        with pytest.raises(ValueError, match="Nyquist"):
            sp.freq_to_bin(spec, spec.sample_rate / 2.0)

    # -- Time stretch --

    def test_time_stretch_identity(self, spec):
        result = sp.time_stretch(spec, 1.0)
        assert result.num_frames == spec.num_frames
        assert result.original_frames == spec.original_frames

    def test_time_stretch_slow(self, spec):
        result = sp.time_stretch(spec, 0.5)
        # Should roughly double num_frames
        assert result.num_frames == pytest.approx(spec.num_frames * 2, abs=1)
        assert result.original_frames == pytest.approx(spec.original_frames * 2, abs=1)

    def test_time_stretch_fast(self, spec):
        result = sp.time_stretch(spec, 2.0)
        # Should roughly halve num_frames
        assert result.num_frames == pytest.approx(spec.num_frames / 2, abs=1)

    def test_time_stretch_roundtrip(self):
        """stft -> stretch(0.5) -> istft produces longer audio."""
        buf = AudioBuffer.sine(440.0, frames=8192, sample_rate=48000.0)
        spec = sp.stft(buf, window_size=1024)
        stretched = sp.time_stretch(spec, 0.5)
        result = sp.istft(stretched)
        assert result.frames > buf.frames

    def test_time_stretch_invalid_rate(self, spec):
        with pytest.raises(ValueError, match="Rate must be > 0"):
            sp.time_stretch(spec, 0.0)
        with pytest.raises(ValueError, match="Rate must be > 0"):
            sp.time_stretch(spec, -1.0)

    def test_time_stretch_extreme_slow(self, spec):
        """Very slow rate (0.05) should produce a much longer spectrogram."""
        result = sp.time_stretch(spec, 0.05)
        assert result.data.shape[1] > spec.data.shape[1] * 10
        assert np.all(np.isfinite(result.data))

    def test_time_stretch_extreme_fast(self, spec):
        """Very fast rate (10.0) should produce a much shorter spectrogram."""
        result = sp.time_stretch(spec, 10.0)
        assert result.data.shape[1] < spec.data.shape[1]
        assert np.all(np.isfinite(result.data))

    # -- Phase lock --

    def test_phase_lock_preserves_magnitude(self, spec):
        result = sp.phase_lock(spec)
        assert result.data.shape == spec.data.shape
        assert result.data.dtype == np.complex64
        np.testing.assert_allclose(np.abs(result.data), np.abs(spec.data), atol=1e-5)

    def test_phase_lock_shape(self, spec):
        result = sp.phase_lock(spec)
        assert result.data.shape == spec.data.shape
        assert result.data.dtype == np.complex64

    # -- Spectral freeze --

    def test_spectral_freeze_shape(self, spec):
        result = sp.spectral_freeze(spec, frame_index=0)
        assert result.data.shape == spec.data.shape
        assert result.data.dtype == np.complex64

    def test_spectral_freeze_all_frames_identical(self, spec):
        result = sp.spectral_freeze(spec, frame_index=3)
        assert result.data.shape == spec.data.shape
        assert result.data.dtype == np.complex64
        for t in range(result.num_frames):
            np.testing.assert_array_equal(result.data[:, t, :], result.data[:, 0, :])

    def test_spectral_freeze_matches_source_frame(self, spec):
        idx = 5
        result = sp.spectral_freeze(spec, frame_index=idx)
        assert result.data.dtype == np.complex64
        assert result.channels == spec.channels
        np.testing.assert_array_equal(result.data[:, 0, :], spec.data[:, idx, :])

    def test_spectral_freeze_negative_index(self, spec):
        result = sp.spectral_freeze(spec, frame_index=-1)
        assert result.data.shape == spec.data.shape
        assert result.data.dtype == np.complex64
        np.testing.assert_array_equal(result.data[:, 0, :], spec.data[:, -1, :])

    def test_spectral_freeze_custom_num_frames(self, spec):
        result = sp.spectral_freeze(spec, frame_index=0, num_frames=10)
        assert result.num_frames == 10

    def test_spectral_freeze_out_of_range_raises(self, spec):
        with pytest.raises(IndexError, match="out of range"):
            sp.spectral_freeze(spec, frame_index=spec.num_frames)
        with pytest.raises(IndexError, match="out of range"):
            sp.spectral_freeze(spec, frame_index=-spec.num_frames - 1)

    def test_spectral_freeze_roundtrip(self, spec):
        """Freeze -> istft should produce audio of the expected length."""
        frozen = sp.spectral_freeze(spec, frame_index=0, num_frames=20)
        audio = sp.istft(frozen)
        expected_len = (20 - 1) * spec.hop_size + spec.window_size
        assert audio.frames == expected_len

    # -- Spectral morph --

    def test_spectral_morph_mix_zero(self, spec):
        result = sp.spectral_morph(spec, spec, mix=0.0)
        assert result.data.shape == spec.data.shape
        assert result.data.dtype == np.complex64
        np.testing.assert_allclose(np.abs(result.data), np.abs(spec.data), atol=1e-5)

    def test_spectral_morph_mix_one(self, spec, spec_stereo):
        """mix=1.0 should return spec_b's magnitudes."""
        # Use two different mono specs
        buf2 = AudioBuffer.noise(channels=1, frames=8192, sample_rate=48000.0, seed=99)
        spec_b = sp.stft(buf2, window_size=1024)
        result = sp.spectral_morph(spec, spec_b, mix=1.0)
        assert result.data.dtype == np.complex64
        assert result.channels == spec.channels
        np.testing.assert_allclose(
            np.abs(result.data),
            np.abs(spec_b.data[:, : result.num_frames, :]),
            atol=1e-5,
        )

    def test_spectral_morph_midpoint(self, spec):
        buf2 = AudioBuffer.noise(channels=1, frames=8192, sample_rate=48000.0, seed=77)
        spec_b = sp.stft(buf2, window_size=1024)
        result = sp.spectral_morph(spec, spec_b, mix=0.5)
        assert result.data.dtype == np.complex64
        assert result.channels == spec.channels
        mag_a = np.abs(spec.data[:, : result.num_frames, :])
        mag_b = np.abs(spec_b.data[:, : result.num_frames, :])
        expected_mag = 0.5 * mag_a + 0.5 * mag_b
        np.testing.assert_allclose(np.abs(result.data), expected_mag, atol=1e-5)

    def test_spectral_morph_different_lengths(self):
        """Shorter spectrogram length should be used."""
        short = AudioBuffer.noise(channels=1, frames=4096, sample_rate=48000.0, seed=0)
        long = AudioBuffer.noise(channels=1, frames=16384, sample_rate=48000.0, seed=1)
        spec_short = sp.stft(short, window_size=1024)
        spec_long = sp.stft(long, window_size=1024)
        result = sp.spectral_morph(spec_short, spec_long, mix=0.5)
        assert result.num_frames == spec_short.num_frames

    def test_spectral_morph_fft_size_mismatch_raises(self):
        buf = AudioBuffer.noise(channels=1, frames=8192, sample_rate=48000.0, seed=0)
        spec_a = sp.stft(buf, window_size=1024)
        spec_b = sp.stft(buf, window_size=512)
        with pytest.raises(ValueError, match="fft_size mismatch"):
            sp.spectral_morph(spec_a, spec_b, mix=0.5)

    def test_spectral_morph_channel_mismatch_raises(self):
        mono = AudioBuffer.noise(channels=1, frames=8192, sample_rate=48000.0, seed=0)
        stereo = AudioBuffer.noise(channels=2, frames=8192, sample_rate=48000.0, seed=0)
        spec_m = sp.stft(mono, window_size=1024)
        spec_s = sp.stft(stereo, window_size=1024)
        with pytest.raises(ValueError, match="channel count mismatch"):
            sp.spectral_morph(spec_m, spec_s, mix=0.5)

    def test_spectral_morph_time_varying_mix(self, spec):
        """Per-frame mix array should work."""
        buf2 = AudioBuffer.noise(channels=1, frames=8192, sample_rate=48000.0, seed=55)
        spec_b = sp.stft(buf2, window_size=1024)
        n_frames = min(spec.num_frames, spec_b.num_frames)
        # Ramp from 0 to 1 across frames: [1, T, 1]
        mix_arr = np.linspace(0, 1, n_frames, dtype=np.float32)[None, :, None]
        result = sp.spectral_morph(spec, spec_b, mix=mix_arr)
        assert result.num_frames == n_frames

    # -- Pitch shift spectral --

    def test_pitch_shift_spectral_identity(self):
        buf = AudioBuffer.sine(440.0, frames=8192, sample_rate=48000.0)
        result = sp.pitch_shift_spectral(buf, semitones=0.0)
        assert result.frames == buf.frames
        assert result.sample_rate == buf.sample_rate
        np.testing.assert_array_equal(result.data, buf.data)

    def test_pitch_shift_spectral_preserves_duration(self):
        buf = AudioBuffer.sine(440.0, frames=8192, sample_rate=48000.0)
        for semi in [3.0, -3.0, 7.0, 12.0]:
            result = sp.pitch_shift_spectral(buf, semitones=semi)
            assert result.frames == buf.frames
            assert result.sample_rate == buf.sample_rate

    def test_pitch_shift_spectral_up_raises_frequency(self):
        """Shifting up should increase dominant frequency."""
        buf = AudioBuffer.sine(440.0, frames=16384, sample_rate=48000.0)
        result = sp.pitch_shift_spectral(buf, semitones=12.0, window_size=2048)
        # Compare spectral centroids as a proxy for pitch
        spec_in = sp.stft(buf, window_size=2048)
        spec_out = sp.stft(result, window_size=2048)
        mag_in = np.mean(np.abs(spec_in.data[0]), axis=0)
        mag_out = np.mean(np.abs(spec_out.data[0]), axis=0)
        bins = np.arange(len(mag_in), dtype=np.float32)
        centroid_in = np.sum(bins * mag_in) / (np.sum(mag_in) + 1e-10)
        centroid_out = np.sum(bins * mag_out) / (np.sum(mag_out) + 1e-10)
        assert centroid_out > centroid_in * 1.3

    def test_pitch_shift_spectral_down_lowers_frequency(self):
        """Shifting down should decrease dominant frequency."""
        buf = AudioBuffer.sine(2000.0, frames=16384, sample_rate=48000.0)
        result = sp.pitch_shift_spectral(buf, semitones=-12.0, window_size=2048)
        spec_in = sp.stft(buf, window_size=2048)
        spec_out = sp.stft(result, window_size=2048)
        mag_in = np.mean(np.abs(spec_in.data[0]), axis=0)
        mag_out = np.mean(np.abs(spec_out.data[0]), axis=0)
        bins = np.arange(len(mag_in), dtype=np.float32)
        centroid_in = np.sum(bins * mag_in) / (np.sum(mag_in) + 1e-10)
        centroid_out = np.sum(bins * mag_out) / (np.sum(mag_out) + 1e-10)
        assert centroid_out < centroid_in * 0.7

    def test_pitch_shift_spectral_multichannel(self):
        buf = AudioBuffer.noise(channels=2, frames=8192, sample_rate=48000.0, seed=0)
        result = sp.pitch_shift_spectral(buf, semitones=5.0)
        assert result.channels == 2
        assert result.frames == buf.frames

    def test_pitch_shift_spectral_metadata(self):
        buf = AudioBuffer.sine(
            440.0,
            channels=2,
            frames=8192,
            sample_rate=44100.0,
            label="test",
        )
        result = sp.pitch_shift_spectral(buf, semitones=3.0)
        assert result.sample_rate == 44100.0
        assert result.channel_layout == "stereo"
        assert result.label == "test"

    # -- Spectral denoise --

    def test_spectral_denoise_shape(self, spec):
        result = sp.spectral_denoise(spec, noise_frames=5)
        assert result.data.shape == spec.data.shape
        assert result.data.dtype == np.complex64

    def test_spectral_denoise_reduces_noise(self):
        """Signal buried in noise should have noise reduced."""
        rng = np.random.RandomState(42)
        noise = rng.randn(1, 16384).astype(np.float32) * 0.01
        # First 4096 samples: pure noise. Rest: noise + signal.
        signal = np.zeros_like(noise)
        t = np.arange(16384, dtype=np.float32) / 48000.0
        signal[0, 4096:] = np.sin(2 * np.pi * 440 * t[4096:]).astype(np.float32) * 0.5
        combined = AudioBuffer(noise + signal, sample_rate=48000.0)
        spec = sp.stft(combined, window_size=1024)
        # ~16 noise-only STFT frames at the start
        noise_f = (4096 - 1024) // 256 + 1
        result = sp.spectral_denoise(spec, noise_frames=noise_f, reduction_db=-40.0)
        # Noise energy should decrease
        noise_region = spec.data[:, :noise_f, :]
        denoised_region = result.data[:, :noise_f, :]
        assert np.sum(np.abs(denoised_region) ** 2) < np.sum(np.abs(noise_region) ** 2)

    def test_spectral_denoise_preserves_loud_signal(self, spec):
        """Bins well above noise floor should pass through."""
        result = sp.spectral_denoise(spec, noise_frames=3, reduction_db=-60.0)
        # Most of the energy is above the noise floor for broadband noise
        energy_ratio = np.sum(np.abs(result.data) ** 2) / np.sum(np.abs(spec.data) ** 2)
        assert energy_ratio > 0.5

    def test_spectral_denoise_smoothing(self, spec):
        """Smoothing should still produce valid output."""
        result = sp.spectral_denoise(spec, noise_frames=5, smoothing=5)
        assert result.data.shape == spec.data.shape
        assert result.data.dtype == np.complex64
        assert np.all(np.isfinite(result.data))

    def test_spectral_denoise_invalid_noise_frames(self, spec):
        with pytest.raises(ValueError, match="noise_frames must be >= 1"):
            sp.spectral_denoise(spec, noise_frames=0)
        with pytest.raises(ValueError, match="exceeds available frames"):
            sp.spectral_denoise(spec, noise_frames=spec.num_frames + 1)


# ---------------------------------------------------------------------------
# EQ matching
# ---------------------------------------------------------------------------


class TestEqMatch:
    def test_identity_match(self):
        """Matching a signal to itself should return roughly the same signal."""
        buf = AudioBuffer.noise(channels=1, frames=8192, sample_rate=48000.0, seed=0)
        result = sp.eq_match(buf, buf)
        assert result.frames == buf.frames
        # STFT roundtrip has edge effects, check interior
        margin = 2048
        np.testing.assert_allclose(
            result.data[0, margin:-margin],
            buf.data[0, margin:-margin],
            atol=0.05,
        )

    def test_shape_preserved(self):
        buf = AudioBuffer.noise(channels=2, frames=8192, sample_rate=48000.0, seed=0)
        target = AudioBuffer.noise(channels=2, frames=8192, sample_rate=48000.0, seed=1)
        result = sp.eq_match(buf, target)
        assert result.channels == buf.channels
        assert result.frames == buf.frames

    def test_spectral_tilt_correction(self):
        """A dark source matched to a bright target should have more high-freq energy."""
        sr = 48000.0
        dark = AudioBuffer.noise(channels=1, frames=16384, sample_rate=sr, seed=0)
        dark = lowpass(dark, 2000.0)
        bright = AudioBuffer.noise(channels=1, frames=16384, sample_rate=sr, seed=1)
        bright = highpass(bright, 2000.0)

        result = sp.eq_match(dark, bright, window_size=2048)
        # Compare spectral centroids
        spec_dark = sp.stft(dark, window_size=2048)
        spec_result = sp.stft(result, window_size=2048)
        mag_dark = np.mean(np.abs(spec_dark.data[0]), axis=0)
        mag_result = np.mean(np.abs(spec_result.data[0]), axis=0)
        bins = np.arange(len(mag_dark), dtype=np.float32)
        centroid_dark = np.sum(bins * mag_dark) / (np.sum(mag_dark) + 1e-10)
        centroid_result = np.sum(bins * mag_result) / (np.sum(mag_result) + 1e-10)
        assert centroid_result > centroid_dark * 1.5

    def test_smoothing(self):
        buf = AudioBuffer.noise(channels=1, frames=8192, sample_rate=48000.0, seed=0)
        target = AudioBuffer.noise(channels=1, frames=8192, sample_rate=48000.0, seed=1)
        result = sp.eq_match(buf, target, smoothing=8)
        assert result.frames == buf.frames

    def test_sample_rate_mismatch_raises(self):
        buf = AudioBuffer.noise(channels=1, frames=4096, sample_rate=48000.0, seed=0)
        target = AudioBuffer.noise(channels=1, frames=4096, sample_rate=44100.0, seed=1)
        with pytest.raises(ValueError, match="Sample rate mismatch"):
            sp.eq_match(buf, target)

    def test_channel_mismatch_raises(self):
        buf = AudioBuffer.noise(channels=1, frames=8192, sample_rate=48000.0, seed=0)
        target = AudioBuffer.noise(channels=2, frames=8192, sample_rate=48000.0, seed=1)
        with pytest.raises(ValueError, match="Channel count mismatch"):
            sp.eq_match(buf, target)

    def test_channel_mismatch_raises_reverse(self):
        buf = AudioBuffer.noise(channels=2, frames=8192, sample_rate=48000.0, seed=0)
        target = AudioBuffer.noise(channels=1, frames=8192, sample_rate=48000.0, seed=1)
        with pytest.raises(ValueError, match="Channel count mismatch"):
            sp.eq_match(buf, target)

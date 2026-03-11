"""Tests for nanodsp.analysis module (loudness, spectral features, pitch/onset, resampling)."""

import numpy as np
import pytest

from nanodsp import analysis
from nanodsp.buffer import AudioBuffer


class TestLoudnessLufs:
    def test_1khz_sine_reference(self):
        sr = 48000.0
        frames = int(sr * 5)
        buf = AudioBuffer.sine(1000.0, channels=1, frames=frames, sample_rate=sr)
        lufs = analysis.loudness_lufs(buf)
        assert -5.0 < lufs < -1.0

    def test_silence_returns_neg_inf(self):
        buf = AudioBuffer.zeros(1, 48000, sample_rate=48000.0)
        lufs = analysis.loudness_lufs(buf)
        assert np.isinf(lufs) and lufs < 0

    def test_short_signal_returns_neg_inf(self):
        sr = 48000.0
        frames = int(sr * 0.3)
        buf = AudioBuffer.sine(1000.0, channels=1, frames=frames, sample_rate=sr)
        lufs = analysis.loudness_lufs(buf)
        assert np.isinf(lufs) and lufs < 0

    def test_6db_gain_tracks_linearly(self):
        sr = 48000.0
        frames = int(sr * 3)
        buf = AudioBuffer.sine(1000.0, channels=1, frames=frames, sample_rate=sr)
        buf = buf * 0.25
        lufs_base = analysis.loudness_lufs(buf)
        boosted = buf.gain_db(6.0)
        lufs_boosted = analysis.loudness_lufs(boosted)
        assert abs((lufs_boosted - lufs_base) - 6.0) < 0.5

    def test_stereo(self):
        sr = 48000.0
        frames = int(sr * 3)
        buf = AudioBuffer.sine(1000.0, channels=2, frames=frames, sample_rate=sr)
        lufs = analysis.loudness_lufs(buf)
        mono_lufs = analysis.loudness_lufs(
            AudioBuffer.sine(1000.0, channels=1, frames=frames, sample_rate=sr)
        )
        assert abs((lufs - mono_lufs) - 3.0) < 0.5

    def test_44100_hz(self):
        sr = 44100.0
        frames = int(sr * 3)
        buf = AudioBuffer.sine(1000.0, channels=1, frames=frames, sample_rate=sr)
        lufs = analysis.loudness_lufs(buf)
        assert -5.0 < lufs < -1.0

    def test_5_1_lfe_ignored(self):
        sr = 48000.0
        frames = int(sr * 3)
        data = np.zeros((6, frames), dtype=np.float32)
        t = np.arange(frames, dtype=np.float32) / sr
        data[3] = np.sin(2.0 * np.pi * 60.0 * t).astype(np.float32)
        buf = AudioBuffer(data, sample_rate=sr)
        lufs = analysis.loudness_lufs(buf)
        assert np.isinf(lufs) and lufs < 0

    def test_5_1_surround_weighted(self):
        sr = 48000.0
        frames = int(sr * 3)
        t = np.arange(frames, dtype=np.float32) / sr
        tone = np.sin(2.0 * np.pi * 1000.0 * t).astype(np.float32) * 0.25
        data_left = np.zeros((6, frames), dtype=np.float32)
        data_left[0] = tone
        lufs_left = analysis.loudness_lufs(AudioBuffer(data_left, sample_rate=sr))
        data_ls = np.zeros((6, frames), dtype=np.float32)
        data_ls[4] = tone
        lufs_ls = analysis.loudness_lufs(AudioBuffer(data_ls, sample_rate=sr))
        delta = lufs_ls - lufs_left
        assert 1.0 < delta < 2.0

    def test_5_1_vs_stereo_front_channels(self):
        sr = 48000.0
        frames = int(sr * 3)
        tone = AudioBuffer.sine(1000.0, channels=1, frames=frames, sample_rate=sr)
        stereo = AudioBuffer(np.tile(tone.data, (2, 1)), sample_rate=sr)
        lufs_stereo = analysis.loudness_lufs(stereo)
        data_51 = np.zeros((6, frames), dtype=np.float32)
        data_51[0] = tone.data[0]
        data_51[1] = tone.data[0]
        lufs_51 = analysis.loudness_lufs(AudioBuffer(data_51, sample_rate=sr))
        assert abs(lufs_51 - lufs_stereo) < 0.5


class TestNormalizeLufs:
    def test_hits_target(self):
        sr = 48000.0
        frames = int(sr * 3)
        buf = AudioBuffer.sine(1000.0, channels=1, frames=frames, sample_rate=sr)
        result = analysis.normalize_lufs(buf, target_lufs=-14.0)
        measured = analysis.loudness_lufs(result)
        assert abs(measured - (-14.0)) < 1.5

    def test_silence_raises(self):
        buf = AudioBuffer.zeros(1, 48000, sample_rate=48000.0)
        with pytest.raises(ValueError, match="silent or too short"):
            analysis.normalize_lufs(buf)

    def test_short_signal_raises(self):
        sr = 48000.0
        frames = int(sr * 0.3)
        buf = AudioBuffer.sine(1000.0, channels=1, frames=frames, sample_rate=sr)
        with pytest.raises(ValueError, match="silent or too short"):
            analysis.normalize_lufs(buf)

    def test_metadata_preserved(self):
        sr = 48000.0
        frames = int(sr * 3)
        buf = AudioBuffer.sine(
            1000.0, channels=1, frames=frames, sample_rate=sr, label="norm_test"
        )
        result = analysis.normalize_lufs(buf, target_lufs=-14.0)
        assert result.sample_rate == sr
        assert result.label == "norm_test"

    def test_idempotent(self):
        sr = 48000.0
        frames = int(sr * 3)
        buf = AudioBuffer.noise(channels=1, frames=frames, sample_rate=sr, seed=42)
        for target in [-14.0, -23.0, -9.0]:
            normalized = analysis.normalize_lufs(buf, target_lufs=target)
            measured = analysis.loudness_lufs(normalized)
            assert abs(measured - target) < 0.5, f"target={target}, measured={measured}"


class TestResample:
    def test_identity(self):
        buf = AudioBuffer.noise(channels=1, frames=1024, sample_rate=48000.0, seed=0)
        result = analysis.resample(buf, 48000.0)
        np.testing.assert_array_equal(result.data, buf.data)

    def test_2x_up_output_sr(self):
        buf = AudioBuffer.noise(channels=1, frames=1024, sample_rate=48000.0, seed=0)
        result = analysis.resample(buf, 96000.0)
        assert result.sample_rate == 96000.0
        assert result.frames == 2048

    def test_2x_down_output_sr(self):
        buf = AudioBuffer.noise(channels=1, frames=1024, sample_rate=48000.0, seed=0)
        result = analysis.resample(buf, 24000.0)
        assert result.sample_rate == 24000.0
        assert result.frames == 512

    def test_arbitrary_ratio(self):
        buf = AudioBuffer.noise(channels=1, frames=4096, sample_rate=48000.0, seed=0)
        result = analysis.resample(buf, 44100.0)
        assert result.sample_rate == 44100.0
        expected_frames = round(4096 * 44100.0 / 48000.0)
        assert result.frames == expected_frames

    def test_multichannel(self):
        buf = AudioBuffer.noise(channels=2, frames=1024, sample_rate=48000.0, seed=0)
        result = analysis.resample(buf, 96000.0)
        assert result.channels == 2
        assert result.frames == 2048

    def test_metadata_preserved(self):
        buf = AudioBuffer.noise(
            channels=1, frames=1024, sample_rate=48000.0, seed=0, label="rs"
        )
        result = analysis.resample(buf, 96000.0)
        assert result.label == "rs"


class TestSpectralCentroid:
    def test_pure_sine_centroid(self):
        buf = AudioBuffer.sine(440.0, frames=16384, sample_rate=48000.0)
        c = analysis.spectral_centroid(buf, window_size=4096)
        mean_c = np.mean(c)
        assert abs(mean_c - 440.0) < 50.0

    def test_high_vs_low_sine(self):
        low = AudioBuffer.sine(200.0, frames=16384, sample_rate=48000.0)
        high = AudioBuffer.sine(5000.0, frames=16384, sample_rate=48000.0)
        c_low = np.mean(analysis.spectral_centroid(low, window_size=4096))
        c_high = np.mean(analysis.spectral_centroid(high, window_size=4096))
        assert c_high > c_low

    def test_shape_mono(self):
        buf = AudioBuffer.noise(1, 8192, seed=0)
        c = analysis.spectral_centroid(buf, window_size=2048)
        assert c.ndim == 1
        assert len(c) > 0

    def test_shape_stereo(self):
        buf = AudioBuffer.noise(2, 8192, seed=0)
        c = analysis.spectral_centroid(buf, window_size=2048)
        assert c.ndim == 2
        assert c.shape[0] == 2


class TestSpectralBandwidth:
    def test_narrow_vs_wide(self):
        narrow = AudioBuffer.sine(1000.0, frames=16384, sample_rate=48000.0)
        wide = AudioBuffer.noise(1, 16384, sample_rate=48000.0, seed=0)
        bw_narrow = np.mean(analysis.spectral_bandwidth(narrow, window_size=4096))
        bw_wide = np.mean(analysis.spectral_bandwidth(wide, window_size=4096))
        assert bw_wide > bw_narrow

    def test_shape(self):
        buf = AudioBuffer.noise(1, 8192, seed=0)
        bw = analysis.spectral_bandwidth(buf, window_size=2048)
        assert bw.ndim == 1


class TestSpectralRolloff:
    def test_lower_percentile_lower_freq(self):
        buf = AudioBuffer.noise(1, 16384, sample_rate=48000.0, seed=0)
        r50 = np.mean(analysis.spectral_rolloff(buf, window_size=4096, percentile=0.5))
        r95 = np.mean(analysis.spectral_rolloff(buf, window_size=4096, percentile=0.95))
        assert r95 > r50

    def test_shape(self):
        buf = AudioBuffer.noise(1, 8192, seed=0)
        r = analysis.spectral_rolloff(buf, window_size=2048)
        assert r.ndim == 1

    def test_stereo(self):
        buf = AudioBuffer.noise(2, 8192, seed=0)
        r = analysis.spectral_rolloff(buf, window_size=2048)
        assert r.shape[0] == 2


class TestSpectralFlux:
    def test_constant_signal_low_flux(self):
        buf = AudioBuffer.sine(440.0, frames=16384, sample_rate=48000.0)
        f = analysis.spectral_flux(buf, window_size=2048)
        assert np.mean(f[2:]) < 50.0

    def test_onset_has_spike(self):
        data = np.zeros((1, 32768), dtype=np.float32)
        data[0, 16384:] = (
            np.random.default_rng(42).standard_normal(32768 - 16384).astype(np.float32)
        )
        buf = AudioBuffer(data, sample_rate=48000.0)
        ws = 2048
        hop = ws // 4
        f = analysis.spectral_flux(buf, window_size=ws, hop_size=hop)
        n_silent_frames = 16384 // hop
        quarter = n_silent_frames // 2
        assert np.mean(f[:quarter]) < 1e-6
        assert np.max(f) > 0.1

    def test_first_frame_zero(self):
        buf = AudioBuffer.noise(1, 8192, seed=0)
        f = analysis.spectral_flux(buf, window_size=2048)
        assert f[0] == 0.0

    def test_rectified_nonnegative(self):
        buf = AudioBuffer.noise(1, 8192, seed=0)
        f = analysis.spectral_flux(buf, window_size=2048, rectify=True)
        assert np.all(f >= 0)


class TestSpectralFlatnessCurve:
    def test_sine_low_flatness(self):
        buf = AudioBuffer.sine(440.0, frames=16384, sample_rate=48000.0)
        fl = analysis.spectral_flatness_curve(buf, window_size=4096)
        assert np.mean(fl) < 0.3

    def test_noise_high_flatness(self):
        buf = AudioBuffer.noise(1, 16384, sample_rate=48000.0, seed=0)
        fl = analysis.spectral_flatness_curve(buf, window_size=4096)
        assert np.mean(fl) > 0.3

    def test_shape(self):
        buf = AudioBuffer.noise(1, 8192, seed=0)
        fl = analysis.spectral_flatness_curve(buf, window_size=2048)
        assert fl.ndim == 1

    def test_range(self):
        buf = AudioBuffer.noise(1, 8192, seed=0)
        fl = analysis.spectral_flatness_curve(buf, window_size=2048)
        assert np.all(fl >= 0)
        assert np.all(fl <= 1.0 + 1e-6)


class TestChromagram:
    def test_a4_peaks_at_a(self):
        buf = AudioBuffer.sine(440.0, frames=32768, sample_rate=48000.0)
        chroma = analysis.chromagram(buf, window_size=8192)
        assert chroma.shape[0] == 12
        mean_chroma = np.mean(chroma, axis=1)
        peak_class = np.argmax(mean_chroma)
        assert peak_class == 0

    def test_shape_mono(self):
        buf = AudioBuffer.noise(1, 16384, seed=0)
        chroma = analysis.chromagram(buf, window_size=4096)
        assert chroma.ndim == 2
        assert chroma.shape[0] == 12

    def test_shape_stereo(self):
        buf = AudioBuffer.noise(2, 16384, seed=0)
        chroma = analysis.chromagram(buf, window_size=4096)
        assert chroma.ndim == 3
        assert chroma.shape[0] == 2
        assert chroma.shape[1] == 12

    def test_custom_n_chroma(self):
        buf = AudioBuffer.noise(1, 16384, seed=0)
        chroma = analysis.chromagram(buf, window_size=4096, n_chroma=24)
        assert chroma.shape[0] == 24


class TestSpectralFeaturesSilence:
    def test_centroid_silence(self):
        buf = AudioBuffer.zeros(1, 8192)
        c = analysis.spectral_centroid(buf, window_size=2048)
        assert np.all(c == 0.0)

    def test_bandwidth_silence(self):
        buf = AudioBuffer.zeros(1, 8192)
        bw = analysis.spectral_bandwidth(buf, window_size=2048)
        assert np.all(bw == 0.0)

    def test_flux_silence(self):
        buf = AudioBuffer.zeros(1, 8192)
        f = analysis.spectral_flux(buf, window_size=2048)
        assert np.all(f == 0.0)


class TestPitchDetect:
    def test_sine_440(self):
        buf = AudioBuffer.sine(440.0, frames=32768, sample_rate=48000.0)
        freqs, confs = analysis.pitch_detect(buf, window_size=4096)
        voiced = confs > 0.5
        assert np.sum(voiced) > len(freqs) // 2
        detected = freqs[voiced]
        assert abs(np.median(detected) - 440.0) < 20.0

    def test_sine_100(self):
        buf = AudioBuffer.sine(100.0, frames=32768, sample_rate=48000.0)
        freqs, confs = analysis.pitch_detect(
            buf, window_size=4096, fmin=50.0, fmax=500.0
        )
        voiced = confs > 0.3
        if np.sum(voiced) > 0:
            detected = freqs[voiced]
            assert abs(np.median(detected) - 100.0) < 30.0

    def test_sine_1000(self):
        buf = AudioBuffer.sine(1000.0, frames=16384, sample_rate=48000.0)
        freqs, confs = analysis.pitch_detect(buf, window_size=2048)
        voiced = confs > 0.5
        if np.sum(voiced) > 0:
            detected = freqs[voiced]
            assert abs(np.median(detected) - 1000.0) < 50.0

    def test_silence_low_confidence(self):
        buf = AudioBuffer.zeros(1, 8192, sample_rate=48000.0)
        freqs, confs = analysis.pitch_detect(buf, window_size=2048)
        assert np.all(confs < 0.5)

    def test_noise_low_confidence(self):
        buf = AudioBuffer.noise(1, 16384, sample_rate=48000.0, seed=0)
        freqs, confs = analysis.pitch_detect(buf, window_size=2048)
        mean_conf = np.mean(confs)
        assert mean_conf < 0.8

    def test_stereo_independent(self):
        t = np.arange(16384, dtype=np.float32) / 48000.0
        data = np.zeros((2, 16384), dtype=np.float32)
        data[0] = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
        data[1] = np.sin(2 * np.pi * 880.0 * t).astype(np.float32)
        buf = AudioBuffer(data, sample_rate=48000.0)
        freqs, confs = analysis.pitch_detect(buf, window_size=4096)
        assert freqs.ndim == 2
        assert freqs.shape[0] == 2

    def test_fmin_fmax_constraints(self):
        buf = AudioBuffer.sine(440.0, frames=16384, sample_rate=48000.0)
        freqs, confs = analysis.pitch_detect(
            buf, window_size=4096, fmin=500.0, fmax=1000.0
        )
        voiced = confs > 0.5
        if np.sum(voiced) > 0:
            assert np.all(freqs[voiced] >= 500.0) or np.all(freqs[voiced] == 0.0)

    def test_return_shapes(self):
        buf = AudioBuffer.noise(1, 8192, seed=0)
        freqs, confs = analysis.pitch_detect(buf, window_size=2048)
        assert freqs.shape == confs.shape
        assert freqs.ndim == 1

    def test_unknown_method_raises(self):
        buf = AudioBuffer.noise(1, 4096, seed=0)
        with pytest.raises(ValueError, match="Unknown"):
            analysis.pitch_detect(buf, method="autocorrelation")


class TestOnsetDetect:
    def test_silence_no_onsets(self):
        buf = AudioBuffer.zeros(1, 16384, sample_rate=48000.0)
        onsets = analysis.onset_detect(buf, window_size=2048)
        assert len(onsets) == 0

    def test_single_impulse(self):
        data = np.zeros((1, 32768), dtype=np.float32)
        data[0, 8000:8200] = (
            np.random.default_rng(0).standard_normal(200).astype(np.float32) * 0.5
        )
        buf = AudioBuffer(data, sample_rate=48000.0)
        onsets = analysis.onset_detect(buf, window_size=2048)
        assert len(onsets) >= 1
        assert np.any(np.abs(onsets - 8000) < 4096)

    def test_periodic_clicks(self):
        data = np.zeros((1, 48000), dtype=np.float32)
        rng = np.random.default_rng(42)
        for start in [4000, 16000, 28000, 40000]:
            data[0, start : start + 200] = (
                rng.standard_normal(200).astype(np.float32) * 0.5
            )
        buf = AudioBuffer(data, sample_rate=48000.0)
        onsets = analysis.onset_detect(buf, window_size=1024, hop_size=256)
        assert len(onsets) >= 2

    def test_threshold_sensitivity(self):
        data = np.zeros((1, 32768), dtype=np.float32)
        data[0, 8000:8200] = (
            np.random.default_rng(0).standard_normal(200).astype(np.float32) * 0.3
        )
        buf = AudioBuffer(data, sample_rate=48000.0)
        onsets_high = analysis.onset_detect(buf, window_size=2048, threshold=1000.0)
        onsets_low = analysis.onset_detect(buf, window_size=2048, threshold=0.01)
        assert len(onsets_low) >= len(onsets_high)

    def test_backtrack(self):
        data = np.zeros((1, 32768), dtype=np.float32)
        ramp = np.linspace(0, 0.5, 500, dtype=np.float32)
        data[0, 8000:8500] = ramp * np.random.default_rng(0).standard_normal(
            500
        ).astype(np.float32)
        data[0, 8500:9000] = (
            np.random.default_rng(1).standard_normal(500).astype(np.float32) * 0.5
        )
        buf = AudioBuffer(data, sample_rate=48000.0)
        onsets_no_bt = analysis.onset_detect(buf, window_size=2048)
        onsets_bt = analysis.onset_detect(buf, window_size=2048, backtrack=True)
        if len(onsets_no_bt) > 0 and len(onsets_bt) > 0:
            assert onsets_bt[0] <= onsets_no_bt[0]

    def test_stereo_mixes_to_mono(self):
        data = np.zeros((2, 32768), dtype=np.float32)
        data[:, 8000:8200] = (
            np.random.default_rng(0).standard_normal((2, 200)).astype(np.float32)
        )
        buf = AudioBuffer(data, sample_rate=48000.0)
        onsets = analysis.onset_detect(buf, window_size=2048)
        assert isinstance(onsets, np.ndarray)

    def test_return_type(self):
        buf = AudioBuffer.noise(1, 16384, seed=0)
        onsets = analysis.onset_detect(buf)
        assert onsets.dtype == np.int64

    def test_unknown_method_raises(self):
        buf = AudioBuffer.noise(1, 4096, seed=0)
        with pytest.raises(ValueError, match="Unknown"):
            analysis.onset_detect(buf, method="energy")


class TestResampleFft:
    def test_identity(self):
        buf = AudioBuffer.noise(1, 1000, sample_rate=48000.0, seed=0)
        result = analysis.resample_fft(buf, 48000.0)
        np.testing.assert_array_equal(result.data, buf.data)
        assert result.sample_rate == 48000.0

    def test_upsample_frame_count(self):
        buf = AudioBuffer.noise(1, 1000, sample_rate=44100.0, seed=0)
        result = analysis.resample_fft(buf, 48000.0)
        expected_frames = round(1000 * 48000.0 / 44100.0)
        assert result.frames == expected_frames
        assert result.sample_rate == 48000.0

    def test_downsample_frame_count(self):
        buf = AudioBuffer.noise(1, 1000, sample_rate=48000.0, seed=0)
        result = analysis.resample_fft(buf, 24000.0)
        expected_frames = round(1000 * 24000.0 / 48000.0)
        assert result.frames == expected_frames
        assert result.sample_rate == 24000.0

    def test_sine_frequency_preserved(self):
        freq = 440.0
        sr = 48000.0
        target_sr = 44100.0
        buf = AudioBuffer.sine(freq, frames=32768, sample_rate=sr)
        result = analysis.resample_fft(buf, target_sr)
        X = np.abs(np.fft.rfft(result.data[0]))
        freqs_axis = np.fft.rfftfreq(result.frames, d=1.0 / target_sr)
        peak_freq = freqs_axis[np.argmax(X)]
        assert abs(peak_freq - freq) < 5.0

    def test_stereo(self):
        buf = AudioBuffer.noise(2, 1000, sample_rate=48000.0, seed=0)
        result = analysis.resample_fft(buf, 24000.0)
        assert result.channels == 2

    def test_metadata_preserved(self):
        buf = AudioBuffer.noise(1, 1000, sample_rate=48000.0, seed=0, label="test")
        result = analysis.resample_fft(buf, 44100.0)
        assert result.label == "test"
        assert result.sample_rate == 44100.0

    def test_negative_sr_raises(self):
        buf = AudioBuffer.noise(1, 100, seed=0)
        with pytest.raises(ValueError, match="positive"):
            analysis.resample_fft(buf, -44100.0)

    def test_roundtrip_approximate(self):
        buf = AudioBuffer.sine(440.0, frames=16384, sample_rate=48000.0)
        up = analysis.resample_fft(buf, 96000.0)
        down = analysis.resample_fft(up, 48000.0)
        min_len = min(buf.frames, down.frames)
        corr = np.corrcoef(buf.data[0, :min_len], down.data[0, :min_len])[0, 1]
        assert corr > 0.95

    def test_downsample_halves(self):
        buf = AudioBuffer.noise(1, 1000, sample_rate=48000.0, seed=0)
        result = analysis.resample_fft(buf, 24000.0)
        assert result.frames == 500

    def test_zero_sr_raises(self):
        buf = AudioBuffer.noise(1, 100, seed=0)
        with pytest.raises(ValueError, match="positive"):
            analysis.resample_fft(buf, 0.0)


# ---------------------------------------------------------------------------
# GCC-PHAT
# ---------------------------------------------------------------------------


class TestGccPhat:
    def test_known_delay_detection(self):
        """GCC-PHAT should detect a known integer sample delay."""
        sr = 48000.0
        n = 4096
        rng = np.random.default_rng(42)
        signal = rng.standard_normal(n).astype(np.float32)
        delay_samples = 50
        delayed = np.zeros(n, dtype=np.float32)
        delayed[delay_samples:] = signal[: n - delay_samples]
        buf = AudioBuffer(delayed.reshape(1, -1), sample_rate=sr)
        ref = AudioBuffer(signal.reshape(1, -1), sample_rate=sr)
        delay_sec, corr = analysis.gcc_phat(buf, ref)
        estimated_samples = round(delay_sec * sr)
        assert estimated_samples == delay_samples

    def test_zero_delay(self):
        """Identical signals should give zero delay."""
        sr = 48000.0
        buf = AudioBuffer.noise(channels=1, frames=2048, sample_rate=sr, seed=42)
        delay_sec, corr = analysis.gcc_phat(buf, buf)
        assert abs(delay_sec) < 1.0 / sr  # within 1 sample

    def test_returns_correlation_array(self):
        sr = 48000.0
        buf = AudioBuffer.noise(channels=1, frames=1024, sample_rate=sr, seed=0)
        ref = AudioBuffer.noise(channels=1, frames=1024, sample_rate=sr, seed=1)
        delay_sec, corr = analysis.gcc_phat(buf, ref)
        assert isinstance(corr, np.ndarray)
        assert corr.dtype == np.float32
        assert len(corr) == 2 * 1024 - 1

    def test_negative_delay(self):
        """If ref is delayed relative to buf, delay should be negative."""
        sr = 48000.0
        n = 4096
        rng = np.random.default_rng(123)
        signal = rng.standard_normal(n).astype(np.float32)
        delay_samples = 30
        delayed = np.zeros(n, dtype=np.float32)
        delayed[delay_samples:] = signal[: n - delay_samples]
        # buf=original, ref=delayed -> buf leads -> negative delay
        buf = AudioBuffer(signal.reshape(1, -1), sample_rate=sr)
        ref = AudioBuffer(delayed.reshape(1, -1), sample_rate=sr)
        delay_sec, corr = analysis.gcc_phat(buf, ref)
        estimated_samples = round(delay_sec * sr)
        assert estimated_samples == -delay_samples

    def test_custom_sample_rate(self):
        sr = 44100.0
        buf = AudioBuffer.noise(channels=1, frames=2048, sample_rate=sr, seed=0)
        delay_sec, corr = analysis.gcc_phat(buf, buf, sample_rate=sr)
        assert abs(delay_sec) < 1.0 / sr

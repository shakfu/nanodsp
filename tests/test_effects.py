"""Tests for nanodsp.effects module (filters, effects, dynamics, reverbs, mastering)."""

import numpy as np
import pytest

from nanodsp import analysis, effects
from nanodsp._core import filters
from nanodsp.buffer import AudioBuffer


# ---------------------------------------------------------------------------
# Filter functions
# ---------------------------------------------------------------------------


class TestFilterFunctions:
    def test_lowpass_attenuates_high_freq(self):
        buf = AudioBuffer.sine(1000.0, frames=4096, sample_rate=48000.0)
        high = AudioBuffer.sine(20000.0, frames=4096, sample_rate=48000.0)
        combined = buf + high
        result = effects.lowpass(combined, 5000.0)
        # High-frequency energy should be attenuated
        assert np.sum(result.data**2) < np.sum(combined.data**2)

    def test_highpass_attenuates_low_freq(self):
        low = AudioBuffer.sine(100.0, frames=4096, sample_rate=48000.0)
        high = AudioBuffer.sine(10000.0, frames=4096, sample_rate=48000.0)
        combined = low + high
        result = effects.highpass(combined, 5000.0)
        assert np.sum(result.data**2) < np.sum(combined.data**2)

    def test_bandpass_passes_center(self):
        center = AudioBuffer.sine(5000.0, frames=4096, sample_rate=48000.0)
        result = effects.bandpass(center, 5000.0, octaves=2.0)
        # Center frequency should pass through with reasonable energy
        energy_ratio = np.sum(result.data**2) / np.sum(center.data**2)
        assert energy_ratio > 0.3

    def test_notch_attenuates_center(self):
        center = AudioBuffer.sine(5000.0, frames=4096, sample_rate=48000.0)
        result = effects.notch(center, 5000.0, octaves=1.0)
        energy_ratio = np.sum(result.data**2) / np.sum(center.data**2)
        assert energy_ratio < 0.5

    def test_peak_boosts(self):
        buf = AudioBuffer.sine(5000.0, frames=4096, sample_rate=48000.0)
        result = effects.peak(buf, 5000.0, gain=4.0)
        assert np.sum(result.data**2) > np.sum(buf.data**2)

    def test_peak_db_boosts(self):
        buf = AudioBuffer.sine(5000.0, frames=4096, sample_rate=48000.0)
        result = effects.peak_db(buf, 5000.0, db=12.0)
        assert np.sum(result.data**2) > np.sum(buf.data**2)

    def test_high_shelf(self):
        buf = AudioBuffer.sine(15000.0, frames=4096, sample_rate=48000.0)
        result = effects.high_shelf(buf, 10000.0, gain=4.0)
        assert np.sum(result.data**2) > np.sum(buf.data**2)

    def test_high_shelf_db(self):
        buf = AudioBuffer.sine(15000.0, frames=4096, sample_rate=48000.0)
        result = effects.high_shelf_db(buf, 10000.0, db=12.0)
        assert np.sum(result.data**2) > np.sum(buf.data**2)

    def test_low_shelf(self):
        buf = AudioBuffer.sine(100.0, frames=4096, sample_rate=48000.0)
        result = effects.low_shelf(buf, 1000.0, gain=4.0)
        assert np.sum(result.data**2) > np.sum(buf.data**2)

    def test_low_shelf_db(self):
        buf = AudioBuffer.sine(100.0, frames=4096, sample_rate=48000.0)
        result = effects.low_shelf_db(buf, 1000.0, db=12.0)
        assert np.sum(result.data**2) > np.sum(buf.data**2)

    def test_allpass_preserves_magnitude(self):
        buf = AudioBuffer.sine(5000.0, frames=4096, sample_rate=48000.0)
        result = effects.allpass(buf, 5000.0)
        in_energy = np.sum(buf.data**2)
        out_energy = np.sum(result.data**2)
        np.testing.assert_allclose(out_energy, in_energy, rtol=0.01)

    def test_metadata_preserved(self):
        buf = AudioBuffer.sine(
            1000.0,
            channels=2,
            frames=1024,
            sample_rate=44100.0,
            label="test",
        )
        result = effects.lowpass(buf, 5000.0)
        assert result.sample_rate == 44100.0
        assert result.channels == 2
        assert result.frames == 1024
        assert result.label == "test"
        assert result.channel_layout == "stereo"

    def test_per_channel_independence(self):
        # Different content per channel
        data = np.zeros((2, 4096), dtype=np.float32)
        t = np.arange(4096, dtype=np.float32) / 48000.0
        data[0] = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
        data[1] = np.sin(2 * np.pi * 15000 * t).astype(np.float32)
        buf = AudioBuffer(data, sample_rate=48000.0)
        result = effects.lowpass(buf, 5000.0)
        # Channel 0 (1kHz) should retain more energy than channel 1 (15kHz)
        ch0_energy = np.sum(result.data[0] ** 2)
        ch1_energy = np.sum(result.data[1] ** 2)
        assert ch0_energy > ch1_energy * 5

    def test_all_filters_produce_correct_shape(self):
        buf = AudioBuffer.noise(channels=1, frames=1024, sample_rate=48000.0, seed=0)
        for fn, kwargs in [
            (effects.lowpass, {"cutoff_hz": 5000.0}),
            (effects.highpass, {"cutoff_hz": 5000.0}),
            (effects.bandpass, {"center_hz": 5000.0}),
            (effects.notch, {"center_hz": 5000.0}),
            (effects.peak, {"center_hz": 5000.0, "gain": 2.0}),
            (effects.peak_db, {"center_hz": 5000.0, "db": 6.0}),
            (effects.high_shelf, {"cutoff_hz": 5000.0, "gain": 2.0}),
            (effects.high_shelf_db, {"cutoff_hz": 5000.0, "db": 6.0}),
            (effects.low_shelf, {"cutoff_hz": 5000.0, "gain": 2.0}),
            (effects.low_shelf_db, {"cutoff_hz": 5000.0, "db": 6.0}),
            (effects.allpass, {"freq_hz": 5000.0}),
        ]:
            result = fn(buf, **kwargs)
            assert result.channels == 1
            assert result.frames == 1024
            assert result.data.dtype == np.float32

    def test_biquad_process_preconfigured(self):
        bq = filters.Biquad()
        bq.lowpass(0.1)
        buf = AudioBuffer.noise(channels=2, frames=1024, sample_rate=48000.0, seed=0)
        result = effects.biquad_process(buf, bq)
        assert result.channels == 2
        assert result.frames == 1024

    def test_lowpass_with_explicit_octaves(self):
        buf = AudioBuffer.noise(channels=1, frames=1024, sample_rate=48000.0, seed=0)
        result = effects.lowpass(buf, 5000.0, octaves=2.0)
        assert result.frames == 1024


# ---------------------------------------------------------------------------
# DaisySP Effects
# ---------------------------------------------------------------------------


class TestDaisySPEffects:
    def _noise(self, channels=1, frames=2048, sample_rate=48000.0):
        return AudioBuffer.noise(
            channels=channels, frames=frames, sample_rate=sample_rate, seed=0
        )

    def _sine(self, freq=440.0, channels=1, frames=2048, sample_rate=48000.0):
        return AudioBuffer.sine(
            freq, channels=channels, frames=frames, sample_rate=sample_rate
        )

    def test_autowah_shape_dtype(self):
        buf = self._sine()
        result = effects.autowah(buf)
        assert result.channels == 1
        assert result.frames == 2048
        assert result.data.dtype == np.float32

    def test_autowah_modifies_signal(self):
        buf = self._sine()
        result = effects.autowah(buf, wah=0.8)
        assert not np.allclose(result.data, buf.data)

    def test_chorus_mono_to_stereo(self):
        buf = self._sine()
        result = effects.chorus(buf)
        assert result.channels == 2
        assert result.frames == 2048

    def test_chorus_multichannel_per_channel(self):
        buf = self._noise(channels=2)
        result = effects.chorus(buf)
        assert result.channels == 2

    def test_decimator_shape(self):
        buf = self._noise()
        result = effects.decimator(buf)
        assert result.frames == 2048
        assert result.data.dtype == np.float32

    def test_decimator_modifies_signal(self):
        buf = self._sine()
        result = effects.decimator(buf, downsample_factor=0.8, bits_to_crush=4)
        assert not np.allclose(result.data, buf.data)

    def test_flanger_shape(self):
        buf = self._sine()
        result = effects.flanger(buf)
        assert result.channels == 1
        assert result.frames == 2048

    def test_overdrive_shape(self):
        buf = self._sine()
        result = effects.overdrive(buf, drive=0.8)
        assert result.frames == 2048
        assert result.data.dtype == np.float32

    def test_overdrive_adds_harmonics(self):
        buf = self._sine()
        result = effects.overdrive(buf, drive=0.9)
        assert not np.allclose(result.data, buf.data)

    def test_phaser_shape(self):
        buf = self._sine()
        result = effects.phaser(buf)
        assert result.frames == 2048

    def test_pitch_shift_shape(self):
        buf = self._sine()
        result = effects.pitch_shift(buf, semitones=5.0)
        assert result.frames == 2048

    def test_sample_rate_reduce_shape(self):
        buf = self._noise()
        result = effects.sample_rate_reduce(buf, freq=0.3)
        assert result.frames == 2048

    def test_tremolo_shape(self):
        buf = self._sine()
        result = effects.tremolo(buf, freq=5.0, depth=1.0)
        assert result.frames == 2048

    def test_wavefold_shape(self):
        buf = self._sine()
        result = effects.wavefold(buf, gain=2.0)
        assert result.frames == 2048

    def test_bitcrush_shape(self):
        buf = self._noise()
        result = effects.bitcrush(buf, bit_depth=4)
        assert result.frames == 2048

    def test_bitcrush_default_crush_rate(self):
        buf = self._noise()
        result = effects.bitcrush(buf)
        assert result.data.dtype == np.float32

    def test_fold_shape(self):
        buf = self._sine()
        result = effects.fold(buf, increment=0.5)
        assert result.frames == 2048

    def test_reverb_sc_mono_to_stereo(self):
        buf = self._sine()
        result = effects.reverb_sc(buf)
        assert result.channels == 2
        assert result.frames == 2048

    def test_reverb_sc_stereo_passthrough(self):
        buf = self._noise(channels=2)
        result = effects.reverb_sc(buf)
        assert result.channels == 2

    def test_reverb_sc_3ch_raises(self):
        buf = self._noise(channels=3)
        with pytest.raises(ValueError, match="mono or stereo"):
            effects.reverb_sc(buf)

    def test_dc_block_shape(self):
        buf = self._noise()
        result = effects.dc_block(buf)
        assert result.frames == 2048

    def test_dc_block_removes_offset(self):
        data = np.ones((1, 4096), dtype=np.float32) * 0.5
        buf = AudioBuffer(data, sample_rate=48000.0)
        result = effects.dc_block(buf)
        # Mean should be much closer to 0 after DC blocking
        assert abs(np.mean(result.data[0, 1024:])) < abs(np.mean(buf.data[0]))

    def test_effects_multichannel(self):
        buf = self._noise(channels=2)
        for fn, kwargs in [
            (effects.autowah, {}),
            (effects.decimator, {}),
            (effects.flanger, {}),
            (effects.overdrive, {}),
            (effects.phaser, {}),
            (effects.tremolo, {}),
            (effects.wavefold, {}),
            (effects.bitcrush, {}),
            (effects.fold, {}),
            (effects.dc_block, {}),
        ]:
            result = fn(buf, **kwargs)
            assert result.channels == 2, f"{fn.__name__} failed multichannel"


# ---------------------------------------------------------------------------
# DaisySP Filters
# ---------------------------------------------------------------------------


class TestDaisySPFilters:
    def _noise(self, channels=1, frames=4096, sample_rate=48000.0):
        return AudioBuffer.noise(
            channels=channels, frames=frames, sample_rate=sample_rate, seed=0
        )

    def test_svf_lowpass_attenuates_high(self):
        low = AudioBuffer.sine(500.0, frames=4096, sample_rate=48000.0)
        high = AudioBuffer.sine(15000.0, frames=4096, sample_rate=48000.0)
        combined = low + high
        result = effects.svf_lowpass(combined, freq_hz=2000.0)
        assert np.sum(result.data**2) < np.sum(combined.data**2)

    def test_svf_highpass_attenuates_low(self):
        low = AudioBuffer.sine(500.0, frames=4096, sample_rate=48000.0)
        high = AudioBuffer.sine(15000.0, frames=4096, sample_rate=48000.0)
        combined = low + high
        result = effects.svf_highpass(combined, freq_hz=5000.0)
        assert np.sum(result.data**2) < np.sum(combined.data**2)

    def test_svf_bandpass_shape(self):
        buf = self._noise()
        result = effects.svf_bandpass(buf, freq_hz=1000.0)
        assert result.frames == 4096

    def test_svf_notch_shape(self):
        buf = self._noise()
        result = effects.svf_notch(buf, freq_hz=1000.0)
        assert result.frames == 4096

    def test_svf_peak_shape(self):
        buf = self._noise()
        result = effects.svf_peak(buf, freq_hz=1000.0)
        assert result.frames == 4096

    def test_ladder_filter_lowpass(self):
        buf = self._noise()
        result = effects.ladder_filter(buf, freq_hz=2000.0, mode="lp24")
        assert result.frames == 4096
        assert np.sum(result.data**2) < np.sum(buf.data**2)

    def test_ladder_filter_modes(self):
        buf = self._noise()
        for mode in ["lp24", "lp12", "bp24", "bp12", "hp24", "hp12"]:
            result = effects.ladder_filter(buf, freq_hz=2000.0, mode=mode)
            assert result.frames == 4096, f"Failed for mode={mode}"

    def test_ladder_filter_invalid_mode(self):
        buf = self._noise()
        with pytest.raises(ValueError, match="Unknown ladder mode"):
            effects.ladder_filter(buf, mode="invalid")

    def test_moog_ladder_shape(self):
        buf = self._noise()
        result = effects.moog_ladder(buf, freq_hz=2000.0, resonance=0.3)
        assert result.frames == 4096

    def test_tone_lowpass_attenuates_high(self):
        low = AudioBuffer.sine(500.0, frames=4096, sample_rate=48000.0)
        high = AudioBuffer.sine(15000.0, frames=4096, sample_rate=48000.0)
        combined = low + high
        result = effects.tone_lowpass(combined, freq_hz=2000.0)
        assert np.sum(result.data**2) < np.sum(combined.data**2)

    def test_tone_highpass_attenuates_low(self):
        low = AudioBuffer.sine(500.0, frames=4096, sample_rate=48000.0)
        high = AudioBuffer.sine(15000.0, frames=4096, sample_rate=48000.0)
        combined = low + high
        result = effects.tone_highpass(combined, freq_hz=5000.0)
        assert np.sum(result.data**2) < np.sum(combined.data**2)

    def test_modal_bandpass_shape(self):
        buf = self._noise()
        result = effects.modal_bandpass(buf, freq_hz=1000.0)
        assert result.frames == 4096

    def test_comb_filter_shape(self):
        buf = self._noise()
        result = effects.comb_filter(buf, freq_hz=500.0)
        assert result.frames == 4096

    def test_filters_multichannel(self):
        buf = self._noise(channels=2)
        for fn, kwargs in [
            (effects.svf_lowpass, {"freq_hz": 2000.0}),
            (effects.svf_highpass, {"freq_hz": 2000.0}),
            (effects.ladder_filter, {"freq_hz": 2000.0}),
            (effects.moog_ladder, {"freq_hz": 2000.0}),
            (effects.tone_lowpass, {"freq_hz": 2000.0}),
            (effects.tone_highpass, {"freq_hz": 2000.0}),
            (effects.modal_bandpass, {"freq_hz": 1000.0}),
            (effects.comb_filter, {"freq_hz": 500.0}),
        ]:
            result = fn(buf, **kwargs)
            assert result.channels == 2, f"{fn.__name__} failed multichannel"


# ---------------------------------------------------------------------------
# DaisySP Dynamics
# ---------------------------------------------------------------------------


class TestDaisySPDynamics:
    def test_compress_shape(self):
        buf = AudioBuffer.noise(channels=1, frames=2048, sample_rate=48000.0, seed=0)
        result = effects.compress(buf, ratio=4.0, threshold=-20.0)
        assert result.frames == 2048
        assert result.data.dtype == np.float32

    def test_compress_reduces_dynamic_range(self):
        buf = AudioBuffer.noise(channels=1, frames=4096, sample_rate=48000.0, seed=0)
        result = effects.compress(buf, ratio=8.0, threshold=-30.0)
        # Compressed signal should have different peak/RMS ratio
        assert not np.allclose(result.data, buf.data)

    def test_compress_multichannel(self):
        buf = AudioBuffer.noise(channels=2, frames=2048, sample_rate=48000.0, seed=0)
        result = effects.compress(buf)
        assert result.channels == 2

    def test_limit_shape(self):
        buf = AudioBuffer.noise(channels=1, frames=2048, sample_rate=48000.0, seed=0)
        result = effects.limit(buf, pre_gain=2.0)
        assert result.frames == 2048
        assert result.data.dtype == np.float32

    def test_limit_multichannel(self):
        buf = AudioBuffer.noise(channels=2, frames=2048, sample_rate=48000.0, seed=0)
        result = effects.limit(buf)
        assert result.channels == 2


# ---------------------------------------------------------------------------
# Saturation
# ---------------------------------------------------------------------------


class TestSaturate:
    def test_shape_preserved(self):
        buf = AudioBuffer.noise(channels=2, frames=2048, sample_rate=48000.0, seed=0)
        result = effects.saturate(buf, drive=0.5, mode="soft")
        assert result.channels == 2
        assert result.frames == 2048
        assert result.data.dtype == np.float32

    def test_drive_zero_near_identity(self):
        """With drive=0 (gain=1x), soft saturation should be near identity for small signals."""
        buf = AudioBuffer.noise(channels=1, frames=2048, sample_rate=48000.0, seed=0)
        buf = buf * 0.1  # keep signal very small so tanh(x) ~= x
        result = effects.saturate(buf, drive=0.0, mode="soft")
        np.testing.assert_allclose(result.data, buf.data, atol=0.02)

    def test_hard_clip_bounded(self):
        buf = AudioBuffer.noise(channels=1, frames=2048, sample_rate=48000.0, seed=0)
        result = effects.saturate(buf, drive=1.0, mode="hard")
        assert np.max(np.abs(result.data)) <= 1.0 + 1e-6

    def test_all_modes_callable(self):
        buf = AudioBuffer.noise(channels=1, frames=1024, sample_rate=48000.0, seed=0)
        for mode in ["soft", "hard", "tape"]:
            result = effects.saturate(buf, drive=0.5, mode=mode)
            assert result.frames == 1024, f"Failed for mode={mode}"

    def test_invalid_mode_raises(self):
        buf = AudioBuffer.noise(channels=1, frames=1024, sample_rate=48000.0, seed=0)
        with pytest.raises(ValueError, match="Unknown saturation mode"):
            effects.saturate(buf, mode="nope")

    def test_soft_preserves_peak(self):
        """Soft saturation should approximately preserve peak amplitude."""
        buf = AudioBuffer.noise(channels=1, frames=4096, sample_rate=48000.0, seed=0)
        result = effects.saturate(buf, drive=0.5, mode="soft")
        peak_in = np.max(np.abs(buf.data))
        peak_out = np.max(np.abs(result.data))
        np.testing.assert_allclose(peak_out, peak_in, rtol=0.05)


# ---------------------------------------------------------------------------
# Exciter
# ---------------------------------------------------------------------------


class TestExciter:
    def test_shape_preserved(self):
        buf = AudioBuffer.noise(channels=1, frames=4096, sample_rate=48000.0, seed=0)
        result = effects.exciter(buf, freq=3000.0, amount=0.3)
        assert result.channels == 1
        assert result.frames == 4096

    def test_amount_zero_near_identity(self):
        buf = AudioBuffer.noise(channels=1, frames=4096, sample_rate=48000.0, seed=0)
        result = effects.exciter(buf, amount=0.0)
        np.testing.assert_allclose(result.data, buf.data, atol=1e-6)

    def test_adds_energy_above_freq(self):
        """Exciter should add harmonic energy in the high-frequency range."""
        buf = AudioBuffer.noise(channels=1, frames=8192, sample_rate=48000.0, seed=0)
        result = effects.exciter(buf, freq=3000.0, amount=0.5)
        # Total energy should increase when adding harmonics
        energy_in = np.sum(buf.data**2)
        energy_out = np.sum(result.data**2)
        assert energy_out > energy_in

    def test_multichannel(self):
        buf = AudioBuffer.noise(channels=2, frames=4096, sample_rate=48000.0, seed=0)
        result = effects.exciter(buf, freq=3000.0)
        assert result.channels == 2


# ---------------------------------------------------------------------------
# De-esser
# ---------------------------------------------------------------------------


class TestDeEsser:
    def test_shape_preserved(self):
        buf = AudioBuffer.noise(channels=1, frames=4096, sample_rate=48000.0, seed=0)
        result = effects.de_esser(buf, freq=6000.0)
        assert result.channels == 1
        assert result.frames == 4096

    def test_reduces_sibilant_energy(self):
        """De-esser should reduce energy in the sibilant band."""
        buf = AudioBuffer.noise(channels=1, frames=8192, sample_rate=48000.0, seed=0)
        result = effects.de_esser(buf, freq=6000.0, threshold_db=-30.0)
        # Measure energy in sibilant band
        bp_in = effects.bandpass(buf, 6000.0, octaves=2.0)
        bp_out = effects.bandpass(result, 6000.0, octaves=2.0)
        energy_in = np.sum(bp_in.data**2)
        energy_out = np.sum(bp_out.data**2)
        assert energy_out < energy_in

    def test_multichannel(self):
        buf = AudioBuffer.noise(channels=2, frames=4096, sample_rate=48000.0, seed=0)
        result = effects.de_esser(buf)
        assert result.channels == 2


# ---------------------------------------------------------------------------
# Parallel compression
# ---------------------------------------------------------------------------


class TestParallelCompress:
    def test_shape_preserved(self):
        buf = AudioBuffer.noise(channels=1, frames=4096, sample_rate=48000.0, seed=0)
        result = effects.parallel_compress(buf, mix=0.5)
        assert result.channels == 1
        assert result.frames == 4096

    def test_mix_zero_identity(self):
        buf = AudioBuffer.noise(channels=1, frames=4096, sample_rate=48000.0, seed=0)
        result = effects.parallel_compress(buf, mix=0.0)
        np.testing.assert_allclose(result.data, buf.data, atol=1e-5)

    def test_modifies_signal(self):
        buf = AudioBuffer.noise(channels=1, frames=4096, sample_rate=48000.0, seed=0)
        result = effects.parallel_compress(buf, mix=0.5)
        assert not np.allclose(result.data, buf.data)

    def test_multichannel(self):
        buf = AudioBuffer.noise(channels=2, frames=4096, sample_rate=48000.0, seed=0)
        result = effects.parallel_compress(buf, mix=0.5)
        assert result.channels == 2


# ---------------------------------------------------------------------------
# Reverb
# ---------------------------------------------------------------------------


class TestReverb:
    def test_mono_to_stereo(self):
        buf = AudioBuffer.noise(channels=1, frames=4096, sample_rate=48000.0, seed=0)
        result = effects.reverb(buf, preset="hall")
        assert result.channels == 2

    def test_stereo_to_stereo(self):
        buf = AudioBuffer.noise(channels=2, frames=4096, sample_rate=48000.0, seed=0)
        result = effects.reverb(buf, preset="room")
        assert result.channels == 2

    def test_all_presets(self):
        buf = AudioBuffer.noise(channels=1, frames=4096, sample_rate=48000.0, seed=0)
        for preset in ["room", "hall", "plate", "chamber", "cathedral"]:
            result = effects.reverb(buf, preset=preset)
            assert result.channels == 2, f"Failed for preset={preset}"
            assert result.frames == buf.frames, f"Frame mismatch for preset={preset}"

    def test_invalid_preset_raises(self):
        buf = AudioBuffer.noise(channels=1, frames=4096, sample_rate=48000.0, seed=0)
        with pytest.raises(ValueError, match="Unknown reverb preset"):
            effects.reverb(buf, preset="garage")

    def test_mix_zero_dry(self):
        """mix=0 should return the dry signal."""
        buf = AudioBuffer.noise(channels=1, frames=4096, sample_rate=48000.0, seed=0)
        result = effects.reverb(buf, mix=0.0, preset="room")
        # Dry stereo = mono duplicated to both channels
        np.testing.assert_allclose(result.data[0], buf.data[0], atol=1e-5)
        np.testing.assert_allclose(result.data[1], buf.data[0], atol=1e-5)

    def test_mix_one_fully_wet(self):
        """mix=1 should contain no dry signal (different from input)."""
        buf = AudioBuffer.noise(channels=1, frames=4096, sample_rate=48000.0, seed=0)
        result = effects.reverb(buf, mix=1.0, preset="room")
        assert not np.allclose(result.data[0], buf.data[0])

    def test_pre_delay(self):
        """Pre-delay should shift the wet signal onset."""
        buf = AudioBuffer.impulse(channels=1, frames=8192, sample_rate=48000.0)
        no_pd = effects.reverb(buf, mix=1.0, preset="room", pre_delay_ms=0.0)
        with_pd = effects.reverb(buf, mix=1.0, preset="room", pre_delay_ms=50.0)
        # With pre-delay, early samples should have less energy
        early_no_pd = np.sum(no_pd.data[:, :240] ** 2)
        early_with_pd = np.sum(with_pd.data[:, :240] ** 2)
        assert early_with_pd < early_no_pd

    def test_output_frames_match_input(self):
        buf = AudioBuffer.noise(channels=1, frames=4096, sample_rate=48000.0, seed=0)
        result = effects.reverb(buf, preset="hall", pre_delay_ms=20.0)
        assert result.frames == buf.frames


# ---------------------------------------------------------------------------
# Mastering chain
# ---------------------------------------------------------------------------


class TestMaster:
    def test_output_shape(self):
        sr = 48000.0
        frames = int(sr * 3)
        buf = AudioBuffer.noise(channels=1, frames=frames, sample_rate=sr, seed=0)
        result = effects.master(buf, target_lufs=-14.0)
        assert result.channels == 1
        assert result.frames == frames

    def test_loudness_near_target(self):
        sr = 48000.0
        frames = int(sr * 3)
        buf = AudioBuffer.noise(channels=1, frames=frames, sample_rate=sr, seed=0)
        result = effects.master(buf, target_lufs=-14.0)
        measured = analysis.loudness_lufs(result)
        assert abs(measured - (-14.0)) < 2.0

    def test_stereo(self):
        sr = 48000.0
        frames = int(sr * 3)
        buf = AudioBuffer.noise(channels=2, frames=frames, sample_rate=sr, seed=0)
        result = effects.master(buf, target_lufs=-14.0)
        assert result.channels == 2

    def test_dc_block_toggle(self):
        sr = 48000.0
        frames = int(sr * 3)
        buf = AudioBuffer.noise(channels=1, frames=frames, sample_rate=sr, seed=0)
        # Should not raise with dc_block on or off
        r1 = effects.master(buf, dc_block_on=True)
        r2 = effects.master(buf, dc_block_on=False)
        assert r1.frames == frames
        assert r2.frames == frames

    def test_compress_toggle(self):
        sr = 48000.0
        frames = int(sr * 3)
        buf = AudioBuffer.noise(channels=1, frames=frames, sample_rate=sr, seed=0)
        r1 = effects.master(buf, compress_on=True)
        r2 = effects.master(buf, compress_on=False)
        assert not np.allclose(r1.data, r2.data)

    def test_eq_dict(self):
        sr = 48000.0
        frames = int(sr * 3)
        buf = AudioBuffer.noise(channels=1, frames=frames, sample_rate=sr, seed=0)
        result = effects.master(
            buf,
            eq={
                "low_shelf": (200.0, -3.0),
                "high_shelf": (8000.0, 2.0),
                "peak": (2000.0, 1.5, 1.0),
            },
        )
        assert result.frames == frames

    def test_eq_multi_peak(self):
        sr = 48000.0
        frames = int(sr * 3)
        buf = AudioBuffer.noise(channels=1, frames=frames, sample_rate=sr, seed=0)
        result = effects.master(
            buf,
            eq={
                "peak": [(1000.0, 2.0), (4000.0, -1.0)],
            },
        )
        assert result.frames == frames


# ---------------------------------------------------------------------------
# Noise gate
# ---------------------------------------------------------------------------


class TestNoiseGate:
    def test_shape_preserved(self):
        buf = AudioBuffer.noise(channels=1, frames=4096, sample_rate=48000.0, seed=0)
        result = effects.noise_gate(buf, threshold_db=-20.0)
        assert result.channels == 1
        assert result.frames == 4096
        assert result.data.dtype == np.float32

    def test_silence_stays_silent(self):
        buf = AudioBuffer.zeros(1, 4096, sample_rate=48000.0)
        result = effects.noise_gate(buf, threshold_db=-60.0)
        np.testing.assert_array_equal(result.data, 0.0)

    def test_loud_signal_passes(self):
        """Signal well above threshold should pass through mostly unchanged."""
        buf = AudioBuffer.sine(440.0, channels=1, frames=4096, sample_rate=48000.0)
        result = effects.noise_gate(buf, threshold_db=-60.0)
        # Most of the signal should survive (after attack settles)
        energy_ratio = np.sum(result.data**2) / np.sum(buf.data**2)
        assert energy_ratio > 0.9

    def test_quiet_signal_gated(self):
        """Signal below threshold should be heavily attenuated."""
        buf = AudioBuffer.sine(440.0, channels=1, frames=4096, sample_rate=48000.0)
        buf = buf * 0.001  # very quiet
        result = effects.noise_gate(buf, threshold_db=-20.0)
        energy_ratio = np.sum(result.data**2) / (np.sum(buf.data**2) + 1e-20)
        assert energy_ratio < 0.1

    def test_multichannel(self):
        buf = AudioBuffer.noise(channels=2, frames=4096, sample_rate=48000.0, seed=0)
        result = effects.noise_gate(buf, threshold_db=-30.0)
        assert result.channels == 2

    def test_gate_opens_and_closes(self):
        """Signal that transitions from loud to silent should show gating."""
        data = np.zeros((1, 4800), dtype=np.float32)
        t = np.arange(2400, dtype=np.float32) / 48000.0
        data[0, :2400] = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        buf = AudioBuffer(data, sample_rate=48000.0)
        result = effects.noise_gate(buf, threshold_db=-40.0, release=0.01)
        # Tail should be mostly gated
        tail_energy = np.sum(result.data[0, 3000:] ** 2)
        head_energy = np.sum(result.data[0, :2400] ** 2)
        assert tail_energy < head_energy * 0.01


# ---------------------------------------------------------------------------
# Stereo delay
# ---------------------------------------------------------------------------


class TestStereoDelay:
    def test_output_stereo(self):
        buf = AudioBuffer.noise(channels=1, frames=4096, sample_rate=48000.0, seed=0)
        result = effects.stereo_delay(buf)
        assert result.channels == 2

    def test_stereo_input(self):
        buf = AudioBuffer.noise(channels=2, frames=4096, sample_rate=48000.0, seed=0)
        result = effects.stereo_delay(buf)
        assert result.channels == 2
        assert result.frames == 4096

    def test_mix_zero_dry(self):
        buf = AudioBuffer.noise(channels=1, frames=4096, sample_rate=48000.0, seed=0)
        result = effects.stereo_delay(buf, mix=0.0)
        # Dry = mono duplicated to stereo
        np.testing.assert_allclose(result.data[0], buf.data[0], atol=1e-6)
        np.testing.assert_allclose(result.data[1], buf.data[0], atol=1e-6)

    def test_delayed_signal_appears(self):
        """Impulse should produce delayed echo."""
        buf = AudioBuffer.impulse(channels=1, frames=48000, sample_rate=48000.0)
        result = effects.stereo_delay(
            buf, left_ms=10.0, right_ms=20.0, feedback=0.0, mix=1.0
        )
        # Left channel should have a peak around sample 480 (10ms @ 48kHz)
        left_peak = np.argmax(np.abs(result.data[0]))
        assert 470 <= left_peak <= 490
        # Right channel around sample 960 (20ms)
        right_peak = np.argmax(np.abs(result.data[1]))
        assert 950 <= right_peak <= 970

    def test_ping_pong(self):
        """Ping-pong should produce cross-channel echoes."""
        buf = AudioBuffer.impulse(channels=1, frames=48000, sample_rate=48000.0)
        result = effects.stereo_delay(
            buf, left_ms=10.0, right_ms=10.0, feedback=0.5, mix=1.0, ping_pong=True
        )
        # Both channels should have significant energy from cross-feeding
        assert np.max(np.abs(result.data[0])) > 0.1
        assert np.max(np.abs(result.data[1])) > 0.1

    def test_multichannel_raises(self):
        buf = AudioBuffer.noise(channels=3, frames=1024, sample_rate=48000.0, seed=0)
        with pytest.raises(ValueError, match="mono or stereo"):
            effects.stereo_delay(buf)

    def test_feedback_produces_repeats(self):
        """Feedback > 0 should produce decaying echoes."""
        buf = AudioBuffer.impulse(channels=1, frames=48000, sample_rate=48000.0)
        no_fb = effects.stereo_delay(
            buf, left_ms=10.0, right_ms=10.0, feedback=0.0, mix=1.0
        )
        with_fb = effects.stereo_delay(
            buf, left_ms=10.0, right_ms=10.0, feedback=0.5, mix=1.0
        )
        # With feedback should have more total energy
        assert np.sum(with_fb.data**2) > np.sum(no_fb.data**2)


# ---------------------------------------------------------------------------
# Multiband compression
# ---------------------------------------------------------------------------


class TestMultibandCompress:
    def test_shape_preserved(self):
        buf = AudioBuffer.noise(channels=1, frames=4096, sample_rate=48000.0, seed=0)
        result = effects.multiband_compress(buf)
        assert result.channels == 1
        assert result.frames == 4096
        assert result.data.dtype == np.float32

    def test_modifies_signal(self):
        buf = AudioBuffer.noise(channels=1, frames=4096, sample_rate=48000.0, seed=0)
        result = effects.multiband_compress(buf)
        assert not np.allclose(result.data, buf.data)

    def test_multichannel(self):
        buf = AudioBuffer.noise(channels=2, frames=4096, sample_rate=48000.0, seed=0)
        result = effects.multiband_compress(buf)
        assert result.channels == 2

    def test_custom_crossovers(self):
        buf = AudioBuffer.noise(channels=1, frames=4096, sample_rate=48000.0, seed=0)
        result = effects.multiband_compress(
            buf,
            crossover_freqs=[500.0, 5000.0],
            ratios=[2.0, 4.0, 2.0],
            thresholds=[-30.0, -20.0, -15.0],
        )
        assert result.frames == 4096

    def test_single_crossover(self):
        buf = AudioBuffer.noise(channels=1, frames=4096, sample_rate=48000.0, seed=0)
        result = effects.multiband_compress(
            buf,
            crossover_freqs=[1000.0],
            ratios=[2.0, 4.0],
            thresholds=[-20.0, -20.0],
        )
        assert result.frames == 4096

    def test_mismatched_ratios_raises(self):
        buf = AudioBuffer.noise(channels=1, frames=4096, sample_rate=48000.0, seed=0)
        with pytest.raises(ValueError, match="ratios length"):
            effects.multiband_compress(buf, crossover_freqs=[1000.0], ratios=[2.0])

    def test_mismatched_thresholds_raises(self):
        buf = AudioBuffer.noise(channels=1, frames=4096, sample_rate=48000.0, seed=0)
        with pytest.raises(ValueError, match="thresholds length"):
            effects.multiband_compress(
                buf, crossover_freqs=[1000.0], thresholds=[-20.0]
            )


# ---------------------------------------------------------------------------
# Vocal chain
# ---------------------------------------------------------------------------


class TestVocalChain:
    def _voice_like(self, sr=48000.0, seconds=3.0):
        """Noise shaped like a vocal (energy around 200-4000 Hz)."""
        frames = int(sr * seconds)
        buf = AudioBuffer.noise(channels=1, frames=frames, sample_rate=sr, seed=42)
        # Bandpass to vocal range
        buf = effects.lowpass(buf, 4000.0)
        buf = effects.highpass(buf, 200.0)
        return buf

    def test_shape_preserved(self):
        buf = self._voice_like()
        result = effects.vocal_chain(buf)
        assert result.channels == 1
        assert result.frames == buf.frames

    def test_modifies_signal(self):
        buf = self._voice_like()
        result = effects.vocal_chain(buf)
        assert not np.allclose(result.data, buf.data)

    def test_with_target_lufs(self):
        buf = self._voice_like()
        result = effects.vocal_chain(buf, target_lufs=-16.0)
        measured = analysis.loudness_lufs(result)
        assert abs(measured - (-16.0)) < 2.0

    def test_de_ess_toggle(self):
        buf = self._voice_like()
        r1 = effects.vocal_chain(buf, de_ess=True)
        r2 = effects.vocal_chain(buf, de_ess=False)
        assert not np.allclose(r1.data, r2.data)

    def test_compress_toggle(self):
        buf = self._voice_like()
        r1 = effects.vocal_chain(buf, compress_on=True, target_lufs=None)
        r2 = effects.vocal_chain(buf, compress_on=False, target_lufs=None)
        assert not np.allclose(r1.data, r2.data)

    def test_custom_eq(self):
        buf = self._voice_like()
        result = effects.vocal_chain(
            buf,
            eq={"peak": (2000.0, 3.0), "high_shelf": (10000.0, 2.0)},
            target_lufs=None,
        )
        assert result.frames == buf.frames

    def test_multichannel(self):
        sr = 48000.0
        frames = int(sr * 3)
        buf = AudioBuffer.noise(channels=2, frames=frames, sample_rate=sr, seed=0)
        result = effects.vocal_chain(buf, target_lufs=None)
        assert result.channels == 2


# ---------------------------------------------------------------------------
# STK reverb
# ---------------------------------------------------------------------------


class TestStkReverb:
    def test_freeverb_stereo_output(self):
        buf = AudioBuffer.noise(channels=1, frames=4096, sample_rate=48000.0, seed=0)
        result = effects.stk_reverb(buf, algorithm="freeverb")
        assert result.channels == 2
        assert result.frames == 4096

    def test_all_algorithms(self):
        buf = AudioBuffer.noise(channels=1, frames=4096, sample_rate=48000.0, seed=0)
        for algo in ["freeverb", "jcrev", "nrev", "prcrev"]:
            result = effects.stk_reverb(buf, algorithm=algo)
            assert result.channels == 2, f"{algo} wrong channels"
            assert result.frames == buf.frames, f"{algo} wrong frames"

    def test_invalid_algorithm_raises(self):
        buf = AudioBuffer.noise(channels=1, frames=1024, sample_rate=48000.0, seed=0)
        with pytest.raises(ValueError, match="Unknown STK reverb"):
            effects.stk_reverb(buf, algorithm="unknown")

    def test_stereo_input(self):
        buf = AudioBuffer.noise(channels=2, frames=4096, sample_rate=48000.0, seed=0)
        result = effects.stk_reverb(buf, algorithm="nrev")
        assert result.channels == 2


# ---------------------------------------------------------------------------
# STK chorus and echo
# ---------------------------------------------------------------------------


class TestStkEffects:
    def test_chorus_stereo_output(self):
        buf = AudioBuffer.noise(channels=1, frames=4096, sample_rate=48000.0, seed=0)
        result = effects.stk_chorus(buf)
        assert result.channels == 2
        assert result.frames == 4096

    def test_chorus_modifies_signal(self):
        buf = AudioBuffer.noise(channels=1, frames=4096, sample_rate=48000.0, seed=0)
        result = effects.stk_chorus(buf)
        assert not np.allclose(result.data[0], buf.data[0])

    def test_echo_shape(self):
        buf = AudioBuffer.noise(channels=1, frames=4096, sample_rate=48000.0, seed=0)
        result = effects.stk_echo(buf, delay_ms=50.0, mix=0.5)
        assert result.channels == 1
        assert result.frames == 4096

    def test_echo_multichannel(self):
        buf = AudioBuffer.noise(channels=2, frames=4096, sample_rate=48000.0, seed=0)
        result = effects.stk_echo(buf, delay_ms=50.0)
        assert result.channels == 2

    def test_echo_modifies_signal(self):
        buf = AudioBuffer.noise(channels=1, frames=4096, sample_rate=48000.0, seed=0)
        result = effects.stk_echo(buf, delay_ms=100.0, mix=0.5)
        assert not np.allclose(result.data, buf.data)


# ---------------------------------------------------------------------------
# AGC
# ---------------------------------------------------------------------------


class TestAgc:
    def test_boosts_quiet_signal(self):
        """AGC should increase the level of a quiet signal."""
        buf = AudioBuffer.sine(440.0, frames=8192, sample_rate=48000.0) * 0.01
        result = effects.agc(buf, target_level=0.5, attack=0.001, release=0.001)
        # After convergence, output RMS should be higher than input
        in_rms = np.sqrt(np.mean(buf.data[:, -2048:] ** 2))
        out_rms = np.sqrt(np.mean(result.data[:, -2048:] ** 2))
        assert out_rms > in_rms * 2

    def test_attenuates_loud_signal(self):
        """AGC should reduce the level of a loud signal."""
        buf = AudioBuffer.sine(440.0, frames=8192, sample_rate=48000.0) * 2.0
        result = effects.agc(buf, target_level=0.1, attack=0.001, release=0.001)
        in_rms = np.sqrt(np.mean(buf.data[:, -2048:] ** 2))
        out_rms = np.sqrt(np.mean(result.data[:, -2048:] ** 2))
        assert out_rms < in_rms

    def test_shape_preserved(self):
        buf = AudioBuffer.noise(channels=2, frames=4096, sample_rate=48000.0, seed=0)
        result = effects.agc(buf)
        assert result.channels == 2
        assert result.frames == 4096
        assert result.data.dtype == np.float32

    def test_max_gain_clamped(self):
        """AGC on silence should not produce huge output."""
        buf = AudioBuffer.zeros(1, 2048, sample_rate=48000.0)
        result = effects.agc(buf, target_level=1.0, max_gain_db=20.0)
        assert np.max(np.abs(result.data)) < 1e-5

    def test_metadata_preserved(self):
        buf = AudioBuffer.noise(
            channels=1, frames=2048, sample_rate=44100.0, seed=0, label="agc"
        )
        result = effects.agc(buf)
        assert result.sample_rate == 44100.0
        assert result.label == "agc"

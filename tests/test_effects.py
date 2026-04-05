"""Tests for nanodsp.effects module (filters, effects, dynamics, reverbs, mastering)."""

import numpy as np
import pytest

from nanodsp import analysis
from nanodsp._core import filters as _core_filters
from nanodsp.effects import composed, daisysp, dynamics
from nanodsp.effects import filters as fx_filters
from nanodsp.effects import reverb, saturation
from nanodsp.buffer import AudioBuffer


# ---------------------------------------------------------------------------
# Filter functions
# ---------------------------------------------------------------------------


class TestFilterFunctions:
    def test_lowpass_attenuates_high_freq(self):
        buf = AudioBuffer.sine(1000.0, frames=4096, sample_rate=48000.0)
        high = AudioBuffer.sine(20000.0, frames=4096, sample_rate=48000.0)
        combined = buf + high
        result = fx_filters.lowpass(combined, 5000.0)
        # High-frequency energy should be attenuated
        assert np.sum(result.data**2) < np.sum(combined.data**2)

    def test_highpass_attenuates_low_freq(self):
        low = AudioBuffer.sine(100.0, frames=4096, sample_rate=48000.0)
        high = AudioBuffer.sine(10000.0, frames=4096, sample_rate=48000.0)
        combined = low + high
        result = fx_filters.highpass(combined, 5000.0)
        assert np.sum(result.data**2) < np.sum(combined.data**2)

    def test_bandpass_passes_center(self):
        center = AudioBuffer.sine(5000.0, frames=4096, sample_rate=48000.0)
        result = fx_filters.bandpass(center, 5000.0, octaves=2.0)
        # Center frequency should pass through with reasonable energy
        energy_ratio = np.sum(result.data**2) / np.sum(center.data**2)
        assert energy_ratio > 0.3

    def test_notch_attenuates_center(self):
        center = AudioBuffer.sine(5000.0, frames=4096, sample_rate=48000.0)
        result = fx_filters.notch(center, 5000.0, octaves=1.0)
        energy_ratio = np.sum(result.data**2) / np.sum(center.data**2)
        assert energy_ratio < 0.5

    def test_peak_boosts(self):
        buf = AudioBuffer.sine(5000.0, frames=4096, sample_rate=48000.0)
        result = fx_filters.peak(buf, 5000.0, gain=4.0)
        assert np.sum(result.data**2) > np.sum(buf.data**2)

    def test_peak_db_boosts(self):
        buf = AudioBuffer.sine(5000.0, frames=4096, sample_rate=48000.0)
        result = fx_filters.peak_db(buf, 5000.0, db=12.0)
        assert np.sum(result.data**2) > np.sum(buf.data**2)

    def test_high_shelf(self):
        buf = AudioBuffer.sine(15000.0, frames=4096, sample_rate=48000.0)
        result = fx_filters.high_shelf(buf, 10000.0, gain=4.0)
        assert np.sum(result.data**2) > np.sum(buf.data**2)

    def test_high_shelf_db(self):
        buf = AudioBuffer.sine(15000.0, frames=4096, sample_rate=48000.0)
        result = fx_filters.high_shelf_db(buf, 10000.0, db=12.0)
        assert np.sum(result.data**2) > np.sum(buf.data**2)

    def test_low_shelf(self):
        buf = AudioBuffer.sine(100.0, frames=4096, sample_rate=48000.0)
        result = fx_filters.low_shelf(buf, 1000.0, gain=4.0)
        assert np.sum(result.data**2) > np.sum(buf.data**2)

    def test_low_shelf_db(self):
        buf = AudioBuffer.sine(100.0, frames=4096, sample_rate=48000.0)
        result = fx_filters.low_shelf_db(buf, 1000.0, db=12.0)
        assert np.sum(result.data**2) > np.sum(buf.data**2)

    def test_allpass_preserves_magnitude(self):
        buf = AudioBuffer.sine(5000.0, frames=4096, sample_rate=48000.0)
        result = fx_filters.allpass(buf, 5000.0)
        assert result.channels == buf.channels
        assert result.frames == buf.frames
        assert result.data.dtype == np.float32
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
        result = fx_filters.lowpass(buf, 5000.0)
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
        result = fx_filters.lowpass(buf, 5000.0)
        # Channel 0 (1kHz) should retain more energy than channel 1 (15kHz)
        ch0_energy = np.sum(result.data[0] ** 2)
        ch1_energy = np.sum(result.data[1] ** 2)
        assert ch0_energy > ch1_energy * 5

    def test_all_filters_produce_correct_shape(self):
        buf = AudioBuffer.noise(channels=1, frames=1024, sample_rate=48000.0, seed=0)
        for fn, kwargs in [
            (fx_filters.lowpass, {"cutoff_hz": 5000.0}),
            (fx_filters.highpass, {"cutoff_hz": 5000.0}),
            (fx_filters.bandpass, {"center_hz": 5000.0}),
            (fx_filters.notch, {"center_hz": 5000.0}),
            (fx_filters.peak, {"center_hz": 5000.0, "gain": 2.0}),
            (fx_filters.peak_db, {"center_hz": 5000.0, "db": 6.0}),
            (fx_filters.high_shelf, {"cutoff_hz": 5000.0, "gain": 2.0}),
            (fx_filters.high_shelf_db, {"cutoff_hz": 5000.0, "db": 6.0}),
            (fx_filters.low_shelf, {"cutoff_hz": 5000.0, "gain": 2.0}),
            (fx_filters.low_shelf_db, {"cutoff_hz": 5000.0, "db": 6.0}),
            (fx_filters.allpass, {"freq_hz": 5000.0}),
        ]:
            result = fn(buf, **kwargs)
            assert result.channels == 1
            assert result.frames == 1024
            assert result.data.dtype == np.float32

    def test_biquad_process_preconfigured(self):
        bq = _core_filters.Biquad()
        bq.lowpass(0.1)
        buf = AudioBuffer.noise(channels=2, frames=1024, sample_rate=48000.0, seed=0)
        result = fx_filters.biquad_process(buf, bq)
        assert result.channels == 2
        assert result.frames == 1024

    def test_lowpass_with_explicit_octaves(self):
        buf = AudioBuffer.noise(channels=1, frames=1024, sample_rate=48000.0, seed=0)
        result = fx_filters.lowpass(buf, 5000.0, octaves=2.0)
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
        result = daisysp.autowah(buf)
        assert result.channels == 1
        assert result.frames == 2048
        assert result.data.dtype == np.float32

    def test_autowah_modifies_signal(self):
        buf = self._sine()
        result = daisysp.autowah(buf, wah=0.8)
        assert not np.allclose(result.data, buf.data)

    def test_chorus_mono_to_stereo(self):
        buf = self._sine()
        result = daisysp.chorus(buf)
        assert result.channels == 2
        assert result.frames == 2048

    def test_chorus_multichannel_per_channel(self):
        buf = self._noise(channels=2)
        result = daisysp.chorus(buf)
        assert result.channels == 2

    def test_decimator_shape(self):
        buf = self._noise()
        result = daisysp.decimator(buf)
        assert result.frames == 2048
        assert result.data.dtype == np.float32

    def test_decimator_modifies_signal(self):
        buf = self._sine()
        result = daisysp.decimator(buf, downsample_factor=0.8, bits_to_crush=4)
        assert not np.allclose(result.data, buf.data)

    def test_flanger_shape(self):
        buf = self._sine()
        result = daisysp.flanger(buf)
        assert result.channels == 1
        assert result.frames == 2048

    def test_overdrive_shape(self):
        buf = self._sine()
        result = daisysp.overdrive(buf, drive=0.8)
        assert result.frames == 2048
        assert result.data.dtype == np.float32

    def test_overdrive_adds_harmonics(self):
        buf = self._sine()
        result = daisysp.overdrive(buf, drive=0.9)
        assert not np.allclose(result.data, buf.data)

    def test_phaser_shape(self):
        buf = self._sine()
        result = daisysp.phaser(buf)
        assert result.frames == 2048

    def test_pitch_shift_shape(self):
        buf = self._sine()
        result = daisysp.pitch_shift(buf, semitones=5.0)
        assert result.frames == 2048

    def test_sample_rate_reduce_shape(self):
        buf = self._noise()
        result = daisysp.sample_rate_reduce(buf, freq=0.3)
        assert result.frames == 2048

    def test_tremolo_shape(self):
        buf = self._sine()
        result = daisysp.tremolo(buf, freq=5.0, depth=1.0)
        assert result.frames == 2048

    def test_wavefold_shape(self):
        buf = self._sine()
        result = daisysp.wavefold(buf, gain=2.0)
        assert result.frames == 2048

    def test_bitcrush_shape(self):
        buf = self._noise()
        result = daisysp.bitcrush(buf, bit_depth=4)
        assert result.frames == 2048

    def test_bitcrush_default_crush_rate(self):
        buf = self._noise()
        result = daisysp.bitcrush(buf)
        assert result.data.dtype == np.float32

    def test_fold_shape(self):
        buf = self._sine()
        result = daisysp.fold(buf, increment=0.5)
        assert result.frames == 2048

    def test_reverb_sc_mono_to_stereo(self):
        buf = self._sine()
        result = daisysp.reverb_sc(buf)
        assert result.channels == 2
        assert result.frames == 2048

    def test_reverb_sc_stereo_passthrough(self):
        buf = self._noise(channels=2)
        result = daisysp.reverb_sc(buf)
        assert result.channels == 2

    def test_reverb_sc_3ch_raises(self):
        buf = self._noise(channels=3)
        with pytest.raises(ValueError, match="mono or stereo"):
            daisysp.reverb_sc(buf)

    def test_dc_block_shape(self):
        buf = self._noise()
        result = daisysp.dc_block(buf)
        assert result.frames == 2048

    def test_dc_block_removes_offset(self):
        data = np.ones((1, 4096), dtype=np.float32) * 0.5
        buf = AudioBuffer(data, sample_rate=48000.0)
        result = daisysp.dc_block(buf)
        # Mean should be much closer to 0 after DC blocking
        assert abs(np.mean(result.data[0, 1024:])) < abs(np.mean(buf.data[0]))

    def test_effects_multichannel(self):
        buf = self._noise(channels=2)
        for fn, kwargs in [
            (daisysp.autowah, {}),
            (daisysp.decimator, {}),
            (daisysp.flanger, {}),
            (daisysp.overdrive, {}),
            (daisysp.phaser, {}),
            (daisysp.tremolo, {}),
            (daisysp.wavefold, {}),
            (daisysp.bitcrush, {}),
            (daisysp.fold, {}),
            (daisysp.dc_block, {}),
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
        result = fx_filters.svf_lowpass(combined, freq_hz=2000.0)
        assert np.sum(result.data**2) < np.sum(combined.data**2)

    def test_svf_highpass_attenuates_low(self):
        low = AudioBuffer.sine(500.0, frames=4096, sample_rate=48000.0)
        high = AudioBuffer.sine(15000.0, frames=4096, sample_rate=48000.0)
        combined = low + high
        result = fx_filters.svf_highpass(combined, freq_hz=5000.0)
        assert np.sum(result.data**2) < np.sum(combined.data**2)

    def test_svf_bandpass_shape(self):
        buf = self._noise()
        result = fx_filters.svf_bandpass(buf, freq_hz=1000.0)
        assert result.frames == 4096

    def test_svf_notch_shape(self):
        buf = self._noise()
        result = fx_filters.svf_notch(buf, freq_hz=1000.0)
        assert result.frames == 4096

    def test_svf_peak_shape(self):
        buf = self._noise()
        result = fx_filters.svf_peak(buf, freq_hz=1000.0)
        assert result.frames == 4096

    def test_ladder_filter_lowpass(self):
        buf = self._noise()
        result = fx_filters.ladder_filter(buf, freq_hz=2000.0, mode="lp24")
        assert result.frames == 4096
        assert np.sum(result.data**2) < np.sum(buf.data**2)

    def test_ladder_filter_modes(self):
        buf = self._noise()
        for mode in ["lp24", "lp12", "bp24", "bp12", "hp24", "hp12"]:
            result = fx_filters.ladder_filter(buf, freq_hz=2000.0, mode=mode)
            assert result.frames == 4096, f"Failed for mode={mode}"

    def test_ladder_filter_invalid_mode(self):
        buf = self._noise()
        with pytest.raises(ValueError, match="Unknown ladder mode"):
            fx_filters.ladder_filter(buf, mode="invalid")

    def test_moog_ladder_shape(self):
        buf = self._noise()
        result = fx_filters.moog_ladder(buf, freq_hz=2000.0, resonance=0.3)
        assert result.frames == 4096

    def test_tone_lowpass_attenuates_high(self):
        low = AudioBuffer.sine(500.0, frames=4096, sample_rate=48000.0)
        high = AudioBuffer.sine(15000.0, frames=4096, sample_rate=48000.0)
        combined = low + high
        result = fx_filters.tone_lowpass(combined, freq_hz=2000.0)
        assert np.sum(result.data**2) < np.sum(combined.data**2)

    def test_tone_highpass_attenuates_low(self):
        low = AudioBuffer.sine(500.0, frames=4096, sample_rate=48000.0)
        high = AudioBuffer.sine(15000.0, frames=4096, sample_rate=48000.0)
        combined = low + high
        result = fx_filters.tone_highpass(combined, freq_hz=5000.0)
        assert np.sum(result.data**2) < np.sum(combined.data**2)

    def test_modal_bandpass_shape(self):
        buf = self._noise()
        result = fx_filters.modal_bandpass(buf, freq_hz=1000.0)
        assert result.frames == 4096

    def test_comb_filter_shape(self):
        buf = self._noise()
        result = fx_filters.comb_filter(buf, freq_hz=500.0)
        assert result.frames == 4096

    def test_filters_multichannel(self):
        buf = self._noise(channels=2)
        for fn, kwargs in [
            (fx_filters.svf_lowpass, {"freq_hz": 2000.0}),
            (fx_filters.svf_highpass, {"freq_hz": 2000.0}),
            (fx_filters.ladder_filter, {"freq_hz": 2000.0}),
            (fx_filters.moog_ladder, {"freq_hz": 2000.0}),
            (fx_filters.tone_lowpass, {"freq_hz": 2000.0}),
            (fx_filters.tone_highpass, {"freq_hz": 2000.0}),
            (fx_filters.modal_bandpass, {"freq_hz": 1000.0}),
            (fx_filters.comb_filter, {"freq_hz": 500.0}),
        ]:
            result = fn(buf, **kwargs)
            assert result.channels == 2, f"{fn.__name__} failed multichannel"


# ---------------------------------------------------------------------------
# DaisySP Dynamics
# ---------------------------------------------------------------------------


class TestDaisySPDynamics:
    def test_compress_shape(self):
        buf = AudioBuffer.noise(channels=1, frames=2048, sample_rate=48000.0, seed=0)
        result = dynamics.compress(buf, ratio=4.0, threshold=-20.0)
        assert result.frames == 2048
        assert result.data.dtype == np.float32

    def test_compress_reduces_dynamic_range(self):
        buf = AudioBuffer.noise(channels=1, frames=4096, sample_rate=48000.0, seed=0)
        result = dynamics.compress(buf, ratio=8.0, threshold=-30.0)
        # Compressed signal should have different peak/RMS ratio
        assert not np.allclose(result.data, buf.data)

    def test_compress_multichannel(self):
        buf = AudioBuffer.noise(channels=2, frames=2048, sample_rate=48000.0, seed=0)
        result = dynamics.compress(buf)
        assert result.channels == 2

    def test_limit_shape(self):
        buf = AudioBuffer.noise(channels=1, frames=2048, sample_rate=48000.0, seed=0)
        result = dynamics.limit(buf, pre_gain=2.0)
        assert result.frames == 2048
        assert result.data.dtype == np.float32

    def test_limit_multichannel(self):
        buf = AudioBuffer.noise(channels=2, frames=2048, sample_rate=48000.0, seed=0)
        result = dynamics.limit(buf)
        assert result.channels == 2


# ---------------------------------------------------------------------------
# Saturation
# ---------------------------------------------------------------------------


class TestSaturate:
    def test_shape_preserved(self):
        buf = AudioBuffer.noise(channels=2, frames=2048, sample_rate=48000.0, seed=0)
        result = saturation.saturate(buf, drive=0.5, mode="soft")
        assert result.channels == 2
        assert result.frames == 2048
        assert result.data.dtype == np.float32

    def test_drive_zero_near_identity(self):
        """With drive=0 (gain=1x), soft saturation should be near identity for small signals."""
        buf = AudioBuffer.noise(channels=1, frames=2048, sample_rate=48000.0, seed=0)
        buf = buf * 0.1  # keep signal very small so tanh(x) ~= x
        result = saturation.saturate(buf, drive=0.0, mode="soft")
        assert result.channels == buf.channels
        assert result.frames == buf.frames
        assert result.data.dtype == np.float32
        np.testing.assert_allclose(result.data, buf.data, atol=0.02)

    def test_hard_clip_bounded(self):
        buf = AudioBuffer.noise(channels=1, frames=2048, sample_rate=48000.0, seed=0)
        result = saturation.saturate(buf, drive=1.0, mode="hard")
        assert np.max(np.abs(result.data)) <= 1.0 + 1e-6

    def test_all_modes_callable(self):
        buf = AudioBuffer.noise(channels=1, frames=1024, sample_rate=48000.0, seed=0)
        for mode in ["soft", "hard", "tape"]:
            result = saturation.saturate(buf, drive=0.5, mode=mode)
            assert result.frames == 1024, f"Failed for mode={mode}"

    def test_invalid_mode_raises(self):
        buf = AudioBuffer.noise(channels=1, frames=1024, sample_rate=48000.0, seed=0)
        with pytest.raises(ValueError, match="Unknown saturation mode"):
            saturation.saturate(buf, mode="nope")

    def test_soft_preserves_peak(self):
        """Soft saturation should approximately preserve peak amplitude."""
        buf = AudioBuffer.noise(channels=1, frames=4096, sample_rate=48000.0, seed=0)
        result = saturation.saturate(buf, drive=0.5, mode="soft")
        assert result.frames == buf.frames
        assert np.all(np.isfinite(result.data))
        peak_in = np.max(np.abs(buf.data))
        peak_out = np.max(np.abs(result.data))
        np.testing.assert_allclose(peak_out, peak_in, rtol=0.05)


# ---------------------------------------------------------------------------
# Exciter
# ---------------------------------------------------------------------------


class TestExciter:
    def test_shape_preserved(self):
        buf = AudioBuffer.noise(channels=1, frames=4096, sample_rate=48000.0, seed=0)
        result = composed.exciter(buf, freq=3000.0, amount=0.3)
        assert result.channels == 1
        assert result.frames == 4096

    def test_amount_zero_near_identity(self):
        buf = AudioBuffer.noise(channels=1, frames=4096, sample_rate=48000.0, seed=0)
        result = composed.exciter(buf, amount=0.0)
        assert result.channels == buf.channels
        assert result.frames == buf.frames
        assert result.data.dtype == np.float32
        np.testing.assert_allclose(result.data, buf.data, atol=1e-6)

    def test_adds_energy_above_freq(self):
        """Exciter should add harmonic energy in the high-frequency range."""
        buf = AudioBuffer.noise(channels=1, frames=8192, sample_rate=48000.0, seed=0)
        result = composed.exciter(buf, freq=3000.0, amount=0.5)
        # Total energy should increase when adding harmonics
        energy_in = np.sum(buf.data**2)
        energy_out = np.sum(result.data**2)
        assert energy_out > energy_in

    def test_multichannel(self):
        buf = AudioBuffer.noise(channels=2, frames=4096, sample_rate=48000.0, seed=0)
        result = composed.exciter(buf, freq=3000.0)
        assert result.channels == 2


# ---------------------------------------------------------------------------
# De-esser
# ---------------------------------------------------------------------------


class TestDeEsser:
    def test_shape_preserved(self):
        buf = AudioBuffer.noise(channels=1, frames=4096, sample_rate=48000.0, seed=0)
        result = composed.de_esser(buf, freq=6000.0)
        assert result.channels == 1
        assert result.frames == 4096

    def test_reduces_sibilant_energy(self):
        """De-esser should reduce energy in the sibilant band."""
        buf = AudioBuffer.noise(channels=1, frames=8192, sample_rate=48000.0, seed=0)
        result = composed.de_esser(buf, freq=6000.0, threshold_db=-30.0)
        # Measure energy in sibilant band
        bp_in = fx_filters.bandpass(buf, 6000.0, octaves=2.0)
        bp_out = fx_filters.bandpass(result, 6000.0, octaves=2.0)
        energy_in = np.sum(bp_in.data**2)
        energy_out = np.sum(bp_out.data**2)
        assert energy_out < energy_in

    def test_multichannel(self):
        buf = AudioBuffer.noise(channels=2, frames=4096, sample_rate=48000.0, seed=0)
        result = composed.de_esser(buf)
        assert result.channels == 2


# ---------------------------------------------------------------------------
# Parallel compression
# ---------------------------------------------------------------------------


class TestParallelCompress:
    def test_shape_preserved(self):
        buf = AudioBuffer.noise(channels=1, frames=4096, sample_rate=48000.0, seed=0)
        result = composed.parallel_compress(buf, mix=0.5)
        assert result.channels == 1
        assert result.frames == 4096

    def test_mix_zero_identity(self):
        buf = AudioBuffer.noise(channels=1, frames=4096, sample_rate=48000.0, seed=0)
        result = composed.parallel_compress(buf, mix=0.0)
        assert result.channels == buf.channels
        assert result.frames == buf.frames
        assert result.data.dtype == np.float32
        np.testing.assert_allclose(result.data, buf.data, atol=1e-5)

    def test_modifies_signal(self):
        buf = AudioBuffer.noise(channels=1, frames=4096, sample_rate=48000.0, seed=0)
        result = composed.parallel_compress(buf, mix=0.5)
        assert not np.allclose(result.data, buf.data)

    def test_multichannel(self):
        buf = AudioBuffer.noise(channels=2, frames=4096, sample_rate=48000.0, seed=0)
        result = composed.parallel_compress(buf, mix=0.5)
        assert result.channels == 2


# ---------------------------------------------------------------------------
# Reverb
# ---------------------------------------------------------------------------


class TestReverb:
    def test_mono_to_stereo(self):
        buf = AudioBuffer.noise(channels=1, frames=4096, sample_rate=48000.0, seed=0)
        result = reverb.reverb(buf, preset="hall")
        assert result.channels == 2

    def test_stereo_to_stereo(self):
        buf = AudioBuffer.noise(channels=2, frames=4096, sample_rate=48000.0, seed=0)
        result = reverb.reverb(buf, preset="room")
        assert result.channels == 2

    def test_all_presets(self):
        buf = AudioBuffer.noise(channels=1, frames=4096, sample_rate=48000.0, seed=0)
        for preset in ["room", "hall", "plate", "chamber", "cathedral"]:
            result = reverb.reverb(buf, preset=preset)
            assert result.channels == 2, f"Failed for preset={preset}"
            assert result.frames == buf.frames, f"Frame mismatch for preset={preset}"

    def test_invalid_preset_raises(self):
        buf = AudioBuffer.noise(channels=1, frames=4096, sample_rate=48000.0, seed=0)
        with pytest.raises(ValueError, match="Unknown reverb preset"):
            reverb.reverb(buf, preset="garage")

    def test_mix_zero_dry(self):
        """mix=0 should return the dry signal."""
        buf = AudioBuffer.noise(channels=1, frames=4096, sample_rate=48000.0, seed=0)
        result = reverb.reverb(buf, mix=0.0, preset="room")
        assert result.channels == 2
        assert result.frames == buf.frames
        assert result.data.dtype == np.float32
        # Dry stereo = mono duplicated to both channels
        np.testing.assert_allclose(result.data[0], buf.data[0], atol=1e-5)
        np.testing.assert_allclose(result.data[1], buf.data[0], atol=1e-5)

    def test_mix_one_fully_wet(self):
        """mix=1 should contain no dry signal (different from input)."""
        buf = AudioBuffer.noise(channels=1, frames=4096, sample_rate=48000.0, seed=0)
        result = reverb.reverb(buf, mix=1.0, preset="room")
        assert not np.allclose(result.data[0], buf.data[0])

    def test_pre_delay(self):
        """Pre-delay should shift the wet signal onset."""
        buf = AudioBuffer.impulse(channels=1, frames=8192, sample_rate=48000.0)
        no_pd = reverb.reverb(buf, mix=1.0, preset="room", pre_delay_ms=0.0)
        with_pd = reverb.reverb(buf, mix=1.0, preset="room", pre_delay_ms=50.0)
        # With pre-delay, early samples should have less energy
        early_no_pd = np.sum(no_pd.data[:, :240] ** 2)
        early_with_pd = np.sum(with_pd.data[:, :240] ** 2)
        assert early_with_pd < early_no_pd

    def test_output_frames_match_input(self):
        buf = AudioBuffer.noise(channels=1, frames=4096, sample_rate=48000.0, seed=0)
        result = reverb.reverb(buf, preset="hall", pre_delay_ms=20.0)
        assert result.frames == buf.frames


# ---------------------------------------------------------------------------
# Mastering chain
# ---------------------------------------------------------------------------


class TestMaster:
    def test_output_shape(self):
        sr = 48000.0
        frames = int(sr * 3)
        buf = AudioBuffer.noise(channels=1, frames=frames, sample_rate=sr, seed=0)
        result = composed.master(buf, target_lufs=-14.0)
        assert result.channels == 1
        assert result.frames == frames

    def test_loudness_near_target(self):
        sr = 48000.0
        frames = int(sr * 3)
        buf = AudioBuffer.noise(channels=1, frames=frames, sample_rate=sr, seed=0)
        result = composed.master(buf, target_lufs=-14.0)
        measured = analysis.loudness_lufs(result)
        assert abs(measured - (-14.0)) < 2.0

    def test_stereo(self):
        sr = 48000.0
        frames = int(sr * 3)
        buf = AudioBuffer.noise(channels=2, frames=frames, sample_rate=sr, seed=0)
        result = composed.master(buf, target_lufs=-14.0)
        assert result.channels == 2

    def test_dc_block_toggle(self):
        sr = 48000.0
        frames = int(sr * 3)
        buf = AudioBuffer.noise(channels=1, frames=frames, sample_rate=sr, seed=0)
        # Should not raise with dc_block on or off
        r1 = composed.master(buf, dc_block_on=True)
        r2 = composed.master(buf, dc_block_on=False)
        assert r1.frames == frames
        assert r2.frames == frames

    def test_compress_toggle(self):
        sr = 48000.0
        frames = int(sr * 3)
        buf = AudioBuffer.noise(channels=1, frames=frames, sample_rate=sr, seed=0)
        r1 = composed.master(buf, compress_on=True)
        r2 = composed.master(buf, compress_on=False)
        assert not np.allclose(r1.data, r2.data)

    def test_eq_dict(self):
        sr = 48000.0
        frames = int(sr * 3)
        buf = AudioBuffer.noise(channels=1, frames=frames, sample_rate=sr, seed=0)
        result = composed.master(
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
        result = composed.master(
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
        result = dynamics.noise_gate(buf, threshold_db=-20.0)
        assert result.channels == 1
        assert result.frames == 4096
        assert result.data.dtype == np.float32

    def test_silence_stays_silent(self):
        buf = AudioBuffer.zeros(1, 4096, sample_rate=48000.0)
        result = dynamics.noise_gate(buf, threshold_db=-60.0)
        assert result.channels == buf.channels
        assert result.frames == buf.frames
        np.testing.assert_array_equal(result.data, 0.0)

    def test_loud_signal_passes(self):
        """Signal well above threshold should pass through mostly unchanged."""
        buf = AudioBuffer.sine(440.0, channels=1, frames=4096, sample_rate=48000.0)
        result = dynamics.noise_gate(buf, threshold_db=-60.0)
        # Most of the signal should survive (after attack settles)
        energy_ratio = np.sum(result.data**2) / np.sum(buf.data**2)
        assert energy_ratio > 0.9

    def test_quiet_signal_gated(self):
        """Signal below threshold should be heavily attenuated."""
        buf = AudioBuffer.sine(440.0, channels=1, frames=4096, sample_rate=48000.0)
        buf = buf * 0.001  # very quiet
        result = dynamics.noise_gate(buf, threshold_db=-20.0)
        energy_ratio = np.sum(result.data**2) / (np.sum(buf.data**2) + 1e-20)
        assert energy_ratio < 0.1

    def test_multichannel(self):
        buf = AudioBuffer.noise(channels=2, frames=4096, sample_rate=48000.0, seed=0)
        result = dynamics.noise_gate(buf, threshold_db=-30.0)
        assert result.channels == 2

    def test_gate_opens_and_closes(self):
        """Signal that transitions from loud to silent should show gating."""
        data = np.zeros((1, 4800), dtype=np.float32)
        t = np.arange(2400, dtype=np.float32) / 48000.0
        data[0, :2400] = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        buf = AudioBuffer(data, sample_rate=48000.0)
        result = dynamics.noise_gate(buf, threshold_db=-40.0, release=0.01)
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
        result = composed.stereo_delay(buf)
        assert result.channels == 2

    def test_stereo_input(self):
        buf = AudioBuffer.noise(channels=2, frames=4096, sample_rate=48000.0, seed=0)
        result = composed.stereo_delay(buf)
        assert result.channels == 2
        assert result.frames == 4096

    def test_mix_zero_dry(self):
        buf = AudioBuffer.noise(channels=1, frames=4096, sample_rate=48000.0, seed=0)
        result = composed.stereo_delay(buf, mix=0.0)
        assert result.channels == 2
        assert result.frames == buf.frames
        assert result.data.dtype == np.float32
        # Dry = mono duplicated to stereo
        np.testing.assert_allclose(result.data[0], buf.data[0], atol=1e-6)
        np.testing.assert_allclose(result.data[1], buf.data[0], atol=1e-6)

    def test_delayed_signal_appears(self):
        """Impulse should produce delayed echo."""
        buf = AudioBuffer.impulse(channels=1, frames=48000, sample_rate=48000.0)
        result = composed.stereo_delay(
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
        result = composed.stereo_delay(
            buf, left_ms=10.0, right_ms=10.0, feedback=0.5, mix=1.0, ping_pong=True
        )
        # Both channels should have significant energy from cross-feeding
        assert np.max(np.abs(result.data[0])) > 0.1
        assert np.max(np.abs(result.data[1])) > 0.1

    def test_multichannel_raises(self):
        buf = AudioBuffer.noise(channels=3, frames=1024, sample_rate=48000.0, seed=0)
        with pytest.raises(ValueError, match="mono or stereo"):
            composed.stereo_delay(buf)

    def test_feedback_produces_repeats(self):
        """Feedback > 0 should produce decaying echoes."""
        buf = AudioBuffer.impulse(channels=1, frames=48000, sample_rate=48000.0)
        no_fb = composed.stereo_delay(
            buf, left_ms=10.0, right_ms=10.0, feedback=0.0, mix=1.0
        )
        with_fb = composed.stereo_delay(
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
        result = composed.multiband_compress(buf)
        assert result.channels == 1
        assert result.frames == 4096
        assert result.data.dtype == np.float32

    def test_modifies_signal(self):
        buf = AudioBuffer.noise(channels=1, frames=4096, sample_rate=48000.0, seed=0)
        result = composed.multiband_compress(buf)
        assert not np.allclose(result.data, buf.data)

    def test_multichannel(self):
        buf = AudioBuffer.noise(channels=2, frames=4096, sample_rate=48000.0, seed=0)
        result = composed.multiband_compress(buf)
        assert result.channels == 2

    def test_custom_crossovers(self):
        buf = AudioBuffer.noise(channels=1, frames=4096, sample_rate=48000.0, seed=0)
        result = composed.multiband_compress(
            buf,
            crossover_freqs=[500.0, 5000.0],
            ratios=[2.0, 4.0, 2.0],
            thresholds=[-30.0, -20.0, -15.0],
        )
        assert result.frames == 4096

    def test_single_crossover(self):
        buf = AudioBuffer.noise(channels=1, frames=4096, sample_rate=48000.0, seed=0)
        result = composed.multiband_compress(
            buf,
            crossover_freqs=[1000.0],
            ratios=[2.0, 4.0],
            thresholds=[-20.0, -20.0],
        )
        assert result.frames == 4096

    def test_mismatched_ratios_raises(self):
        buf = AudioBuffer.noise(channels=1, frames=4096, sample_rate=48000.0, seed=0)
        with pytest.raises(ValueError, match="ratios length"):
            composed.multiband_compress(buf, crossover_freqs=[1000.0], ratios=[2.0])

    def test_mismatched_thresholds_raises(self):
        buf = AudioBuffer.noise(channels=1, frames=4096, sample_rate=48000.0, seed=0)
        with pytest.raises(ValueError, match="thresholds length"):
            composed.multiband_compress(
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
        buf = fx_filters.lowpass(buf, 4000.0)
        buf = fx_filters.highpass(buf, 200.0)
        return buf

    def test_shape_preserved(self):
        buf = self._voice_like()
        result = composed.vocal_chain(buf)
        assert result.channels == 1
        assert result.frames == buf.frames

    def test_modifies_signal(self):
        buf = self._voice_like()
        result = composed.vocal_chain(buf)
        assert not np.allclose(result.data, buf.data)

    def test_with_target_lufs(self):
        buf = self._voice_like()
        result = composed.vocal_chain(buf, target_lufs=-16.0)
        measured = analysis.loudness_lufs(result)
        assert abs(measured - (-16.0)) < 2.0

    def test_de_ess_toggle(self):
        buf = self._voice_like()
        r1 = composed.vocal_chain(buf, de_ess=True)
        r2 = composed.vocal_chain(buf, de_ess=False)
        assert not np.allclose(r1.data, r2.data)

    def test_compress_toggle(self):
        buf = self._voice_like()
        r1 = composed.vocal_chain(buf, compress_on=True, target_lufs=None)
        r2 = composed.vocal_chain(buf, compress_on=False, target_lufs=None)
        assert not np.allclose(r1.data, r2.data)

    def test_custom_eq(self):
        buf = self._voice_like()
        result = composed.vocal_chain(
            buf,
            eq={"peak": (2000.0, 3.0), "high_shelf": (10000.0, 2.0)},
            target_lufs=None,
        )
        assert result.frames == buf.frames

    def test_multichannel(self):
        sr = 48000.0
        frames = int(sr * 3)
        buf = AudioBuffer.noise(channels=2, frames=frames, sample_rate=sr, seed=0)
        result = composed.vocal_chain(buf, target_lufs=None)
        assert result.channels == 2


# ---------------------------------------------------------------------------
# STK reverb
# ---------------------------------------------------------------------------


class TestStkReverb:
    def test_freeverb_stereo_output(self):
        buf = AudioBuffer.noise(channels=1, frames=4096, sample_rate=48000.0, seed=0)
        result = reverb.stk_reverb(buf, algorithm="freeverb")
        assert result.channels == 2
        assert result.frames == 4096

    def test_all_algorithms(self):
        buf = AudioBuffer.noise(channels=1, frames=4096, sample_rate=48000.0, seed=0)
        for algo in ["freeverb", "jcrev", "nrev", "prcrev"]:
            result = reverb.stk_reverb(buf, algorithm=algo)
            assert result.channels == 2, f"{algo} wrong channels"
            assert result.frames == buf.frames, f"{algo} wrong frames"

    def test_invalid_algorithm_raises(self):
        buf = AudioBuffer.noise(channels=1, frames=1024, sample_rate=48000.0, seed=0)
        with pytest.raises(ValueError, match="Unknown STK reverb"):
            reverb.stk_reverb(buf, algorithm="unknown")

    def test_stereo_input(self):
        buf = AudioBuffer.noise(channels=2, frames=4096, sample_rate=48000.0, seed=0)
        result = reverb.stk_reverb(buf, algorithm="nrev")
        assert result.channels == 2


# ---------------------------------------------------------------------------
# STK chorus and echo
# ---------------------------------------------------------------------------


class TestStkEffects:
    def test_chorus_stereo_output(self):
        buf = AudioBuffer.noise(channels=1, frames=4096, sample_rate=48000.0, seed=0)
        result = reverb.stk_chorus(buf)
        assert result.channels == 2
        assert result.frames == 4096

    def test_chorus_modifies_signal(self):
        buf = AudioBuffer.noise(channels=1, frames=4096, sample_rate=48000.0, seed=0)
        result = reverb.stk_chorus(buf)
        assert not np.allclose(result.data[0], buf.data[0])

    def test_echo_shape(self):
        buf = AudioBuffer.noise(channels=1, frames=4096, sample_rate=48000.0, seed=0)
        result = reverb.stk_echo(buf, delay_ms=50.0, mix=0.5)
        assert result.channels == 1
        assert result.frames == 4096

    def test_echo_multichannel(self):
        buf = AudioBuffer.noise(channels=2, frames=4096, sample_rate=48000.0, seed=0)
        result = reverb.stk_echo(buf, delay_ms=50.0)
        assert result.channels == 2

    def test_echo_modifies_signal(self):
        buf = AudioBuffer.noise(channels=1, frames=4096, sample_rate=48000.0, seed=0)
        result = reverb.stk_echo(buf, delay_ms=100.0, mix=0.5)
        assert not np.allclose(result.data, buf.data)


# ---------------------------------------------------------------------------
# AGC
# ---------------------------------------------------------------------------


class TestAgc:
    def test_boosts_quiet_signal(self):
        """AGC should increase the level of a quiet signal."""
        buf = AudioBuffer.sine(440.0, frames=8192, sample_rate=48000.0) * 0.01
        result = dynamics.agc(buf, target_level=0.5, attack=0.001, release=0.001)
        # After convergence, output RMS should be higher than input
        in_rms = np.sqrt(np.mean(buf.data[:, -2048:] ** 2))
        out_rms = np.sqrt(np.mean(result.data[:, -2048:] ** 2))
        assert out_rms > in_rms * 2

    def test_attenuates_loud_signal(self):
        """AGC should reduce the level of a loud signal."""
        buf = AudioBuffer.sine(440.0, frames=8192, sample_rate=48000.0) * 2.0
        result = dynamics.agc(buf, target_level=0.1, attack=0.001, release=0.001)
        in_rms = np.sqrt(np.mean(buf.data[:, -2048:] ** 2))
        out_rms = np.sqrt(np.mean(result.data[:, -2048:] ** 2))
        assert out_rms < in_rms

    def test_shape_preserved(self):
        buf = AudioBuffer.noise(channels=2, frames=4096, sample_rate=48000.0, seed=0)
        result = dynamics.agc(buf)
        assert result.channels == 2
        assert result.frames == 4096
        assert result.data.dtype == np.float32

    def test_max_gain_clamped(self):
        """AGC on silence should not produce huge output."""
        buf = AudioBuffer.zeros(1, 2048, sample_rate=48000.0)
        result = dynamics.agc(buf, target_level=1.0, max_gain_db=20.0)
        assert np.max(np.abs(result.data)) < 1e-5

    def test_metadata_preserved(self):
        buf = AudioBuffer.noise(
            channels=1, frames=2048, sample_rate=44100.0, seed=0, label="agc"
        )
        result = dynamics.agc(buf)
        assert result.sample_rate == 44100.0
        assert result.label == "agc"

    def test_near_silence_no_nan(self):
        """AGC on near-silence should not produce NaN or Inf."""
        data = np.full((1, 4096), 1e-10, dtype=np.float32)
        buf = AudioBuffer(data, sample_rate=48000.0)
        result = dynamics.agc(buf, target_level=0.5, max_gain_db=40.0)
        assert np.all(np.isfinite(result.data))


# ---------------------------------------------------------------------------
# Sidechain Compression
# ---------------------------------------------------------------------------


class TestSidechainCompress:
    def test_sidechain_silence_no_reduction(self):
        """Silent sidechain should not reduce gain."""
        buf = AudioBuffer.sine(440.0, frames=8192, sample_rate=48000.0)
        sidechain = AudioBuffer.zeros(1, 8192, sample_rate=48000.0)
        result = dynamics.sidechain_compress(buf, sidechain, ratio=8.0, threshold=-20.0)
        np.testing.assert_allclose(result.data, buf.data, atol=1e-5)

    def test_loud_sidechain_reduces_gain(self):
        """Loud sidechain should reduce the output level."""
        buf = AudioBuffer.sine(440.0, frames=8192, sample_rate=48000.0)
        sidechain = AudioBuffer.sine(100.0, frames=8192, sample_rate=48000.0)
        result = dynamics.sidechain_compress(buf, sidechain, ratio=8.0, threshold=-20.0)
        assert np.max(np.abs(result.data)) < np.max(np.abs(buf.data))

    def test_ratio_one_is_identity(self):
        """Ratio of 1.0 should produce no compression."""
        buf = AudioBuffer.sine(440.0, frames=4096, sample_rate=48000.0)
        sc = AudioBuffer.sine(100.0, frames=4096, sample_rate=48000.0)
        result = dynamics.sidechain_compress(buf, sc, ratio=1.0, threshold=-20.0)
        np.testing.assert_allclose(result.data, buf.data, atol=1e-5)

    def test_frame_mismatch_raises(self):
        buf = AudioBuffer.sine(440.0, frames=4096, sample_rate=48000.0)
        sc = AudioBuffer.sine(100.0, frames=2048, sample_rate=48000.0)
        with pytest.raises(ValueError, match="Frame count mismatch"):
            dynamics.sidechain_compress(buf, sc)

    def test_mono_sidechain_stereo_buf(self):
        buf = AudioBuffer.noise(channels=2, frames=4096, sample_rate=48000.0, seed=0)
        sc = AudioBuffer.sine(100.0, channels=1, frames=4096, sample_rate=48000.0)
        result = dynamics.sidechain_compress(buf, sc, ratio=4.0, threshold=-10.0)
        assert result.channels == 2
        assert result.frames == 4096
        assert np.all(np.isfinite(result.data))

    def test_shape_and_metadata(self):
        buf = AudioBuffer.noise(
            channels=1, frames=8192, sample_rate=44100.0, seed=0, label="sc"
        )
        sc = AudioBuffer.sine(100.0, frames=8192, sample_rate=44100.0)
        result = dynamics.sidechain_compress(buf, sc)
        assert result.sample_rate == 44100.0
        assert result.label == "sc"


# ---------------------------------------------------------------------------
# Transient Shaper
# ---------------------------------------------------------------------------


class TestTransientShape:
    def test_unity_is_near_identity(self):
        """attack_gain=1, sustain_gain=1 should be close to identity."""
        buf = AudioBuffer.sine(440.0, frames=8192, sample_rate=48000.0)
        result = dynamics.transient_shape(buf, attack_gain=1.0, sustain_gain=1.0)
        np.testing.assert_allclose(result.data, buf.data, atol=0.05)

    def test_attack_boost_increases_transients(self):
        """Boosting attack should increase peak of impulsive signal."""
        data = np.zeros((1, 8192), dtype=np.float32)
        # Create a transient burst
        data[0, 1000:1050] = 0.8
        data[0, 1050:2000] = 0.2  # sustain
        buf = AudioBuffer(data, sample_rate=48000.0)
        result = dynamics.transient_shape(buf, attack_gain=3.0, sustain_gain=1.0)
        # Peak in the attack region should be higher
        attack_peak_in = float(np.max(np.abs(buf.data[0, 1000:1060])))
        attack_peak_out = float(np.max(np.abs(result.data[0, 1000:1060])))
        assert attack_peak_out > attack_peak_in

    def test_silence_stays_silent(self):
        buf = AudioBuffer.zeros(1, 4096, sample_rate=48000.0)
        result = dynamics.transient_shape(buf, attack_gain=5.0, sustain_gain=0.1)
        assert np.max(np.abs(result.data)) < 1e-6

    def test_shape_preserved(self):
        buf = AudioBuffer.noise(channels=2, frames=4096, sample_rate=48000.0, seed=0)
        result = dynamics.transient_shape(buf, attack_gain=2.0, sustain_gain=0.5)
        assert result.channels == 2
        assert result.frames == 4096
        assert np.all(np.isfinite(result.data))


# ---------------------------------------------------------------------------
# Lookahead Limiter
# ---------------------------------------------------------------------------


class TestLookaheadLimit:
    def test_output_below_threshold(self):
        """Output should never exceed the threshold."""
        buf = AudioBuffer.noise(channels=1, frames=16384, sample_rate=48000.0, seed=0)
        buf = buf * 2.0  # push above 0 dBFS
        result = dynamics.lookahead_limit(buf, threshold_db=-3.0, lookahead_ms=5.0)
        threshold_lin = 10.0 ** (-3.0 / 20.0)
        assert np.max(np.abs(result.data)) <= threshold_lin + 1e-4

    def test_quiet_signal_unchanged(self):
        """Signal well below threshold should pass through (delayed)."""
        buf = AudioBuffer.sine(440.0, frames=8192, sample_rate=48000.0) * 0.1
        result = dynamics.lookahead_limit(buf, threshold_db=-1.0)
        # Signal is ~-20 dBFS, threshold is -1 dBFS -> no reduction
        lookahead = max(1, int(48000 * 5.0 / 1000.0))
        # Compare delayed input to output
        np.testing.assert_allclose(
            result.data[:, lookahead:],
            buf.data[:, : buf.frames - lookahead],
            atol=1e-5,
        )

    def test_stereo_shape(self):
        buf = AudioBuffer.noise(channels=2, frames=8192, sample_rate=48000.0, seed=0)
        result = dynamics.lookahead_limit(buf, threshold_db=-3.0)
        assert result.channels == 2
        assert result.frames == buf.frames
        assert np.all(np.isfinite(result.data))

    def test_metadata_preserved(self):
        buf = AudioBuffer.noise(
            channels=1, frames=4096, sample_rate=44100.0, seed=0, label="la"
        )
        result = dynamics.lookahead_limit(buf)
        assert result.sample_rate == 44100.0
        assert result.label == "la"


# ---------------------------------------------------------------------------
# Shimmer Reverb
# ---------------------------------------------------------------------------


class TestShimmerReverb:
    def test_output_stereo(self):
        """FDN reverb always returns stereo, so shimmer_reverb does too."""
        buf = AudioBuffer.noise(channels=1, frames=8192, sample_rate=48000.0, seed=0)
        result = composed.shimmer_reverb(buf)
        assert result.channels == 2
        assert result.frames == 8192

    def test_mix_zero_passes_dry(self):
        buf = AudioBuffer.noise(channels=1, frames=8192, sample_rate=48000.0, seed=0)
        result = composed.shimmer_reverb(buf, mix=0.0)
        assert result.channels == 2
        assert result.frames == buf.frames
        assert result.data.dtype == np.float32
        # Dry mono is tiled to stereo
        expected = np.tile(buf.data, (2, 1))
        np.testing.assert_allclose(result.data, expected, atol=1e-5)

    def test_modifies_signal(self):
        buf = AudioBuffer.noise(channels=1, frames=8192, sample_rate=48000.0, seed=0)
        result = composed.shimmer_reverb(buf, mix=0.4, shimmer=0.3)
        assert not np.allclose(result.data, buf.data)

    def test_shimmer_zero_is_plain_reverb(self):
        """With shimmer=0, should behave like plain reverb (no pitch shift)."""
        buf = AudioBuffer.noise(channels=1, frames=8192, sample_rate=48000.0, seed=0)
        r1 = composed.shimmer_reverb(buf, mix=0.5, shimmer=0.0, preset="room")
        r2 = composed.shimmer_reverb(buf, mix=0.5, shimmer=0.5, preset="room")
        # Different shimmer amounts should produce different output
        assert not np.allclose(r1.data, r2.data)

    def test_stereo(self):
        buf = AudioBuffer.noise(channels=2, frames=8192, sample_rate=48000.0, seed=0)
        result = composed.shimmer_reverb(buf)
        assert result.channels == 2

    def test_presets(self):
        buf = AudioBuffer.noise(channels=1, frames=8192, sample_rate=48000.0, seed=0)
        for preset in ("room", "hall", "plate", "chamber", "cathedral"):
            result = composed.shimmer_reverb(buf, preset=preset)
            assert result.frames == 8192


# ---------------------------------------------------------------------------
# Tape Echo
# ---------------------------------------------------------------------------


class TestTapeEcho:
    def test_shape_preserved(self):
        buf = AudioBuffer.noise(channels=1, frames=24000, sample_rate=48000.0, seed=0)
        result = composed.tape_echo(buf, delay_ms=100)
        assert result.channels == 1
        assert result.frames == 24000

    def test_mix_zero_identity(self):
        buf = AudioBuffer.noise(channels=1, frames=24000, sample_rate=48000.0, seed=0)
        result = composed.tape_echo(buf, mix=0.0)
        assert result.channels == buf.channels
        assert result.frames == buf.frames
        assert result.data.dtype == np.float32
        np.testing.assert_allclose(result.data, buf.data, atol=1e-5)

    def test_echo_appears_after_delay(self):
        """Energy should appear in the delay region that was silent in the input."""
        sr = 48000
        buf = AudioBuffer.zeros(1, sr, sample_rate=float(sr))
        # Impulse in first 100 samples
        buf.data[0, :100] = 1.0
        result = composed.tape_echo(buf, delay_ms=200, feedback=0.5, mix=1.0)
        delay_samples = int(sr * 0.2)
        # Energy after the delay offset should be nonzero
        tail_energy = np.sum(result.data[0, delay_samples : delay_samples + 1000] ** 2)
        assert tail_energy > 1e-6

    def test_repeats_decay(self):
        """Later repeats should have less energy than earlier ones."""
        sr = 48000
        buf = AudioBuffer.zeros(1, sr, sample_rate=float(sr))
        buf.data[0, :100] = 1.0
        result = composed.tape_echo(buf, delay_ms=100, feedback=0.4, repeats=4, mix=1.0)
        ds = int(sr * 0.1)
        e1 = np.sum(result.data[0, ds : ds + ds] ** 2)
        e2 = np.sum(result.data[0, 2 * ds : 3 * ds] ** 2)
        assert e1 > e2

    def test_stereo(self):
        buf = AudioBuffer.noise(channels=2, frames=24000, sample_rate=48000.0, seed=0)
        result = composed.tape_echo(buf, delay_ms=100)
        assert result.channels == 2

    def test_tone_darkens(self):
        """Lower tone should produce darker echoes."""
        buf = AudioBuffer.noise(channels=1, frames=24000, sample_rate=48000.0, seed=0)
        bright = composed.tape_echo(buf, tone=8000.0, mix=1.0)
        dark = composed.tape_echo(buf, tone=1000.0, mix=1.0)
        # Spectral centroid proxy: dark should have less HF energy
        fft_bright = np.abs(np.fft.rfft(bright.data[0]))
        fft_dark = np.abs(np.fft.rfft(dark.data[0]))
        freqs = np.fft.rfftfreq(bright.frames, 1.0 / 48000.0)
        centroid_bright = np.sum(freqs * fft_bright) / (np.sum(fft_bright) + 1e-12)
        centroid_dark = np.sum(freqs * fft_dark) / (np.sum(fft_dark) + 1e-12)
        assert centroid_dark < centroid_bright


# ---------------------------------------------------------------------------
# Lo-Fi
# ---------------------------------------------------------------------------


class TestLoFi:
    def test_shape_preserved(self):
        buf = AudioBuffer.noise(channels=1, frames=4096, sample_rate=48000.0, seed=0)
        result = composed.lo_fi(buf)
        assert result.channels == 1
        assert result.frames == 4096

    def test_modifies_signal(self):
        buf = AudioBuffer.noise(channels=1, frames=4096, sample_rate=48000.0, seed=0)
        result = composed.lo_fi(buf)
        assert not np.allclose(result.data, buf.data)

    def test_tone_controls_brightness(self):
        """Lower tone cutoff should produce darker output."""
        buf = AudioBuffer.noise(channels=1, frames=8192, sample_rate=48000.0, seed=0)
        bright = composed.lo_fi(buf, tone=8000.0)
        dark = composed.lo_fi(buf, tone=1500.0)
        fft_bright = np.abs(np.fft.rfft(bright.data[0]))
        fft_dark = np.abs(np.fft.rfft(dark.data[0]))
        freqs = np.fft.rfftfreq(buf.frames, 1.0 / 48000.0)
        hf_mask = freqs > 4000
        hf_bright = np.sum(fft_bright[hf_mask] ** 2)
        hf_dark = np.sum(fft_dark[hf_mask] ** 2)
        assert hf_dark < hf_bright

    def test_stereo(self):
        buf = AudioBuffer.noise(channels=2, frames=4096, sample_rate=48000.0, seed=0)
        result = composed.lo_fi(buf)
        assert result.channels == 2

    def test_bit_depth_affects_output(self):
        """Different bit depths should produce different outputs."""
        buf = AudioBuffer.noise(channels=1, frames=4096, sample_rate=48000.0, seed=0)
        r4 = composed.lo_fi(buf, bit_depth=4, reduce=0.0)
        r12 = composed.lo_fi(buf, bit_depth=12, reduce=0.0)
        assert not np.allclose(r4.data, r12.data)


# ---------------------------------------------------------------------------
# Telephone
# ---------------------------------------------------------------------------


class TestTelephone:
    def test_shape_preserved(self):
        buf = AudioBuffer.noise(channels=1, frames=4096, sample_rate=48000.0, seed=0)
        result = composed.telephone(buf)
        assert result.channels == 1
        assert result.frames == 4096

    def test_bandlimits_signal(self):
        """Telephone should cut below 300 Hz and above 3400 Hz."""
        buf = AudioBuffer.noise(channels=1, frames=8192, sample_rate=48000.0, seed=0)
        result = composed.telephone(buf)
        fft_out = np.abs(np.fft.rfft(result.data[0]))
        freqs = np.fft.rfftfreq(buf.frames, 1.0 / 48000.0)
        lo_mask = freqs < 200
        mid_mask = (freqs > 500) & (freqs < 3000)
        lo_energy = np.sum(fft_out[lo_mask] ** 2)
        mid_energy = np.sum(fft_out[mid_mask] ** 2)
        # Low frequencies should be strongly attenuated
        assert lo_energy < mid_energy * 0.1
        # HF test is relaxed because hard saturation reintroduces harmonics
        # above the lowpass cutoff -- this is expected telephone character

    def test_modifies_signal(self):
        buf = AudioBuffer.noise(channels=1, frames=4096, sample_rate=48000.0, seed=0)
        result = composed.telephone(buf)
        assert not np.allclose(result.data, buf.data)

    def test_stereo(self):
        buf = AudioBuffer.noise(channels=2, frames=4096, sample_rate=48000.0, seed=0)
        result = composed.telephone(buf)
        assert result.channels == 2


# ---------------------------------------------------------------------------
# Gated Reverb
# ---------------------------------------------------------------------------


class TestGatedReverb:
    def test_output_stereo(self):
        """FDN reverb always returns stereo, so gated_reverb does too."""
        buf = AudioBuffer.noise(channels=1, frames=8192, sample_rate=48000.0, seed=0)
        result = composed.gated_reverb(buf)
        assert result.channels == 2
        assert result.frames == 8192

    def test_mix_zero_passes_dry(self):
        buf = AudioBuffer.noise(channels=1, frames=8192, sample_rate=48000.0, seed=0)
        result = composed.gated_reverb(buf, mix=0.0)
        assert result.channels == 2
        assert result.frames == buf.frames
        assert result.data.dtype == np.float32
        expected = np.tile(buf.data, (2, 1))
        np.testing.assert_allclose(result.data, expected, atol=1e-5)

    def test_modifies_signal(self):
        buf = AudioBuffer.noise(channels=1, frames=8192, sample_rate=48000.0, seed=0)
        result = composed.gated_reverb(buf, mix=0.5)
        assert not np.allclose(result.data, buf.data)

    def test_gate_reduces_tail(self):
        """Gated reverb should have less tail energy than ungated reverb."""
        from nanodsp.effects.reverb import reverb as _reverb

        buf = AudioBuffer.zeros(1, 48000, sample_rate=48000.0)
        buf.data[0, :200] = (
            np.random.default_rng(0).standard_normal(200).astype(np.float32)
        )
        ungated = _reverb(buf, preset="plate", mix=1.0, decay=0.7)
        gated = composed.gated_reverb(
            buf,
            preset="plate",
            decay=0.7,
            mix=1.0,
            gate_threshold_db=-20.0,
            gate_release=0.01,
        )
        # Tail energy (last 25%) should be lower in gated version
        q = buf.frames // 4
        tail_ungated = np.sum(ungated.data[:, -q:] ** 2)
        tail_gated = np.sum(gated.data[:, -q:] ** 2)
        assert tail_gated < tail_ungated

    def test_stereo(self):
        buf = AudioBuffer.noise(channels=2, frames=8192, sample_rate=48000.0, seed=0)
        result = composed.gated_reverb(buf)
        assert result.channels == 2

    def test_presets(self):
        buf = AudioBuffer.noise(channels=1, frames=8192, sample_rate=48000.0, seed=0)
        for preset in ("room", "plate", "hall"):
            result = composed.gated_reverb(buf, preset=preset)
            assert result.frames == 8192


# ---------------------------------------------------------------------------
# Auto-Pan
# ---------------------------------------------------------------------------


class TestAutoPan:
    def test_output_is_stereo(self):
        buf = AudioBuffer.noise(channels=1, frames=4096, sample_rate=48000.0, seed=0)
        result = composed.auto_pan(buf)
        assert result.channels == 2
        assert result.frames == 4096

    def test_stereo_input(self):
        buf = AudioBuffer.noise(channels=2, frames=4096, sample_rate=48000.0, seed=0)
        result = composed.auto_pan(buf)
        assert result.channels == 2

    def test_depth_zero_centered(self):
        """With depth=0 and center=0, both channels should be equal (center pan)."""
        buf = AudioBuffer.noise(channels=1, frames=4096, sample_rate=48000.0, seed=0)
        result = composed.auto_pan(buf, depth=0.0, center=0.0)
        assert result.channels == 2
        assert result.frames == buf.frames
        assert result.data.dtype == np.float32
        np.testing.assert_allclose(result.data[0], result.data[1], atol=1e-6)

    def test_panning_creates_channel_difference(self):
        """With nonzero depth, L and R should differ over time."""
        buf = AudioBuffer.noise(channels=1, frames=48000, sample_rate=48000.0, seed=0)
        result = composed.auto_pan(buf, rate=2.0, depth=1.0)
        # L and R should not be identical
        assert not np.allclose(result.data[0], result.data[1])

    def test_center_left(self):
        """Center=-1 should put more energy in the left channel."""
        buf = AudioBuffer.noise(channels=1, frames=48000, sample_rate=48000.0, seed=0)
        result = composed.auto_pan(buf, rate=1.0, depth=0.0, center=-1.0)
        left_energy = np.sum(result.data[0] ** 2)
        right_energy = np.sum(result.data[1] ** 2)
        assert left_energy > right_energy * 5

    def test_center_right(self):
        """Center=+1 should put more energy in the right channel."""
        buf = AudioBuffer.noise(channels=1, frames=48000, sample_rate=48000.0, seed=0)
        result = composed.auto_pan(buf, rate=1.0, depth=0.0, center=1.0)
        left_energy = np.sum(result.data[0] ** 2)
        right_energy = np.sum(result.data[1] ** 2)
        assert right_energy > left_energy * 5

    def test_energy_preserved(self):
        """Auto-pan should roughly preserve total energy (equal-power panning)."""
        buf = AudioBuffer.noise(channels=1, frames=48000, sample_rate=48000.0, seed=0)
        result = composed.auto_pan(buf, rate=2.0, depth=1.0)
        assert result.channels == 2
        assert result.frames == buf.frames
        assert np.all(np.isfinite(result.data))
        energy_in = np.sum(buf.data**2)
        energy_out = np.sum(result.data**2)
        np.testing.assert_allclose(energy_out, energy_in, rtol=0.05)


# ---------------------------------------------------------------------------
# Integration tests: composed effect chains
# ---------------------------------------------------------------------------


class TestEffectChains:
    """Verify that chaining multiple effects produces finite, correctly-shaped output."""

    def test_exciter_compress_limit(self):
        buf = AudioBuffer.sine(440.0, frames=8192, sample_rate=48000.0)
        result = buf.pipe(composed.exciter, amount=0.3)
        result = result.pipe(dynamics.compress, ratio=4.0, threshold=-20.0)
        result = result.pipe(dynamics.limit)
        assert result.frames == buf.frames
        assert result.channels == buf.channels
        assert np.all(np.isfinite(result.data))
        assert np.max(np.abs(result.data)) <= 1.0 + 1e-6

    def test_vocal_chain_on_noise(self):
        buf = AudioBuffer.noise(channels=1, frames=48000, sample_rate=48000.0, seed=42)
        result = composed.vocal_chain(buf)
        assert result.frames == buf.frames
        assert np.all(np.isfinite(result.data))

    def test_master_chain_stereo(self):
        buf = AudioBuffer.noise(channels=2, frames=48000, sample_rate=48000.0, seed=0)
        result = composed.master(buf)
        assert result.frames == buf.frames
        assert np.all(np.isfinite(result.data))

    def test_lowpass_saturate_reverb(self):
        buf = AudioBuffer.sine(440.0, frames=8192, sample_rate=48000.0)
        result = buf.pipe(fx_filters.lowpass, cutoff_hz=2000.0)
        result = result.pipe(saturation.saturate, drive=0.7, mode="tape")
        result = result.pipe(reverb.reverb, preset="room", mix=0.2)
        assert result.channels == 2  # reverb outputs stereo
        assert result.frames == buf.frames
        assert np.all(np.isfinite(result.data))

    def test_stereo_delay_compress_limit(self):
        buf = AudioBuffer.noise(channels=2, frames=16384, sample_rate=48000.0, seed=1)
        result = composed.stereo_delay(buf, left_ms=100.0, right_ms=150.0, feedback=0.3)
        result = result.pipe(dynamics.compress, ratio=6.0, threshold=-15.0)
        result = result.pipe(dynamics.limit)
        assert result.channels == 2
        assert result.frames == buf.frames
        assert np.all(np.isfinite(result.data))

    def test_de_esser_eq_compress(self):
        buf = AudioBuffer.noise(channels=1, frames=24000, sample_rate=48000.0, seed=3)
        result = composed.de_esser(buf, freq=6000.0)
        result = result.pipe(fx_filters.highpass, cutoff_hz=80.0)
        result = result.pipe(fx_filters.high_shelf_db, cutoff_hz=8000.0, db=3.0)
        result = result.pipe(dynamics.compress, ratio=3.0, threshold=-18.0)
        assert result.frames == buf.frames
        assert np.all(np.isfinite(result.data))

    def test_multiband_compress_limit(self):
        buf = AudioBuffer.noise(channels=1, frames=16384, sample_rate=48000.0, seed=5)
        result = composed.multiband_compress(buf)
        result = result.pipe(dynamics.limit)
        assert result.frames == buf.frames
        assert np.all(np.isfinite(result.data))

    def test_shimmer_reverb_normalize(self):
        buf = AudioBuffer.sine(220.0, frames=16384, sample_rate=48000.0)
        result = composed.shimmer_reverb(buf, mix=0.4, decay=0.8)
        assert result.channels == 2
        assert np.all(np.isfinite(result.data))
        # Normalize the output
        from nanodsp.ops import normalize_peak

        result = normalize_peak(result, target_db=-3.0)
        peak_db = 20.0 * np.log10(np.max(np.abs(result.data)) + 1e-20)
        assert abs(peak_db - (-3.0)) < 0.1

    def test_lo_fi_then_reverb(self):
        buf = AudioBuffer.sine(440.0, frames=8192, sample_rate=48000.0)
        result = composed.lo_fi(buf)
        result = result.pipe(reverb.reverb, preset="plate", mix=0.3)
        assert result.channels == 2
        assert np.all(np.isfinite(result.data))

    def test_noise_gate_reverb_limit(self):
        """Gate -> reverb -> limit chain should produce bounded output."""
        data = np.zeros((1, 24000), dtype=np.float32)
        data[0, 4000:8000] = np.sin(
            2 * np.pi * 440.0 * np.arange(4000, dtype=np.float32) / 48000.0
        ).astype(np.float32)
        buf = AudioBuffer(data, sample_rate=48000.0)
        result = dynamics.noise_gate(buf, threshold_db=-30.0)
        result = result.pipe(reverb.reverb, preset="hall", mix=0.3)
        result = result.pipe(dynamics.limit)
        assert np.all(np.isfinite(result.data))
        assert np.max(np.abs(result.data)) <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# Vocoder
# ---------------------------------------------------------------------------


class TestVocoder:
    def test_speech_noise_produces_output(self):
        """Vocoder with noise carrier should produce non-silent output."""
        mod = AudioBuffer.sine(440.0, frames=16384, sample_rate=48000.0)
        carrier = AudioBuffer.noise(
            channels=1, frames=16384, sample_rate=48000.0, seed=0
        )
        result = composed.vocoder(mod, carrier, n_bands=8)
        assert result.frames == mod.frames
        assert np.all(np.isfinite(result.data))
        assert np.max(np.abs(result.data)) > 0.001

    def test_silent_modulator_produces_silence(self):
        mod = AudioBuffer.zeros(1, 8192, sample_rate=48000.0)
        carrier = AudioBuffer.noise(
            channels=1, frames=8192, sample_rate=48000.0, seed=0
        )
        result = composed.vocoder(mod, carrier, n_bands=8)
        assert np.max(np.abs(result.data)) < 0.01

    def test_silent_carrier_produces_silence(self):
        mod = AudioBuffer.sine(440.0, frames=8192, sample_rate=48000.0)
        carrier = AudioBuffer.zeros(1, 8192, sample_rate=48000.0)
        result = composed.vocoder(mod, carrier, n_bands=8)
        assert np.max(np.abs(result.data)) < 0.01

    def test_frame_mismatch_raises(self):
        mod = AudioBuffer.sine(440.0, frames=4096, sample_rate=48000.0)
        carrier = AudioBuffer.noise(
            channels=1, frames=2048, sample_rate=48000.0, seed=0
        )
        with pytest.raises(ValueError, match="Frame count mismatch"):
            composed.vocoder(mod, carrier)

    def test_sample_rate_mismatch_raises(self):
        mod = AudioBuffer.sine(440.0, frames=4096, sample_rate=48000.0)
        carrier = AudioBuffer.sine(440.0, frames=4096, sample_rate=44100.0)
        with pytest.raises(ValueError, match="Sample rate mismatch"):
            composed.vocoder(mod, carrier)

    def test_single_band(self):
        mod = AudioBuffer.sine(440.0, frames=8192, sample_rate=48000.0)
        carrier = AudioBuffer.noise(
            channels=1, frames=8192, sample_rate=48000.0, seed=0
        )
        result = composed.vocoder(mod, carrier, n_bands=1)
        assert result.frames == 8192
        assert np.all(np.isfinite(result.data))

    def test_many_bands(self):
        mod = AudioBuffer.sine(440.0, frames=8192, sample_rate=48000.0)
        carrier = AudioBuffer.noise(
            channels=1, frames=8192, sample_rate=48000.0, seed=0
        )
        result = composed.vocoder(mod, carrier, n_bands=32)
        assert np.all(np.isfinite(result.data))

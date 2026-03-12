"""Tests for virtual analog filters (vafilters submodule)."""

import numpy as np
import pytest

from nanodsp._core import vafilters
from nanodsp.buffer import AudioBuffer
from nanodsp.effects.filters import (
    va_moog_ladder,
    va_moog_half_ladder,
    va_diode_ladder,
    va_korg35_lpf,
    va_korg35_hpf,
    va_oberheim,
)


SR = 44100.0


def make_impulse(n=1024, sr=SR):
    x = np.zeros(n, dtype=np.float32)
    x[0] = 1.0
    return x


def make_sine(freq=440.0, n=4096, sr=SR):
    t = np.arange(n, dtype=np.float32) / sr
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def make_noise(n=4096, sr=SR, seed=42):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n).astype(np.float32) * 0.5


def make_buf(data, sr=SR, channels=1):
    if channels == 1:
        return AudioBuffer(data.reshape(1, -1), sample_rate=int(sr))
    return AudioBuffer(np.tile(data, (channels, 1)), sample_rate=int(sr))


# ---- C++ binding-level tests ----


class TestMoogLadder:
    def test_construction_and_init(self):
        f = vafilters.MoogLadder()
        f.init(SR)
        assert f.cutoff == pytest.approx(20000.0)
        assert f.q == pytest.approx(1.0)

    def test_set_cutoff_q(self):
        f = vafilters.MoogLadder()
        f.init(SR)
        f.cutoff = 500.0
        f.q = 5.0
        assert f.cutoff == pytest.approx(500.0)
        assert f.q == pytest.approx(5.0)

    def test_impulse_no_nan(self):
        f = vafilters.MoogLadder()
        f.init(SR)
        f.cutoff = 1000.0
        out = f.process(make_impulse())
        assert not np.any(np.isnan(out))
        assert out.dtype == np.float32

    def test_lowpass_behavior(self):
        """Low cutoff should attenuate high-frequency content."""
        f = vafilters.MoogLadder()
        f.init(SR)
        f.cutoff = 200.0
        f.q = 1.0
        noise = make_noise()
        out = f.process(noise)
        # Energy above cutoff should be reduced
        fft_in = np.abs(np.fft.rfft(noise))
        fft_out = np.abs(np.fft.rfft(out))
        freqs = np.fft.rfftfreq(len(noise), 1.0 / SR)
        high_mask = freqs > 2000.0
        ratio = np.mean(fft_out[high_mask]) / (np.mean(fft_in[high_mask]) + 1e-10)
        assert ratio < 0.1, f"High-freq ratio {ratio} too large for 200 Hz LP"

    def test_reset_clears_state(self):
        f = vafilters.MoogLadder()
        f.init(SR)
        f.cutoff = 1000.0
        f.process(make_noise(n=512))
        f.reset()
        out = f.process(np.zeros(64, dtype=np.float32))
        assert np.allclose(out, 0.0, atol=1e-10)


class TestMoogHalfLadder:
    def test_impulse_no_nan(self):
        f = vafilters.MoogHalfLadder()
        f.init(SR)
        f.cutoff = 1000.0
        out = f.process(make_impulse())
        assert not np.any(np.isnan(out))

    def test_12db_slope(self):
        """Half-ladder (12 dB/oct) should attenuate less than full ladder (24 dB/oct)."""
        noise = make_noise()
        full = vafilters.MoogLadder()
        full.init(SR)
        full.cutoff = 500.0
        half = vafilters.MoogHalfLadder()
        half.init(SR)
        half.cutoff = 500.0
        out_full = full.process(noise)
        out_half = half.process(noise)
        # Half-ladder should preserve more high-freq energy
        fft_full = np.abs(np.fft.rfft(out_full))
        fft_half = np.abs(np.fft.rfft(out_half))
        freqs = np.fft.rfftfreq(len(noise), 1.0 / SR)
        high_mask = freqs > 5000.0
        assert np.mean(fft_half[high_mask]) > np.mean(fft_full[high_mask])


class TestDiodeLadder:
    def test_impulse_no_nan(self):
        f = vafilters.DiodeLadder()
        f.init(SR)
        f.cutoff = 1000.0
        out = f.process(make_impulse())
        assert not np.any(np.isnan(out))

    def test_lowpass_behavior(self):
        f = vafilters.DiodeLadder()
        f.init(SR)
        f.cutoff = 300.0
        noise = make_noise()
        out = f.process(noise)
        fft_in = np.abs(np.fft.rfft(noise))
        fft_out = np.abs(np.fft.rfft(out))
        freqs = np.fft.rfftfreq(len(noise), 1.0 / SR)
        high_mask = freqs > 3000.0
        ratio = np.mean(fft_out[high_mask]) / (np.mean(fft_in[high_mask]) + 1e-10)
        assert ratio < 0.1

    def test_nonlinear_soft_clip(self):
        """DiodeLadder should soft-clip high input levels."""
        f = vafilters.DiodeLadder()
        f.init(SR)
        f.cutoff = 10000.0
        loud = make_sine(440.0) * 10.0
        out = f.process(loud)
        assert not np.any(np.isnan(out))


class TestKorg35LPF:
    def test_impulse_no_nan(self):
        f = vafilters.Korg35LPF()
        f.init(SR)
        f.cutoff = 1000.0
        out = f.process(make_impulse())
        assert not np.any(np.isnan(out))

    def test_lowpass_behavior(self):
        f = vafilters.Korg35LPF()
        f.init(SR)
        f.cutoff = 500.0
        noise = make_noise()
        out = f.process(noise)
        fft_out = np.abs(np.fft.rfft(out))
        freqs = np.fft.rfftfreq(len(noise), 1.0 / SR)
        # Energy should decrease with frequency above cutoff
        low_mask = (freqs > 100) & (freqs < 400)
        high_mask = freqs > 5000
        assert np.mean(fft_out[low_mask]) > np.mean(fft_out[high_mask])


class TestKorg35HPF:
    def test_impulse_no_nan(self):
        f = vafilters.Korg35HPF()
        f.init(SR)
        f.cutoff = 1000.0
        out = f.process(make_impulse())
        assert not np.any(np.isnan(out))

    def test_highpass_behavior(self):
        """HPF should pass high frequencies and attenuate low frequencies."""
        f = vafilters.Korg35HPF()
        f.init(SR)
        f.cutoff = 5000.0
        noise = make_noise()
        out = f.process(noise)
        fft_out = np.abs(np.fft.rfft(out))
        freqs = np.fft.rfftfreq(len(noise), 1.0 / SR)
        low_mask = freqs < 500
        high_mask = freqs > 10000
        assert np.mean(fft_out[high_mask]) > np.mean(fft_out[low_mask])

    def test_dc_rejection(self):
        """HPF should remove DC offset."""
        f = vafilters.Korg35HPF()
        f.init(SR)
        f.cutoff = 100.0
        dc_signal = np.ones(4096, dtype=np.float32) * 0.5
        out = f.process(dc_signal)
        # After settling, output should be near zero
        assert abs(np.mean(out[-1024:])) < 0.05


class TestOberheimSVF:
    def test_construction_and_init(self):
        f = vafilters.OberheimSVF()
        f.init(SR)
        assert f.cutoff == pytest.approx(20000.0)
        assert f.q == pytest.approx(1.0)

    def test_single_mode_process(self):
        f = vafilters.OberheimSVF()
        f.init(SR)
        f.cutoff = 1000.0
        imp = make_impulse()
        for mode in range(4):
            out = f.process(imp, mode)
            f.reset()
            assert not np.any(np.isnan(out))
            assert out.shape == imp.shape

    def test_multi_output(self):
        f = vafilters.OberheimSVF()
        f.init(SR)
        f.cutoff = 1000.0
        imp = make_impulse()
        lpf, hpf, bpf, bsf = f.process_multi(imp)
        assert lpf.shape == imp.shape
        assert hpf.shape == imp.shape
        assert bpf.shape == imp.shape
        assert bsf.shape == imp.shape
        for arr in [lpf, hpf, bpf, bsf]:
            assert not np.any(np.isnan(arr))

    def test_lpf_hpf_complementary(self):
        """LPF + HPF should approximately equal the original (when BPF is small)."""
        f = vafilters.OberheimSVF()
        f.init(SR)
        f.cutoff = 1000.0
        f.q = 0.7
        noise = make_noise()
        lpf, hpf, bpf, bsf = f.process_multi(noise)
        # BSF = LP + HP approximately (notch filter = complement of bandpass)
        # The sum LPF + HPF should be close to BSF
        reconstructed = lpf + hpf
        assert np.allclose(reconstructed, bsf, atol=0.1)

    def test_lpf_mode_is_lowpass(self):
        f = vafilters.OberheimSVF()
        f.init(SR)
        f.cutoff = 300.0
        noise = make_noise()
        out = f.process(noise, 0)  # LPF mode
        fft_out = np.abs(np.fft.rfft(out))
        freqs = np.fft.rfftfreq(len(noise), 1.0 / SR)
        low_mask = (freqs > 50) & (freqs < 200)
        high_mask = freqs > 5000
        assert np.mean(fft_out[low_mask]) > np.mean(fft_out[high_mask])

    def test_hpf_mode_is_highpass(self):
        f = vafilters.OberheimSVF()
        f.init(SR)
        f.cutoff = 5000.0
        noise = make_noise()
        out = f.process(noise, 1)  # HPF mode
        fft_out = np.abs(np.fft.rfft(out))
        freqs = np.fft.rfftfreq(len(noise), 1.0 / SR)
        low_mask = freqs < 500
        high_mask = freqs > 10000
        assert np.mean(fft_out[high_mask]) > np.mean(fft_out[low_mask])


# ---- High-level Python API tests ----


class TestVAEffects:
    def test_va_moog_ladder(self):
        buf = make_buf(make_noise())
        out = va_moog_ladder(buf, cutoff_hz=1000.0, q=2.0)
        assert out.data.shape == buf.data.shape
        assert out.sample_rate == buf.sample_rate
        assert not np.any(np.isnan(out.data))

    def test_va_moog_half_ladder(self):
        buf = make_buf(make_noise())
        out = va_moog_half_ladder(buf, cutoff_hz=1000.0, q=2.0)
        assert out.data.shape == buf.data.shape
        assert not np.any(np.isnan(out.data))

    def test_va_diode_ladder(self):
        buf = make_buf(make_noise())
        out = va_diode_ladder(buf, cutoff_hz=1000.0, q=2.0)
        assert out.data.shape == buf.data.shape
        assert not np.any(np.isnan(out.data))

    def test_va_korg35_lpf(self):
        buf = make_buf(make_noise())
        out = va_korg35_lpf(buf, cutoff_hz=1000.0, q=2.0)
        assert out.data.shape == buf.data.shape
        assert not np.any(np.isnan(out.data))

    def test_va_korg35_hpf(self):
        buf = make_buf(make_noise())
        out = va_korg35_hpf(buf, cutoff_hz=1000.0, q=2.0)
        assert out.data.shape == buf.data.shape
        assert not np.any(np.isnan(out.data))

    def test_va_oberheim_modes(self):
        buf = make_buf(make_noise())
        for mode in ["lpf", "hpf", "bpf", "bsf"]:
            out = va_oberheim(buf, cutoff_hz=1000.0, q=2.0, mode=mode)
            assert out.data.shape == buf.data.shape
            assert not np.any(np.isnan(out.data))

    def test_va_oberheim_invalid_mode(self):
        buf = make_buf(make_noise())
        with pytest.raises(ValueError, match="Unknown mode"):
            va_oberheim(buf, mode="invalid")

    def test_stereo_processing(self):
        buf = make_buf(make_noise(), channels=2)
        out = va_moog_ladder(buf, cutoff_hz=1000.0)
        assert out.data.shape == buf.data.shape

    def test_different_sample_rates(self):
        for sr in [22050, 44100, 48000, 96000]:
            buf = make_buf(make_noise(sr=sr), sr=sr)
            out = va_moog_ladder(buf, cutoff_hz=1000.0)
            assert not np.any(np.isnan(out.data))

    def test_cutoff_range_extremes(self):
        buf = make_buf(make_noise())
        for cutoff in [20.0, 100.0, 1000.0, 10000.0, 20000.0]:
            out = va_moog_ladder(buf, cutoff_hz=cutoff)
            assert not np.any(np.isnan(out.data))

    def test_high_resonance(self):
        buf = make_buf(make_noise())
        out = va_moog_ladder(buf, cutoff_hz=1000.0, q=20.0)
        assert not np.any(np.isnan(out.data))


class TestVAFilterProperties:
    """Test common properties across all filter types."""

    @pytest.mark.parametrize(
        "cls_name",
        [
            "MoogLadder",
            "MoogHalfLadder",
            "DiodeLadder",
            "Korg35LPF",
            "Korg35HPF",
            "OberheimSVF",
        ],
    )
    def test_output_length_matches_input(self, cls_name):
        cls = getattr(vafilters, cls_name)
        f = cls()
        f.init(SR)
        f.cutoff = 1000.0
        for n in [1, 64, 256, 1024]:
            inp = np.zeros(n, dtype=np.float32)
            inp[0] = 1.0
            out = f.process(inp) if cls_name != "OberheimSVF" else f.process(inp, 0)
            f.reset()
            assert len(out) == n

    @pytest.mark.parametrize(
        "cls_name",
        [
            "MoogLadder",
            "MoogHalfLadder",
            "DiodeLadder",
            "Korg35LPF",
            "Korg35HPF",
            "OberheimSVF",
        ],
    )
    def test_silence_produces_silence(self, cls_name):
        cls = getattr(vafilters, cls_name)
        f = cls()
        f.init(SR)
        f.cutoff = 1000.0
        silence = np.zeros(256, dtype=np.float32)
        out = f.process(silence) if cls_name != "OberheimSVF" else f.process(silence, 0)
        assert np.allclose(out, 0.0, atol=1e-10)

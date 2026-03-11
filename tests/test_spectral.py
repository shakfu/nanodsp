"""Tests for nanodsp.spectral module (signalsmith STFT)."""

import numpy as np
from nanodsp._core import spectral


class TestSTFTConstruction:
    def test_construction(self):
        stft = spectral.STFT(1, 256, 128)
        assert stft is not None

    def test_properties(self):
        stft = spectral.STFT(2, 512, 256)
        assert stft.window_size() == 512
        assert stft.interval() == 256
        assert stft.fft_size() >= 512
        assert stft.bands() > 0
        assert stft.latency() >= 0

    def test_fft_size_geq_window(self):
        stft = spectral.STFT(1, 256, 128)
        assert stft.fft_size() >= stft.window_size()

    def test_bands_half_fft_size(self):
        stft = spectral.STFT(1, 256, 128)
        # bands = fft_size / 2 + 1 or fft_size / 2 depending on implementation
        assert stft.bands() > 0
        assert stft.bands() <= stft.fft_size()


class TestSTFTAnalyse:
    def test_analyse_channel(self):
        stft = spectral.STFT(1, 256, 128)
        inp = np.zeros(256, dtype=np.float32)
        inp[0] = 1.0
        stft.analyse_channel(0, inp)
        spec = stft.get_spectrum()
        assert spec.dtype == np.complex64
        assert spec.shape[0] == stft.bands()

    def test_analyse_multichannel(self):
        stft = spectral.STFT(2, 256, 128)
        inp = np.zeros((2, 256), dtype=np.float32)
        inp[0, 0] = 1.0
        inp[1, 10] = 1.0
        stft.analyse(inp)
        spec0 = stft.get_spectrum_channel(0)
        spec1 = stft.get_spectrum_channel(1)
        assert spec0.shape == spec1.shape
        # Channels should have different spectra
        assert not np.allclose(spec0, spec1)

    def test_get_spectrum_default_channel_0(self):
        stft = spectral.STFT(2, 256, 128)
        inp = np.zeros((2, 256), dtype=np.float32)
        inp[0, 0] = 1.0
        stft.analyse(inp)
        spec_default = stft.get_spectrum()
        spec_ch0 = stft.get_spectrum_channel(0)
        np.testing.assert_allclose(spec_default, spec_ch0)

    def test_sine_peak_in_spectrum(self):
        n = 1024
        stft = spectral.STFT(1, 256, 128)
        freq = 0.1  # normalized
        t = np.arange(n, dtype=np.float32)
        inp = np.sin(2 * np.pi * freq * t).astype(np.float32)
        stft.analyse_channel(0, inp)
        spec = stft.get_spectrum()
        magnitudes = np.abs(spec)
        # Should have a peak somewhere in the spectrum
        assert np.max(magnitudes) > 0

    def test_reset(self):
        stft = spectral.STFT(1, 256, 128)
        inp = np.ones(512, dtype=np.float32)
        stft.analyse_channel(0, inp)
        stft.reset()
        # After reset, spectrum should be zero
        spec = stft.get_spectrum()
        np.testing.assert_allclose(np.abs(spec), 0.0, atol=1e-7)


class TestSTFTSetSpectrum:
    def test_set_spectrum_channel(self):
        stft = spectral.STFT(1, 256, 128)
        bands = stft.bands()
        new_spec = np.ones(bands, dtype=np.complex64) * (1 + 2j)
        stft.set_spectrum_channel(0, new_spec)
        retrieved = stft.get_spectrum_channel(0)
        np.testing.assert_allclose(retrieved, new_spec, atol=1e-6)

    def test_set_spectrum_wrong_size_raises(self):
        stft = spectral.STFT(1, 256, 128)
        wrong = np.ones(10, dtype=np.complex64)
        try:
            stft.set_spectrum_channel(0, wrong)
            assert False, "Should have raised"
        except (ValueError, RuntimeError):
            pass

    def test_set_and_get_roundtrip(self):
        stft = spectral.STFT(2, 256, 128)
        bands = stft.bands()
        rng = np.random.default_rng(42)
        spec = (rng.standard_normal(bands) + 1j * rng.standard_normal(bands)).astype(
            np.complex64
        )
        stft.set_spectrum_channel(1, spec)
        retrieved = stft.get_spectrum_channel(1)
        np.testing.assert_allclose(retrieved, spec, atol=1e-6)


class TestSTFTZeroPadding:
    def test_zero_padding_increases_fft_size(self):
        stft_no_pad = spectral.STFT(1, 256, 128, 0, 0)
        stft_pad = spectral.STFT(1, 256, 128, 0, 256)
        assert stft_pad.fft_size() > stft_no_pad.fft_size()
        assert stft_pad.bands() > stft_no_pad.bands()

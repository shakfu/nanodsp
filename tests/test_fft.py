"""Tests for nanodsp.fft module (signalsmith FFT)."""

import numpy as np
import pytest
from nanodsp._core import fft as fft_mod


class TestFFTConstruction:
    def test_default_construction(self):
        f = fft_mod.FFT(64)
        assert f.size() == 64

    def test_set_size(self):
        f = fft_mod.FFT(64)
        f.set_size(128)
        assert f.size() == 128

    def test_fast_size_above(self):
        # Should return a size >= input that is efficient for the FFT
        result = fft_mod.FFT.fast_size_above(100)
        assert result >= 100
        # Verify it's a power-of-2-based fast size
        assert result == 128

    def test_fast_size_below(self):
        result = fft_mod.FFT.fast_size_below(100)
        assert result <= 100
        # 96 = 32 * 3 = 2^5 * 3^1
        assert result == 96


class TestComplexFFT:
    def test_fft_shape(self):
        f = fft_mod.FFT(128)
        inp = np.zeros(128, dtype=np.complex64)
        inp[0] = 1.0
        out = f.fft(inp)
        assert out.shape == (128,)
        assert out.dtype == np.complex64

    def test_ifft_shape(self):
        f = fft_mod.FFT(128)
        inp = np.ones(128, dtype=np.complex64)
        out = f.ifft(inp)
        assert out.shape == (128,)
        assert out.dtype == np.complex64

    def test_impulse_has_flat_spectrum(self):
        n = 256
        f = fft_mod.FFT(n)
        inp = np.zeros(n, dtype=np.complex64)
        inp[0] = 1.0
        out = f.fft(inp)
        assert out.shape == (n,)
        # Impulse FFT should have magnitude ~1 at all bins
        magnitudes = np.abs(out)
        np.testing.assert_allclose(magnitudes, 1.0, atol=1e-5)

    def test_dc_signal(self):
        n = 64
        f = fft_mod.FFT(n)
        inp = np.ones(n, dtype=np.complex64)
        out = f.fft(inp)
        # DC bin should be n, all others 0
        assert abs(out[0] - n) < 1e-3
        np.testing.assert_allclose(np.abs(out[1:]), 0.0, atol=1e-3)

    def test_roundtrip(self):
        n = 256
        f = fft_mod.FFT(n)
        rng = np.random.default_rng(42)
        inp = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(
            np.complex64
        )
        spectrum = f.fft(inp)
        recovered = f.ifft(spectrum)
        assert recovered.shape == inp.shape
        # IFFT is unscaled, so result = inp * N
        np.testing.assert_allclose(recovered / n, inp, atol=1e-4)

    def test_single_tone(self):
        n = 128
        f = fft_mod.FFT(n)
        k = 5  # bin index
        t = np.arange(n, dtype=np.float32)
        # Complex exponential at bin k
        inp = np.exp(2j * np.pi * k * t / n).astype(np.complex64)
        out = f.fft(inp)
        magnitudes = np.abs(out)
        # Bin k should have magnitude ~n, others ~0
        assert magnitudes[k] > n * 0.99
        others = np.delete(magnitudes, k)
        np.testing.assert_allclose(others, 0.0, atol=0.5)

    def test_wrong_size_raises(self):
        f = fft_mod.FFT(64)
        inp = np.zeros(32, dtype=np.complex64)
        with pytest.raises(ValueError):
            f.fft(inp)

    def test_wrong_size_ifft_raises(self):
        f = fft_mod.FFT(64)
        inp = np.zeros(32, dtype=np.complex64)
        with pytest.raises(ValueError):
            f.ifft(inp)


class TestRealFFTConstruction:
    def test_construction(self):
        rfft = fft_mod.RealFFT(256)
        assert rfft.size() == 256

    def test_set_size(self):
        rfft = fft_mod.RealFFT(256)
        rfft.set_size(512)
        assert rfft.size() == 512

    def test_fast_size_above(self):
        result = fft_mod.RealFFT.fast_size_above(100)
        assert result >= 100

    def test_fast_size_below(self):
        result = fft_mod.RealFFT.fast_size_below(100)
        assert result <= 100


class TestRealFFT:
    def test_fft_shape(self):
        n = 256
        rfft = fft_mod.RealFFT(n)
        inp = np.zeros(n, dtype=np.float32)
        inp[0] = 1.0
        out = rfft.fft(inp)
        assert out.shape == (n // 2,)
        assert out.dtype == np.complex64

    def test_ifft_shape(self):
        n = 256
        rfft = fft_mod.RealFFT(n)
        inp = np.zeros(n // 2, dtype=np.complex64)
        out = rfft.ifft(inp)
        assert out.shape == (n,)
        assert out.dtype == np.float32

    def test_impulse_flat_spectrum(self):
        n = 256
        rfft = fft_mod.RealFFT(n)
        inp = np.zeros(n, dtype=np.float32)
        inp[0] = 1.0
        out = rfft.fft(inp)
        magnitudes = np.abs(out)
        assert magnitudes.shape == (n // 2,)
        # signalsmith's modified RealFFT scales DC bin by sqrt(2)
        np.testing.assert_allclose(magnitudes[0], np.sqrt(2), atol=1e-5)
        np.testing.assert_allclose(magnitudes[1:], 1.0, atol=1e-5)

    def test_dc_signal(self):
        n = 128
        rfft = fft_mod.RealFFT(n)
        inp = np.ones(n, dtype=np.float32)
        out = rfft.fft(inp)
        # DC bin (index 0) should be n
        assert abs(out[0].real - n) < 1e-3
        # Other bins should be ~0
        np.testing.assert_allclose(np.abs(out[1:]), 0.0, atol=1e-3)

    def test_roundtrip(self):
        n = 256
        rfft = fft_mod.RealFFT(n)
        rng = np.random.default_rng(42)
        inp = rng.standard_normal(n).astype(np.float32)
        spectrum = rfft.fft(inp)
        recovered = rfft.ifft(spectrum)
        # IFFT is unscaled, so result = inp * N
        np.testing.assert_allclose(recovered / n, inp, atol=1e-4)

    def test_sine_peak(self):
        n = 256
        rfft = fft_mod.RealFFT(n)
        k = 10  # bin index
        t = np.arange(n, dtype=np.float32)
        inp = np.sin(2 * np.pi * k * t / n).astype(np.float32)
        out = rfft.fft(inp)
        magnitudes = np.abs(out)
        # The peak should be at bin k
        peak_bin = np.argmax(magnitudes)
        assert peak_bin == k

    def test_wrong_size_raises(self):
        rfft = fft_mod.RealFFT(256)
        inp = np.zeros(128, dtype=np.float32)
        with pytest.raises(ValueError):
            rfft.fft(inp)

    def test_wrong_bins_ifft_raises(self):
        rfft = fft_mod.RealFFT(256)
        inp = np.zeros(64, dtype=np.complex64)
        with pytest.raises(ValueError):
            rfft.ifft(inp)

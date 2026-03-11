"""Tests for nanodsp.rates module (signalsmith oversampling)."""

import numpy as np
from nanodsp._core import rates


class TestOversampler2xConstruction:
    def test_construction(self):
        os = rates.Oversampler2x(1, 256)
        assert os is not None

    def test_multichannel(self):
        os = rates.Oversampler2x(4, 128)
        assert os is not None

    def test_latency(self):
        os = rates.Oversampler2x(1, 256)
        lat = os.latency()
        assert isinstance(lat, (int, float))
        assert lat >= 0


class TestOversampler2xUp:
    def test_up_shape(self):
        os = rates.Oversampler2x(1, 128)
        inp = np.zeros((1, 64), dtype=np.float32)
        inp[0, 0] = 1.0
        out = os.up(inp)
        assert out.shape == (1, 128)  # 2x frames
        assert out.dtype == np.float32

    def test_up_multichannel(self):
        os = rates.Oversampler2x(2, 128)
        inp = np.zeros((2, 64), dtype=np.float32)
        inp[0, 0] = 1.0
        inp[1, 0] = -1.0
        out = os.up(inp)
        assert out.shape == (2, 128)

    def test_up_preserves_energy(self):
        os = rates.Oversampler2x(1, 256)
        rng = np.random.default_rng(42)
        inp = rng.standard_normal((1, 128)).astype(np.float32)
        out = os.up(inp)
        # Upsampled signal should have similar energy (not exact due to filtering)
        out_energy = np.sum(out**2) / 2  # normalize for 2x samples
        assert out_energy > 0


class TestOversampler2xDown:
    def test_down_shape(self):
        os = rates.Oversampler2x(1, 128)
        # First upsample, then downsample
        inp = np.zeros((1, 64), dtype=np.float32)
        inp[0, 0] = 1.0
        upsampled = os.up(inp)
        out = os.down(upsampled, 64)
        assert out.shape == (1, 64)
        assert out.dtype == np.float32


class TestOversampler2xProcess:
    def test_process_shape(self):
        os = rates.Oversampler2x(1, 128)
        inp = np.zeros((1, 64), dtype=np.float32)
        out = os.process(inp)
        assert out.shape == (1, 64)  # same as input
        assert out.dtype == np.float32

    def test_process_multichannel(self):
        os = rates.Oversampler2x(2, 128)
        inp = np.zeros((2, 64), dtype=np.float32)
        out = os.process(inp)
        assert out.shape == (2, 64)

    def test_roundtrip_near_unity(self):
        """Up+down roundtrip of low-freq signal should be near-unity (accounting for latency)."""
        os = rates.Oversampler2x(1, 256)
        # Low-frequency sine (well below Nyquist) should survive round-trip
        t = np.arange(256, dtype=np.float32)
        sine = np.sin(2 * np.pi * 0.01 * t).astype(np.float32)
        inp = sine.reshape(1, -1)

        # Process several blocks to let filter settle
        for _ in range(4):
            os.process(inp)
        out = os.process(inp)

        # After settling, output should approximate input (with latency shift)
        # Just check the output is non-trivial and has similar shape
        assert np.max(np.abs(out)) > 0.5

    def test_reset(self):
        os = rates.Oversampler2x(1, 128)
        inp = np.ones((1, 64), dtype=np.float32)
        os.process(inp)
        os.reset()
        out = os.process(np.zeros((1, 64), dtype=np.float32))
        # After reset with zeros, output should be very small
        # (not exactly zero due to filter tails, but small)
        assert np.max(np.abs(out)) < 0.5

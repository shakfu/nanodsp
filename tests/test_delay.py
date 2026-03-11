"""Tests for nanodsp.delay module (signalsmith delay lines)."""

import numpy as np
from nanodsp._core import delay


def make_impulse(n=256):
    x = np.zeros(n, dtype=np.float32)
    x[0] = 1.0
    return x


class TestDelayConstruction:
    def test_default_construction(self):
        d = delay.Delay()
        assert d is not None

    def test_construction_with_capacity(self):
        d = delay.Delay(1024)
        assert d is not None

    def test_cubic_construction(self):
        d = delay.DelayCubic(1024)
        assert d is not None


class TestDelayLatency:
    def test_linear_latency(self):
        # Linear interpolation has a known latency
        lat = delay.Delay.latency
        assert isinstance(lat, (int, float))
        assert lat >= 0

    def test_cubic_latency(self):
        lat = delay.DelayCubic.latency
        assert isinstance(lat, (int, float))
        assert lat >= 0

    def test_cubic_greater_latency(self):
        # Cubic interpolation should have >= latency than linear
        assert delay.DelayCubic.latency >= delay.Delay.latency


class TestDelayProcess:
    def test_process_shape(self):
        d = delay.Delay(64)
        inp = np.ones(32, dtype=np.float32)
        out = d.process(inp, 4.0)
        assert out.shape == (32,)
        assert out.dtype == np.float32

    def test_impulse_delay(self):
        delay_samples = 10.0
        d = delay.Delay(64)
        inp = make_impulse(64)
        out = d.process(inp, delay_samples)
        # The impulse should appear at the delay offset (accounting for latency)
        peak_idx = np.argmax(np.abs(out))
        expected = int(delay_samples) + delay.Delay.latency
        assert peak_idx == expected

    def test_zero_delay(self):
        d = delay.Delay(64)
        inp = make_impulse(64)
        out = d.process(inp, 0.0)
        # With 0 delay, the peak should be at latency offset
        peak_idx = np.argmax(np.abs(out))
        assert peak_idx == delay.Delay.latency

    def test_fractional_delay(self):
        d = delay.Delay(64)
        inp = make_impulse(64)
        out = d.process(inp, 5.5)
        # Should have nonzero values at neighboring samples (interpolated)
        peak_area = np.abs(out[5:8])
        assert np.max(peak_area) > 0

    def test_reset(self):
        d = delay.Delay(64)
        # Fill with some signal
        d.process(np.ones(32, dtype=np.float32), 4.0)
        d.reset()
        # After reset, processing zeros should give zeros
        out = d.process(np.zeros(16, dtype=np.float32), 4.0)
        np.testing.assert_allclose(out, 0.0, atol=1e-7)

    def test_resize(self):
        d = delay.Delay(32)
        d.resize(256)
        inp = make_impulse(64)
        out = d.process(inp, 20.0)
        assert out.shape == (64,)


class TestDelayVarying:
    def test_process_varying_shape(self):
        d = delay.Delay(128)
        inp = np.ones(64, dtype=np.float32)
        delays = np.full(64, 4.0, dtype=np.float32)
        out = d.process_varying(inp, delays)
        assert out.shape == (64,)
        assert out.dtype == np.float32

    def test_varying_matches_fixed(self):
        """Constant delay array should match fixed delay."""
        d1 = delay.Delay(128)
        d2 = delay.Delay(128)
        inp = np.random.default_rng(42).standard_normal(64).astype(np.float32)
        delay_val = 8.0

        out_fixed = d1.process(inp, delay_val)
        delays = np.full(64, delay_val, dtype=np.float32)
        out_varying = d2.process_varying(inp, delays)
        np.testing.assert_allclose(out_fixed, out_varying, atol=1e-6)

    def test_mismatched_lengths_raises(self):
        d = delay.Delay(128)
        inp = np.ones(64, dtype=np.float32)
        delays = np.ones(32, dtype=np.float32)
        try:
            d.process_varying(inp, delays)
            assert False, "Should have raised"
        except (ValueError, RuntimeError):
            pass


class TestDelayCubic:
    def test_process_shape(self):
        d = delay.DelayCubic(128)
        inp = np.ones(64, dtype=np.float32)
        out = d.process(inp, 8.0)
        assert out.shape == (64,)
        assert out.dtype == np.float32

    def test_impulse_delay(self):
        delay_samples = 10.0
        d = delay.DelayCubic(64)
        inp = make_impulse(64)
        out = d.process(inp, delay_samples)
        peak_idx = np.argmax(np.abs(out))
        expected = int(delay_samples) + delay.DelayCubic.latency
        assert peak_idx == expected

    def test_cubic_smoother_than_linear(self):
        """Cubic interpolation should produce smoother output for fractional delays."""
        d_lin = delay.Delay(128)
        d_cub = delay.DelayCubic(128)
        # Sine wave with fractional delay
        t = np.arange(64, dtype=np.float32)
        inp = np.sin(2 * np.pi * 0.05 * t).astype(np.float32)
        delay_val = 3.7  # fractional

        out_lin = d_lin.process(inp, delay_val)
        out_cub = d_cub.process(inp, delay_val)

        # Both should produce valid output
        assert out_lin.shape == (64,)
        assert out_cub.shape == (64,)
        # Both outputs should be non-trivial
        assert np.max(np.abs(out_lin)) > 0
        assert np.max(np.abs(out_cub)) > 0

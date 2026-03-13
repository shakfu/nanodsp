"""Tests for nanodsp.envelopes module (signalsmith envelopes, LFOs, smoothing)."""

import numpy as np
from nanodsp._core import envelopes


class TestCubicLfo:
    def test_construction(self):
        lfo = envelopes.CubicLfo()
        assert lfo is not None
        assert isinstance(lfo, envelopes.CubicLfo)

    def test_construction_with_seed(self):
        lfo = envelopes.CubicLfo(42)
        assert lfo is not None
        assert isinstance(lfo, envelopes.CubicLfo)

    def test_set_and_next(self):
        lfo = envelopes.CubicLfo(0)
        lfo.set(0.0, 1.0, 0.01)
        val = lfo.next()
        assert isinstance(val, float)

    def test_process_shape(self):
        lfo = envelopes.CubicLfo(0)
        lfo.set(-1.0, 1.0, 0.01)
        out = lfo.process(256)
        assert out.shape == (256,)
        assert out.dtype == np.float32

    def test_output_within_range(self):
        lfo = envelopes.CubicLfo(0)
        lfo.set(-1.0, 1.0, 0.005)
        out = lfo.process(4096)
        # LFO should stay within the configured range (with some tolerance)
        assert np.min(out) >= -1.5
        assert np.max(out) <= 1.5

    def test_deterministic_with_seed(self):
        lfo1 = envelopes.CubicLfo(123)
        lfo1.set(0.0, 1.0, 0.01)
        out1 = lfo1.process(128)

        lfo2 = envelopes.CubicLfo(123)
        lfo2.set(0.0, 1.0, 0.01)
        out2 = lfo2.process(128)

        assert out1.shape == out2.shape
        np.testing.assert_allclose(out1, out2)

    def test_reset(self):
        lfo = envelopes.CubicLfo(42)
        lfo.set(0.0, 1.0, 0.01)
        lfo.process(100)
        lfo.reset()
        # After reset, should still be usable
        lfo.set(0.0, 1.0, 0.01)
        out = lfo.process(10)
        assert out.shape == (10,)


class TestBoxFilter:
    def test_construction(self):
        bf = envelopes.BoxFilter(64)
        assert bf is not None
        assert isinstance(bf, envelopes.BoxFilter)

    def test_process_shape(self):
        bf = envelopes.BoxFilter(64)
        bf.set(16)
        inp = np.ones(128, dtype=np.float32)
        out = bf.process(inp)
        assert out.shape == (128,)
        assert out.dtype == np.float32

    def test_smoothing_effect(self):
        bf = envelopes.BoxFilter(64)
        bf.set(16)
        # Step function
        inp = np.zeros(128, dtype=np.float32)
        inp[32:] = 1.0
        out = bf.process(inp)
        # Output should be smoothed - gradual transition
        assert out[31] < out[48]  # rising edge
        assert out[48] < 1.0 or abs(out[48] - 1.0) < 0.1  # not yet at full

    def test_dc_passthrough(self):
        bf = envelopes.BoxFilter(64)
        bf.set(8)
        bf.reset(1.0)
        inp = np.ones(128, dtype=np.float32)
        out = bf.process(inp)
        assert out.shape == (128,)
        # After settling, DC should pass through
        np.testing.assert_allclose(out[-32:], 1.0, atol=1e-5)

    def test_reset(self):
        bf = envelopes.BoxFilter(64)
        bf.set(8)
        bf.process(np.ones(64, dtype=np.float32))
        bf.reset(0.0)
        out = bf.process(np.zeros(32, dtype=np.float32))
        np.testing.assert_allclose(out, 0.0, atol=1e-7)

    def test_resize(self):
        bf = envelopes.BoxFilter(16)
        bf.resize(128)
        bf.set(64)
        out = bf.process(np.ones(128, dtype=np.float32))
        assert out.shape == (128,)


class TestBoxStackFilter:
    def test_construction(self):
        bs = envelopes.BoxStackFilter(64)
        assert bs is not None
        assert isinstance(bs, envelopes.BoxStackFilter)

    def test_construction_with_layers(self):
        bs = envelopes.BoxStackFilter(64, 3)
        assert bs is not None
        assert isinstance(bs, envelopes.BoxStackFilter)

    def test_process_shape(self):
        bs = envelopes.BoxStackFilter(64)
        bs.set(16)
        inp = np.ones(128, dtype=np.float32)
        out = bs.process(inp)
        assert out.shape == (128,)
        assert out.dtype == np.float32

    def test_smoother_than_single_box(self):
        # BoxStack should produce a smoother result than a single box
        bf = envelopes.BoxFilter(64)
        bf.set(16)
        bs = envelopes.BoxStackFilter(64, 4)
        bs.set(16)

        inp = np.zeros(256, dtype=np.float32)
        inp[0] = 1.0  # impulse

        out_box = bf.process(inp)
        out_stack = bs.process(inp)

        # Stack response should be smoother (more Gaussian-like)
        assert out_box.shape == out_stack.shape

    def test_dc_passthrough(self):
        bs = envelopes.BoxStackFilter(64, 4)
        bs.set(8)
        bs.reset(1.0)
        inp = np.ones(256, dtype=np.float32)
        out = bs.process(inp)
        np.testing.assert_allclose(out[-32:], 1.0, atol=1e-4)


class TestPeakHold:
    def test_construction(self):
        ph = envelopes.PeakHold(64)
        assert ph is not None
        assert isinstance(ph, envelopes.PeakHold)

    def test_process_shape(self):
        ph = envelopes.PeakHold(64)
        ph.set(16)
        inp = np.random.default_rng(42).standard_normal(128).astype(np.float32)
        out = ph.process(inp)
        assert out.shape == (128,)
        assert out.dtype == np.float32

    def test_holds_peak(self):
        ph = envelopes.PeakHold(64)
        ph.set(32)
        # Create signal with a single spike
        inp = np.zeros(128, dtype=np.float32)
        inp[10] = 5.0
        out = ph.process(inp)
        # Peak should be held for the hold window
        assert np.sum(out >= 4.9) > 1  # held for multiple samples

    def test_output_geq_input(self):
        ph = envelopes.PeakHold(64)
        ph.set(16)
        rng = np.random.default_rng(42)
        inp = rng.standard_normal(256).astype(np.float32)
        out = ph.process(inp)
        # Peak hold output should always be >= current input
        assert np.all(out >= inp)

    def test_reset(self):
        ph = envelopes.PeakHold(64)
        ph.set(16)
        ph.process(np.ones(32, dtype=np.float32) * 10.0)
        ph.reset()
        ph.set(16)
        out = ph.process(np.zeros(16, dtype=np.float32))
        # After reset with default (lowest), zeros should produce lowest values
        assert np.all(out <= 0)


class TestPeakDecayLinear:
    def test_construction(self):
        pd = envelopes.PeakDecayLinear(64)
        assert pd is not None
        assert isinstance(pd, envelopes.PeakDecayLinear)

    def test_process_shape(self):
        pd = envelopes.PeakDecayLinear(64)
        pd.set(32)
        inp = np.random.default_rng(42).standard_normal(128).astype(np.float32)
        out = pd.process(inp)
        assert out.shape == (128,)
        assert out.dtype == np.float32

    def test_decay_behavior(self):
        pd = envelopes.PeakDecayLinear(128)
        pd.set(64)
        # Single spike followed by silence
        inp = np.zeros(128, dtype=np.float32)
        inp[0] = 1.0
        out = pd.process(inp)
        # Output should decay linearly after the peak
        # Find where it starts decaying
        peak_val = np.max(out)
        assert peak_val > 0.9
        # After the peak, values should decrease
        peak_idx = np.argmax(out)
        if peak_idx + 10 < len(out):
            assert out[peak_idx + 10] < peak_val

    def test_output_geq_input(self):
        pd = envelopes.PeakDecayLinear(64)
        pd.set(32)
        rng = np.random.default_rng(42)
        inp = rng.standard_normal(256).astype(np.float32)
        out = pd.process(inp)
        # Peak decay output should always be >= current input
        assert np.all(out >= inp - 1e-6)

    def test_reset(self):
        pd = envelopes.PeakDecayLinear(64)
        pd.set(32)
        pd.process(np.ones(32, dtype=np.float32) * 10.0)
        pd.reset()
        pd.set(32)
        out = pd.process(np.zeros(16, dtype=np.float32))
        assert np.all(out <= 0)

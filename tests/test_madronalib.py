"""Tests for nanodsp.madronalib bindings."""

import numpy as np
import pytest

from nanodsp._core import madronalib

ml = madronalib


# ============================================================================
# Constants
# ============================================================================


class TestConstants:
    def test_block_size(self):
        assert ml.BLOCK_SIZE == 64


# ============================================================================
# Scalar Math
# ============================================================================


class TestScalarMath:
    def test_amp_to_db_unity(self):
        assert ml.amp_to_db(1.0) == pytest.approx(0.0)

    def test_db_to_amp_zero(self):
        assert ml.db_to_amp(0.0) == pytest.approx(1.0)

    def test_roundtrip(self):
        for amp in [0.1, 0.5, 1.0, 2.0, 10.0]:
            db = ml.amp_to_db(amp)
            assert ml.db_to_amp(db) == pytest.approx(amp, rel=1e-5)

    def test_amp_to_db_vectorized(self):
        amps = np.array([0.1, 0.5, 1.0, 2.0], dtype=np.float32)
        dbs = ml.amp_to_db(amps)
        assert dbs.shape == (4,)
        assert dbs.dtype == np.float32
        assert dbs[2] == pytest.approx(0.0, abs=1e-5)

    def test_db_to_amp_vectorized(self):
        dbs = np.array([-20.0, -6.0, 0.0, 6.0], dtype=np.float32)
        amps = ml.db_to_amp(dbs)
        assert amps.shape == (4,)
        assert amps.dtype == np.float32
        assert amps[2] == pytest.approx(1.0, abs=1e-5)


# ============================================================================
# FDN Reverbs
# ============================================================================


class TestFDN4:
    def test_construction(self):
        fdn = ml.reverbs.FDN4()
        assert fdn is not None
        assert isinstance(fdn, ml.reverbs.FDN4)

    def test_stereo_output_shape(self):
        fdn = ml.reverbs.FDN4()
        fdn.set_delays_in_samples([100, 200, 300, 400])
        fdn.set_filter_cutoffs([0.5, 0.5, 0.5, 0.5])
        fdn.set_feedback_gains([0.9, 0.9, 0.9, 0.9])
        x = np.zeros(128, dtype=np.float32)
        x[0] = 1.0
        out = fdn.process(x)
        assert out.shape == (2, 128)
        assert out.dtype == np.float32

    def test_produces_output(self):
        fdn = ml.reverbs.FDN4()
        fdn.set_delays_in_samples([100, 200, 300, 400])
        fdn.set_filter_cutoffs([0.5, 0.5, 0.5, 0.5])
        fdn.set_feedback_gains([0.9, 0.9, 0.9, 0.9])
        x = np.zeros(512, dtype=np.float32)
        x[0] = 1.0
        out = fdn.process(x)
        assert np.any(out != 0)

    def test_non_aligned_length(self):
        """Non-64-aligned input should work (zero-padded internally)."""
        fdn = ml.reverbs.FDN4()
        fdn.set_delays_in_samples([100, 200, 300, 400])
        fdn.set_filter_cutoffs([0.5, 0.5, 0.5, 0.5])
        fdn.set_feedback_gains([0.9, 0.9, 0.9, 0.9])
        x = np.zeros(100, dtype=np.float32)
        x[0] = 1.0
        out = fdn.process(x)
        assert out.shape == (2, 100)

    def test_wrong_size_raises(self):
        fdn = ml.reverbs.FDN4()
        with pytest.raises(Exception):
            fdn.set_delays_in_samples([100, 200])  # need 4

    def test_feedback_gains_roundtrip(self):
        fdn = ml.reverbs.FDN4()
        fdn.set_feedback_gains([0.1, 0.2, 0.3, 0.4])
        gains = fdn.get_feedback_gains()
        assert len(gains) == 4
        assert gains[0] == pytest.approx(0.1)
        assert gains[3] == pytest.approx(0.4)


class TestFDN8:
    def test_construction_and_process(self):
        fdn = ml.reverbs.FDN8()
        delays = [100 + i * 50 for i in range(8)]
        fdn.set_delays_in_samples(delays)
        fdn.set_filter_cutoffs([0.5] * 8)
        fdn.set_feedback_gains([0.8] * 8)
        x = np.zeros(128, dtype=np.float32)
        x[0] = 1.0
        out = fdn.process(x)
        assert out.shape == (2, 128)

    def test_wrong_size_raises(self):
        fdn = ml.reverbs.FDN8()
        with pytest.raises(Exception):
            fdn.set_delays_in_samples([100] * 4)


class TestFDN16:
    def test_construction_and_process(self):
        fdn = ml.reverbs.FDN16()
        delays = [100 + i * 30 for i in range(16)]
        fdn.set_delays_in_samples(delays)
        fdn.set_filter_cutoffs([0.4] * 16)
        fdn.set_feedback_gains([0.7] * 16)
        x = np.zeros(128, dtype=np.float32)
        x[0] = 1.0
        out = fdn.process(x)
        assert out.shape == (2, 128)


# ============================================================================
# PitchbendableDelay
# ============================================================================


class TestPitchbendableDelay:
    def test_construction(self):
        d = ml.delays.PitchbendableDelay()
        assert d is not None
        assert isinstance(d, ml.delays.PitchbendableDelay)

    def test_process_shape(self):
        d = ml.delays.PitchbendableDelay()
        d.set_max_delay_in_samples(1000.0)
        inp = np.ones(128, dtype=np.float32) * 0.5
        delays = np.ones(128, dtype=np.float32) * 100.0
        out = d.process(inp, delays)
        assert out.shape == (128,)
        assert out.dtype == np.float32

    def test_length_mismatch_raises(self):
        d = ml.delays.PitchbendableDelay()
        d.set_max_delay_in_samples(1000.0)
        with pytest.raises(Exception):
            d.process(
                np.ones(64, dtype=np.float32),
                np.ones(128, dtype=np.float32),
            )

    def test_non_aligned_length(self):
        d = ml.delays.PitchbendableDelay()
        d.set_max_delay_in_samples(1000.0)
        inp = np.ones(100, dtype=np.float32)
        delays = np.ones(100, dtype=np.float32) * 50.0
        out = d.process(inp, delays)
        assert out.shape == (100,)

    def test_clear(self):
        d = ml.delays.PitchbendableDelay()
        d.set_max_delay_in_samples(500.0)
        d.clear()
        # After clear, processing should produce zeros for zero input
        inp = np.zeros(128, dtype=np.float32)
        delays = np.ones(128, dtype=np.float32) * 50.0
        out = d.process(inp, delays)
        assert out.shape == (128,)
        assert out.dtype == np.float32


# ============================================================================
# Downsampler
# ============================================================================


class TestDownsampler:
    def test_2x_downsampling(self):
        ds = ml.resampling.Downsampler(1)
        inp = np.ones(128, dtype=np.float32)
        out = ds.process(inp)
        assert out.shape == (64,)
        assert out.dtype == np.float32

    def test_4x_downsampling(self):
        ds = ml.resampling.Downsampler(2)
        inp = np.ones(256, dtype=np.float32)
        out = ds.process(inp)
        assert out.shape == (64,)

    def test_non_aligned_raises(self):
        ds = ml.resampling.Downsampler(1)
        with pytest.raises(Exception):
            ds.process(np.ones(100, dtype=np.float32))

    def test_clear(self):
        ds = ml.resampling.Downsampler(1)
        ds.clear()
        # After clear, processing should still work
        inp = np.ones(128, dtype=np.float32)
        out = ds.process(inp)
        assert out.shape == (64,)
        assert out.dtype == np.float32


# ============================================================================
# Upsampler
# ============================================================================


class TestUpsampler:
    def test_2x_upsampling(self):
        us = ml.resampling.Upsampler(1)
        inp = np.ones(64, dtype=np.float32)
        out = us.process(inp, 1)
        assert out.shape == (128,)
        assert out.dtype == np.float32

    def test_4x_upsampling(self):
        us = ml.resampling.Upsampler(2)
        inp = np.ones(64, dtype=np.float32)
        out = us.process(inp, 2)
        assert out.shape == (256,)

    def test_non_aligned_raises(self):
        us = ml.resampling.Upsampler(1)
        with pytest.raises(Exception):
            us.process(np.ones(100, dtype=np.float32), 1)

    def test_clear(self):
        us = ml.resampling.Upsampler(1)
        us.clear()
        # After clear, processing should still work
        inp = np.ones(64, dtype=np.float32)
        out = us.process(inp, 1)
        assert out.shape == (128,)
        assert out.dtype == np.float32


# ============================================================================
# OneShotGen
# ============================================================================


class TestOneShotGen:
    def test_construction(self):
        g = ml.generators.OneShotGen()
        assert g is not None
        assert isinstance(g, ml.generators.OneShotGen)

    def test_trigger_and_ramp(self):
        g = ml.generators.OneShotGen()
        g.trigger()
        # cycles_per_sample = 1/128 -> full ramp in 128 samples
        cps = np.full(128, 1.0 / 128, dtype=np.float32)
        out = g.process(cps)
        assert out.shape == (128,)
        # ramp should start near 0 and reach near 1
        assert out[0] < 0.1
        # the ramp peaks just before wrapping to 0 at the end
        assert np.max(out) > 0.9

    def test_process_sample(self):
        g = ml.generators.OneShotGen()
        g.trigger()
        val = g.process_sample(0.1)
        assert isinstance(val, float)

    def test_no_trigger_no_output(self):
        g = ml.generators.OneShotGen()
        cps = np.full(64, 1.0 / 64, dtype=np.float32)
        out = g.process(cps)
        # without trigger, output should be all zeros
        assert np.allclose(out, 0.0)


# ============================================================================
# LinearGlide
# ============================================================================


class TestLinearGlide:
    def test_construction(self):
        g = ml.generators.LinearGlide()
        assert g is not None
        assert isinstance(g, ml.generators.LinearGlide)

    def test_glide_to_target(self):
        g = ml.generators.LinearGlide()
        g.set_glide_time_in_samples(128.0)
        out = g.process(1.0, 256)
        assert out.shape == (256,)
        # should reach target by the end
        assert out[-1] == pytest.approx(1.0, abs=0.01)

    def test_set_value_immediate(self):
        g = ml.generators.LinearGlide()
        g.set_glide_time_in_samples(1000.0)
        g.set_value(5.0)
        out = g.process(5.0, 64)
        # setValue jumps immediately
        assert out[0] == pytest.approx(5.0, abs=0.01)

    def test_clear(self):
        g = ml.generators.LinearGlide()
        g.clear()
        # After clear, glide should start from zero
        g.set_glide_time_in_samples(1.0)
        out = g.process(0.0, 64)
        assert out.shape == (64,)
        assert out.dtype == np.float32


# ============================================================================
# SampleAccurateLinearGlide
# ============================================================================


class TestSampleAccurateLinearGlide:
    def test_construction(self):
        g = ml.generators.SampleAccurateLinearGlide()
        assert g is not None
        assert isinstance(g, ml.generators.SampleAccurateLinearGlide)

    def test_glide(self):
        g = ml.generators.SampleAccurateLinearGlide()
        g.set_glide_time_in_samples(10.0)
        out = g.process(1.0, 20)
        assert out.shape == (20,)
        # Should reach target
        assert out[-1] == pytest.approx(1.0, abs=0.01)

    def test_process_sample(self):
        g = ml.generators.SampleAccurateLinearGlide()
        g.set_glide_time_in_samples(5.0)
        val = g.process_sample(1.0)
        assert isinstance(val, float)

    def test_clear(self):
        g = ml.generators.SampleAccurateLinearGlide()
        g.clear()
        # After clear, should still be usable
        g.set_glide_time_in_samples(5.0)
        out = g.process(0.0, 10)
        assert out.shape == (10,)
        assert out.dtype == np.float32


# ============================================================================
# TempoLock
# ============================================================================


class TestTempoLock:
    def test_construction(self):
        tl = ml.generators.TempoLock()
        assert tl is not None
        assert isinstance(tl, ml.generators.TempoLock)

    def test_process_shape(self):
        tl = ml.generators.TempoLock()
        phasor = np.linspace(0, 1, 128, endpoint=False, dtype=np.float32)
        out = tl.process(phasor, 1.0, 1.0 / 44100.0)
        assert out.shape == (128,)
        assert out.dtype == np.float32

    def test_clear(self):
        tl = ml.generators.TempoLock()
        tl.clear()
        # After clear, processing should still work
        phasor = np.linspace(0, 1, 128, endpoint=False, dtype=np.float32)
        out = tl.process(phasor, 1.0, 1.0 / 44100.0)
        assert out.shape == (128,)
        assert out.dtype == np.float32


# ============================================================================
# Projections
# ============================================================================


class TestProjections:
    def test_smoothstep_boundaries(self):
        assert ml.projections.smoothstep(0.0) == pytest.approx(0.0)
        assert ml.projections.smoothstep(1.0) == pytest.approx(1.0)
        assert ml.projections.smoothstep(0.5) == pytest.approx(0.5)

    def test_bell_peak(self):
        # bell peaks at x=0.5
        assert ml.projections.bell(0.5) == pytest.approx(1.0, abs=0.01)
        assert ml.projections.bell(0.0) < 0.01
        assert ml.projections.bell(1.0) < 0.01

    def test_ease_in_boundaries(self):
        assert ml.projections.ease_in(0.0) == pytest.approx(0.0)
        assert ml.projections.ease_in(1.0) == pytest.approx(1.0)

    def test_ease_out_boundaries(self):
        assert ml.projections.ease_out(0.0) == pytest.approx(0.0)
        assert ml.projections.ease_out(1.0) == pytest.approx(1.0)

    def test_ease_in_out_boundaries(self):
        assert ml.projections.ease_in_out(0.0) == pytest.approx(0.0)
        assert ml.projections.ease_in_out(1.0) == pytest.approx(1.0)

    def test_flip(self):
        assert ml.projections.flip(0.0) == pytest.approx(1.0)
        assert ml.projections.flip(1.0) == pytest.approx(0.0)

    def test_squared(self):
        assert ml.projections.squared(0.5) == pytest.approx(0.25)

    def test_clip(self):
        assert ml.projections.clip(-0.5) == pytest.approx(0.0)
        assert ml.projections.clip(1.5) == pytest.approx(1.0)
        assert ml.projections.clip(0.5) == pytest.approx(0.5)

    def test_vectorized_output(self):
        arr = np.linspace(0, 1, 10, dtype=np.float32)
        out = ml.projections.smoothstep(arr)
        assert out.shape == (10,)
        assert out.dtype == np.float32
        assert out[0] == pytest.approx(0.0, abs=1e-5)
        assert out[-1] == pytest.approx(1.0, abs=1e-5)

    def test_all_projections_callable(self):
        """Every projection should be callable with scalar 0.5."""
        names = [
            "smoothstep",
            "bell",
            "ease_in",
            "ease_out",
            "ease_in_out",
            "ease_in_cubic",
            "ease_out_cubic",
            "ease_in_out_cubic",
            "ease_in_quartic",
            "ease_out_quartic",
            "ease_in_out_quartic",
            "overshoot",
            "flip",
            "squared",
            "flatcenter",
            "bisquared",
            "inv_bisquared",
            "clip",
        ]
        for name in names:
            fn = getattr(ml.projections, name)
            val = fn(0.5)
            assert isinstance(val, float), f"{name} did not return float"

    def test_overshoot(self):
        # overshoot(1) = 3*1 - 2*1 = 1
        assert ml.projections.overshoot(1.0) == pytest.approx(1.0)
        # overshoot peaks above 1 near x~0.75
        assert ml.projections.overshoot(0.75) > 1.0

    def test_bisquared_inv_bisquared_roundtrip(self):
        for x in [-0.5, 0.0, 0.5, 0.8]:
            val = ml.projections.inv_bisquared(ml.projections.bisquared(x))
            assert val == pytest.approx(x, abs=1e-4)

    def test_cubic_easing(self):
        assert ml.projections.ease_in_cubic(0.0) == pytest.approx(0.0)
        assert ml.projections.ease_in_cubic(1.0) == pytest.approx(1.0)
        assert ml.projections.ease_out_cubic(0.0) == pytest.approx(0.0)
        assert ml.projections.ease_out_cubic(1.0) == pytest.approx(1.0)

    def test_quartic_easing(self):
        assert ml.projections.ease_in_quartic(0.0) == pytest.approx(0.0)
        assert ml.projections.ease_in_quartic(1.0) == pytest.approx(1.0)
        assert ml.projections.ease_out_quartic(0.0) == pytest.approx(0.0)
        assert ml.projections.ease_out_quartic(1.0) == pytest.approx(1.0)


# ============================================================================
# Windows
# ============================================================================


class TestWindows:
    def test_hamming_shape_dtype(self):
        w = ml.windows.hamming(64)
        assert w.shape == (64,)
        assert w.dtype == np.float32

    def test_blackman_shape(self):
        w = ml.windows.blackman(128)
        assert w.shape == (128,)

    def test_flat_top_shape(self):
        w = ml.windows.flat_top(256)
        assert w.shape == (256,)

    def test_triangle_peak(self):
        w = ml.windows.triangle(64)
        # peak should be near the middle
        mid = len(w) // 2
        assert w[mid] > w[0]

    def test_raised_cosine_endpoints(self):
        w = ml.windows.raised_cosine(64)
        # raised cosine should be near 0 at endpoints
        assert w[0] < 0.01

    def test_rectangle(self):
        w = ml.windows.rectangle(64)
        # rectangle is 1 in the middle, 0 at edges
        assert w[32] == pytest.approx(1.0)

    def test_all_windows_callable(self):
        names = [
            "hamming",
            "blackman",
            "flat_top",
            "triangle",
            "raised_cosine",
            "rectangle",
        ]
        for name in names:
            fn = getattr(ml.windows, name)
            w = fn(64)
            assert w.shape == (64,)
            assert w.dtype == np.float32

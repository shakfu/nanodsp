"""Tests for HISSTools Library bindings."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from nanodsp._core import hisstools


class TestWindows:
    """Tests for hisstools.windows submodule."""

    def test_hann_shape(self):
        w = hisstools.windows.hann(256)
        assert w.shape == (256,)
        assert w.dtype == np.float32

    def test_hann_symmetry(self):
        w = hisstools.windows.hann(256)
        assert w.shape == (256,)
        assert_allclose(w, w[::-1], atol=1e-6)

    def test_hann_endpoints(self):
        w = hisstools.windows.hann(256)
        assert w[0] == pytest.approx(0.0, abs=1e-6)
        assert w[-1] == pytest.approx(0.0, abs=1e-6)

    def test_hann_peak(self):
        w = hisstools.windows.hann(257)
        assert w[128] == pytest.approx(1.0, abs=1e-6)

    def test_rect_all_ones(self):
        w = hisstools.windows.rect(64)
        assert w.shape == (64,)
        assert_allclose(w, np.ones(64, dtype=np.float32))

    def test_all_simple_windows_callable(self):
        """All simple window functions return correct-sized arrays."""
        fns = [
            hisstools.windows.rect,
            hisstools.windows.triangle,
            hisstools.windows.welch,
            hisstools.windows.parzen,
            hisstools.windows.sine,
            hisstools.windows.hann,
            hisstools.windows.hamming,
            hisstools.windows.blackman,
            hisstools.windows.exact_blackman,
            hisstools.windows.blackman_harris_62dB,
            hisstools.windows.blackman_harris_71dB,
            hisstools.windows.blackman_harris_74dB,
            hisstools.windows.blackman_harris_92dB,
            hisstools.windows.nuttall_1st_64dB,
            hisstools.windows.nuttall_1st_93dB,
            hisstools.windows.nuttall_3rd_47dB,
            hisstools.windows.nuttall_3rd_83dB,
            hisstools.windows.nuttall_5th_61dB,
            hisstools.windows.nuttall_minimal_71dB,
            hisstools.windows.nuttall_minimal_98dB,
            hisstools.windows.ni_flat_top,
            hisstools.windows.hp_flat_top,
            hisstools.windows.stanford_flat_top,
            hisstools.windows.heinzel_flat_top_70dB,
            hisstools.windows.heinzel_flat_top_90dB,
            hisstools.windows.heinzel_flat_top_95dB,
        ]
        for fn in fns:
            w = fn(32)
            assert w.shape == (32,), f"{fn.__name__} returned wrong shape"
            assert w.dtype == np.float32

    def test_kaiser_beta(self):
        w = hisstools.windows.kaiser(64, beta=5.0)
        assert w.shape == (64,)
        assert w[0] < 0.1  # small at edges
        assert w[32] > 0.9  # large at center

    def test_tukey_alpha_zero(self):
        """tukey(alpha=0) should be all ones (like rect)."""
        w = hisstools.windows.tukey(64, alpha=0.0)
        assert w.shape == (64,)
        assert_allclose(w, np.ones(64, dtype=np.float32), atol=1e-6)

    def test_tukey_alpha_one(self):
        """tukey(alpha=1) should approximate hann."""
        w_tukey = hisstools.windows.tukey(256, alpha=1.0)
        w_hann = hisstools.windows.hann(256)
        assert w_tukey.shape == w_hann.shape
        assert_allclose(w_tukey, w_hann, atol=1e-5)

    def test_trapezoid_params(self):
        w = hisstools.windows.trapezoid(64, a=0.25, b=0.75)
        assert w.shape == (64,)
        # Middle portion should be 1.0
        mid = w[20:44]
        assert_allclose(mid, np.ones_like(mid), atol=1e-5)

    def test_window_size_one(self):
        w = hisstools.windows.hann(1)
        assert w.shape == (1,)


class TestAnalysis:
    """Tests for hisstools.analysis submodule."""

    def test_stat_mean(self):
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        assert hisstools.analysis.stat_mean(x) == pytest.approx(2.5)

    def test_stat_rms(self):
        x = np.ones(100, dtype=np.float32) * 3.0
        assert hisstools.analysis.stat_rms(x) == pytest.approx(3.0, abs=1e-5)

    def test_stat_sum(self):
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert hisstools.analysis.stat_sum(x) == pytest.approx(6.0)

    def test_stat_min_max(self):
        x = np.array([3.0, 1.0, 4.0, 1.0, 5.0], dtype=np.float32)
        assert hisstools.analysis.stat_min(x) == pytest.approx(1.0)
        assert hisstools.analysis.stat_max(x) == pytest.approx(5.0)

    def test_stat_variance(self):
        x = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0], dtype=np.float32)
        var = hisstools.analysis.stat_variance(x)
        expected = np.var(x)
        assert var == pytest.approx(expected, rel=1e-5)

    def test_stat_standard_deviation(self):
        x = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0], dtype=np.float32)
        std = hisstools.analysis.stat_standard_deviation(x)
        expected = np.std(x)
        assert std == pytest.approx(expected, rel=1e-5)

    def test_stat_centroid(self):
        # Centroid of [0,0,0,1,0,0,0] should be ~3
        x = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        c = hisstools.analysis.stat_centroid(x)
        assert c == pytest.approx(3.0)

    def test_stat_flatness(self):
        # Flatness of constant signal should be 1.0
        x = np.ones(64, dtype=np.float32) * 5.0
        assert hisstools.analysis.stat_flatness(x) == pytest.approx(1.0, abs=1e-5)

    def test_stat_crest(self):
        # Crest of constant signal should be 1.0
        x = np.ones(64, dtype=np.float32) * 3.0
        assert hisstools.analysis.stat_crest(x) == pytest.approx(1.0, abs=1e-5)

    def test_stat_product(self):
        x = np.array([2.0, 3.0, 4.0], dtype=np.float32)
        assert hisstools.analysis.stat_product(x) == pytest.approx(24.0)

    def test_stat_geometric_mean(self):
        x = np.array([4.0, 9.0], dtype=np.float32)
        assert hisstools.analysis.stat_geometric_mean(x) == pytest.approx(6.0, rel=1e-5)

    def test_stat_pdf_percentile(self):
        # Uniform distribution: 50th percentile should be near middle
        x = np.ones(100, dtype=np.float32)
        p = hisstools.analysis.stat_pdf_percentile(x, 50.0)
        assert 48 <= p <= 52

    def test_stat_length(self):
        x = np.zeros(42, dtype=np.float32)
        assert hisstools.analysis.stat_length(x) == pytest.approx(42.0)

    def test_partial_tracker_basic(self):
        pt = hisstools.analysis.PartialTracker(8, 16)
        peaks = [
            hisstools.analysis.Peak(440.0, 0.5),
            hisstools.analysis.Peak(880.0, 0.3),
        ]
        pt.process(peaks, 0.01)
        t0 = pt.get_track(0)
        assert t0.active()
        assert t0.state == hisstools.analysis.TrackState.start
        assert t0.peak.freq() == pytest.approx(440.0)

    def test_partial_tracker_track_states(self):
        pt = hisstools.analysis.PartialTracker(4, 8)
        peaks1 = [hisstools.analysis.Peak(440.0, 0.5)]
        pt.process(peaks1, 0.01)
        assert pt.get_track(0).state == hisstools.analysis.TrackState.start

        # Process again with same freq -> should continue
        peaks2 = [hisstools.analysis.Peak(441.0, 0.5)]
        pt.process(peaks2, 0.01)
        assert pt.get_track(0).state == hisstools.analysis.TrackState.continue_

    def test_partial_tracker_reset(self):
        pt = hisstools.analysis.PartialTracker(4, 8)
        peaks = [hisstools.analysis.Peak(440.0, 0.5)]
        pt.process(peaks, 0.01)
        assert pt.get_track(0).active()
        pt.reset()
        assert not pt.get_track(0).active()


class TestSpectral:
    """Tests for hisstools.spectral submodule."""

    def test_spectral_processor_construction(self):
        sp = hisstools.spectral.SpectralProcessor()
        assert sp.max_fft_size() == 32768

    def test_convolve_linear_length(self):
        sp = hisstools.spectral.SpectralProcessor()
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([1.0, 1.0], dtype=np.float32)
        out = sp.convolve(a, b)
        # Linear convolution: len(a) + len(b) - 1 = 4
        assert out.shape == (4,)

    def test_convolve_delta_passthrough(self):
        sp = hisstools.spectral.SpectralProcessor()
        delta = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        signal = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        out = sp.convolve(delta, signal)
        assert len(out) >= 3
        # First 3 samples should match signal
        assert_allclose(out[:3], signal, atol=1e-5)

    def test_correlate_shape(self):
        sp = hisstools.spectral.SpectralProcessor()
        a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        b = np.array([1.0, 0.0], dtype=np.float32)
        out = sp.correlate(a, b)
        expected_size = sp.correlated_size(4, 2)
        assert out.shape == (expected_size,)

    def test_convolved_size(self):
        sp = hisstools.spectral.SpectralProcessor()
        assert sp.convolved_size(10, 5) == 14  # 10 + 5 - 1

    def test_change_phase(self):
        sp = hisstools.spectral.SpectralProcessor()
        x = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        out = sp.change_phase(x, 0.0)  # zero phase = same
        # First sample should be preserved (delta stays delta with zero phase)
        assert out[0] == pytest.approx(1.0, abs=0.1)

    def test_kernel_smoother_construction(self):
        ks = hisstools.spectral.KernelSmoother()
        assert ks is not None
        assert isinstance(ks, hisstools.spectral.KernelSmoother)

    def test_kernel_smoother_smooth(self):
        ks = hisstools.spectral.KernelSmoother()
        data = np.random.randn(128).astype(np.float32)
        kernel = np.array([1.0], dtype=np.float32)
        out = ks.smooth(data, kernel, width_lo=3.0, width_hi=3.0)
        assert out.shape == (128,)


class TestConvolution:
    """Tests for hisstools.convolution submodule."""

    def test_mono_convolve_construction(self):
        mc = hisstools.convolution.MonoConvolve(1024)
        assert mc is not None
        assert isinstance(mc, hisstools.convolution.MonoConvolve)

    def test_mono_convolve_delta_passthrough(self):
        mc = hisstools.convolution.MonoConvolve(1024, latency=0)
        ir = np.zeros(32, dtype=np.float32)
        ir[0] = 1.0
        mc.set_ir(ir)
        inp = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32)
        out = mc.process(inp)
        assert out.shape == inp.shape
        assert_allclose(out, inp, atol=1e-5)

    def test_mono_convolve_reset(self):
        mc = hisstools.convolution.MonoConvolve(1024, latency=0)
        ir = np.zeros(32, dtype=np.float32)
        ir[0] = 1.0
        mc.set_ir(ir)
        mc.reset()
        # After reset, should still process correctly
        inp = np.ones(8, dtype=np.float32)
        out = mc.process(inp)
        assert out.shape == (8,)

    def test_mono_convolve_empty_input(self):
        mc = hisstools.convolution.MonoConvolve(1024, latency=0)
        ir = np.array([1.0], dtype=np.float32)
        mc.set_ir(ir)
        inp = np.array([], dtype=np.float32)
        out = mc.process(inp)
        assert out.shape == (0,)

    def test_convolver_construction(self):
        cv = hisstools.convolution.Convolver(2, 2, latency=0)
        assert cv.num_ins == 2
        assert cv.num_outs == 2

    def test_convolver_set_ir_process(self):
        cv = hisstools.convolution.Convolver(1, 1, latency=0)
        ir = np.zeros(32, dtype=np.float32)
        ir[0] = 1.0
        cv.set_ir(0, 0, ir)
        inp = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]], dtype=np.float32)
        out = cv.process(inp)
        assert out.shape == (1, 8)
        assert_allclose(out[0], inp[0], atol=1e-5)

    def test_convolver_channel_mismatch_raises(self):
        cv = hisstools.convolution.Convolver(2, 1, latency=0)
        inp = np.ones((1, 8), dtype=np.float32)  # 1 channel, expects 2
        with pytest.raises(ValueError):
            cv.process(inp)

    def test_convolver_reset(self):
        cv = hisstools.convolution.Convolver(1, 1, latency=0)
        ir = np.zeros(32, dtype=np.float32)
        ir[0] = 1.0
        cv.set_ir(0, 0, ir)
        cv.reset()
        # After reset, processing should still work and return correct shape
        inp = np.ones((1, 8), dtype=np.float32)
        out = cv.process(inp)
        assert out.shape == (1, 8)
        assert out.dtype == np.float32

    def test_convolver_clear(self):
        cv = hisstools.convolution.Convolver(1, 1, latency=0)
        ir = np.array([1.0], dtype=np.float32)
        cv.set_ir(0, 0, ir)
        cv.clear()
        # After clear, processing should produce silence
        inp = np.ones((1, 8), dtype=np.float32)
        out = cv.process(inp)
        assert out.shape == (1, 8)
        assert out.dtype == np.float32

    def test_latency_modes(self):
        for mode_val in [0, 1, 2]:
            mc = hisstools.convolution.MonoConvolve(1024, latency=mode_val)
            assert mc is not None
            assert isinstance(mc, hisstools.convolution.MonoConvolve)
            # Should be able to process after construction with any latency mode
            ir = np.array([1.0], dtype=np.float32)
            mc.set_ir(ir)
            out = mc.process(np.ones(8, dtype=np.float32))
            assert out.shape == (8,)
            assert out.dtype == np.float32

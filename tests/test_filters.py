"""Tests for nanodsp.filters module (signalsmith biquad filters)."""

import numpy as np
import pytest
from nanodsp._core import filters


def make_impulse(n=1024):
    x = np.zeros(n, dtype=np.float32)
    x[0] = 1.0
    return x


def make_sine(freq, n=4096, sr=48000.0):
    """Generate a sine at normalized frequency freq (0-0.5) relative to sr."""
    t = np.arange(n, dtype=np.float32)
    return np.sin(2 * np.pi * freq * sr * t / sr).astype(np.float32)


class TestBiquadConstruction:
    def test_default_construction(self):
        bq = filters.Biquad()
        assert bq is not None
        assert isinstance(bq, filters.Biquad)

    def test_lowpass_returns_self(self):
        bq = filters.Biquad()
        result = bq.lowpass(0.1)
        assert result is bq

    def test_chaining(self):
        bq = filters.Biquad()
        result = bq.lowpass(0.1).add_gain(0.5)
        assert result is bq


class TestBiquadFilterTypes:
    """Verify each filter type can be configured and processes without error."""

    @pytest.mark.parametrize(
        "filter_method, args, check_dtype",
        [
            ("lowpass", (0.1,), True),
            ("lowpass_q", (0.1, 0.707), False),
            ("highpass", (0.1,), False),
            ("bandpass", (0.25,), False),
            ("notch", (0.25,), False),
            ("peak", (0.25, 2.0), False),
            ("peak_db", (0.25, 6.0), False),
            ("high_shelf", (0.25, 2.0), False),
            ("high_shelf_db", (0.25, 6.0), False),
            ("low_shelf", (0.25, 2.0), False),
            ("low_shelf_db", (0.25, 6.0), False),
            ("allpass", (0.25,), False),
            ("allpass_q", (0.25, 1.0), False),
        ],
        ids=[
            "lowpass",
            "lowpass_q",
            "highpass",
            "bandpass",
            "notch",
            "peak",
            "peak_db",
            "high_shelf",
            "high_shelf_db",
            "low_shelf",
            "low_shelf_db",
            "allpass",
            "allpass_q",
        ],
    )
    def test_filter_type(self, filter_method, args, check_dtype):
        bq = filters.Biquad()
        getattr(bq, filter_method)(*args)
        out = bq.process(make_impulse())
        assert out.shape == (1024,)
        if check_dtype:
            assert out.dtype == np.float32


class TestBiquadBehavior:
    def test_lowpass_attenuates_high_frequencies(self):
        bq = filters.Biquad()
        bq.lowpass(0.1)  # cutoff at 0.1 * Nyquist
        # Response well below cutoff should be near 0 dB
        low_resp = bq.response_db(0.01)
        # Response well above cutoff should be significantly attenuated
        high_resp = bq.response_db(0.4)
        assert low_resp > high_resp
        assert abs(low_resp) < 1.0  # passband close to 0 dB
        assert high_resp < -10.0  # significant attenuation

    def test_highpass_attenuates_low_frequencies(self):
        bq = filters.Biquad()
        bq.highpass(0.3)
        low_resp = bq.response_db(0.01)
        high_resp = bq.response_db(0.45)
        assert high_resp > low_resp
        assert low_resp < -10.0

    def test_notch_attenuates_at_center(self):
        bq = filters.Biquad()
        bq.notch(0.25, 1.0)
        center_resp = bq.response_db(0.25)
        off_resp = bq.response_db(0.1)
        assert center_resp < -20.0
        assert abs(off_resp) < 3.0

    def test_allpass_unity_magnitude(self):
        bq = filters.Biquad()
        bq.allpass(0.25)
        for freq in [0.05, 0.1, 0.2, 0.3, 0.4]:
            resp = bq.response(freq)
            mag = abs(resp)
            assert abs(mag - 1.0) < 0.01, f"Allpass magnitude at {freq}: {mag}"

    def test_peak_boost(self):
        bq = filters.Biquad()
        bq.peak_db(0.25, 12.0)
        center_resp = bq.response_db(0.25)
        # Should be close to +12 dB at center
        assert center_resp > 10.0

    def test_reset_clears_state(self):
        bq = filters.Biquad()
        bq.lowpass(0.1)
        # Process some signal to build up state
        bq.process(np.ones(100, dtype=np.float32))
        bq.reset()
        # After reset, processing an impulse should give clean response
        out = bq.process(make_impulse(64))
        # First sample of impulse through lowpass should be small positive
        assert out[0] > 0


class TestBiquadBlockProcessing:
    def test_process_returns_correct_shape(self):
        bq = filters.Biquad()
        bq.lowpass(0.2)
        inp = np.random.randn(512).astype(np.float32)
        out = bq.process(inp)
        assert out.shape == inp.shape
        assert out.dtype == np.float32

    def test_process_inplace(self):
        bq = filters.Biquad()
        bq.lowpass(0.2)
        data = np.ones(64, dtype=np.float32)
        bq.process_inplace(data)
        # Data should be modified in-place
        assert not np.allclose(data, np.ones(64, dtype=np.float32))

    def test_process_matches_sample_by_sample(self):
        bq1 = filters.Biquad()
        bq1.lowpass(0.15)
        bq2 = filters.Biquad()
        bq2.lowpass(0.15)

        inp = np.random.randn(256).astype(np.float32)
        block_out = bq1.process(inp)

        # Sample-by-sample via repeated single-sample calls
        sample_out = np.zeros_like(inp)
        for i in range(len(inp)):
            sample_out[i : i + 1] = bq2.process(inp[i : i + 1])

        assert block_out.shape == sample_out.shape
        np.testing.assert_allclose(block_out, sample_out, atol=1e-6)


class TestBiquadDesign:
    def test_design_enum_values(self):
        assert filters.BiquadDesign.bilinear is not None
        assert filters.BiquadDesign.cookbook is not None
        assert filters.BiquadDesign.one_sided is not None
        assert filters.BiquadDesign.vicanek is not None

    def test_vicanek_design(self):
        bq = filters.Biquad()
        bq.lowpass(0.1, design=filters.BiquadDesign.vicanek)
        out = bq.process(make_impulse())
        assert out.shape == (1024,)

    def test_add_gain(self):
        bq = filters.Biquad()
        bq.lowpass(0.2)
        resp_before = bq.response_db(0.05)
        bq.add_gain_db(6.0)
        resp_after = bq.response_db(0.05)
        assert abs(resp_after - resp_before - 6.0) < 0.5

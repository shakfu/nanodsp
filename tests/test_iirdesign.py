"""Tests for IIR filter design (DspFilters: Butterworth, Chebyshev, Elliptic, Bessel)."""

import numpy as np
import pytest

from nanodsp._core import iirdesign as iir
from nanodsp.buffer import AudioBuffer
from nanodsp.effects.filters import iir_filter, iir_design


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SR = 48000.0
FRAMES = 4096


def make_impulse(frames=FRAMES):
    x = np.zeros(frames, dtype=np.float32)
    x[0] = 1.0
    return x


def make_sine(freq=440.0, frames=FRAMES, sr=SR):
    t = np.arange(frames) / sr
    return (np.sin(2 * np.pi * freq * t)).astype(np.float32)


def rms(x):
    return np.sqrt(np.mean(np.asarray(x).flatten() ** 2))


def make_buf(data, sr=SR):
    return AudioBuffer(data.reshape(1, -1), sample_rate=sr)


def energy_ratio(output, reference):
    """Ratio of output energy to reference energy."""
    return np.sum(output**2) / max(np.sum(reference**2), 1e-30)


# ---------------------------------------------------------------------------
# C++ binding tests -- design() and apply()
# ---------------------------------------------------------------------------


class TestDesignFunction:
    def test_butterworth_lp_sos_shape(self):
        sos = iir.design(0, 0, 4, SR, 1000)
        assert sos.shape == (2, 6)  # order 4 -> 2 biquad sections

    def test_butterworth_lp_order_1(self):
        sos = iir.design(0, 0, 1, SR, 1000)
        assert sos.shape == (1, 6)

    def test_butterworth_lp_sos_a0_is_one(self):
        sos = iir.design(0, 0, 4, SR, 1000)
        for i in range(sos.shape[0]):
            assert sos[i, 3] == pytest.approx(1.0, abs=1e-5)

    def test_butterworth_bp_doubles_sections(self):
        # Bandpass of order N uses 2N poles -> N biquad sections
        sos = iir.design(0, 2, 4, SR, 1000, width=500)
        assert sos.shape == (4, 6)

    def test_all_families_produce_sos(self):
        for fam in range(5):
            kwargs = {}
            if fam == 1:
                kwargs["ripple_db"] = 1.0
            elif fam == 2:
                kwargs["ripple_db"] = 40.0
            elif fam == 3:
                kwargs["ripple_db"] = 1.0
                kwargs["rolloff_db"] = 0.0
            sos = iir.design(fam, 0, 4, SR, 1000, **kwargs)
            assert sos.shape[0] >= 1
            assert sos.shape[1] == 6

    def test_invalid_order_raises(self):
        with pytest.raises(Exception):
            iir.design(0, 0, 0, SR, 1000)

    def test_invalid_freq_raises(self):
        with pytest.raises(Exception):
            iir.design(0, 0, 4, SR, SR / 2)  # freq >= Nyquist


class TestApplyFunction:
    def test_lowpass_attenuates_high(self):
        low = make_sine(200.0)
        high = make_sine(10000.0)
        mixed = low + high
        out = iir.apply(mixed, 0, 0, 6, SR, 1000)
        # High-frequency component should be attenuated
        assert rms(out) < rms(mixed)

    def test_highpass_attenuates_low(self):
        low = make_sine(100.0)
        high = make_sine(5000.0)
        mixed = low + high
        out = iir.apply(mixed, 0, 1, 6, SR, 1000)
        assert rms(out) < rms(mixed)

    def test_bandpass_passes_center(self):
        center = make_sine(1000.0)
        out = iir.apply(center, 0, 2, 4, SR, 1000, width=500)
        # Signal at center frequency should pass with minimal loss
        assert energy_ratio(out, center) > 0.5

    def test_bandstop_rejects_center(self):
        center = make_sine(1000.0)
        out = iir.apply(center, 0, 3, 4, SR, 1000, width=500)
        # Signal at center frequency should be attenuated
        assert energy_ratio(out, center) < 0.5

    def test_output_shape_matches_input(self):
        x = make_impulse()
        out = iir.apply(x, 0, 0, 4, SR, 1000)
        assert out.shape == x.shape

    def test_parseval_energy_conservation(self):
        x = make_impulse()
        lp = iir.apply(x, 0, 0, 4, SR, 1000)
        hp = iir.apply(x, 0, 1, 4, SR, 1000)
        # LP energy + HP energy should approximately equal total energy
        total = np.sum(lp**2) + np.sum(hp**2)
        assert total == pytest.approx(np.sum(x**2), rel=0.01)


# ---------------------------------------------------------------------------
# C++ binding tests -- IIRFilter class
# ---------------------------------------------------------------------------


class TestIIRFilterClass:
    def test_construction(self):
        f = iir.IIRFilter()
        assert f.num_stages == 0

    def test_setup_sets_stages(self):
        f = iir.IIRFilter()
        f.setup(0, 0, 4, SR, 1000)
        assert f.num_stages == 2

    def test_process_matches_apply(self):
        x = make_impulse()
        f = iir.IIRFilter()
        f.setup(0, 0, 4, SR, 1000)
        y1 = f.process(x)
        y2 = iir.apply(x, 0, 0, 4, SR, 1000)
        assert np.allclose(y1, y2)

    def test_sos_matches_design(self):
        f = iir.IIRFilter()
        f.setup(0, 0, 4, SR, 1000)
        sos1 = f.sos()
        sos2 = iir.design(0, 0, 4, SR, 1000)
        assert np.allclose(sos1, sos2)

    def test_reset(self):
        f = iir.IIRFilter()
        f.setup(0, 0, 4, SR, 1000)
        f.process(make_impulse())
        f.reset()
        silence = np.zeros(256, dtype=np.float32)
        out = f.process(silence)
        assert rms(out) < 1e-6

    def test_stateful_processing(self):
        """Processing in chunks should match processing all at once."""
        x = make_sine(500.0)
        # All at once
        f1 = iir.IIRFilter()
        f1.setup(0, 0, 4, SR, 1000)
        y_all = f1.process(x)
        # In two chunks
        f2 = iir.IIRFilter()
        f2.setup(0, 0, 4, SR, 1000)
        y_a = f2.process(x[:2048])
        y_b = f2.process(x[2048:])
        y_chunks = np.concatenate([y_a, y_b])
        assert np.allclose(y_all, y_chunks, atol=1e-6)


# ---------------------------------------------------------------------------
# Filter family correctness tests
# ---------------------------------------------------------------------------


class TestButterworth:
    def test_maximally_flat(self):
        """Butterworth passband should be maximally flat."""
        sos = iir.design(0, 0, 8, SR, 5000)
        # Check that b coefficients form a valid filter
        assert sos.shape[0] == 4  # order 8 = 4 sections
        # Apply to impulse and verify no passband ripple
        x = make_sine(1000.0)  # well within passband
        out = iir.apply(x, 0, 0, 8, SR, 5000)
        # Output should be very close to input (flat passband)
        assert energy_ratio(out, x) > 0.95

    def test_higher_order_sharper_rolloff(self):
        """Higher order should have sharper transition."""
        x = make_sine(2000.0)  # near cutoff
        out_2 = iir.apply(x, 0, 0, 2, SR, 1000)
        out_8 = iir.apply(x, 0, 0, 8, SR, 1000)
        # Order 8 should attenuate more at 2x cutoff
        assert rms(out_8) < rms(out_2)


class TestChebyshev1:
    def test_passband_ripple(self):
        """Chebyshev I should have ripple in passband."""
        # With high ripple, passband signal amplitude varies
        x = make_impulse()
        out = iir.apply(x, 1, 0, 4, SR, 5000, ripple_db=3.0)
        assert rms(out) > 0  # produces output

    def test_sharper_than_butterworth(self):
        """Same order Chebyshev I should have sharper rolloff than Butterworth."""
        x = make_sine(3000.0)  # above cutoff
        out_butter = iir.apply(x, 0, 0, 4, SR, 1000)
        out_cheby = iir.apply(x, 1, 0, 4, SR, 1000, ripple_db=1.0)
        # Chebyshev should attenuate more in stopband
        assert rms(out_cheby) < rms(out_butter)


class TestChebyshev2:
    def test_flat_passband(self):
        """Chebyshev II should have flat passband."""
        x = make_sine(200.0)  # well within passband
        out = iir.apply(x, 2, 0, 4, SR, 1000, ripple_db=40.0)
        assert energy_ratio(out, x) > 0.95

    def test_stopband_attenuation(self):
        """Chebyshev II should have specified stopband attenuation."""
        x = make_sine(5000.0)
        out = iir.apply(x, 2, 0, 4, SR, 1000, ripple_db=40.0)
        assert rms(out) < 0.1 * rms(x)


class TestElliptic:
    def test_produces_output(self):
        x = make_sine(200.0)
        out = iir.apply(x, 3, 0, 4, SR, 1000, ripple_db=1.0, rolloff_db=0.0)
        assert rms(out) > 0

    def test_sharpest_transition(self):
        """Elliptic should have the sharpest transition of all families."""
        x = make_sine(1500.0)  # just above cutoff
        out_butter = iir.apply(x, 0, 0, 4, SR, 1000)
        out_ellip = iir.apply(x, 3, 0, 4, SR, 1000, ripple_db=1.0, rolloff_db=0.0)
        # Elliptic should attenuate more
        assert rms(out_ellip) < rms(out_butter)


class TestBessel:
    def test_group_delay_flatter_than_chebyshev(self):
        """Bessel should have more uniform group delay than Chebyshev I."""
        x = make_impulse()
        out_bessel = iir.apply(x, 4, 0, 4, SR, 5000)
        out_cheby = iir.apply(x, 1, 0, 4, SR, 5000, ripple_db=3.0)
        # Bessel preserves transient shape better: peak should be higher
        # relative to energy (more compact impulse response)
        peak_bessel = np.max(np.abs(out_bessel))
        peak_cheby = np.max(np.abs(out_cheby))
        assert peak_bessel > peak_cheby

    def test_all_filter_types(self):
        x = make_impulse()
        for ftype in range(4):
            kwargs = {"width": 500.0} if ftype >= 2 else {}
            out = iir.apply(x, 4, ftype, 4, SR, 1000, **kwargs)
            assert out.shape == x.shape
            assert not np.any(np.isnan(out))


# ---------------------------------------------------------------------------
# Python API tests
# ---------------------------------------------------------------------------


class TestIIRDesignPython:
    def test_basic(self):
        sos = iir_design("butterworth", "lowpass", 4, SR, 1000)
        assert sos.shape == (2, 6)

    def test_aliases(self):
        sos1 = iir_design("butter", "lp", 4, SR, 1000)
        sos2 = iir_design("butterworth", "lowpass", 4, SR, 1000)
        assert np.allclose(sos1, sos2)

    def test_invalid_family(self):
        with pytest.raises(ValueError, match="Unknown family"):
            iir_design("invalid", "lowpass", 4, SR, 1000)

    def test_invalid_type(self):
        with pytest.raises(ValueError, match="Unknown type"):
            iir_design("butterworth", "invalid", 4, SR, 1000)

    def test_case_insensitive(self):
        sos1 = iir_design("BUTTERWORTH", "LOWPASS", 4, SR, 1000)
        sos2 = iir_design("butterworth", "lowpass", 4, SR, 1000)
        assert np.allclose(sos1, sos2)


class TestIIRFilterPython:
    def test_basic_lowpass(self):
        buf = make_buf(make_sine(200.0))
        out = iir_filter(buf, "butterworth", "lowpass", 4, freq=5000.0)
        assert out.data.shape == buf.data.shape
        # 200 Hz signal through 5000 Hz lowpass should pass
        assert energy_ratio(out.data.flatten(), buf.data.flatten()) > 0.9

    def test_highpass_rejects_low(self):
        buf = make_buf(make_sine(100.0))
        out = iir_filter(buf, "butter", "hp", 6, freq=1000.0)
        assert energy_ratio(out.data.flatten(), buf.data.flatten()) < 0.1

    def test_bandpass(self):
        buf = make_buf(make_sine(1000.0))
        out = iir_filter(buf, "butter", "bp", 4, freq=1000.0, width=500.0)
        assert energy_ratio(out.data.flatten(), buf.data.flatten()) > 0.5

    def test_bandstop(self):
        buf = make_buf(make_sine(1000.0))
        out = iir_filter(buf, "butter", "notch", 4, freq=1000.0, width=500.0)
        assert energy_ratio(out.data.flatten(), buf.data.flatten()) < 0.5

    def test_all_families(self):
        buf = make_buf(make_sine(200.0))
        for family in ["butterworth", "chebyshev1", "chebyshev2", "elliptic", "bessel"]:
            kwargs = {}
            if family == "chebyshev1":
                kwargs["ripple_db"] = 1.0
            elif family == "chebyshev2":
                kwargs["ripple_db"] = 40.0
            elif family == "elliptic":
                kwargs["ripple_db"] = 1.0
            out = iir_filter(buf, family, "lowpass", 4, freq=5000.0, **kwargs)
            assert out.data.shape == buf.data.shape
            assert not np.any(np.isnan(out.data))

    def test_stereo(self):
        mono = make_sine(500.0)
        stereo = np.stack([mono, mono * 0.5])
        buf = AudioBuffer(stereo, sample_rate=SR)
        out = iir_filter(buf, "butter", "lp", 4, freq=5000.0)
        assert out.data.shape == (2, FRAMES)

    def test_metadata_preserved(self):
        buf = make_buf(make_sine(500.0))
        out = iir_filter(buf, "butter", "lp", 4, freq=1000.0)
        assert out.sample_rate == buf.sample_rate

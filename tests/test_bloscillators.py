"""Tests for band-limited oscillators (PolyBLEP, BLIT, DPW)."""

import numpy as np
import pytest

from nanodsp._core import bloscillators as bl
from nanodsp.synthesis import polyblep, blit_saw, blit_square, dpw_saw, dpw_pulse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SR = 44100.0
FRAMES = 4096


def rms(x):
    return np.sqrt(np.mean(x**2))


def peak_freq(data, sr):
    """Return the dominant frequency via FFT."""
    x = data.flatten()
    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
    return freqs[np.argmax(spectrum[1:]) + 1]  # skip DC


# ===========================================================================
# C++ Binding Tests
# ===========================================================================


class TestPolyBLEPConstruction:
    def test_default(self):
        osc = bl.PolyBLEP()
        assert osc.frequency == pytest.approx(440.0)

    def test_custom_sr_waveform(self):
        osc = bl.PolyBLEP(48000.0, bl.PolyBLEP.Waveform.SINE)
        assert osc.frequency == pytest.approx(440.0)

    def test_set_frequency(self):
        osc = bl.PolyBLEP(SR)
        osc.frequency = 880.0
        assert osc.frequency == pytest.approx(880.0)

    def test_set_waveform(self):
        osc = bl.PolyBLEP(SR)
        osc.waveform = bl.PolyBLEP.Waveform.TRIANGLE
        assert osc.waveform == bl.PolyBLEP.Waveform.TRIANGLE

    def test_set_pulse_width(self):
        osc = bl.PolyBLEP(SR)
        osc.pulse_width = 0.3
        assert osc.pulse_width == pytest.approx(0.3)

    def test_set_phase(self):
        osc = bl.PolyBLEP(SR)
        osc.phase = 0.25
        assert osc.phase == pytest.approx(0.25)


class TestPolyBLEPGeneration:
    def test_output_shape(self):
        osc = bl.PolyBLEP(SR)
        out = osc.generate(FRAMES)
        assert out.shape == (FRAMES,)
        assert out.dtype == np.float32

    def test_tick_matches_generate(self):
        osc1 = bl.PolyBLEP(SR, bl.PolyBLEP.Waveform.SAWTOOTH)
        osc1.frequency = 440.0
        osc2 = bl.PolyBLEP(SR, bl.PolyBLEP.Waveform.SAWTOOTH)
        osc2.frequency = 440.0
        ticked = np.array([osc1.tick() for _ in range(256)], dtype=np.float32)
        generated = osc2.generate(256)
        assert ticked.shape == generated.shape
        np.testing.assert_allclose(ticked, generated, atol=1e-6)

    def test_sine_bounded(self):
        osc = bl.PolyBLEP(SR, bl.PolyBLEP.Waveform.SINE)
        osc.frequency = 440.0
        out = osc.generate(FRAMES)
        assert np.all(out >= -1.01) and np.all(out <= 1.01)

    def test_sine_frequency(self):
        osc = bl.PolyBLEP(SR, bl.PolyBLEP.Waveform.SINE)
        osc.frequency = 440.0
        out = osc.generate(FRAMES)
        assert peak_freq(out, SR) == pytest.approx(440.0, rel=0.02)

    def test_sawtooth_has_energy(self):
        osc = bl.PolyBLEP(SR, bl.PolyBLEP.Waveform.SAWTOOTH)
        osc.frequency = 440.0
        out = osc.generate(FRAMES)
        assert rms(out) > 0.3

    def test_square_has_energy(self):
        osc = bl.PolyBLEP(SR, bl.PolyBLEP.Waveform.SQUARE)
        osc.frequency = 440.0
        out = osc.generate(FRAMES)
        assert rms(out) > 0.5

    def test_reset(self):
        osc = bl.PolyBLEP(SR, bl.PolyBLEP.Waveform.SAWTOOTH)
        osc.frequency = 440.0
        out1 = osc.generate(256)
        osc.reset()
        out2 = osc.generate(256)
        assert out1.shape == (256,)
        np.testing.assert_allclose(out1, out2, atol=1e-6)

    def test_sync(self):
        osc = bl.PolyBLEP(SR, bl.PolyBLEP.Waveform.SINE)
        osc.frequency = 440.0
        osc.generate(100)
        osc.sync(0.0)
        # After sync to phase 0, first tick should be near sin(0) = 0
        val = osc.tick()
        assert abs(val) < 0.1

    @pytest.mark.parametrize(
        "wf",
        [
            "SINE",
            "COSINE",
            "TRIANGLE",
            "SQUARE",
            "RECTANGLE",
            "SAWTOOTH",
            "RAMP",
            "MODIFIED_TRIANGLE",
            "MODIFIED_SQUARE",
            "HALF_WAVE_RECTIFIED_SINE",
            "FULL_WAVE_RECTIFIED_SINE",
            "TRIANGULAR_PULSE",
            "TRAPEZOID_FIXED",
            "TRAPEZOID_VARIABLE",
        ],
    )
    def test_all_waveforms_produce_output(self, wf):
        waveform = getattr(bl.PolyBLEP.Waveform, wf)
        osc = bl.PolyBLEP(SR, waveform)
        osc.frequency = 440.0
        out = osc.generate(FRAMES)
        assert out.shape == (FRAMES,)
        assert rms(out) > 0.01  # should produce some signal


class TestBlitSaw:
    def test_construction(self):
        osc = bl.BlitSaw(SR, 220.0)
        assert osc.frequency == pytest.approx(220.0)

    def test_default_construction(self):
        osc = bl.BlitSaw()
        assert osc.frequency == pytest.approx(220.0)

    def test_set_frequency(self):
        osc = bl.BlitSaw(SR)
        osc.frequency = 440.0
        assert osc.frequency == pytest.approx(440.0)

    def test_output_shape(self):
        osc = bl.BlitSaw(SR, 220.0)
        out = osc.generate(FRAMES)
        assert out.shape == (FRAMES,)
        assert out.dtype == np.float32

    def test_has_energy(self):
        osc = bl.BlitSaw(SR, 220.0)
        out = osc.generate(FRAMES)
        assert rms(out) > 0.1

    def test_frequency_content(self):
        osc = bl.BlitSaw(SR, 440.0)
        out = osc.generate(FRAMES * 4)
        assert peak_freq(out, SR) == pytest.approx(440.0, rel=0.02)

    def test_set_harmonics(self):
        osc = bl.BlitSaw(SR, 220.0)
        osc.set_harmonics(5)
        out = osc.generate(FRAMES)
        assert out.shape == (FRAMES,)
        assert rms(out) > 0.01

    def test_reset(self):
        osc = bl.BlitSaw(SR, 220.0)
        out1 = osc.generate(512)
        osc.reset()
        out2 = osc.generate(512)
        np.testing.assert_allclose(out1, out2, atol=1e-5)

    def test_tick_matches_generate(self):
        osc1 = bl.BlitSaw(SR, 220.0)
        osc2 = bl.BlitSaw(SR, 220.0)
        ticked = np.array([osc1.tick() for _ in range(256)], dtype=np.float32)
        generated = osc2.generate(256)
        np.testing.assert_allclose(ticked, generated, atol=1e-6)


class TestBlitSquare:
    def test_construction(self):
        osc = bl.BlitSquare(SR, 220.0)
        assert osc.frequency == pytest.approx(220.0)

    def test_output_shape(self):
        out = bl.BlitSquare(SR, 220.0).generate(FRAMES)
        assert out.shape == (FRAMES,)

    def test_has_energy(self):
        out = bl.BlitSquare(SR, 220.0).generate(FRAMES)
        assert rms(out) > 0.1

    def test_frequency_content(self):
        osc = bl.BlitSquare(SR, 440.0)
        out = osc.generate(FRAMES * 4)
        assert peak_freq(out, SR) == pytest.approx(440.0, rel=0.02)

    def test_set_harmonics(self):
        osc = bl.BlitSquare(SR, 220.0)
        osc.set_harmonics(5)
        out = osc.generate(FRAMES)
        assert rms(out) > 0.01

    def test_reset(self):
        osc = bl.BlitSquare(SR, 220.0)
        out1 = osc.generate(512)
        osc.reset()
        out2 = osc.generate(512)
        np.testing.assert_allclose(out1, out2, atol=1e-5)


class TestDPWSaw:
    def test_construction(self):
        osc = bl.DPWSaw(SR, 440.0)
        assert osc.frequency == pytest.approx(440.0)

    def test_output_shape(self):
        out = bl.DPWSaw(SR, 440.0).generate(FRAMES)
        assert out.shape == (FRAMES,)

    def test_bounded(self):
        out = bl.DPWSaw(SR, 440.0).generate(FRAMES)
        assert np.all(out >= -1.1) and np.all(out <= 1.1)

    def test_no_startup_transient(self):
        out = bl.DPWSaw(SR, 440.0).generate(FRAMES)
        # First sample should not spike above 1
        assert abs(out[0]) < 1.1

    def test_has_energy(self):
        out = bl.DPWSaw(SR, 440.0).generate(FRAMES)
        assert rms(out) > 0.3

    def test_frequency_content(self):
        osc = bl.DPWSaw(SR, 440.0)
        out = osc.generate(FRAMES * 4)
        assert peak_freq(out, SR) == pytest.approx(440.0, rel=0.02)

    def test_reset(self):
        osc = bl.DPWSaw(SR, 440.0)
        out1 = osc.generate(512)
        osc.reset()
        out2 = osc.generate(512)
        np.testing.assert_allclose(out1, out2, atol=1e-5)


class TestDPWPulse:
    def test_construction(self):
        osc = bl.DPWPulse(SR, 440.0)
        assert osc.frequency == pytest.approx(440.0)

    def test_output_shape(self):
        out = bl.DPWPulse(SR, 440.0).generate(FRAMES)
        assert out.shape == (FRAMES,)

    def test_bounded(self):
        out = bl.DPWPulse(SR, 440.0).generate(FRAMES)
        assert np.all(out >= -1.1) and np.all(out <= 1.1)

    def test_no_startup_transient(self):
        out = bl.DPWPulse(SR, 440.0).generate(FRAMES)
        assert abs(out[0]) < 1.1

    def test_duty_cycle(self):
        osc = bl.DPWPulse(SR, 440.0)
        osc.duty = 0.25
        assert osc.duty == pytest.approx(0.25)
        out = osc.generate(FRAMES)
        assert rms(out) > 0.01

    def test_has_energy(self):
        out = bl.DPWPulse(SR, 440.0).generate(FRAMES)
        assert rms(out) > 0.1

    def test_frequency_content(self):
        osc = bl.DPWPulse(SR, 440.0)
        out = osc.generate(FRAMES * 4)
        assert peak_freq(out, SR) == pytest.approx(440.0, rel=0.02)

    def test_reset(self):
        osc = bl.DPWPulse(SR, 440.0)
        out1 = osc.generate(512)
        osc.reset()
        out2 = osc.generate(512)
        np.testing.assert_allclose(out1, out2, atol=1e-5)


# ===========================================================================
# Python API Tests
# ===========================================================================


class TestPolyBLEPPython:
    def test_basic(self):
        buf = polyblep(FRAMES, freq=440.0)
        assert buf.data.shape == (1, FRAMES)
        assert buf.sample_rate == 48000.0

    def test_waveform_string(self):
        buf = polyblep(FRAMES, freq=440.0, waveform="sine")
        assert rms(buf.data) > 0.3

    def test_waveform_aliases(self):
        for alias in ["tri", "saw", "rect"]:
            buf = polyblep(FRAMES, freq=440.0, waveform=alias)
            assert buf.data.shape == (1, FRAMES)

    def test_invalid_waveform(self):
        with pytest.raises(ValueError, match="Unknown waveform"):
            polyblep(FRAMES, waveform="invalid")

    def test_custom_sample_rate(self):
        buf = polyblep(FRAMES, freq=440.0, sample_rate=96000.0)
        assert buf.sample_rate == 96000.0

    def test_pulse_width(self):
        buf = polyblep(FRAMES, freq=440.0, waveform="rectangle", pulse_width=0.25)
        assert buf.data.shape == (1, FRAMES)

    @pytest.mark.parametrize(
        "wf",
        [
            "sine",
            "cosine",
            "triangle",
            "square",
            "sawtooth",
            "ramp",
            "modified_triangle",
            "modified_square",
            "trapezoid_fixed",
        ],
    )
    def test_all_waveforms_via_python(self, wf):
        buf = polyblep(FRAMES, freq=440.0, waveform=wf)
        assert rms(buf.data) > 0.01


class TestBlitSawPython:
    def test_basic(self):
        buf = blit_saw(FRAMES, freq=220.0)
        assert buf.data.shape == (1, FRAMES)
        assert rms(buf.data) > 0.1

    def test_harmonics(self):
        buf = blit_saw(FRAMES, freq=220.0, harmonics=5)
        assert buf.data.shape == (1, FRAMES)

    def test_custom_sr(self):
        buf = blit_saw(FRAMES, freq=220.0, sample_rate=96000.0)
        assert buf.sample_rate == 96000.0


class TestBlitSquarePython:
    def test_basic(self):
        buf = blit_square(FRAMES, freq=220.0)
        assert buf.data.shape == (1, FRAMES)
        assert rms(buf.data) > 0.1

    def test_harmonics(self):
        buf = blit_square(FRAMES, freq=220.0, harmonics=5)
        assert buf.data.shape == (1, FRAMES)


class TestDPWSawPython:
    def test_basic(self):
        buf = dpw_saw(FRAMES, freq=440.0)
        assert buf.data.shape == (1, FRAMES)
        assert rms(buf.data) > 0.3

    def test_no_transient(self):
        buf = dpw_saw(FRAMES, freq=440.0)
        assert abs(buf.data[0, 0]) < 1.1


class TestDPWPulsePython:
    def test_basic(self):
        buf = dpw_pulse(FRAMES, freq=440.0)
        assert buf.data.shape == (1, FRAMES)
        assert rms(buf.data) > 0.1

    def test_duty(self):
        buf = dpw_pulse(FRAMES, freq=440.0, duty=0.25)
        assert buf.data.shape == (1, FRAMES)

    def test_no_transient(self):
        buf = dpw_pulse(FRAMES, freq=440.0)
        assert abs(buf.data[0, 0]) < 1.1

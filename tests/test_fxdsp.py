"""Tests for fxdsp algorithms (waveshaping, reverbs, minBLEP, PSOLA, formant)."""

import numpy as np
import pytest

from nanodsp._core import fxdsp as fx
from nanodsp.buffer import AudioBuffer
from nanodsp.effects import (
    aa_hard_clip,
    aa_soft_clip,
    aa_wavefold,
    schroeder_reverb,
    moorer_reverb,
    formant_filter,
    psola_pitch_shift,
)
from nanodsp.synthesis import minblep


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SR = 44100.0
FRAMES = 4096


def rms(x):
    return np.sqrt(np.mean(np.asarray(x).flatten() ** 2))


def make_sine(freq=440.0, frames=FRAMES, sr=SR, amp=1.0):
    t = np.arange(frames) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def make_impulse(frames=FRAMES):
    x = np.zeros(frames, dtype=np.float32)
    x[0] = 1.0
    return x


def make_buf(data, sr=SR):
    return AudioBuffer(data.reshape(1, -1), sample_rate=sr)


def peak_freq(data, sr):
    x = np.asarray(data).flatten()
    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
    return freqs[np.argmax(spectrum[1:]) + 1]


# ===========================================================================
# C++ Binding Tests -- Waveshaping
# ===========================================================================


class TestHardClipper:
    def test_construction(self):
        hc = fx.HardClipper()
        assert hc is not None

    def test_clips_to_range(self):
        hc = fx.HardClipper()
        x = make_sine(amp=3.0)
        out = hc.process(x)
        assert out.max() <= 1.01
        assert out.min() >= -1.01

    def test_passthrough_below_threshold(self):
        hc = fx.HardClipper()
        x = make_sine(amp=0.5)
        out = hc.process(x)
        # With AA, not exactly equal but close
        np.testing.assert_allclose(out, x, atol=0.15)

    def test_output_shape(self):
        hc = fx.HardClipper()
        x = make_sine()
        out = hc.process(x)
        assert out.shape == x.shape

    def test_reset(self):
        hc = fx.HardClipper()
        x = make_sine(amp=2.0, frames=256)
        out1 = hc.process(x)
        hc.reset()
        out2 = hc.process(x)
        np.testing.assert_allclose(out1, out2, atol=1e-6)

    def test_tick_matches_process(self):
        hc1 = fx.HardClipper()
        hc2 = fx.HardClipper()
        x = make_sine(amp=2.0, frames=64)
        ticked = np.array([hc1.tick(float(s)) for s in x], dtype=np.float32)
        processed = hc2.process(x)
        np.testing.assert_allclose(ticked, processed, atol=1e-6)


class TestSoftClipper:
    def test_clips_to_range(self):
        sc = fx.SoftClipper()
        x = make_sine(amp=3.0)
        out = sc.process(x)
        assert out.max() <= 1.01
        assert out.min() >= -1.01

    def test_smooth_saturation(self):
        sc = fx.SoftClipper()
        x = make_sine(amp=2.0)
        out = sc.process(x)
        # Soft clipping should produce output with energy
        assert rms(out) > 0.3

    def test_reset(self):
        sc = fx.SoftClipper()
        x = make_sine(amp=2.0, frames=256)
        out1 = sc.process(x)
        sc.reset()
        out2 = sc.process(x)
        np.testing.assert_allclose(out1, out2, atol=1e-6)


class TestWavefolder:
    def test_folds_signal(self):
        wf = fx.Wavefolder()
        x = make_sine(amp=2.0)
        out = wf.process(x)
        # Folded signal should have more harmonic content
        assert rms(out) > 0.1

    def test_bounded(self):
        wf = fx.Wavefolder()
        x = make_sine(amp=2.0)
        out = wf.process(x)
        # Folded should stay bounded
        assert out.max() <= 1.5
        assert out.min() >= -1.5

    def test_reset(self):
        wf = fx.Wavefolder()
        x = make_sine(amp=2.0, frames=256)
        out1 = wf.process(x)
        wf.reset()
        out2 = wf.process(x)
        np.testing.assert_allclose(out1, out2, atol=1e-6)


# ===========================================================================
# C++ Binding Tests -- Reverbs
# ===========================================================================


class TestSchroederReverb:
    def test_construction_and_init(self):
        rev = fx.SchroederReverb()
        rev.init(48000.0)
        assert rev.feedback == pytest.approx(0.7)

    def test_impulse_response_has_tail(self):
        rev = fx.SchroederReverb()
        rev.init(48000.0)
        out = rev.process(make_impulse())
        # Reverb tail should have energy after the initial impulse
        assert rms(out[100:]) > 0.001

    def test_silence_produces_silence(self):
        rev = fx.SchroederReverb()
        rev.init(48000.0)
        x = np.zeros(FRAMES, dtype=np.float32)
        out = rev.process(x)
        assert rms(out) < 1e-6

    def test_feedback_control(self):
        rev = fx.SchroederReverb()
        rev.init(48000.0)
        rev.feedback = 0.3
        assert rev.feedback == pytest.approx(0.3)

    def test_diffusion_control(self):
        rev = fx.SchroederReverb()
        rev.init(48000.0)
        rev.diffusion = 0.8
        assert rev.diffusion == pytest.approx(0.8)

    def test_reset(self):
        rev = fx.SchroederReverb()
        rev.init(48000.0)
        rev.process(make_impulse())
        rev.reset()
        out = rev.process(np.zeros(256, dtype=np.float32))
        assert rms(out) < 1e-6

    def test_output_shape(self):
        rev = fx.SchroederReverb()
        rev.init(48000.0)
        out = rev.process(make_sine(frames=1024))
        assert out.shape == (1024,)


class TestMoorerReverb:
    def test_construction_and_init(self):
        rev = fx.MoorerReverb()
        rev.init(48000.0)
        assert rev.feedback == pytest.approx(0.7)

    def test_impulse_response_has_tail(self):
        rev = fx.MoorerReverb()
        rev.init(48000.0)
        out = rev.process(make_impulse())
        assert rms(out[100:]) > 0.001

    def test_has_early_reflections(self):
        rev = fx.MoorerReverb()
        rev.init(48000.0)
        out = rev.process(make_impulse())
        # Early reflections should produce non-zero output quickly
        assert np.any(np.abs(out[:500]) > 0.001)

    def test_feedback_and_diffusion(self):
        rev = fx.MoorerReverb()
        rev.init(48000.0)
        rev.feedback = 0.5
        rev.diffusion = 0.6
        assert rev.feedback == pytest.approx(0.5)
        assert rev.diffusion == pytest.approx(0.6)

    def test_reset(self):
        rev = fx.MoorerReverb()
        rev.init(48000.0)
        rev.process(make_impulse())
        rev.reset()
        out = rev.process(np.zeros(256, dtype=np.float32))
        assert rms(out) < 1e-6


# ===========================================================================
# C++ Binding Tests -- MinBLEP
# ===========================================================================


class TestMinBLEP:
    def test_construction(self):
        mb = fx.MinBLEP(44100.0, 440.0)
        assert mb.frequency == pytest.approx(440.0)

    def test_default_construction(self):
        mb = fx.MinBLEP()
        assert mb.frequency == pytest.approx(440.0)

    def test_set_frequency(self):
        mb = fx.MinBLEP(44100.0)
        mb.frequency = 880.0
        assert mb.frequency == pytest.approx(880.0)

    def test_set_waveform(self):
        mb = fx.MinBLEP(44100.0)
        mb.waveform = fx.MinBLEP.Waveform.SQUARE
        assert mb.waveform == fx.MinBLEP.Waveform.SQUARE

    def test_output_shape(self):
        mb = fx.MinBLEP(44100.0, 440.0)
        out = mb.generate(FRAMES)
        assert out.shape == (FRAMES,)
        assert out.dtype == np.float32

    def test_saw_has_energy(self):
        mb = fx.MinBLEP(44100.0, 440.0)
        mb.waveform = fx.MinBLEP.Waveform.SAW
        out = mb.generate(FRAMES)
        assert rms(out) > 0.3

    def test_square_has_energy(self):
        mb = fx.MinBLEP(44100.0, 440.0)
        mb.waveform = fx.MinBLEP.Waveform.SQUARE
        out = mb.generate(FRAMES)
        assert rms(out) > 0.3

    def test_triangle_has_energy(self):
        mb = fx.MinBLEP(44100.0, 440.0)
        mb.waveform = fx.MinBLEP.Waveform.TRIANGLE
        out = mb.generate(FRAMES)
        assert rms(out) > 0.1

    def test_rsaw_has_energy(self):
        mb = fx.MinBLEP(44100.0, 440.0)
        mb.waveform = fx.MinBLEP.Waveform.RSAW
        out = mb.generate(FRAMES)
        assert rms(out) > 0.3

    def test_frequency_content(self):
        mb = fx.MinBLEP(44100.0, 440.0)
        out = mb.generate(FRAMES * 16)
        # Saw has strong harmonics; check fundamental is present with significant energy
        spectrum = np.abs(np.fft.rfft(out * np.hanning(len(out))))
        freqs = np.fft.rfftfreq(len(out), 1.0 / 44100.0)
        fund_idx = np.argmin(np.abs(freqs - 440.0))
        fund_energy = spectrum[fund_idx]
        assert fund_energy > 0.1 * np.max(spectrum)

    def test_pulse_width(self):
        mb = fx.MinBLEP(44100.0, 440.0)
        mb.waveform = fx.MinBLEP.Waveform.SQUARE
        mb.pulse_width = 0.25
        assert mb.pulse_width == pytest.approx(0.25)
        out = mb.generate(FRAMES)
        assert rms(out) > 0.1

    def test_reset(self):
        mb = fx.MinBLEP(44100.0, 440.0)
        out1 = mb.generate(512)
        mb.reset()
        out2 = mb.generate(512)
        np.testing.assert_allclose(out1, out2, atol=1e-5)

    def test_tick_matches_generate(self):
        mb1 = fx.MinBLEP(44100.0, 440.0)
        mb2 = fx.MinBLEP(44100.0, 440.0)
        ticked = np.array([mb1.tick() for _ in range(256)], dtype=np.float32)
        generated = mb2.generate(256)
        np.testing.assert_allclose(ticked, generated, atol=1e-6)


# ===========================================================================
# C++ Binding Tests -- PSOLA
# ===========================================================================


class TestPSOLA:
    def test_basic_shift(self):
        x = make_sine(freq=220.0, frames=8000)
        out = fx.psola_pitch_shift(x, SR, 2.0)
        assert out.shape == (8000,)

    def test_no_shift(self):
        x = make_sine(freq=220.0, frames=4000)
        out = fx.psola_pitch_shift(x, SR, 0.0)
        # Zero semitones should return original
        np.testing.assert_array_equal(out, x)

    def test_shift_up(self):
        x = make_sine(freq=220.0, frames=8000)
        out = fx.psola_pitch_shift(x, SR, 12.0)
        assert out.shape == (8000,)
        assert rms(out) > 0.01

    def test_shift_down(self):
        x = make_sine(freq=440.0, frames=8000)
        out = fx.psola_pitch_shift(x, SR, -12.0)
        assert out.shape == (8000,)
        assert rms(out) > 0.01

    def test_output_length_matches_input(self):
        x = make_sine(frames=10000)
        out = fx.psola_pitch_shift(x, SR, 5.0)
        assert out.shape == (10000,)


# ===========================================================================
# C++ Binding Tests -- Formant Filter
# ===========================================================================


class TestFormantFilter:
    def test_construction(self):
        ff = fx.FormantFilter()
        ff.init(44100.0)
        assert ff.vowel == 0  # A

    def test_set_vowel(self):
        ff = fx.FormantFilter()
        ff.init(44100.0)
        ff.vowel = 2  # I
        assert ff.vowel == 2

    def test_all_vowels_produce_output(self):
        ff = fx.FormantFilter()
        ff.init(44100.0)
        noise = np.random.randn(FRAMES).astype(np.float32) * 0.5
        for v in range(5):
            ff.vowel = v
            ff.reset()
            out = ff.process(noise)
            assert rms(out) > 0.01

    def test_vowel_blend(self):
        ff = fx.FormantFilter()
        ff.init(44100.0)
        ff.set_vowel_blend(0, 4, 0.5)
        noise = np.random.randn(FRAMES).astype(np.float32) * 0.5
        out = ff.process(noise)
        assert rms(out) > 0.01

    def test_output_shape(self):
        ff = fx.FormantFilter()
        ff.init(44100.0)
        x = make_sine()
        out = ff.process(x)
        assert out.shape == x.shape

    def test_reset(self):
        ff = fx.FormantFilter()
        ff.init(44100.0)
        noise = np.random.randn(256).astype(np.float32)
        out1 = ff.process(noise)
        ff.reset()
        out2 = ff.process(noise)
        np.testing.assert_allclose(out1, out2, atol=1e-5)

    def test_silence_produces_silence(self):
        ff = fx.FormantFilter()
        ff.init(44100.0)
        x = np.zeros(FRAMES, dtype=np.float32)
        out = ff.process(x)
        assert rms(out) < 1e-6


# ===========================================================================
# Python API Tests
# ===========================================================================


class TestAAWaveshapingPython:
    def test_aa_hard_clip(self):
        buf = make_buf(make_sine(amp=2.0))
        out = aa_hard_clip(buf, drive=1.0)
        assert out.data.shape == buf.data.shape
        assert out.data.max() <= 1.05

    def test_aa_hard_clip_with_drive(self):
        buf = make_buf(make_sine(amp=0.5))
        out = aa_hard_clip(buf, drive=4.0)
        assert out.data.max() <= 1.05

    def test_aa_soft_clip(self):
        buf = make_buf(make_sine(amp=2.0))
        out = aa_soft_clip(buf)
        assert out.data.shape == buf.data.shape
        assert out.data.max() <= 1.05

    def test_aa_wavefold(self):
        buf = make_buf(make_sine(amp=2.0))
        out = aa_wavefold(buf)
        assert out.data.shape == buf.data.shape
        assert rms(out.data) > 0.1

    def test_stereo(self):
        data = np.stack([make_sine(amp=2.0), make_sine(amp=1.5)])
        buf = AudioBuffer(data, sample_rate=SR)
        out = aa_hard_clip(buf)
        assert out.data.shape == (2, FRAMES)


class TestReverbsPython:
    def test_schroeder(self):
        buf = make_buf(make_impulse())
        out = schroeder_reverb(buf)
        assert out.data.shape == buf.data.shape
        assert rms(out.data[0, 100:]) > 0.001

    def test_schroeder_params(self):
        buf = make_buf(make_impulse())
        out = schroeder_reverb(buf, feedback=0.5, diffusion=0.8, mod_depth=0.1)
        assert out.data.shape == buf.data.shape

    def test_moorer(self):
        buf = make_buf(make_impulse())
        out = moorer_reverb(buf)
        assert out.data.shape == buf.data.shape
        assert rms(out.data[0, 100:]) > 0.001

    def test_moorer_params(self):
        buf = make_buf(make_impulse())
        out = moorer_reverb(buf, feedback=0.5, diffusion=0.6, mod_depth=0.05)
        assert out.data.shape == buf.data.shape


class TestFormantFilterPython:
    def test_vowel_string(self):
        noise = np.random.randn(FRAMES).astype(np.float32) * 0.5
        buf = make_buf(noise)
        out = formant_filter(buf, vowel="a")
        assert out.data.shape == buf.data.shape
        assert rms(out.data) > 0.01

    def test_all_vowels(self):
        noise = np.random.randn(FRAMES).astype(np.float32) * 0.5
        buf = make_buf(noise)
        for v in ["a", "e", "i", "o", "u"]:
            out = formant_filter(buf, vowel=v)
            assert rms(out.data) > 0.001

    def test_vowel_int(self):
        noise = np.random.randn(FRAMES).astype(np.float32) * 0.5
        buf = make_buf(noise)
        out = formant_filter(buf, vowel=3)
        assert out.data.shape == buf.data.shape

    def test_invalid_vowel(self):
        buf = make_buf(make_sine())
        with pytest.raises(ValueError, match="Unknown vowel"):
            formant_filter(buf, vowel="x")


class TestPSOLAPython:
    def test_basic(self):
        buf = make_buf(make_sine(freq=220.0, frames=8000))
        out = psola_pitch_shift(buf, semitones=2.0)
        assert out.data.shape == buf.data.shape

    def test_no_shift(self):
        buf = make_buf(make_sine(freq=220.0, frames=4000))
        out = psola_pitch_shift(buf, semitones=0.0)
        assert out.data.shape == buf.data.shape


class TestMinBLEPPython:
    def test_basic(self):
        buf = minblep(FRAMES, freq=440.0)
        assert buf.data.shape == (1, FRAMES)
        assert rms(buf.data) > 0.3

    def test_waveform_string(self):
        for wf in ["saw", "rsaw", "square", "triangle"]:
            buf = minblep(FRAMES, freq=440.0, waveform=wf)
            assert buf.data.shape == (1, FRAMES)
            assert rms(buf.data) > 0.05

    def test_invalid_waveform(self):
        with pytest.raises(ValueError, match="Unknown waveform"):
            minblep(FRAMES, waveform="invalid")

    def test_custom_sr(self):
        buf = minblep(FRAMES, freq=440.0, sample_rate=96000.0)
        assert buf.sample_rate == 96000.0

    def test_pulse_width(self):
        buf = minblep(FRAMES, freq=440.0, waveform="square", pulse_width=0.25)
        assert buf.data.shape == (1, FRAMES)

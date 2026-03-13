"""Tests for fxdsp algorithms (waveshaping, reverbs, minBLEP, PSOLA, formant)."""

import numpy as np
import pytest

from nanodsp._core import fxdsp as fx
from nanodsp.buffer import AudioBuffer
from nanodsp.effects.saturation import aa_hard_clip, aa_soft_clip, aa_wavefold
from nanodsp.effects.reverb import schroeder_reverb, moorer_reverb
from nanodsp.effects.composed import (
    formant_filter,
    psola_pitch_shift,
    ping_pong_delay,
    freq_shift,
    ring_mod,
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
        assert out.shape == x.shape
        assert out.dtype == np.float32
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
        assert out1.shape == x.shape
        assert out2.dtype == np.float32
        np.testing.assert_allclose(out1, out2, atol=1e-6)

    def test_tick_matches_process(self):
        hc1 = fx.HardClipper()
        hc2 = fx.HardClipper()
        x = make_sine(amp=2.0, frames=64)
        ticked = np.array([hc1.tick(float(s)) for s in x], dtype=np.float32)
        processed = hc2.process(x)
        assert ticked.shape == processed.shape
        assert np.all(np.isfinite(ticked))
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
        assert out1.shape == x.shape
        assert out2.dtype == np.float32
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
        assert out1.shape == x.shape
        assert out2.dtype == np.float32
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
        assert out1.shape == (512,)
        assert out1.dtype == np.float32
        np.testing.assert_allclose(out1, out2, atol=1e-5)

    def test_tick_matches_generate(self):
        mb1 = fx.MinBLEP(44100.0, 440.0)
        mb2 = fx.MinBLEP(44100.0, 440.0)
        ticked = np.array([mb1.tick() for _ in range(256)], dtype=np.float32)
        generated = mb2.generate(256)
        assert ticked.shape == generated.shape
        assert np.all(np.isfinite(ticked))
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
        assert out.shape == x.shape
        assert out.dtype == np.float32
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
        assert out1.shape == noise.shape
        assert out1.dtype == np.float32
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
        assert buf.data.shape == (1, FRAMES)  # input unchanged


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
        assert buf.data.dtype == np.float32
        assert rms(buf.data) > 0.1


# ===========================================================================
# C++ Binding Tests -- Ping-Pong Delay
# ===========================================================================


class TestPingPongDelay:
    def test_construction_and_init(self):
        ppd = fx.PingPongDelay()
        ppd.init(44100.0)
        assert ppd.delay_ms == pytest.approx(500.0)

    def test_set_params(self):
        ppd = fx.PingPongDelay()
        ppd.init(44100.0)
        ppd.delay_ms = 250.0
        ppd.feedback = 0.3
        ppd.mix = 0.7
        assert ppd.delay_ms == pytest.approx(250.0)
        assert ppd.feedback == pytest.approx(0.3)
        assert ppd.mix == pytest.approx(0.7)

    def test_tick_returns_pair(self):
        ppd = fx.PingPongDelay()
        ppd.init(44100.0)
        result = ppd.tick(1.0, 0.0)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_process_stereo(self):
        ppd = fx.PingPongDelay()
        ppd.init(SR)
        ppd.delay_ms = 100.0
        ppd.feedback = 0.3
        ppd.mix = 0.5
        stereo = np.zeros((2, FRAMES), dtype=np.float32)
        stereo[0] = make_impulse()
        out = ppd.process(stereo)
        assert out.shape == (2, FRAMES)

    def test_impulse_produces_ping_pong(self):
        ppd = fx.PingPongDelay()
        ppd.init(SR)
        ppd.delay_ms = 10.0  # short delay so it fits in FRAMES
        ppd.feedback = 0.5
        ppd.mix = 1.0
        stereo = np.zeros((2, FRAMES), dtype=np.float32)
        stereo[0, 0] = 1.0  # impulse on left only
        out = ppd.process(stereo)
        # Crossed feedback: impulse in L delay -> fb_l set after 1 delay period
        # -> written into R delay -> appears in R after 2 delay periods
        delay_samples = int(SR * 0.01)
        assert np.max(np.abs(out[1, 2 * delay_samples:])) > 0.01

    def test_reset(self):
        ppd = fx.PingPongDelay()
        ppd.init(SR)
        ppd.delay_ms = 50.0
        stereo = np.zeros((2, FRAMES), dtype=np.float32)
        stereo[0] = make_impulse()
        ppd.process(stereo)
        ppd.reset()
        silence = np.zeros((2, 256), dtype=np.float32)
        out = ppd.process(silence)
        assert rms(out) < 1e-6

    def test_feedback_clamped(self):
        ppd = fx.PingPongDelay()
        ppd.init(SR)
        ppd.feedback = 2.0
        assert ppd.feedback == pytest.approx(0.99, abs=1e-5)

    def test_mix_clamped(self):
        ppd = fx.PingPongDelay()
        ppd.init(SR)
        ppd.mix = 1.5
        assert ppd.mix <= 1.0


# ===========================================================================
# C++ Binding Tests -- Frequency Shifter
# ===========================================================================


class TestFreqShifter:
    def test_construction_and_init(self):
        fs = fx.FreqShifter()
        fs.init(44100.0)
        assert fs.shift_hz == pytest.approx(0.0)

    def test_set_shift(self):
        fs = fx.FreqShifter()
        fs.init(44100.0)
        fs.shift_hz = 100.0
        assert fs.shift_hz == pytest.approx(100.0)

    def test_zero_shift_passthrough(self):
        fs = fx.FreqShifter()
        fs.init(SR)
        fs.shift_hz = 0.0
        x = make_sine(freq=440.0, frames=4096)
        out = fs.process(x)
        assert out.shape == x.shape
        # With zero shift the output should be close to input
        # (allpass filters introduce phase shift but preserve magnitude)
        assert rms(out) > 0.5 * rms(x)

    def test_positive_shift_changes_frequency(self):
        fs = fx.FreqShifter()
        fs.init(SR)
        fs.shift_hz = 200.0
        x = make_sine(freq=440.0, frames=FRAMES * 4)
        out = fs.process(x)
        # Peak frequency should be near 640 Hz
        pf = peak_freq(out, SR)
        assert abs(pf - 640.0) < 50.0

    def test_negative_shift(self):
        fs = fx.FreqShifter()
        fs.init(SR)
        fs.shift_hz = -200.0
        x = make_sine(freq=440.0, frames=FRAMES * 4)
        out = fs.process(x)
        # Peak frequency should be near 240 Hz
        pf = peak_freq(out, SR)
        assert abs(pf - 240.0) < 50.0

    def test_output_shape(self):
        fs = fx.FreqShifter()
        fs.init(SR)
        fs.shift_hz = 50.0
        x = make_sine(frames=1024)
        out = fs.process(x)
        assert out.shape == (1024,)

    def test_reset(self):
        fs = fx.FreqShifter()
        fs.init(SR)
        fs.shift_hz = 100.0
        x = make_sine(frames=512)
        out1 = fs.process(x)
        fs.reset()
        out2 = fs.process(x)
        assert out1.shape == x.shape
        assert out1.dtype == np.float32
        np.testing.assert_allclose(out1, out2, atol=1e-5)

    def test_tick_matches_process(self):
        fs1 = fx.FreqShifter()
        fs2 = fx.FreqShifter()
        fs1.init(SR)
        fs2.init(SR)
        fs1.shift_hz = 100.0
        fs2.shift_hz = 100.0
        x = make_sine(frames=64)
        ticked = np.array([fs1.tick(float(s)) for s in x], dtype=np.float32)
        processed = fs2.process(x)
        assert ticked.shape == processed.shape
        assert np.all(np.isfinite(ticked))
        np.testing.assert_allclose(ticked, processed, atol=1e-6)


# ===========================================================================
# C++ Binding Tests -- Ring Modulator
# ===========================================================================


class TestRingMod:
    def test_construction_and_init(self):
        rm = fx.RingMod()
        rm.init(44100.0)
        assert rm.carrier_freq == pytest.approx(440.0)

    def test_set_params(self):
        rm = fx.RingMod()
        rm.init(SR)
        rm.carrier_freq = 200.0
        rm.lfo_freq = 5.0
        rm.lfo_width = 10.0
        rm.mix = 0.8
        assert rm.carrier_freq == pytest.approx(200.0)
        assert rm.lfo_freq == pytest.approx(5.0)
        assert rm.lfo_width == pytest.approx(10.0)
        assert rm.mix == pytest.approx(0.8)

    def test_produces_sum_and_difference_tones(self):
        rm = fx.RingMod()
        rm.init(SR)
        rm.carrier_freq = 300.0
        rm.mix = 1.0
        x = make_sine(freq=440.0, frames=FRAMES * 4)
        out = rm.process(x)
        # Ring mod of 440 Hz * 300 Hz carrier should produce
        # 140 Hz (difference) and 740 Hz (sum)
        spectrum = np.abs(np.fft.rfft(out * np.hanning(len(out))))
        freqs = np.fft.rfftfreq(len(out), 1.0 / SR)
        # Check sum tone near 740 Hz
        idx_740 = np.argmin(np.abs(freqs - 740.0))
        # Check difference tone near 140 Hz
        idx_140 = np.argmin(np.abs(freqs - 140.0))
        assert spectrum[idx_740] > 0.1 * np.max(spectrum)
        assert spectrum[idx_140] > 0.1 * np.max(spectrum)

    def test_dry_mix(self):
        rm = fx.RingMod()
        rm.init(SR)
        rm.carrier_freq = 300.0
        rm.mix = 0.0
        x = make_sine(freq=440.0, frames=1024)
        out = rm.process(x)
        assert out.shape == x.shape
        assert out.dtype == np.float32
        # With mix=0, output should be the dry signal
        np.testing.assert_allclose(out, x, atol=1e-5)

    def test_silence_produces_silence(self):
        rm = fx.RingMod()
        rm.init(SR)
        x = np.zeros(FRAMES, dtype=np.float32)
        out = rm.process(x)
        assert rms(out) < 1e-6

    def test_output_shape(self):
        rm = fx.RingMod()
        rm.init(SR)
        x = make_sine(frames=1024)
        out = rm.process(x)
        assert out.shape == (1024,)

    def test_reset(self):
        rm = fx.RingMod()
        rm.init(SR)
        rm.carrier_freq = 300.0
        x = make_sine(frames=512)
        out1 = rm.process(x)
        rm.reset()
        out2 = rm.process(x)
        assert out1.shape == x.shape
        assert out1.dtype == np.float32
        np.testing.assert_allclose(out1, out2, atol=1e-5)

    def test_tick_matches_process(self):
        rm1 = fx.RingMod()
        rm2 = fx.RingMod()
        rm1.init(SR)
        rm2.init(SR)
        rm1.carrier_freq = 300.0
        rm2.carrier_freq = 300.0
        x = make_sine(frames=64)
        ticked = np.array([rm1.tick(float(s)) for s in x], dtype=np.float32)
        processed = rm2.process(x)
        assert ticked.shape == processed.shape
        assert np.all(np.isfinite(ticked))
        np.testing.assert_allclose(ticked, processed, atol=1e-6)

    def test_lfo_modulation(self):
        rm = fx.RingMod()
        rm.init(SR)
        rm.carrier_freq = 300.0
        rm.lfo_freq = 5.0
        rm.lfo_width = 50.0
        rm.mix = 1.0
        x = make_sine(freq=440.0, frames=FRAMES)
        out = rm.process(x)
        # Should produce output with energy
        assert rms(out) > 0.1


# ===========================================================================
# Python API Tests -- New Effects
# ===========================================================================


class TestPingPongDelayPython:
    def test_mono_input(self):
        buf = make_buf(make_impulse())
        out = ping_pong_delay(buf, delay_ms=100.0)
        assert out.data.shape[0] == 2  # stereo output
        assert out.data.shape[1] == FRAMES

    def test_stereo_input(self):
        data = np.stack([make_impulse(), np.zeros(FRAMES, dtype=np.float32)])
        buf = AudioBuffer(data, sample_rate=SR)
        out = ping_pong_delay(buf, delay_ms=100.0, feedback=0.3)
        assert out.data.shape == (2, FRAMES)

    def test_feedback_and_mix(self):
        buf = make_buf(make_impulse())
        out = ping_pong_delay(buf, delay_ms=50.0, feedback=0.5, mix=0.8)
        assert out.data.shape[0] == 2

    def test_multichannel_rejected(self):
        data = np.zeros((3, FRAMES), dtype=np.float32)
        buf = AudioBuffer(data, sample_rate=SR)
        with pytest.raises(ValueError, match="mono or stereo"):
            ping_pong_delay(buf)
        assert buf.channels == 3  # input unchanged


class TestFreqShiftPython:
    def test_basic(self):
        buf = make_buf(make_sine(freq=440.0))
        out = freq_shift(buf, shift_hz=100.0)
        assert out.data.shape == buf.data.shape

    def test_negative_shift(self):
        buf = make_buf(make_sine(freq=440.0))
        out = freq_shift(buf, shift_hz=-100.0)
        assert out.data.shape == buf.data.shape
        assert rms(out.data) > 0.1

    def test_stereo(self):
        data = np.stack([make_sine(freq=440.0), make_sine(freq=880.0)])
        buf = AudioBuffer(data, sample_rate=SR)
        out = freq_shift(buf, shift_hz=50.0)
        assert out.data.shape == (2, FRAMES)


class TestRingModPython:
    def test_basic(self):
        buf = make_buf(make_sine(freq=440.0))
        out = ring_mod(buf, carrier_freq=300.0)
        assert out.data.shape == buf.data.shape
        assert rms(out.data) > 0.1

    def test_with_lfo(self):
        buf = make_buf(make_sine(freq=440.0))
        out = ring_mod(buf, carrier_freq=300.0, lfo_freq=5.0, lfo_width=20.0)
        assert out.data.shape == buf.data.shape

    def test_mix(self):
        buf = make_buf(make_sine(freq=440.0))
        out = ring_mod(buf, carrier_freq=300.0, mix=0.5)
        assert out.data.shape == buf.data.shape

    def test_stereo(self):
        data = np.stack([make_sine(freq=440.0), make_sine(freq=880.0)])
        buf = AudioBuffer(data, sample_rate=SR)
        out = ring_mod(buf, carrier_freq=300.0)
        assert out.data.shape == (2, FRAMES)

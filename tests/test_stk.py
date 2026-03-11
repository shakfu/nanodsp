"""Tests for STK (Synthesis ToolKit) bindings."""

import numpy as np
import pytest

from nanodsp._core import stk

SR = 44100.0


@pytest.fixture(autouse=True)
def reset_sample_rate():
    """Ensure sample rate is reset for each test."""
    stk.set_sample_rate(SR)
    yield
    stk.set_sample_rate(SR)


# ---------------------------------------------------------------------------
# Global
# ---------------------------------------------------------------------------


class TestGlobal:
    def test_sample_rate_default(self):
        assert stk.sample_rate() == pytest.approx(SR)

    def test_set_sample_rate(self):
        stk.set_sample_rate(48000.0)
        assert stk.sample_rate() == pytest.approx(48000.0)

    def test_sample_rate_roundtrip(self):
        for rate in [22050.0, 44100.0, 48000.0, 96000.0]:
            stk.set_sample_rate(rate)
            assert stk.sample_rate() == pytest.approx(rate)


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------


class TestSineWave:
    def test_construction(self):
        sw = stk.generators.SineWave()
        assert sw is not None

    def test_process_shape_dtype(self):
        sw = stk.generators.SineWave()
        sw.set_frequency(440.0)
        out = sw.process(1024)
        assert out.shape == (1024,)
        assert out.dtype == np.float32

    def test_process_sample(self):
        sw = stk.generators.SineWave()
        sw.set_frequency(440.0)
        val = sw.process_sample()
        assert isinstance(val, float)

    def test_frequency_output(self):
        sw = stk.generators.SineWave()
        sw.set_frequency(440.0)
        out = sw.process(4096)
        # Should have significant amplitude
        assert np.max(np.abs(out)) > 0.9

    def test_reset(self):
        sw = stk.generators.SineWave()
        sw.set_frequency(440.0)
        sw.process(100)
        sw.reset()
        # After reset, first sample should be near 0
        val = sw.process_sample()
        assert abs(val) < 0.1

    def test_add_phase(self):
        sw = stk.generators.SineWave()
        sw.set_frequency(440.0)
        sw.reset()
        # Adding 0.25 phase (90 degrees) should produce near 1.0
        sw.add_phase(0.25)
        val = sw.process_sample()
        assert abs(val - 1.0) < 0.05


class TestNoise:
    def test_construction(self):
        n = stk.generators.Noise()
        assert n is not None

    def test_with_seed(self):
        n = stk.generators.Noise(42)
        val = n.process_sample()
        assert isinstance(val, float)

    def test_range(self):
        n = stk.generators.Noise(1)
        out = n.process(10000)
        assert np.all(out >= -1.0)
        assert np.all(out <= 1.0)
        # Should not be all zeros
        assert np.std(out) > 0.3

    def test_set_seed(self):
        n = stk.generators.Noise()
        n.set_seed(123)
        val = n.process_sample()
        assert isinstance(val, float)


class TestBlit:
    def test_construction(self):
        b = stk.generators.Blit()
        assert b is not None

    def test_construction_with_freq(self):
        b = stk.generators.Blit(440.0)
        out = b.process(1024)
        assert out.shape == (1024,)

    def test_set_frequency(self):
        b = stk.generators.Blit()
        b.set_frequency(880.0)
        out = b.process(1024)
        assert np.max(np.abs(out)) > 0.5

    def test_set_harmonics(self):
        b = stk.generators.Blit(440.0)
        b.set_harmonics(5)
        out = b.process(1024)
        assert out.shape == (1024,)

    def test_phase(self):
        b = stk.generators.Blit()
        b.set_phase(0.5)
        assert abs(b.get_phase() - 0.5) < 0.01


class TestBlitSaw:
    def test_process(self):
        b = stk.generators.BlitSaw(220.0)
        out = b.process(4096)
        assert out.shape == (4096,)
        # Sawtooth should eventually have significant amplitude
        assert np.max(np.abs(out)) > 0.1


class TestBlitSquare:
    def test_process(self):
        b = stk.generators.BlitSquare(220.0)
        out = b.process(4096)
        assert out.shape == (4096,)
        assert np.max(np.abs(out)) > 0.1

    def test_phase(self):
        b = stk.generators.BlitSquare()
        b.set_phase(0.25)
        assert abs(b.get_phase() - 0.25) < 0.01


class TestADSR:
    def test_construction(self):
        a = stk.generators.ADSR()
        assert a.get_state() == stk.generators.ADSR_IDLE

    def test_key_on_off(self):
        a = stk.generators.ADSR()
        a.set_all_times(0.001, 0.01, 0.5, 0.01)
        a.key_on()
        assert a.get_state() == stk.generators.ADSR_ATTACK
        # Process through attack
        a.process(1000)
        a.key_off()
        assert a.get_state() == stk.generators.ADSR_RELEASE

    def test_envelope_shape(self):
        a = stk.generators.ADSR()
        a.set_all_times(0.01, 0.01, 0.5, 0.01)
        a.key_on()
        # Attack phase
        out = a.process(int(SR * 0.02))
        # Should reach near 1.0 during attack
        assert np.max(out) > 0.8
        a.key_off()
        # Release phase
        release = a.process(int(SR * 0.02))
        # Should decay toward 0
        assert release[-1] < 0.1

    def test_set_value(self):
        a = stk.generators.ADSR()
        a.set_value(0.7)
        val = a.process_sample()
        assert abs(val - 0.7) < 0.01

    def test_states_constants(self):
        assert stk.generators.ADSR_ATTACK == 0
        assert stk.generators.ADSR_DECAY == 1
        assert stk.generators.ADSR_SUSTAIN == 2
        assert stk.generators.ADSR_RELEASE == 3
        assert stk.generators.ADSR_IDLE == 4


class TestAsymp:
    def test_key_on_off(self):
        a = stk.generators.Asymp()
        a.set_tau(0.005)
        a.key_on()
        out = a.process(int(SR * 0.05))
        # Should approach 1.0
        assert out[-1] > 0.8
        a.key_off()
        out = a.process(int(SR * 0.05))
        # Should approach 0.0
        assert out[-1] < 0.2

    def test_set_target(self):
        a = stk.generators.Asymp()
        a.set_tau(0.001)
        a.set_target(0.5)
        out = a.process(int(SR * 0.1))
        assert abs(out[-1] - 0.5) < 0.01


class TestEnvelope:
    def test_basic(self):
        e = stk.generators.Envelope()
        e.set_rate(0.01)
        e.key_on()
        out = e.process(200)
        assert out[-1] > 0.0


class TestModulate:
    def test_basic(self):
        m = stk.generators.Modulate()
        m.set_vibrato_rate(5.0)
        m.set_vibrato_gain(1.0)
        out = m.process(1024)
        assert out.shape == (1024,)
        assert np.max(np.abs(out)) > 0.0


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


class TestBiQuad:
    def test_construction(self):
        bq = stk.filters.BiQuad()
        assert bq is not None

    def test_lowpass_attenuation(self):
        bq = stk.filters.BiQuad()
        bq.set_low_pass(1000.0)
        # Feed a high-frequency signal (10kHz)
        t = np.arange(4096, dtype=np.float32) / SR
        high_freq = np.sin(2 * np.pi * 10000 * t).astype(np.float32)
        out = bq.process(high_freq)
        # High frequency should be attenuated
        assert np.max(np.abs(out[100:])) < 0.3

    def test_highpass(self):
        bq = stk.filters.BiQuad()
        bq.set_high_pass(5000.0)
        # Feed a low-frequency signal (100Hz)
        t = np.arange(4096, dtype=np.float32) / SR
        low_freq = np.sin(2 * np.pi * 100 * t).astype(np.float32)
        out = bq.process(low_freq)
        # Low frequency should be attenuated
        assert np.max(np.abs(out[200:])) < 0.3

    def test_set_coefficients(self):
        bq = stk.filters.BiQuad()
        bq.set_coefficients(1.0, 0.0, 0.0, 0.0, 0.0)
        # Pass-through: input = output
        t = np.arange(100, dtype=np.float32) / SR
        sig = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        out = bq.process(sig)
        np.testing.assert_allclose(out, sig, atol=1e-5)

    def test_gain(self):
        bq = stk.filters.BiQuad()
        bq.set_gain(2.0)
        assert abs(bq.get_gain() - 2.0) < 1e-5

    def test_clear(self):
        bq = stk.filters.BiQuad()
        bq.set_low_pass(1000.0)
        sig = np.ones(100, dtype=np.float32)
        bq.process(sig)
        bq.clear()
        val = bq.process_sample(0.0)
        assert abs(val) < 1e-6

    def test_bandpass(self):
        bq = stk.filters.BiQuad()
        bq.set_band_pass(1000.0, 2.0)
        out = bq.process(np.ones(100, dtype=np.float32))
        assert out.shape == (100,)

    def test_bandreject(self):
        bq = stk.filters.BiQuad()
        bq.set_band_reject(1000.0, 2.0)
        out = bq.process(np.ones(100, dtype=np.float32))
        assert out.shape == (100,)

    def test_allpass(self):
        bq = stk.filters.BiQuad()
        bq.set_all_pass(1000.0, 1.0)
        out = bq.process(np.ones(100, dtype=np.float32))
        assert out.shape == (100,)

    def test_resonance(self):
        bq = stk.filters.BiQuad()
        bq.set_resonance(1000.0, 0.99, True)
        out = bq.process(np.ones(100, dtype=np.float32))
        assert out.shape == (100,)

    def test_notch(self):
        bq = stk.filters.BiQuad()
        bq.set_notch(1000.0, 0.99)
        out = bq.process(np.ones(100, dtype=np.float32))
        assert out.shape == (100,)


class TestOnePole:
    def test_construction(self):
        op = stk.filters.OnePole()
        assert op is not None

    def test_set_pole(self):
        op = stk.filters.OnePole()
        op.set_pole(0.95)
        out = op.process(np.ones(100, dtype=np.float32))
        assert out.shape == (100,)

    def test_lowpass_behavior(self):
        op = stk.filters.OnePole(0.9)
        # Step input
        out = op.process(np.ones(100, dtype=np.float32))
        # Should approach 1.0 for lowpass
        assert out[-1] > 0.5


class TestOneZero:
    def test_construction(self):
        oz = stk.filters.OneZero()
        assert oz is not None

    def test_process(self):
        oz = stk.filters.OneZero(-1.0)
        out = oz.process(np.ones(100, dtype=np.float32))
        assert out.shape == (100,)


class TestTwoPole:
    def test_resonance(self):
        tp = stk.filters.TwoPole()
        tp.set_resonance(1000.0, 0.99, True)
        out = tp.process(np.ones(100, dtype=np.float32))
        assert out.shape == (100,)


class TestTwoZero:
    def test_notch(self):
        tz = stk.filters.TwoZero()
        tz.set_notch(1000.0, 0.99)
        out = tz.process(np.ones(100, dtype=np.float32))
        assert out.shape == (100,)


class TestPoleZero:
    def test_allpass(self):
        pz = stk.filters.PoleZero()
        pz.set_allpass(0.5)
        out = pz.process(np.ones(100, dtype=np.float32))
        assert out.shape == (100,)

    def test_block_zero(self):
        pz = stk.filters.PoleZero()
        pz.set_block_zero(0.99)
        out = pz.process(np.ones(100, dtype=np.float32))
        assert out.shape == (100,)


class TestResonate:
    def test_basic(self):
        r = stk.filters.Resonate()
        r.set_resonance(1000.0, 0.99)
        r.key_on()
        out = r.process(1024)
        assert out.shape == (1024,)
        assert np.max(np.abs(out)) > 0.0


class TestFormSwep:
    def test_basic(self):
        fs = stk.filters.FormSwep()
        fs.set_states(500.0, 0.99, 1.0)
        fs.set_targets(2000.0, 0.99, 1.0)
        fs.set_sweep_rate(0.01)
        out = fs.process(np.ones(100, dtype=np.float32))
        assert out.shape == (100,)


# ---------------------------------------------------------------------------
# Delays
# ---------------------------------------------------------------------------


class TestDelay:
    def test_integer_roundtrip(self):
        d = stk.delays.Delay(10, 100)
        # Push impulse through
        d.process_sample(1.0)
        for _ in range(9):
            d.process_sample(0.0)
        # Should come out at delay 10
        val = d.process_sample(0.0)
        assert abs(val - 1.0) < 1e-5

    def test_set_delay(self):
        d = stk.delays.Delay(5, 100)
        d.set_delay(5)
        assert d.get_delay() == 5

    def test_tap_in_out(self):
        d = stk.delays.Delay(10, 100)
        d.tap_in(0.5, 3)
        val = d.tap_out(3)
        assert abs(val - 0.5) < 1e-5

    def test_energy(self):
        d = stk.delays.Delay(10, 100)
        d.process_sample(1.0)
        assert d.energy() > 0.0


class TestDelayA:
    def test_construction(self):
        d = stk.delays.DelayA()
        assert d is not None

    def test_output(self):
        d = stk.delays.DelayA(5.5, 100)
        d.process_sample(1.0)
        for _ in range(10):
            d.process_sample(0.0)
        # Output should not be all zero (impulse passed through)
        # DelayA uses allpass interpolation


class TestDelayL:
    def test_construction(self):
        d = stk.delays.DelayL()
        assert d is not None

    def test_interpolation(self):
        d = stk.delays.DelayL(5.5, 100)
        d.process_sample(1.0)
        out = []
        for _ in range(10):
            out.append(d.process_sample(0.0))
        # Should see impulse come out around sample 5-6 (interpolated)
        assert max(abs(v) for v in out) > 0.3

    def test_tap_out(self):
        d = stk.delays.DelayL(10.0, 100)
        d.process_sample(1.0)
        val = d.tap_out(0)
        assert isinstance(val, float)


class TestTapDelay:
    def test_construction(self):
        td = stk.delays.TapDelay([5, 10, 20], 100)
        assert td is not None

    def test_get_tap_delays(self):
        td = stk.delays.TapDelay([5, 10], 100)
        taps = td.get_tap_delays()
        assert taps == [5, 10]

    def test_set_tap_delays(self):
        td = stk.delays.TapDelay([5], 100)
        td.set_tap_delays([3, 7])
        taps = td.get_tap_delays()
        assert taps == [3, 7]


# ---------------------------------------------------------------------------
# Effects
# ---------------------------------------------------------------------------


class TestFreeVerb:
    def test_construction(self):
        fv = stk.effects.FreeVerb()
        assert fv is not None

    def test_stereo_output(self):
        fv = stk.effects.FreeVerb()
        left, right = fv.process_sample(0.5)
        assert isinstance(left, float)
        assert isinstance(right, float)

    def test_stereo_process_mono(self):
        fv = stk.effects.FreeVerb()
        mono = np.random.randn(512).astype(np.float32) * 0.1
        out = fv.process_mono(mono)
        assert out.shape == (2, 512)
        assert out.dtype == np.float32

    def test_stereo_process(self):
        fv = stk.effects.FreeVerb()
        stereo = np.random.randn(2, 512).astype(np.float32) * 0.1
        out = fv.process(stereo)
        assert out.shape == (2, 512)

    def test_room_size(self):
        fv = stk.effects.FreeVerb()
        fv.set_room_size(0.5)
        assert abs(fv.get_room_size() - 0.5) < 0.01

    def test_damping(self):
        fv = stk.effects.FreeVerb()
        fv.set_damping(0.3)
        assert abs(fv.get_damping() - 0.3) < 0.01

    def test_width(self):
        fv = stk.effects.FreeVerb()
        fv.set_width(0.8)
        assert abs(fv.get_width() - 0.8) < 0.01

    def test_effect_mix(self):
        fv = stk.effects.FreeVerb()
        fv.set_effect_mix(0.5)
        # Mix affects output blend


class TestJCRev:
    def test_stereo_output(self):
        r = stk.effects.JCRev(1.0)
        left, r_out = r.process_sample(0.5)
        assert isinstance(left, float)
        assert isinstance(r_out, float)

    def test_process(self):
        r = stk.effects.JCRev(1.0)
        mono = np.random.randn(512).astype(np.float32) * 0.1
        out = r.process(mono)
        assert out.shape == (2, 512)

    def test_set_t60(self):
        r = stk.effects.JCRev()
        r.set_t60(2.0)


class TestNRev:
    def test_stereo_output(self):
        r = stk.effects.NRev(1.0)
        left, r_out = r.process_sample(0.5)
        assert isinstance(left, float)


class TestPRCRev:
    def test_stereo_output(self):
        r = stk.effects.PRCRev(1.0)
        left, r_out = r.process_sample(0.5)
        assert isinstance(left, float)


class TestEcho:
    def test_construction(self):
        e = stk.effects.Echo()
        assert e is not None

    def test_delay(self):
        e = stk.effects.Echo(44100)
        e.set_delay(100)
        e.set_effect_mix(1.0)
        # Feed impulse and collect output over delay length
        inp = np.zeros(200, dtype=np.float32)
        inp[0] = 1.0
        out = e.process(inp)
        # Echo should appear at sample index 100
        assert abs(out[100] - 1.0) < 0.01

    def test_process(self):
        e = stk.effects.Echo(44100)
        e.set_delay(100)
        mono = np.zeros(200, dtype=np.float32)
        mono[0] = 1.0
        out = e.process(mono)
        assert out.shape == (200,)


class TestChorus:
    def test_stereo_output(self):
        c = stk.effects.Chorus()
        left, right = c.process_sample(0.5)
        assert isinstance(left, float)
        assert isinstance(right, float)

    def test_process(self):
        c = stk.effects.Chorus()
        mono = np.random.randn(512).astype(np.float32) * 0.1
        out = c.process(mono)
        assert out.shape == (2, 512)


class TestPitShift:
    def test_construction(self):
        ps = stk.effects.PitShift()
        assert ps is not None

    def test_identity_shift(self):
        ps = stk.effects.PitShift()
        ps.set_shift(1.0)
        ps.set_effect_mix(1.0)
        sig = np.sin(2 * np.pi * 440 * np.arange(4096) / SR).astype(np.float32)
        out = ps.process(sig)
        assert out.shape == (4096,)
        # With shift=1.0, output should correlate with input (after transient)
        # Just check it produces output
        assert np.max(np.abs(out[500:])) > 0.1

    def test_shift_change(self):
        ps = stk.effects.PitShift()
        ps.set_shift(2.0)
        out = ps.process(
            np.sin(np.arange(1024, dtype=np.float32) * 0.1).astype(np.float32)
        )
        assert out.shape == (1024,)


class TestLentPitShift:
    def test_construction(self):
        lps = stk.effects.LentPitShift()
        assert lps is not None

    def test_process(self):
        lps = stk.effects.LentPitShift(1.0, 512)
        sig = np.sin(2 * np.pi * 440 * np.arange(2048) / SR).astype(np.float32)
        out = lps.process(sig)
        assert out.shape == (2048,)


# ---------------------------------------------------------------------------
# Instruments
# ---------------------------------------------------------------------------


class TestClarinet:
    def test_construction(self):
        c = stk.instruments.Clarinet()
        assert c is not None

    def test_note_on_off(self):
        c = stk.instruments.Clarinet()
        c.note_on(440.0, 0.8)
        out = c.process(4096)
        assert out.shape == (4096,)
        assert np.max(np.abs(out)) > 0.01
        c.note_off(0.5)
        decay = c.process(4096)
        # Decay should reduce amplitude
        assert np.max(np.abs(decay[-1024:])) < np.max(np.abs(out))

    def test_frequency(self):
        c = stk.instruments.Clarinet()
        c.set_frequency(880.0)
        c.note_on(880.0, 0.8)
        out = c.process(2048)
        assert out.shape == (2048,)

    def test_control_change(self):
        c = stk.instruments.Clarinet()
        c.control_change(2, 64.0)  # Reed stiffness

    def test_blowing(self):
        c = stk.instruments.Clarinet()
        c.start_blowing(0.8, 0.02)
        c.process(2048)
        c.stop_blowing(0.02)


class TestFlute:
    def test_lifecycle(self):
        f = stk.instruments.Flute()
        f.note_on(440.0, 0.8)
        out = f.process(2048)
        assert np.max(np.abs(out)) > 0.001
        f.note_off(0.5)

    def test_jet_methods(self):
        f = stk.instruments.Flute()
        f.set_jet_reflection(0.5)
        f.set_end_reflection(0.5)
        f.set_jet_delay(0.5)


class TestBrass:
    def test_lifecycle(self):
        b = stk.instruments.Brass()
        b.note_on(220.0, 0.8)
        out = b.process(2048)
        assert out.shape == (2048,)
        b.note_off(0.5)

    def test_set_lip(self):
        b = stk.instruments.Brass()
        b.set_lip(440.0)


class TestBowed:
    def test_lifecycle(self):
        b = stk.instruments.Bowed()
        b.note_on(440.0, 0.8)
        out = b.process(4096)
        assert np.max(np.abs(out)) > 0.001
        b.note_off(0.5)

    def test_bowing(self):
        b = stk.instruments.Bowed()
        b.start_bowing(0.8, 0.02)
        b.process(2048)
        b.stop_bowing(0.02)


class TestPlucked:
    def test_lifecycle(self):
        p = stk.instruments.Plucked()
        p.note_on(440.0, 0.8)
        out = p.process(4096)
        assert np.max(np.abs(out)) > 0.01
        p.note_off(0.5)


class TestSitar:
    def test_lifecycle(self):
        s = stk.instruments.Sitar()
        s.note_on(220.0, 0.8)
        out = s.process(4096)
        assert np.max(np.abs(out)) > 0.01


class TestStifKarp:
    def test_lifecycle(self):
        sk = stk.instruments.StifKarp()
        sk.note_on(440.0, 0.8)
        out = sk.process(4096)
        assert np.max(np.abs(out)) > 0.01

    def test_stretch(self):
        sk = stk.instruments.StifKarp()
        sk.set_stretch(0.5)
        sk.set_base_loop_gain(0.99)


class TestSaxofony:
    def test_lifecycle(self):
        s = stk.instruments.Saxofony()
        s.note_on(440.0, 0.8)
        out = s.process(2048)
        assert out.shape == (2048,)
        s.note_off(0.5)


class TestRecorder:
    def test_lifecycle(self):
        r = stk.instruments.Recorder()
        r.note_on(440.0, 0.8)
        out = r.process(2048)
        assert out.shape == (2048,)
        r.note_off(0.5)


class TestBlowBotl:
    def test_lifecycle(self):
        b = stk.instruments.BlowBotl()
        b.note_on(440.0, 0.8)
        out = b.process(2048)
        assert out.shape == (2048,)
        b.note_off(0.5)


class TestBlowHole:
    def test_lifecycle(self):
        b = stk.instruments.BlowHole(8.0)
        b.note_on(440.0, 0.8)
        out = b.process(2048)
        assert out.shape == (2048,)
        b.note_off(0.5)

    def test_tonehole_vent(self):
        b = stk.instruments.BlowHole(8.0)
        b.set_tonehole(0.5)
        b.set_vent(0.5)


class TestWhistle:
    def test_lifecycle(self):
        w = stk.instruments.Whistle()
        w.note_on(880.0, 0.8)
        out = w.process(2048)
        assert out.shape == (2048,)
        w.note_off(0.5)


class TestGuitar:
    def test_construction(self):
        g = stk.instruments.Guitar()
        assert g is not None

    def test_note_on_off(self):
        g = stk.instruments.Guitar(6)
        g.note_on(440.0, 0.8, 0)
        out = g.process(4096)
        assert out.shape == (4096,)
        assert np.max(np.abs(out)) > 0.01
        g.note_off(0.5, 0)

    def test_multiple_strings(self):
        g = stk.instruments.Guitar(6)
        g.note_on(220.0, 0.8, 0)
        g.note_on(330.0, 0.8, 1)
        out = g.process(2048)
        assert out.shape == (2048,)


class TestTwang:
    def test_construction(self):
        t = stk.instruments.Twang()
        assert t is not None

    def test_process(self):
        t = stk.instruments.Twang()
        t.set_frequency(440.0)
        t.set_loop_gain(0.99)
        # Feed impulse
        excitation = np.zeros(4096, dtype=np.float32)
        excitation[0] = 1.0
        out = t.process(excitation)
        assert out.shape == (4096,)
        assert np.max(np.abs(out)) > 0.01

    def test_set_loop_filter(self):
        t = stk.instruments.Twang()
        t.set_loop_filter([0.5, 0.5])

"""Tests for DaisySP bindings (nanodsp.daisysp)."""

import numpy as np
import pytest

from nanodsp._core import daisysp

SR = 48000.0

# ---------------------------------------------------------------------------
# Oscillators
# ---------------------------------------------------------------------------


class TestOscillator:
    def test_construction_and_init(self):
        osc = daisysp.oscillators.Oscillator()
        osc.init(SR)

    def test_waveform_constants(self):
        m = daisysp.oscillators
        assert isinstance(m.WAVE_SIN, int)
        assert isinstance(m.WAVE_TRI, int)
        assert isinstance(m.WAVE_SAW, int)
        assert isinstance(m.WAVE_SQUARE, int)

    def test_process_shape_dtype(self):
        osc = daisysp.oscillators.Oscillator()
        osc.init(SR)
        osc.set_freq(440.0)
        osc.set_amp(1.0)
        osc.set_waveform(daisysp.oscillators.WAVE_SIN)
        out = osc.process(1024)
        assert out.shape == (1024,)
        assert out.dtype == np.float32

    def test_sine_range(self):
        osc = daisysp.oscillators.Oscillator()
        osc.init(SR)
        osc.set_freq(440.0)
        osc.set_amp(1.0)
        osc.set_waveform(daisysp.oscillators.WAVE_SIN)
        out = osc.process(4096)
        assert np.max(np.abs(out)) > 0.9
        assert np.max(np.abs(out)) <= 1.01

    def test_process_sample(self):
        osc = daisysp.oscillators.Oscillator()
        osc.init(SR)
        osc.set_freq(440.0)
        osc.set_amp(1.0)
        osc.set_waveform(daisysp.oscillators.WAVE_SIN)
        val = osc.process_sample()
        assert isinstance(val, float)

    def test_zero_amp_produces_silence(self):
        osc = daisysp.oscillators.Oscillator()
        osc.init(SR)
        osc.set_freq(440.0)
        osc.set_amp(0.0)
        osc.set_waveform(daisysp.oscillators.WAVE_SIN)
        out = osc.process(512)
        assert np.max(np.abs(out)) < 1e-6

    def test_all_waveforms(self):
        m = daisysp.oscillators
        for wf in [
            m.WAVE_SIN,
            m.WAVE_TRI,
            m.WAVE_SAW,
            m.WAVE_RAMP,
            m.WAVE_SQUARE,
            m.WAVE_POLYBLEP_TRI,
            m.WAVE_POLYBLEP_SAW,
            m.WAVE_POLYBLEP_SQUARE,
        ]:
            osc = m.Oscillator()
            osc.init(SR)
            osc.set_freq(440.0)
            osc.set_amp(1.0)
            osc.set_waveform(wf)
            out = osc.process(512)
            assert np.max(np.abs(out)) > 0.1, f"Waveform {wf} produced near-silence"


class TestFm2:
    def test_produces_output(self):
        fm = daisysp.oscillators.Fm2()
        fm.init(SR)
        fm.set_frequency(440.0)
        fm.set_ratio(2.0)
        fm.set_index(1.0)
        out = fm.process(1024)
        assert out.shape == (1024,)
        assert np.max(np.abs(out)) > 0.01


class TestFormantOscillator:
    def test_produces_output(self):
        fo = daisysp.oscillators.FormantOscillator()
        fo.init(SR)
        fo.set_carrier_freq(220.0)
        fo.set_formant_freq(800.0)
        out = fo.process(1024)
        assert out.shape == (1024,)
        assert np.max(np.abs(out)) > 0.01


class TestHarmonicOscillator:
    def test_produces_output(self):
        ho = daisysp.oscillators.HarmonicOscillator()
        ho.init(SR)
        ho.set_freq(440.0)
        amps = np.ones(16, dtype=np.float32)
        ho.set_amplitudes(amps)
        out = ho.process(1024)
        assert out.shape == (1024,)
        assert np.max(np.abs(out)) > 0.01


class TestOscillatorBank:
    def test_produces_output(self):
        ob = daisysp.oscillators.OscillatorBank()
        ob.init(SR)
        ob.set_freq(220.0)
        ob.set_gain(1.0)
        amps = np.ones(7, dtype=np.float32)
        ob.set_amplitudes(amps)
        out = ob.process(1024)
        assert out.shape == (1024,)
        assert np.max(np.abs(out)) > 0.01


class TestVariableSawOscillator:
    def test_produces_output(self):
        v = daisysp.oscillators.VariableSawOscillator()
        v.init(SR)
        v.set_freq(440.0)
        v.set_pw(0.5)
        v.set_waveshape(0.5)
        out = v.process(1024)
        assert out.shape == (1024,)
        assert np.max(np.abs(out)) > 0.01


class TestVariableShapeOscillator:
    def test_produces_output(self):
        v = daisysp.oscillators.VariableShapeOscillator()
        v.init(SR)
        v.set_freq(440.0)
        v.set_pw(0.5)
        v.set_waveshape(0.5)
        out = v.process(1024)
        assert out.shape == (1024,)
        assert np.max(np.abs(out)) > 0.01


class TestVosimOscillator:
    def test_produces_output(self):
        v = daisysp.oscillators.VosimOscillator()
        v.init(SR)
        v.set_freq(220.0)
        v.set_form1_freq(800.0)
        v.set_form2_freq(1200.0)
        v.set_shape(0.5)
        out = v.process(1024)
        assert out.shape == (1024,)
        assert np.max(np.abs(out)) > 0.01


class TestZOscillator:
    def test_produces_output(self):
        z = daisysp.oscillators.ZOscillator()
        z.init(SR)
        z.set_freq(220.0)
        z.set_formant_freq(800.0)
        z.set_shape(0.5)
        z.set_mode(0.5)
        out = z.process(1024)
        assert out.shape == (1024,)
        assert np.max(np.abs(out)) > 0.01


class TestBlOsc:
    def test_produces_output(self):
        bl = daisysp.oscillators.BlOsc()
        bl.init(SR)
        bl.set_freq(440.0)
        bl.set_amp(1.0)
        bl.set_waveform(daisysp.oscillators.BLOSC_WAVE_SAW)
        out = bl.process(1024)
        assert out.shape == (1024,)
        assert np.max(np.abs(out)) > 0.01


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


class TestSvf:
    def test_lowpass_attenuates_high(self):
        svf = daisysp.filters.Svf()
        svf.init(SR)
        svf.set_freq(1000.0)
        svf.set_res(0.0)
        # Feed white noise
        np.random.seed(42)
        noise = np.random.randn(4096).astype(np.float32)
        low = svf.process_low(noise)
        assert low.shape == (4096,)
        assert low.dtype == np.float32
        # Low energy should be less than input energy for wideband input
        assert np.sum(low**2) < np.sum(noise**2)

    def test_multi_output(self):
        svf = daisysp.filters.Svf()
        svf.init(SR)
        svf.set_freq(2000.0)
        svf.set_res(0.5)
        result = svf.process_sample(0.5)
        assert len(result) == 5  # (low, high, band, notch, peak)

    def test_all_outputs(self):
        svf = daisysp.filters.Svf()
        svf.init(SR)
        svf.set_freq(1000.0)
        svf.set_res(0.5)
        np.random.seed(42)
        noise = np.random.randn(1024).astype(np.float32)
        for method in [
            "process_low",
            "process_high",
            "process_band",
            "process_notch",
            "process_peak",
        ]:
            svf2 = daisysp.filters.Svf()
            svf2.init(SR)
            svf2.set_freq(1000.0)
            svf2.set_res(0.5)
            out = getattr(svf2, method)(noise)
            assert out.shape == (1024,), f"{method} shape wrong"


class TestOnePole:
    def test_lowpass(self):
        op = daisysp.filters.OnePole()
        op.init()
        op.set_frequency(0.1)
        op.set_filter_mode(daisysp.filters.OnePoleFM.LOW_PASS)
        np.random.seed(42)
        noise = np.random.randn(1024).astype(np.float32)
        out = op.process(noise)
        assert out.shape == (1024,)
        assert np.sum(out**2) < np.sum(noise**2)


class TestLadderFilter:
    def test_filter_modes(self):
        modes = daisysp.filters
        lf = modes.LadderFilter()
        lf.init(SR)
        lf.set_freq(1000.0)
        lf.set_res(0.5)
        lf.set_filter_mode(modes.LadderFilterMode.LP24)
        np.random.seed(42)
        noise = np.random.randn(1024).astype(np.float32)
        out = lf.process(noise)
        assert out.shape == (1024,)


class TestSoap:
    def test_bandpass_bandreject(self):
        sp = daisysp.filters.Soap()
        sp.init(SR)
        sp.set_center_freq(1000.0)
        sp.set_filter_bandwidth(200.0)
        bp, br = sp.process_sample(1.0)
        assert isinstance(bp, float)
        assert isinstance(br, float)


class TestLGPLFilters:
    def test_allpass(self):
        ap = daisysp.filters.Allpass(SR, 4096)
        ap.set_freq(0.5)
        ap.set_rev_time(1.0)
        np.random.seed(42)
        noise = np.random.randn(512).astype(np.float32)
        out = ap.process(noise)
        assert out.shape == (512,)

    def test_atone(self):
        at = daisysp.filters.ATone()
        at.init(SR)
        at.set_freq(1000.0)
        assert at.get_freq() == pytest.approx(1000.0)
        np.random.seed(42)
        noise = np.random.randn(512).astype(np.float32)
        out = at.process(noise)
        assert out.shape == (512,)

    def test_daisy_biquad(self):
        bq = daisysp.filters.DaisyBiquad()
        bq.init(SR)
        bq.set_cutoff(1000.0)
        bq.set_res(0.7)
        np.random.seed(42)
        noise = np.random.randn(512).astype(np.float32)
        out = bq.process(noise)
        assert out.shape == (512,)

    def test_comb(self):
        cb = daisysp.filters.Comb(SR, 4096)
        cb.set_freq(500.0)
        cb.set_rev_time(0.5)
        np.random.seed(42)
        noise = np.random.randn(512).astype(np.float32)
        out = cb.process(noise)
        assert out.shape == (512,)

    def test_mode(self):
        md = daisysp.filters.Mode()
        md.init(SR)
        md.set_freq(1000.0)
        md.set_q(100.0)
        np.random.seed(42)
        noise = np.random.randn(512).astype(np.float32)
        out = md.process(noise)
        assert out.shape == (512,)

    def test_moog_ladder(self):
        ml = daisysp.filters.MoogLadder()
        ml.init(SR)
        ml.set_freq(1000.0)
        ml.set_res(0.7)
        np.random.seed(42)
        noise = np.random.randn(512).astype(np.float32)
        out = ml.process(noise)
        assert out.shape == (512,)

    def test_nlfilt(self):
        nf = daisysp.filters.NlFilt()
        nf.init()
        nf.set_coefficients(0.5, 0.5, 0.0, 0.0, 0.0)
        np.random.seed(42)
        noise = np.random.randn(512).astype(np.float32) * 0.1
        out = nf.process(noise)
        assert out.shape == (512,)

    def test_tone(self):
        tn = daisysp.filters.Tone()
        tn.init(SR)
        tn.set_freq(1000.0)
        assert tn.get_freq() == pytest.approx(1000.0)
        np.random.seed(42)
        noise = np.random.randn(512).astype(np.float32)
        out = tn.process(noise)
        assert out.shape == (512,)


# ---------------------------------------------------------------------------
# Effects
# ---------------------------------------------------------------------------


class TestAutowah:
    def test_process(self):
        aw = daisysp.effects.Autowah()
        aw.init(SR)
        aw.set_wah(0.5)
        aw.set_dry_wet(0.5)
        aw.set_level(0.5)
        inp = np.sin(np.linspace(0, 100, 1024)).astype(np.float32) * 0.5
        out = aw.process(inp)
        assert out.shape == (1024,)


class TestChorus:
    def test_mono(self):
        ch = daisysp.effects.Chorus()
        ch.init(SR)
        ch.set_lfo_freq(1.0)
        ch.set_lfo_depth(0.5)
        ch.set_feedback(0.3)
        inp = np.sin(np.linspace(0, 100, 1024)).astype(np.float32) * 0.5
        out = ch.process(inp)
        assert out.shape == (1024,)

    def test_stereo(self):
        ch = daisysp.effects.Chorus()
        ch.init(SR)
        ch.set_lfo_freq(1.0)
        ch.set_lfo_depth(0.5)
        inp = np.sin(np.linspace(0, 100, 1024)).astype(np.float32) * 0.5
        out = ch.process_stereo(inp)
        assert out.shape == (2, 1024)
        assert out.dtype == np.float32


class TestDecimator:
    def test_process(self):
        dc = daisysp.effects.Decimator()
        dc.init()
        dc.set_downsample_factor(0.5)
        dc.set_bitcrush_factor(0.5)
        dc.set_bits_to_crush(4)
        inp = np.sin(np.linspace(0, 100, 512)).astype(np.float32)
        out = dc.process(inp)
        assert out.shape == (512,)


class TestFlanger:
    def test_process(self):
        fl = daisysp.effects.Flanger()
        fl.init(SR)
        fl.set_feedback(0.5)
        fl.set_lfo_depth(0.5)
        fl.set_lfo_freq(0.5)
        inp = np.sin(np.linspace(0, 100, 512)).astype(np.float32) * 0.5
        out = fl.process(inp)
        assert out.shape == (512,)


class TestOverdrive:
    def test_drive(self):
        od = daisysp.effects.Overdrive()
        od.init()
        od.set_drive(0.8)
        inp = np.sin(np.linspace(0, 100, 512)).astype(np.float32) * 0.5
        out = od.process(inp)
        assert out.shape == (512,)
        assert np.max(np.abs(out)) > 0.01


class TestPhaser:
    def test_process(self):
        ph = daisysp.effects.Phaser()
        ph.init(SR)
        ph.set_lfo_freq(0.5)
        ph.set_lfo_depth(0.5)
        ph.set_feedback(0.5)
        inp = np.sin(np.linspace(0, 100, 512)).astype(np.float32) * 0.5
        out = ph.process(inp)
        assert out.shape == (512,)


class TestPitchShifter:
    def test_process(self):
        ps = daisysp.effects.PitchShifter()
        ps.init(SR)
        ps.set_transposition(7.0)
        inp = np.sin(np.linspace(0, 100, 512)).astype(np.float32) * 0.5
        out = ps.process(inp)
        assert out.shape == (512,)


class TestSampleRateReducer:
    def test_process(self):
        srr = daisysp.effects.SampleRateReducer()
        srr.init()
        srr.set_freq(0.5)
        inp = np.sin(np.linspace(0, 100, 512)).astype(np.float32)
        out = srr.process(inp)
        assert out.shape == (512,)


class TestTremolo:
    def test_process(self):
        tr = daisysp.effects.Tremolo()
        tr.init(SR)
        tr.set_freq(5.0)
        tr.set_depth(0.8)
        inp = np.sin(np.linspace(0, 100, 512)).astype(np.float32) * 0.5
        out = tr.process(inp)
        assert out.shape == (512,)


class TestWavefolder:
    def test_process(self):
        wf = daisysp.effects.Wavefolder()
        wf.init()
        wf.set_gain(2.0)
        inp = np.sin(np.linspace(0, 100, 512)).astype(np.float32)
        out = wf.process(inp)
        assert out.shape == (512,)


class TestBitcrush:
    def test_process(self):
        bc = daisysp.effects.Bitcrush()
        bc.init(SR)
        bc.set_bit_depth(8)
        bc.set_crush_rate(SR / 4)
        inp = np.sin(np.linspace(0, 100, 512)).astype(np.float32)
        out = bc.process(inp)
        assert out.shape == (512,)


class TestFold:
    def test_process(self):
        fd = daisysp.effects.Fold()
        fd.init()
        fd.set_increment(0.5)
        inp = np.sin(np.linspace(0, 100, 512)).astype(np.float32)
        out = fd.process(inp)
        assert out.shape == (512,)


class TestReverbSc:
    def test_stereo(self):
        rv = daisysp.effects.ReverbSc()
        rv.init(SR)
        rv.set_feedback(0.9)
        rv.set_lp_freq(10000.0)
        # ReverbSc has internal delay lines; need enough samples for output
        inp = np.zeros((2, 8192), dtype=np.float32)
        inp[0, 0] = 1.0  # impulse
        inp[1, 0] = 1.0
        out = rv.process(inp)
        assert out.shape == (2, 8192)
        assert out.dtype == np.float32
        assert np.max(np.abs(out)) > 0.001

    def test_process_sample(self):
        rv = daisysp.effects.ReverbSc()
        rv.init(SR)
        rv.set_feedback(0.8)
        rv.set_lp_freq(10000.0)
        o1, o2 = rv.process_sample(1.0, 1.0)
        assert isinstance(o1, float)
        assert isinstance(o2, float)


# ---------------------------------------------------------------------------
# Dynamics
# ---------------------------------------------------------------------------


class TestCrossFade:
    def test_linear(self):
        cf = daisysp.dynamics.CrossFade()
        cf.init(daisysp.dynamics.CROSSFADE_LIN)
        cf.set_pos(0.0)
        a = np.ones(100, dtype=np.float32)
        b = np.zeros(100, dtype=np.float32)
        out = cf.process(a, b)
        assert out.shape == (100,)
        # pos=0 should be all input1
        assert np.allclose(out, 1.0, atol=0.05)

    def test_midpoint(self):
        cf = daisysp.dynamics.CrossFade()
        cf.init(daisysp.dynamics.CROSSFADE_LIN)
        cf.set_pos(0.5)
        val = cf.process_sample(1.0, 0.0)
        assert val == pytest.approx(0.5, abs=0.05)


class TestLimiter:
    def test_limits(self):
        lm = daisysp.dynamics.Limiter()
        lm.init()
        inp = np.array([0.0, 0.5, 1.0, 2.0, 5.0], dtype=np.float32)
        out = lm.process(inp, 1.0)
        assert out.shape == (5,)


class TestBalance:
    def test_process(self):
        bl = daisysp.dynamics.Balance()
        bl.init(SR)
        bl.set_cutoff(1000.0)
        sig = np.sin(np.linspace(0, 100, 512)).astype(np.float32) * 0.5
        comp = np.ones(512, dtype=np.float32)
        out = bl.process(sig, comp)
        assert out.shape == (512,)


class TestCompressor:
    def test_process(self):
        cp = daisysp.dynamics.Compressor()
        cp.init(SR)
        cp.set_ratio(4.0)
        cp.set_threshold(-20.0)
        cp.set_attack(0.01)
        cp.set_release(0.1)
        inp = np.sin(np.linspace(0, 100, 512)).astype(np.float32) * 0.8
        out = cp.process(inp)
        assert out.shape == (512,)

    def test_getters(self):
        cp = daisysp.dynamics.Compressor()
        cp.init(SR)
        cp.set_ratio(4.0)
        cp.set_threshold(-20.0)
        assert cp.get_ratio() == pytest.approx(4.0)
        assert cp.get_threshold() == pytest.approx(-20.0)


# ---------------------------------------------------------------------------
# Control
# ---------------------------------------------------------------------------


class TestAdEnv:
    def test_envelope(self):
        ae = daisysp.control.AdEnv()
        ae.init(SR)
        ae.set_time(daisysp.control.ADENV_SEG_ATTACK, 0.01)
        ae.set_time(daisysp.control.ADENV_SEG_DECAY, 0.1)
        ae.set_min(0.0)
        ae.set_max(1.0)
        ae.trigger()
        out = ae.process(2048)
        assert out.shape == (2048,)
        assert np.max(out) > 0.5  # should reach near max


class TestAdsr:
    def test_envelope(self):
        ad = daisysp.control.Adsr()
        ad.init(SR)
        ad.set_attack_time(0.01)
        ad.set_decay_time(0.05)
        ad.set_sustain_level(0.5)
        ad.set_release_time(0.1)
        # Gate on for 2048 samples
        gate = np.ones(2048, dtype=np.float32)
        out = ad.process(gate)
        assert out.shape == (2048,)
        assert np.max(out) > 0.5

    def test_segment(self):
        ad = daisysp.control.Adsr()
        ad.init(SR)
        seg = ad.get_current_segment()
        assert isinstance(seg, int)


class TestPhasor:
    def test_ramp(self):
        ph = daisysp.control.Phasor()
        ph.init(SR, 1.0)
        out = ph.process(int(SR))  # 1 second = 1 full cycle
        assert out.shape == (int(SR),)
        assert out[0] < out[int(SR) // 2]  # should be increasing
        assert np.min(out) >= 0.0
        assert np.max(out) < 1.0


class TestLine:
    def test_ramp(self):
        ln = daisysp.control.Line()
        ln.init(SR)
        ln.start(0.0, 1.0, 0.1)
        samples, finished = ln.process(int(SR * 0.2))
        assert samples.shape == (int(SR * 0.2),)
        assert isinstance(finished, bool)


# ---------------------------------------------------------------------------
# Noise
# ---------------------------------------------------------------------------


class TestWhiteNoise:
    def test_output(self):
        wn = daisysp.noise.WhiteNoise()
        wn.init()
        wn.set_amp(1.0)
        out = wn.process(4096)
        assert out.shape == (4096,)
        assert out.dtype == np.float32
        assert np.std(out) > 0.1  # should have significant variance

    def test_zero_amp(self):
        wn = daisysp.noise.WhiteNoise()
        wn.init()
        wn.set_amp(0.0)
        out = wn.process(512)
        assert np.max(np.abs(out)) < 1e-6


class TestDust:
    def test_output(self):
        du = daisysp.noise.Dust()
        du.init()
        du.set_density(0.5)
        out = du.process(4096)
        assert out.shape == (4096,)


class TestClockedNoise:
    def test_output(self):
        cn = daisysp.noise.ClockedNoise()
        cn.init(SR)
        cn.set_freq(100.0)
        out = cn.process(1024)
        assert out.shape == (1024,)


class TestFractalRandomGenerator:
    def test_output(self):
        frg = daisysp.noise.FractalRandomGenerator()
        frg.init(SR)
        frg.set_freq(10.0)
        frg.set_color(0.5)
        out = frg.process(1024)
        assert out.shape == (1024,)


class TestGrainletOscillator:
    def test_output(self):
        gl = daisysp.noise.GrainletOscillator()
        gl.init(SR)
        gl.set_freq(100.0)
        gl.set_formant_freq(500.0)
        gl.set_shape(0.5)
        gl.set_bleed(0.5)
        out = gl.process(1024)
        assert out.shape == (1024,)


class TestParticle:
    def test_output(self):
        pt = daisysp.noise.Particle()
        pt.init(SR)
        pt.set_freq(200.0)
        pt.set_resonance(0.9)
        pt.set_density(0.5)
        pt.set_gain(1.0)
        out = pt.process(1024)
        assert out.shape == (1024,)


class TestSmoothRandomGenerator:
    def test_output(self):
        sr = daisysp.noise.SmoothRandomGenerator()
        sr.init(SR)
        sr.set_freq(5.0)
        out = sr.process(1024)
        assert out.shape == (1024,)


# ---------------------------------------------------------------------------
# Drums
# ---------------------------------------------------------------------------


class TestAnalogBassDrum:
    def test_trigger_and_output(self):
        bd = daisysp.drums.AnalogBassDrum()
        bd.init(SR)
        bd.set_freq(60.0)
        bd.set_tone(0.5)
        bd.set_decay(0.5)
        out = bd.process(4096)
        assert out.shape == (4096,)
        assert np.max(np.abs(out)) > 0.01


class TestAnalogSnareDrum:
    def test_trigger_and_output(self):
        sd = daisysp.drums.AnalogSnareDrum()
        sd.init(SR)
        sd.set_freq(200.0)
        sd.set_tone(0.5)
        sd.set_decay(0.5)
        sd.set_snappy(0.5)
        out = sd.process(4096)
        assert out.shape == (4096,)
        assert np.max(np.abs(out)) > 0.01


class TestHiHat:
    def test_trigger_and_output(self):
        hh = daisysp.drums.HiHat()
        hh.init(SR)
        hh.set_freq(3000.0)
        hh.set_tone(0.5)
        hh.set_decay(0.3)
        hh.set_noisiness(0.8)
        out = hh.process(4096)
        assert out.shape == (4096,)
        assert np.max(np.abs(out)) > 0.001


class TestSyntheticBassDrum:
    def test_trigger_and_output(self):
        bd = daisysp.drums.SyntheticBassDrum()
        bd.init(SR)
        bd.set_freq(60.0)
        bd.set_tone(0.5)
        bd.set_decay(0.5)
        out = bd.process(4096)
        assert out.shape == (4096,)
        assert np.max(np.abs(out)) > 0.01


class TestSyntheticSnareDrum:
    def test_trigger_and_output(self):
        sd = daisysp.drums.SyntheticSnareDrum()
        sd.init(SR)
        sd.set_freq(200.0)
        sd.set_decay(0.5)
        sd.set_snappy(0.5)
        out = sd.process(4096)
        assert out.shape == (4096,)
        assert np.max(np.abs(out)) > 0.01


# ---------------------------------------------------------------------------
# Physical Modeling
# ---------------------------------------------------------------------------


class TestDrip:
    def test_trigger_and_output(self):
        dr = daisysp.physical_modeling.Drip()
        dr.init(SR, 0.004)
        out = dr.process(4096)
        assert out.shape == (4096,)


class TestString:
    def test_excitation(self):
        st = daisysp.physical_modeling.String()
        st.init(SR)
        st.set_freq(220.0)
        st.set_brightness(0.5)
        st.set_damping(0.5)
        # Impulse excitation
        inp = np.zeros(4096, dtype=np.float32)
        inp[0] = 1.0
        out = st.process(inp)
        assert out.shape == (4096,)
        assert np.max(np.abs(out)) > 0.001


class TestModalVoice:
    def test_trigger(self):
        mv = daisysp.physical_modeling.ModalVoice()
        mv.init(SR)
        mv.set_freq(440.0)
        mv.set_accent(0.8)
        mv.set_structure(0.5)
        mv.set_brightness(0.5)
        mv.set_damping(0.5)
        out = mv.process(4096)
        assert out.shape == (4096,)
        assert np.max(np.abs(out)) > 0.001


class TestResonator:
    def test_process(self):
        rs = daisysp.physical_modeling.Resonator()
        rs.init(0.3, 24, SR)
        rs.set_freq(440.0)
        rs.set_structure(0.5)
        rs.set_brightness(0.5)
        rs.set_damping(0.5)
        inp = np.zeros(2048, dtype=np.float32)
        inp[0] = 1.0
        out = rs.process(inp)
        assert out.shape == (2048,)


class TestStringVoice:
    def test_trigger(self):
        sv = daisysp.physical_modeling.StringVoice()
        sv.init(SR)
        sv.set_freq(220.0)
        sv.set_accent(0.8)
        sv.set_structure(0.5)
        sv.set_brightness(0.5)
        sv.set_damping(0.5)
        out = sv.process(4096)
        assert out.shape == (4096,)
        assert np.max(np.abs(out)) > 0.001


class TestPluck:
    def test_trigger(self):
        pl = daisysp.physical_modeling.Pluck(SR, 256, 0)
        pl.set_freq(440.0)
        pl.set_amp(1.0)
        pl.set_decay(0.95)
        pl.set_damp(0.5)
        out = pl.process(4096)
        assert out.shape == (4096,)
        assert np.max(np.abs(out)) > 0.001


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


class TestDcBlock:
    def test_removes_dc(self):
        dcb = daisysp.utility.DcBlock()
        dcb.init(SR)
        # Signal with DC offset
        inp = np.ones(4096, dtype=np.float32) * 0.5
        inp += np.sin(np.linspace(0, 100, 4096)).astype(np.float32) * 0.3
        out = dcb.process(inp)
        assert out.shape == (4096,)
        # DC should be attenuated in later samples
        assert abs(np.mean(out[-1024:])) < abs(np.mean(inp[-1024:]))


class TestDelayLine:
    def test_delay(self):
        dl = daisysp.utility.DelayLine()
        dl.init()
        dl.set_delay(100.0)
        inp = np.zeros(200, dtype=np.float32)
        inp[0] = 1.0
        out = dl.process(inp, 100.0)
        assert out.shape == (200,)
        # The impulse should appear around sample 100
        peak_idx = np.argmax(np.abs(out))
        assert 99 <= peak_idx <= 101


class TestLooper:
    def test_basic(self):
        lo = daisysp.utility.Looper(48000)
        # Record
        lo.trig_record()
        inp = np.sin(np.linspace(0, 10, 1000)).astype(np.float32)
        out = lo.process(inp)
        assert out.shape == (1000,)


class TestMaytrig:
    def test_probabilistic(self):
        mt = daisysp.utility.Maytrig()
        # With prob=1.0, should always trigger
        results = [mt.process(1.0) for _ in range(100)]
        assert all(r for r in results)


class TestMetro:
    def test_ticks(self):
        me = daisysp.utility.Metro()
        me.init(10.0, SR)  # 10 Hz
        out = me.process(int(SR))  # 1 second
        assert out.shape == (int(SR),)
        # Should have approximately 10 ticks in 1 second
        ticks = np.sum(out > 0.5)
        assert 8 <= ticks <= 12


class TestSampleHold:
    def test_process(self):
        sh = daisysp.utility.SampleHold()
        val = sh.process(True, 0.5)
        assert isinstance(val, float)
        # Hold: should return same value without trigger
        val2 = sh.process(False, 0.9)
        assert val2 == pytest.approx(0.5)


class TestJitter:
    def test_output(self):
        jt = daisysp.utility.Jitter()
        jt.init(SR)
        jt.set_amp(1.0)
        jt.set_cps_min(0.5)
        jt.set_cps_max(4.0)
        out = jt.process(1024)
        assert out.shape == (1024,)


class TestPort:
    def test_glide(self):
        po = daisysp.utility.Port()
        po.init(SR, 0.01)
        # Step from 0 to 1 -- output should smoothly approach 1
        inp = np.ones(4096, dtype=np.float32)
        out = po.process(inp)
        assert out.shape == (4096,)
        # First sample should be less than 1, last should be close to 1
        assert out[0] < 1.0
        assert out[-1] == pytest.approx(1.0, abs=0.1)

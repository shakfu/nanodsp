"""Sound synthesis: oscillators, noise, drums, physical modeling, STK instruments."""

from __future__ import annotations

import numpy as np

from nanodsp.buffer import AudioBuffer
from nanodsp._helpers import (
    _process_per_channel,
    _resolve_waveform,
    _WAVEFORM_MAP,
    _BLOSC_WAVEFORM_MAP,
    _dsy_osc,
    _dsy_noise,
    _dsy_drums,
    _dsy_pm,
    _STK_INSTRUMENTS,
)
from nanodsp._core import stk as _stk
from nanodsp._core import bloscillators as _blosc
from nanodsp._core import fxdsp as _fxdsp


# ---------------------------------------------------------------------------
# DaisySP Oscillators
# ---------------------------------------------------------------------------


def oscillator(
    frames: int,
    freq: float = 440.0,
    amp: float = 1.0,
    waveform: int | str = "sine",
    pw: float = 0.5,
    sample_rate: float = 48000.0,
) -> AudioBuffer:
    """Generate a waveform using DaisySP Oscillator.

    Parameters
    ----------
    waveform : int or str
        Waveform constant or name: "sine", "tri", "saw", "ramp", "square",
        "polyblep_tri", "polyblep_saw", "polyblep_square".
    """
    wf = _resolve_waveform(waveform, _WAVEFORM_MAP)
    osc = _dsy_osc.Oscillator()
    osc.init(sample_rate)
    osc.set_freq(freq)
    osc.set_amp(amp)
    osc.set_waveform(wf)
    osc.set_pw(pw)
    data = osc.process(frames)
    return AudioBuffer(np.asarray(data).reshape(1, -1), sample_rate=sample_rate)


def fm2(
    frames: int,
    freq: float = 440.0,
    ratio: float = 2.0,
    index: float = 1.0,
    sample_rate: float = 48000.0,
) -> AudioBuffer:
    """Generate 2-operator FM synthesis."""
    fm = _dsy_osc.Fm2()
    fm.init(sample_rate)
    fm.set_frequency(freq)
    fm.set_ratio(ratio)
    fm.set_index(index)
    data = fm.process(frames)
    return AudioBuffer(np.asarray(data).reshape(1, -1), sample_rate=sample_rate)


def formant_oscillator(
    frames: int,
    carrier_freq: float = 440.0,
    formant_freq: float = 1000.0,
    phase_shift: float = 0.0,
    sample_rate: float = 48000.0,
) -> AudioBuffer:
    """Generate formant oscillator signal."""
    fo = _dsy_osc.FormantOscillator()
    fo.init(sample_rate)
    fo.set_carrier_freq(carrier_freq)
    fo.set_formant_freq(formant_freq)
    fo.set_phase_shift(phase_shift)
    data = fo.process(frames)
    return AudioBuffer(np.asarray(data).reshape(1, -1), sample_rate=sample_rate)


def bl_oscillator(
    frames: int,
    freq: float = 440.0,
    amp: float = 1.0,
    waveform: int | str = "saw",
    pw: float = 0.5,
    sample_rate: float = 48000.0,
) -> AudioBuffer:
    """Generate a band-limited waveform using DaisySP BlOsc.

    Parameters
    ----------
    waveform : int or str
        Waveform constant or name: "triangle"/"tri", "saw", "square", "off".
    """
    wf = _resolve_waveform(waveform, _BLOSC_WAVEFORM_MAP)
    osc = _dsy_osc.BlOsc()
    osc.init(sample_rate)
    osc.set_freq(freq)
    osc.set_amp(amp)
    osc.set_waveform(wf)
    osc.set_pw(pw)
    data = osc.process(frames)
    return AudioBuffer(np.asarray(data).reshape(1, -1), sample_rate=sample_rate)


# ---------------------------------------------------------------------------
# DaisySP Noise
# ---------------------------------------------------------------------------


def white_noise(
    frames: int,
    amp: float = 1.0,
    sample_rate: float = 48000.0,
) -> AudioBuffer:
    """Generate white noise."""
    wn = _dsy_noise.WhiteNoise()
    wn.init()
    wn.set_amp(amp)
    data = wn.process(frames)
    return AudioBuffer(np.asarray(data).reshape(1, -1), sample_rate=sample_rate)


def clocked_noise(
    frames: int,
    freq: float = 1000.0,
    sample_rate: float = 48000.0,
) -> AudioBuffer:
    """Generate clocked (sample-and-hold) noise."""
    cn = _dsy_noise.ClockedNoise()
    cn.init(sample_rate)
    cn.set_freq(freq)
    data = cn.process(frames)
    return AudioBuffer(np.asarray(data).reshape(1, -1), sample_rate=sample_rate)


def dust(
    frames: int,
    density: float = 1.0,
    sample_rate: float = 48000.0,
) -> AudioBuffer:
    """Generate Dust (random impulses at given density)."""
    d = _dsy_noise.Dust()
    d.init()
    d.set_density(density)
    data = d.process(frames)
    return AudioBuffer(np.asarray(data).reshape(1, -1), sample_rate=sample_rate)


# ---------------------------------------------------------------------------
# DaisySP Drums
# ---------------------------------------------------------------------------


def analog_bass_drum(
    frames: int,
    freq: float = 60.0,
    tone: float = 0.5,
    decay: float = 0.5,
    accent: float = 0.5,
    sustain: bool = False,
    attack_fm: float = 0.5,
    self_fm: float = 0.5,
    sample_rate: float = 48000.0,
) -> AudioBuffer:
    """Generate an analog bass drum hit (triggered at sample 0)."""
    d = _dsy_drums.AnalogBassDrum()
    d.init(sample_rate)
    d.set_freq(freq)
    d.set_tone(tone)
    d.set_decay(decay)
    d.set_accent(accent)
    d.set_sustain(sustain)
    d.set_attack_fm_amount(attack_fm)
    d.set_self_fm_amount(self_fm)
    d.trig()
    data = d.process(frames)
    return AudioBuffer(np.asarray(data).reshape(1, -1), sample_rate=sample_rate)


def analog_snare_drum(
    frames: int,
    freq: float = 200.0,
    tone: float = 0.5,
    decay: float = 0.5,
    snappy: float = 0.5,
    accent: float = 0.5,
    sustain: bool = False,
    sample_rate: float = 48000.0,
) -> AudioBuffer:
    """Generate an analog snare drum hit (triggered at sample 0)."""
    d = _dsy_drums.AnalogSnareDrum()
    d.init(sample_rate)
    d.set_freq(freq)
    d.set_tone(tone)
    d.set_decay(decay)
    d.set_snappy(snappy)
    d.set_accent(accent)
    d.set_sustain(sustain)
    d.trig()
    data = d.process(frames)
    return AudioBuffer(np.asarray(data).reshape(1, -1), sample_rate=sample_rate)


def hihat(
    frames: int,
    freq: float = 3000.0,
    tone: float = 0.5,
    decay: float = 0.3,
    noisiness: float = 0.8,
    accent: float = 0.5,
    sustain: bool = False,
    sample_rate: float = 48000.0,
) -> AudioBuffer:
    """Generate a hi-hat hit (triggered at sample 0)."""
    d = _dsy_drums.HiHat()
    d.init(sample_rate)
    d.set_freq(freq)
    d.set_tone(tone)
    d.set_decay(decay)
    d.set_noisiness(noisiness)
    d.set_accent(accent)
    d.set_sustain(sustain)
    d.trig()
    data = d.process(frames)
    return AudioBuffer(np.asarray(data).reshape(1, -1), sample_rate=sample_rate)


def synthetic_bass_drum(
    frames: int,
    freq: float = 60.0,
    tone: float = 0.5,
    decay: float = 0.5,
    dirtiness: float = 0.3,
    fm_env_amount: float = 0.5,
    fm_env_decay: float = 0.3,
    accent: float = 0.5,
    sustain: bool = False,
    sample_rate: float = 48000.0,
) -> AudioBuffer:
    """Generate a synthetic bass drum hit (triggered at sample 0)."""
    d = _dsy_drums.SyntheticBassDrum()
    d.init(sample_rate)
    d.set_freq(freq)
    d.set_tone(tone)
    d.set_decay(decay)
    d.set_dirtiness(dirtiness)
    d.set_fm_envelope_amount(fm_env_amount)
    d.set_fm_envelope_decay(fm_env_decay)
    d.set_accent(accent)
    d.set_sustain(sustain)
    d.trig()
    data = d.process(frames)
    return AudioBuffer(np.asarray(data).reshape(1, -1), sample_rate=sample_rate)


def synthetic_snare_drum(
    frames: int,
    freq: float = 200.0,
    decay: float = 0.5,
    snappy: float = 0.5,
    fm_amount: float = 0.3,
    accent: float = 0.5,
    sustain: bool = False,
    sample_rate: float = 48000.0,
) -> AudioBuffer:
    """Generate a synthetic snare drum hit (triggered at sample 0)."""
    d = _dsy_drums.SyntheticSnareDrum()
    d.init(sample_rate)
    d.set_freq(freq)
    d.set_decay(decay)
    d.set_snappy(snappy)
    d.set_fm_amount(fm_amount)
    d.set_accent(accent)
    d.set_sustain(sustain)
    d.trig()
    data = d.process(frames)
    return AudioBuffer(np.asarray(data).reshape(1, -1), sample_rate=sample_rate)


# ---------------------------------------------------------------------------
# DaisySP Physical Modeling
# ---------------------------------------------------------------------------


def karplus_strong(
    buf: AudioBuffer,
    freq_hz: float = 440.0,
    brightness: float = 0.5,
    damping: float = 0.5,
    non_linearity: float = 0.0,
) -> AudioBuffer:
    """Karplus-Strong string model (excitation input, per channel)."""

    def _process(x):
        s = _dsy_pm.String()
        s.init(buf.sample_rate)
        s.set_freq(freq_hz)
        s.set_brightness(brightness)
        s.set_damping(damping)
        s.set_non_linearity(non_linearity)
        return s.process(x)

    return _process_per_channel(buf, _process)


def modal_voice(
    frames: int,
    freq: float = 440.0,
    accent: float = 0.5,
    structure: float = 0.5,
    brightness: float = 0.5,
    damping: float = 0.5,
    sustain: bool = False,
    sample_rate: float = 48000.0,
) -> AudioBuffer:
    """Generate a modal voice hit (triggered at sample 0)."""
    mv = _dsy_pm.ModalVoice()
    mv.init(sample_rate)
    mv.set_freq(freq)
    mv.set_accent(accent)
    mv.set_structure(structure)
    mv.set_brightness(brightness)
    mv.set_damping(damping)
    mv.set_sustain(sustain)
    mv.trig()
    data = mv.process(frames)
    return AudioBuffer(np.asarray(data).reshape(1, -1), sample_rate=sample_rate)


def string_voice(
    frames: int,
    freq: float = 440.0,
    accent: float = 0.5,
    structure: float = 0.5,
    brightness: float = 0.5,
    damping: float = 0.5,
    sustain: bool = False,
    sample_rate: float = 48000.0,
) -> AudioBuffer:
    """Generate a string voice hit (triggered at sample 0)."""
    sv = _dsy_pm.StringVoice()
    sv.init(sample_rate)
    sv.set_freq(freq)
    sv.set_accent(accent)
    sv.set_structure(structure)
    sv.set_brightness(brightness)
    sv.set_damping(damping)
    sv.set_sustain(sustain)
    sv.trig()
    data = sv.process(frames)
    return AudioBuffer(np.asarray(data).reshape(1, -1), sample_rate=sample_rate)


def pluck(
    frames: int,
    freq: float = 440.0,
    amp: float = 0.8,
    decay: float = 0.95,
    damp: float = 0.9,
    sample_rate: float = 48000.0,
) -> AudioBuffer:
    """Generate a plucked string sound (triggered at sample 0)."""
    npt = max(256, int(sample_rate / freq) + 1)
    p = _dsy_pm.Pluck(sample_rate, npt)
    p.set_freq(freq)
    p.set_amp(amp)
    p.set_decay(decay)
    p.set_damp(damp)
    data = p.process(frames)
    return AudioBuffer(np.asarray(data).reshape(1, -1), sample_rate=sample_rate)


def drip(
    frames: int,
    dettack: float = 0.01,
    sample_rate: float = 48000.0,
) -> AudioBuffer:
    """Generate a water-drip sound (triggered at sample 0)."""
    d = _dsy_pm.Drip()
    d.init(sample_rate, dettack)
    data = d.process(frames)
    return AudioBuffer(np.asarray(data).reshape(1, -1), sample_rate=sample_rate)


# ---------------------------------------------------------------------------
# STK Synthesis
# ---------------------------------------------------------------------------


def synth_note(
    instrument: str,
    freq: float = 440.0,
    duration: float = 1.0,
    velocity: float = 0.8,
    release: float = 0.1,
    sample_rate: float = 48000.0,
) -> AudioBuffer:
    """Synthesize a single note using an STK physical model.

    Parameters
    ----------
    instrument : str
        Instrument name: 'clarinet', 'flute', 'brass', 'bowed', 'plucked',
        'sitar', 'stifkarp', 'saxofony', 'recorder', 'blowbotl',
        'blowhole', 'whistle'.
    freq : float
        Note frequency in Hz.
    duration : float
        Note duration in seconds (sustain portion before release).
    velocity : float
        Note velocity / amplitude (0.0 to 1.0).
    release : float
        Release time in seconds after note-off.
    sample_rate : float
        Output sample rate.
    """
    key = instrument.lower()
    if key not in _STK_INSTRUMENTS:
        raise ValueError(
            f"Unknown instrument {instrument!r}, valid: {list(_STK_INSTRUMENTS.keys())}"
        )
    cls, ctor_style = _STK_INSTRUMENTS[key]

    # Set STK global sample rate
    _stk.set_sample_rate(sample_rate)

    # Construct instrument
    if ctor_style == "void":
        inst = cls()
    elif ctor_style == "freq_required":
        inst = cls(freq)
    else:
        inst = cls()

    # Note on
    inst.note_on(freq, velocity)

    # Generate sustain portion
    sustain_frames = max(1, int(sample_rate * duration))
    sustain_data = np.asarray(inst.process(sustain_frames), dtype=np.float32)

    # Note off
    inst.note_off(velocity)

    # Generate release portion
    release_frames = max(1, int(sample_rate * release))
    release_data = np.asarray(inst.process(release_frames), dtype=np.float32)

    # Concatenate
    data = np.concatenate([sustain_data, release_data])
    return AudioBuffer(
        data.reshape(1, -1),
        sample_rate=sample_rate,
        label=instrument,
    )


# ---------------------------------------------------------------------------
# Band-Limited Oscillators (PolyBLEP, BLIT, DPW)
# ---------------------------------------------------------------------------

# Map string names to PolyBLEP.Waveform enum values
_POLYBLEP_WAVEFORMS = {
    "sine": _blosc.PolyBLEP.Waveform.SINE,
    "cosine": _blosc.PolyBLEP.Waveform.COSINE,
    "triangle": _blosc.PolyBLEP.Waveform.TRIANGLE,
    "tri": _blosc.PolyBLEP.Waveform.TRIANGLE,
    "square": _blosc.PolyBLEP.Waveform.SQUARE,
    "rectangle": _blosc.PolyBLEP.Waveform.RECTANGLE,
    "rect": _blosc.PolyBLEP.Waveform.RECTANGLE,
    "sawtooth": _blosc.PolyBLEP.Waveform.SAWTOOTH,
    "saw": _blosc.PolyBLEP.Waveform.SAWTOOTH,
    "ramp": _blosc.PolyBLEP.Waveform.RAMP,
    "modified_triangle": _blosc.PolyBLEP.Waveform.MODIFIED_TRIANGLE,
    "modified_square": _blosc.PolyBLEP.Waveform.MODIFIED_SQUARE,
    "half_wave_rectified_sine": _blosc.PolyBLEP.Waveform.HALF_WAVE_RECTIFIED_SINE,
    "full_wave_rectified_sine": _blosc.PolyBLEP.Waveform.FULL_WAVE_RECTIFIED_SINE,
    "triangular_pulse": _blosc.PolyBLEP.Waveform.TRIANGULAR_PULSE,
    "trapezoid_fixed": _blosc.PolyBLEP.Waveform.TRAPEZOID_FIXED,
    "trapezoid_variable": _blosc.PolyBLEP.Waveform.TRAPEZOID_VARIABLE,
}


def polyblep(
    frames: int,
    freq: float = 440.0,
    waveform: str = "sawtooth",
    pulse_width: float = 0.5,
    sample_rate: float = 48000.0,
) -> AudioBuffer:
    """Generate a band-limited waveform using PolyBLEP anti-aliasing.

    Parameters
    ----------
    frames : int
        Number of samples to generate.
    freq : float
        Frequency in Hz.
    waveform : str
        One of: 'sine', 'cosine', 'triangle'/'tri', 'square',
        'rectangle'/'rect', 'sawtooth'/'saw', 'ramp',
        'modified_triangle', 'modified_square',
        'half_wave_rectified_sine', 'full_wave_rectified_sine',
        'triangular_pulse', 'trapezoid_fixed', 'trapezoid_variable'.
    pulse_width : float
        Pulse width for rectangle/variable waveforms (0.0 to 1.0).
    sample_rate : float
        Output sample rate.
    """
    key = waveform.lower()
    if key not in _POLYBLEP_WAVEFORMS:
        raise ValueError(
            f"Unknown waveform {waveform!r}, valid: {list(_POLYBLEP_WAVEFORMS.keys())}"
        )
    osc = _blosc.PolyBLEP(sample_rate, _POLYBLEP_WAVEFORMS[key])
    osc.frequency = freq
    osc.pulse_width = pulse_width
    data = np.asarray(osc.generate(frames))
    return AudioBuffer(data.reshape(1, -1), sample_rate=sample_rate)


def blit_saw(
    frames: int,
    freq: float = 220.0,
    harmonics: int = 0,
    sample_rate: float = 48000.0,
) -> AudioBuffer:
    """Generate a BLIT (Band-Limited Impulse Train) sawtooth.

    Parameters
    ----------
    frames : int
        Number of samples to generate.
    freq : float
        Frequency in Hz.
    harmonics : int
        Number of harmonics (0 = maximum up to Nyquist).
    sample_rate : float
        Output sample rate.
    """
    osc = _blosc.BlitSaw(sample_rate, freq)
    if harmonics > 0:
        osc.set_harmonics(harmonics)
    data = np.asarray(osc.generate(frames))
    return AudioBuffer(data.reshape(1, -1), sample_rate=sample_rate)


def blit_square(
    frames: int,
    freq: float = 220.0,
    harmonics: int = 0,
    sample_rate: float = 48000.0,
) -> AudioBuffer:
    """Generate a BLIT square wave with DC blocker.

    Parameters
    ----------
    frames : int
        Number of samples to generate.
    freq : float
        Frequency in Hz.
    harmonics : int
        Number of harmonics (0 = maximum up to Nyquist).
    sample_rate : float
        Output sample rate.
    """
    osc = _blosc.BlitSquare(sample_rate, freq)
    if harmonics > 0:
        osc.set_harmonics(harmonics)
    data = np.asarray(osc.generate(frames))
    return AudioBuffer(data.reshape(1, -1), sample_rate=sample_rate)


def dpw_saw(
    frames: int,
    freq: float = 440.0,
    sample_rate: float = 48000.0,
) -> AudioBuffer:
    """Generate a DPW (Differentiated Parabolic Wave) sawtooth.

    Parameters
    ----------
    frames : int
        Number of samples to generate.
    freq : float
        Frequency in Hz.
    sample_rate : float
        Output sample rate.
    """
    osc = _blosc.DPWSaw(sample_rate, freq)
    data = np.asarray(osc.generate(frames))
    return AudioBuffer(data.reshape(1, -1), sample_rate=sample_rate)


def dpw_pulse(
    frames: int,
    freq: float = 440.0,
    duty: float = 0.5,
    sample_rate: float = 48000.0,
) -> AudioBuffer:
    """Generate a DPW pulse with variable duty cycle.

    Parameters
    ----------
    frames : int
        Number of samples to generate.
    freq : float
        Frequency in Hz.
    duty : float
        Duty cycle (0.01 to 0.99, default 0.5 for square).
    sample_rate : float
        Output sample rate.
    """
    osc = _blosc.DPWPulse(sample_rate, freq)
    osc.duty = duty
    data = np.asarray(osc.generate(frames))
    return AudioBuffer(data.reshape(1, -1), sample_rate=sample_rate)


# ---------------------------------------------------------------------------
# MinBLEP Oscillator
# ---------------------------------------------------------------------------

_MINBLEP_WAVEFORMS = {
    "saw": _fxdsp.MinBLEP.Waveform.SAW,
    "sawtooth": _fxdsp.MinBLEP.Waveform.SAW,
    "rsaw": _fxdsp.MinBLEP.Waveform.RSAW,
    "square": _fxdsp.MinBLEP.Waveform.SQUARE,
    "triangle": _fxdsp.MinBLEP.Waveform.TRIANGLE,
    "tri": _fxdsp.MinBLEP.Waveform.TRIANGLE,
}


def minblep(
    frames: int,
    freq: float = 440.0,
    waveform: str = "saw",
    pulse_width: float = 0.5,
    sample_rate: float = 48000.0,
) -> AudioBuffer:
    """Generate a band-limited waveform using MinBLEP anti-aliasing.

    Parameters
    ----------
    frames : int
        Number of samples to generate.
    freq : float
        Frequency in Hz.
    waveform : str
        One of: 'saw'/'sawtooth', 'rsaw', 'square', 'triangle'/'tri'.
    pulse_width : float
        Pulse width for square waveform (0.01 to 0.99).
    sample_rate : float
        Output sample rate.
    """
    key = waveform.lower()
    if key not in _MINBLEP_WAVEFORMS:
        raise ValueError(
            f"Unknown waveform {waveform!r}, valid: {list(_MINBLEP_WAVEFORMS.keys())}"
        )
    osc = _fxdsp.MinBLEP(sample_rate, freq)
    osc.waveform = _MINBLEP_WAVEFORMS[key]
    osc.pulse_width = pulse_width
    data = np.asarray(osc.generate(frames))
    return AudioBuffer(data.reshape(1, -1), sample_rate=sample_rate)


def synth_sequence(
    instrument: str,
    notes: list[tuple[float, float, float]],
    sample_rate: float = 48000.0,
    release: float = 0.1,
    velocity: float = 0.8,
) -> AudioBuffer:
    """Synthesize a sequence of notes.

    Parameters
    ----------
    instrument : str
        Instrument name (see :func:`synth_note`).
    notes : list of (freq_hz, start_time_s, duration_s)
        Each tuple is (frequency, start_time, duration) in seconds.
    sample_rate : float
        Output sample rate.
    release : float
        Release time for each note in seconds.
    velocity : float
        Default velocity for all notes.
    """
    if not notes:
        raise ValueError("notes list must not be empty")

    # Find total duration
    total_end = max(start + dur + release for _, start, dur in notes)
    total_frames = int(sample_rate * total_end) + 1

    out = np.zeros(total_frames, dtype=np.float32)

    for freq, start, dur in notes:
        note_buf = synth_note(
            instrument,
            freq=freq,
            duration=dur,
            velocity=velocity,
            release=release,
            sample_rate=sample_rate,
        )
        start_frame = int(sample_rate * start)
        end_frame = min(start_frame + note_buf.frames, total_frames)
        available = end_frame - start_frame
        out[start_frame:end_frame] += note_buf.data[0, :available]

    return AudioBuffer(
        out.reshape(1, -1),
        sample_rate=sample_rate,
        label=instrument,
    )

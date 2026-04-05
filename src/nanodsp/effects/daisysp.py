"""DaisySP effects -- autowah, chorus, flanger, overdrive, etc."""

from __future__ import annotations

import numpy as np

from ..buffer import AudioBuffer
from .._helpers import _process_per_channel, _dsy_fx, _dsy_util


# ---------------------------------------------------------------------------
# DaisySP Effects
# ---------------------------------------------------------------------------


def autowah(
    buf: AudioBuffer,
    wah: float = 0.5,
    dry_wet: float = 1.0,
    level: float = 0.5,
) -> AudioBuffer:
    """Apply auto-wah effect per channel.

    Parameters
    ----------
    buf : AudioBuffer
        Input audio.
    wah : float
        Wah amount, 0.0--1.0.
    dry_wet : float
        Dry/wet mix, 0.0--1.0 (0.0 = dry, 1.0 = wet).
    level : float
        Output level, 0.0--1.0.

    Returns
    -------
    AudioBuffer
        Processed audio.
    """

    def _process(x):
        aw = _dsy_fx.Autowah()
        aw.init(buf.sample_rate)
        aw.set_wah(wah)
        aw.set_dry_wet(dry_wet)
        aw.set_level(level)
        return aw.process(x)

    return _process_per_channel(buf, _process)


def chorus(
    buf: AudioBuffer,
    lfo_freq: float = 0.3,
    lfo_depth: float = 0.5,
    delay_ms: float = 5.0,
    feedback: float = 0.2,
) -> AudioBuffer:
    """Apply chorus effect.

    Mono input produces stereo output via process_stereo.
    Multi-channel input is processed per-channel (mono chorus).

    Parameters
    ----------
    buf : AudioBuffer
        Input audio.
    lfo_freq : float
        LFO rate in Hz, > 0. Typical: 0.1--5.0.
    lfo_depth : float
        LFO modulation depth, 0.0--1.0.
    delay_ms : float
        Base delay time in milliseconds, > 0. Typical: 1--30.
    feedback : float
        Feedback amount, 0.0--1.0.

    Returns
    -------
    AudioBuffer
        Processed audio (stereo if mono input).
    """
    if buf.channels == 1:
        ch = _dsy_fx.Chorus()
        ch.init(buf.sample_rate)
        ch.set_lfo_freq(lfo_freq)
        ch.set_lfo_depth(lfo_depth)
        ch.set_delay_ms(delay_ms)
        ch.set_feedback(feedback)
        stereo = ch.process_stereo(buf.ensure_1d(0))
        return AudioBuffer(
            stereo,
            sample_rate=buf.sample_rate,
            channel_layout="stereo",
            label=buf.label,
        )

    def _process(x):
        ch = _dsy_fx.Chorus()
        ch.init(buf.sample_rate)
        ch.set_lfo_freq(lfo_freq)
        ch.set_lfo_depth(lfo_depth)
        ch.set_delay_ms(delay_ms)
        ch.set_feedback(feedback)
        return ch.process(x)

    return _process_per_channel(buf, _process)


def decimator(
    buf: AudioBuffer,
    downsample_factor: float = 0.5,
    bitcrush_factor: float = 0.5,
    bits_to_crush: int = 8,
    smooth: bool = False,
) -> AudioBuffer:
    """Apply decimator (bitcrushing / downsampling) per channel.

    Parameters
    ----------
    buf : AudioBuffer
        Input audio.
    downsample_factor : float
        Downsampling amount, 0.0--1.0.
    bitcrush_factor : float
        Bit reduction amount, 0.0--1.0.
    bits_to_crush : int
        Number of bits to quantize to, 1--32.
    smooth : bool
        If True, apply smoothing to crushed output.

    Returns
    -------
    AudioBuffer
        Decimated audio.
    """

    def _process(x):
        d = _dsy_fx.Decimator()
        d.init()
        d.set_downsample_factor(downsample_factor)
        d.set_bitcrush_factor(bitcrush_factor)
        d.set_bits_to_crush(bits_to_crush)
        d.set_smooth_crushing(smooth)
        return d.process(x)

    return _process_per_channel(buf, _process)


def flanger(
    buf: AudioBuffer,
    lfo_freq: float = 0.2,
    lfo_depth: float = 0.5,
    feedback: float = 0.3,
    delay_ms: float = 1.0,
) -> AudioBuffer:
    """Apply flanger effect per channel.

    Parameters
    ----------
    buf : AudioBuffer
        Input audio.
    lfo_freq : float
        LFO rate in Hz, > 0. Typical: 0.05--2.0.
    lfo_depth : float
        LFO modulation depth, 0.0--1.0.
    feedback : float
        Feedback amount, 0.0--1.0.
    delay_ms : float
        Base delay time in milliseconds, > 0. Typical: 0.5--10.

    Returns
    -------
    AudioBuffer
        Flanged audio.
    """

    def _process(x):
        f = _dsy_fx.Flanger()
        f.init(buf.sample_rate)
        f.set_lfo_freq(lfo_freq)
        f.set_lfo_depth(lfo_depth)
        f.set_feedback(feedback)
        f.set_delay_ms(delay_ms)
        return f.process(x)

    return _process_per_channel(buf, _process)


def overdrive(buf: AudioBuffer, drive: float = 0.5) -> AudioBuffer:
    """Apply overdrive distortion per channel.

    Parameters
    ----------
    buf : AudioBuffer
        Input audio.
    drive : float
        Drive amount, 0.0--1.0.

    Returns
    -------
    AudioBuffer
        Overdriven audio.
    """

    def _process(x):
        od = _dsy_fx.Overdrive()
        od.init()
        od.set_drive(drive)
        return od.process(x)

    return _process_per_channel(buf, _process)


def phaser(
    buf: AudioBuffer,
    lfo_freq: float = 0.3,
    lfo_depth: float = 0.5,
    freq: float = 1000.0,
    feedback: float = 0.5,
    poles: int = 4,
) -> AudioBuffer:
    """Apply phaser effect per channel.

    Parameters
    ----------
    buf : AudioBuffer
        Input audio.
    lfo_freq : float
        LFO rate in Hz, > 0. Typical: 0.1--2.0.
    lfo_depth : float
        LFO modulation depth, 0.0--1.0.
    freq : float
        Base allpass frequency in Hz, > 0 and < Nyquist.
    feedback : float
        Feedback amount, 0.0--1.0.
    poles : int
        Number of allpass stages, >= 1. Typical: 2--8.

    Returns
    -------
    AudioBuffer
        Phased audio.
    """

    def _process(x):
        p = _dsy_fx.Phaser()
        p.init(buf.sample_rate)
        p.set_lfo_freq(lfo_freq)
        p.set_lfo_depth(lfo_depth)
        p.set_freq(freq)
        p.set_feedback(feedback)
        p.set_poles(poles)
        return p.process(x)

    return _process_per_channel(buf, _process)


def pitch_shift(
    buf: AudioBuffer,
    semitones: float = 0.0,
    del_size: int = 256,
    fun: float = 0.0,
) -> AudioBuffer:
    """Apply pitch shifting per channel.

    Parameters
    ----------
    buf : AudioBuffer
        Input audio.
    semitones : float
        Pitch shift in semitones.
    del_size : int
        Internal delay buffer size.
    fun : float
        Fun parameter (spectral smearing), 0.0--1.0.

    Returns
    -------
    AudioBuffer
        Pitch-shifted audio.
    """

    def _process(x):
        ps = _dsy_fx.PitchShifter()
        ps.init(buf.sample_rate)
        ps.set_transposition(semitones)
        ps.set_del_size(del_size)
        ps.set_fun(fun)
        return ps.process(x)

    return _process_per_channel(buf, _process)


def sample_rate_reduce(buf: AudioBuffer, freq: float = 0.5) -> AudioBuffer:
    """Apply sample-rate reduction per channel.

    Parameters
    ----------
    buf : AudioBuffer
        Input audio.
    freq : float
        Normalized frequency, 0.0--1.0. Lower = more reduction.

    Returns
    -------
    AudioBuffer
        Sample-rate-reduced audio.
    """

    def _process(x):
        srr = _dsy_fx.SampleRateReducer()
        srr.init()
        srr.set_freq(freq)
        return srr.process(x)

    return _process_per_channel(buf, _process)


def tremolo(
    buf: AudioBuffer,
    freq: float = 5.0,
    depth: float = 0.5,
    waveform: int = 0,
) -> AudioBuffer:
    """Apply tremolo effect per channel.

    Parameters
    ----------
    buf : AudioBuffer
        Input audio.
    freq : float
        Tremolo rate in Hz, > 0. Typical: 1--20.
    depth : float
        Modulation depth, 0.0--1.0.
    waveform : int
        LFO waveform index, 0--7.

    Returns
    -------
    AudioBuffer
        Tremolo-modulated audio.
    """

    def _process(x):
        t = _dsy_fx.Tremolo()
        t.init(buf.sample_rate)
        t.set_freq(freq)
        t.set_depth(depth)
        t.set_waveform(waveform)
        return t.process(x)

    return _process_per_channel(buf, _process)


def wavefold(
    buf: AudioBuffer,
    gain: float = 1.0,
    offset: float = 0.0,
) -> AudioBuffer:
    """Apply wavefolding per channel.

    Parameters
    ----------
    buf : AudioBuffer
        Input audio.
    gain : float
        Input gain before folding, > 0. Higher values create more folds.
    offset : float
        DC offset applied before folding. Typical: -1.0 to 1.0.

    Returns
    -------
    AudioBuffer
        Wavefolded audio.
    """

    def _process(x):
        wf = _dsy_fx.Wavefolder()
        wf.init()
        wf.set_gain(gain)
        wf.set_offset(offset)
        return wf.process(x)

    return _process_per_channel(buf, _process)


def bitcrush(
    buf: AudioBuffer,
    bit_depth: int = 8,
    crush_rate: float | None = None,
) -> AudioBuffer:
    """Apply bitcrushing per channel.

    Parameters
    ----------
    buf : AudioBuffer
        Input audio.
    bit_depth : int
        Number of bits for quantization, 1--32. Lower = more distortion.
    crush_rate : float or None
        Sample-and-hold rate in Hz, > 0. Defaults to sample_rate / 4 if None.

    Returns
    -------
    AudioBuffer
        Bitcrushed audio.
    """
    rate = crush_rate if crush_rate is not None else buf.sample_rate / 4.0

    def _process(x):
        bc = _dsy_fx.Bitcrush()
        bc.init(buf.sample_rate)
        bc.set_bit_depth(bit_depth)
        bc.set_crush_rate(rate)
        return bc.process(x)

    return _process_per_channel(buf, _process)


def fold(buf: AudioBuffer, increment: float = 1.0) -> AudioBuffer:
    """Apply fold distortion per channel.

    Parameters
    ----------
    buf : AudioBuffer
        Input audio.
    increment : float
        Fold increment amount, > 0.

    Returns
    -------
    AudioBuffer
        Folded audio.
    """

    def _process(x):
        f = _dsy_fx.Fold()
        f.init()
        f.set_increment(increment)
        return f.process(x)

    return _process_per_channel(buf, _process)


def reverb_sc(
    buf: AudioBuffer,
    feedback: float = 0.7,
    lp_freq: float = 10000.0,
) -> AudioBuffer:
    """Apply ReverbSc stereo reverb.

    Mono input is duplicated to stereo. Stereo input is passed through.
    3+ channels raises ValueError.

    Parameters
    ----------
    buf : AudioBuffer
        Input audio (mono or stereo).
    feedback : float
        Reverb feedback / decay, 0.0--<1.0 (values >= 1.0 are unstable).
    lp_freq : float
        Lowpass filter frequency in Hz for damping, > 0 and < Nyquist.
        Typical: 1000--20000.

    Returns
    -------
    AudioBuffer
        Stereo reverb output.
    """
    if buf.channels > 2:
        raise ValueError(
            f"reverb_sc requires mono or stereo input, got {buf.channels} channels"
        )
    rv = _dsy_fx.ReverbSc()
    rv.init(buf.sample_rate)
    rv.set_feedback(feedback)
    rv.set_lp_freq(lp_freq)
    if buf.channels == 1:
        stereo_in = np.vstack([buf.data[0], buf.data[0]])
    else:
        stereo_in = buf.data
    out = rv.process(stereo_in)
    return AudioBuffer(
        out,
        sample_rate=buf.sample_rate,
        channel_layout="stereo",
        label=buf.label,
    )


def dc_block(buf: AudioBuffer) -> AudioBuffer:
    """Remove DC offset per channel using DaisySP DcBlock.

    Parameters
    ----------
    buf : AudioBuffer
        Input audio.

    Returns
    -------
    AudioBuffer
        Audio with DC offset removed.
    """

    def _process(x):
        dc = _dsy_util.DcBlock()
        dc.init(buf.sample_rate)
        return dc.process(x)

    return _process_per_channel(buf, _process)

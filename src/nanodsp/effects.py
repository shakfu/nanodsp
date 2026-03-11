"""Filters, effects, dynamics, reverbs, and mastering chains."""

from __future__ import annotations

import numpy as np

from nanodsp.buffer import AudioBuffer
from nanodsp._helpers import (
    _hz_to_normalized,
    _process_per_channel,
    _dsy_fx,
    _dsy_filt,
    _dsy_dyn,
    _dsy_util,
    _stk_fx,
    _LADDER_MODE_MAP,
)
from nanodsp._core import filters, madronalib as _madronalib
from nanodsp._core import stk as _stk
from nanodsp._core import vafilters as _va
from nanodsp._core import fxdsp as _fxdsp
from nanodsp._core import iirdesign as _iir


# ---------------------------------------------------------------------------
# Filter functions (signalsmith)
# ---------------------------------------------------------------------------


def lowpass(
    buf: AudioBuffer,
    cutoff_hz: float,
    octaves: float | None = None,
    design=filters.BiquadDesign.bilinear,
) -> AudioBuffer:
    freq = _hz_to_normalized(cutoff_hz, buf.sample_rate)

    def _process(x):
        bq = filters.Biquad()
        if octaves is not None:
            bq.lowpass(freq, octaves, design)
        else:
            bq.lowpass(freq, design=design)
        return bq.process(x)

    return _process_per_channel(buf, _process)


def highpass(
    buf: AudioBuffer,
    cutoff_hz: float,
    octaves: float | None = None,
    design=filters.BiquadDesign.bilinear,
) -> AudioBuffer:
    freq = _hz_to_normalized(cutoff_hz, buf.sample_rate)

    def _process(x):
        bq = filters.Biquad()
        if octaves is not None:
            bq.highpass(freq, octaves, design)
        else:
            bq.highpass(freq, design=design)
        return bq.process(x)

    return _process_per_channel(buf, _process)


def bandpass(
    buf: AudioBuffer,
    center_hz: float,
    octaves: float | None = None,
    design=filters.BiquadDesign.one_sided,
) -> AudioBuffer:
    freq = _hz_to_normalized(center_hz, buf.sample_rate)

    def _process(x):
        bq = filters.Biquad()
        if octaves is not None:
            bq.bandpass(freq, octaves, design)
        else:
            bq.bandpass(freq, design=design)
        return bq.process(x)

    return _process_per_channel(buf, _process)


def notch(
    buf: AudioBuffer,
    center_hz: float,
    octaves: float | None = None,
    design=filters.BiquadDesign.one_sided,
) -> AudioBuffer:
    freq = _hz_to_normalized(center_hz, buf.sample_rate)

    def _process(x):
        bq = filters.Biquad()
        if octaves is not None:
            bq.notch(freq, octaves, design)
        else:
            bq.notch(freq, design=design)
        return bq.process(x)

    return _process_per_channel(buf, _process)


def peak(
    buf: AudioBuffer,
    center_hz: float,
    gain: float,
    octaves: float = 1.0,
    design=filters.BiquadDesign.one_sided,
) -> AudioBuffer:
    freq = _hz_to_normalized(center_hz, buf.sample_rate)

    def _process(x):
        bq = filters.Biquad()
        bq.peak(freq, gain, octaves, design)
        return bq.process(x)

    return _process_per_channel(buf, _process)


def peak_db(
    buf: AudioBuffer,
    center_hz: float,
    db: float,
    octaves: float = 1.0,
    design=filters.BiquadDesign.one_sided,
) -> AudioBuffer:
    freq = _hz_to_normalized(center_hz, buf.sample_rate)

    def _process(x):
        bq = filters.Biquad()
        bq.peak_db(freq, db, octaves, design)
        return bq.process(x)

    return _process_per_channel(buf, _process)


def high_shelf(
    buf: AudioBuffer,
    cutoff_hz: float,
    gain: float,
    octaves: float | None = None,
    design=filters.BiquadDesign.one_sided,
) -> AudioBuffer:
    freq = _hz_to_normalized(cutoff_hz, buf.sample_rate)

    def _process(x):
        bq = filters.Biquad()
        if octaves is not None:
            bq.high_shelf(freq, gain, octaves, design)
        else:
            bq.high_shelf(freq, gain, design=design)
        return bq.process(x)

    return _process_per_channel(buf, _process)


def high_shelf_db(
    buf: AudioBuffer,
    cutoff_hz: float,
    db: float,
    octaves: float | None = None,
    design=filters.BiquadDesign.one_sided,
) -> AudioBuffer:
    freq = _hz_to_normalized(cutoff_hz, buf.sample_rate)

    def _process(x):
        bq = filters.Biquad()
        if octaves is not None:
            bq.high_shelf_db(freq, db, octaves, design)
        else:
            bq.high_shelf_db(freq, db, design=design)
        return bq.process(x)

    return _process_per_channel(buf, _process)


def low_shelf(
    buf: AudioBuffer,
    cutoff_hz: float,
    gain: float,
    octaves: float = 2.0,
    design=filters.BiquadDesign.one_sided,
) -> AudioBuffer:
    freq = _hz_to_normalized(cutoff_hz, buf.sample_rate)

    def _process(x):
        bq = filters.Biquad()
        bq.low_shelf(freq, gain, octaves, design)
        return bq.process(x)

    return _process_per_channel(buf, _process)


def low_shelf_db(
    buf: AudioBuffer,
    cutoff_hz: float,
    db: float,
    octaves: float = 2.0,
    design=filters.BiquadDesign.one_sided,
) -> AudioBuffer:
    freq = _hz_to_normalized(cutoff_hz, buf.sample_rate)

    def _process(x):
        bq = filters.Biquad()
        bq.low_shelf_db(freq, db, octaves, design)
        return bq.process(x)

    return _process_per_channel(buf, _process)


def allpass(
    buf: AudioBuffer,
    freq_hz: float,
    octaves: float = 1.0,
    design=filters.BiquadDesign.one_sided,
) -> AudioBuffer:
    freq = _hz_to_normalized(freq_hz, buf.sample_rate)

    def _process(x):
        bq = filters.Biquad()
        bq.allpass(freq, octaves, design)
        return bq.process(x)

    return _process_per_channel(buf, _process)


def biquad_process(buf: AudioBuffer, biquad) -> AudioBuffer:
    """Process buffer through a pre-configured Biquad, resetting between channels."""

    def _process(x):
        biquad.reset()
        return biquad.process(x)

    return _process_per_channel(buf, _process)


# ---------------------------------------------------------------------------
# DaisySP Effects
# ---------------------------------------------------------------------------


def autowah(
    buf: AudioBuffer,
    wah: float = 0.5,
    dry_wet: float = 1.0,
    level: float = 0.5,
) -> AudioBuffer:
    """Apply auto-wah effect per channel."""

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
    """Apply decimator (bitcrushing / downsampling) per channel."""

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
    """Apply flanger effect per channel."""

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
    """Apply overdrive distortion per channel."""

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
    """Apply phaser effect per channel."""

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
    """Apply pitch shifting per channel."""

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
    freq : float
        Normalized frequency 0-1 controlling the reduction amount.
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
    """Apply tremolo effect per channel."""

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
    """Apply wavefolding per channel."""

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
    crush_rate : float or None
        Sample-and-hold rate. Defaults to sample_rate / 4 if None.
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
    """Apply fold distortion per channel."""

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
    """Remove DC offset per channel using DaisySP DcBlock."""

    def _process(x):
        dc = _dsy_util.DcBlock()
        dc.init(buf.sample_rate)
        return dc.process(x)

    return _process_per_channel(buf, _process)


# ---------------------------------------------------------------------------
# DaisySP Filters
# ---------------------------------------------------------------------------


def _make_svf(buf, freq_hz, resonance, drive, process_method):
    """Internal helper for SVF filter variants."""

    def _process(x):
        svf = _dsy_filt.Svf()
        svf.init(buf.sample_rate)
        svf.set_freq(freq_hz)
        svf.set_res(resonance)
        svf.set_drive(drive)
        return getattr(svf, process_method)(x)

    return _process_per_channel(buf, _process)


def svf_lowpass(
    buf: AudioBuffer,
    freq_hz: float = 1000.0,
    resonance: float = 0.0,
    drive: float = 0.0,
) -> AudioBuffer:
    """State-variable filter lowpass."""
    return _make_svf(buf, freq_hz, resonance, drive, "process_low")


def svf_highpass(
    buf: AudioBuffer,
    freq_hz: float = 1000.0,
    resonance: float = 0.0,
    drive: float = 0.0,
) -> AudioBuffer:
    """State-variable filter highpass."""
    return _make_svf(buf, freq_hz, resonance, drive, "process_high")


def svf_bandpass(
    buf: AudioBuffer,
    freq_hz: float = 1000.0,
    resonance: float = 0.0,
    drive: float = 0.0,
) -> AudioBuffer:
    """State-variable filter bandpass."""
    return _make_svf(buf, freq_hz, resonance, drive, "process_band")


def svf_notch(
    buf: AudioBuffer,
    freq_hz: float = 1000.0,
    resonance: float = 0.0,
    drive: float = 0.0,
) -> AudioBuffer:
    """State-variable filter notch."""
    return _make_svf(buf, freq_hz, resonance, drive, "process_notch")


def svf_peak(
    buf: AudioBuffer,
    freq_hz: float = 1000.0,
    resonance: float = 0.0,
    drive: float = 0.0,
) -> AudioBuffer:
    """State-variable filter peak."""
    return _make_svf(buf, freq_hz, resonance, drive, "process_peak")


def ladder_filter(
    buf: AudioBuffer,
    freq_hz: float = 1000.0,
    resonance: float = 0.0,
    mode: str = "lp24",
    drive: float = 0.0,
) -> AudioBuffer:
    """Ladder filter with selectable mode.

    Parameters
    ----------
    mode : str
        One of "lp24", "lp12", "bp24", "bp12", "hp24", "hp12".
    """
    mode_key = mode.lower()
    if mode_key not in _LADDER_MODE_MAP:
        raise ValueError(
            f"Unknown ladder mode {mode!r}, valid: {list(_LADDER_MODE_MAP.keys())}"
        )
    mode_val = _LADDER_MODE_MAP[mode_key]

    def _process(x):
        lf = _dsy_filt.LadderFilter()
        lf.init(buf.sample_rate)
        lf.set_freq(freq_hz)
        lf.set_res(resonance)
        lf.set_filter_mode(mode_val)
        lf.set_input_drive(drive)
        return lf.process(x)

    return _process_per_channel(buf, _process)


def moog_ladder(
    buf: AudioBuffer,
    freq_hz: float = 1000.0,
    resonance: float = 0.0,
) -> AudioBuffer:
    """Moog-style ladder lowpass filter."""

    def _process(x):
        ml = _dsy_filt.MoogLadder()
        ml.init(buf.sample_rate)
        ml.set_freq(freq_hz)
        ml.set_res(resonance)
        return ml.process(x)

    return _process_per_channel(buf, _process)


def tone_lowpass(buf: AudioBuffer, freq_hz: float = 1000.0) -> AudioBuffer:
    """One-pole lowpass filter (Tone)."""

    def _process(x):
        t = _dsy_filt.Tone()
        t.init(buf.sample_rate)
        t.set_freq(freq_hz)
        return t.process(x)

    return _process_per_channel(buf, _process)


def tone_highpass(buf: AudioBuffer, freq_hz: float = 1000.0) -> AudioBuffer:
    """One-pole highpass filter (ATone)."""

    def _process(x):
        at = _dsy_filt.ATone()
        at.init(buf.sample_rate)
        at.set_freq(freq_hz)
        return at.process(x)

    return _process_per_channel(buf, _process)


def modal_bandpass(
    buf: AudioBuffer,
    freq_hz: float = 1000.0,
    q: float = 500.0,
) -> AudioBuffer:
    """Modal resonator bandpass filter."""

    def _process(x):
        m = _dsy_filt.Mode()
        m.init(buf.sample_rate)
        m.set_freq(freq_hz)
        m.set_q(q)
        return m.process(x)

    return _process_per_channel(buf, _process)


def comb_filter(
    buf: AudioBuffer,
    freq_hz: float = 500.0,
    rev_time: float = 0.5,
    max_size: int = 4096,
) -> AudioBuffer:
    """Comb filter."""

    def _process(x):
        c = _dsy_filt.Comb(buf.sample_rate, max_size)
        c.set_freq(freq_hz)
        c.set_rev_time(rev_time)
        return c.process(x)

    return _process_per_channel(buf, _process)


# ---------------------------------------------------------------------------
# DaisySP Dynamics
# ---------------------------------------------------------------------------


def compress(
    buf: AudioBuffer,
    ratio: float = 4.0,
    threshold: float = -20.0,
    attack: float = 0.01,
    release: float = 0.1,
    makeup: float = 0.0,
    auto_makeup: bool = False,
) -> AudioBuffer:
    """Apply compression per channel."""

    def _process(x):
        c = _dsy_dyn.Compressor()
        c.init(buf.sample_rate)
        c.set_ratio(ratio)
        c.set_threshold(threshold)
        c.set_attack(attack)
        c.set_release(release)
        c.set_makeup(makeup)
        c.auto_makeup(auto_makeup)
        return c.process(x)

    return _process_per_channel(buf, _process)


def limit(buf: AudioBuffer, pre_gain: float = 1.0) -> AudioBuffer:
    """Apply limiter per channel."""

    def _process(x):
        lm = _dsy_dyn.Limiter()
        lm.init()
        return lm.process(x, pre_gain)

    return _process_per_channel(buf, _process)


# ---------------------------------------------------------------------------
# Saturation
# ---------------------------------------------------------------------------


def saturate(
    buf: AudioBuffer,
    drive: float = 0.5,
    mode: str = "soft",
) -> AudioBuffer:
    """Apply saturation/distortion.

    Modes:
    - ``'soft'``: tanh soft clipping, normalized to preserve peak.
    - ``'hard'``: hard clipping to [-1, 1].
    - ``'tape'``: asymmetric soft clip ``x - x^3/3``.

    *drive*: 0.0 to 1.0 controls intensity (maps to gain 1x-10x).
    """
    drive_scaled = np.float32(1.0 + drive * 9.0)
    data = buf.data * drive_scaled

    if mode == "soft":
        out = np.tanh(data)
        # Normalize to preserve original peak
        peak_in = np.max(np.abs(buf.data))
        peak_out = np.max(np.abs(out))
        if peak_out > 0 and peak_in > 0:
            out *= np.float32(peak_in / peak_out)
    elif mode == "hard":
        out = np.clip(data, -1.0, 1.0)
    elif mode == "tape":
        # Asymmetric soft clip: x - x^3/3 for |x| < 1, clamped otherwise
        out = np.where(
            np.abs(data) < 1.0,
            data - (data**3) / 3.0,
            np.sign(data) * 2.0 / 3.0,
        )
        # Normalize to preserve original peak
        peak_in = np.max(np.abs(buf.data))
        peak_out = np.max(np.abs(out))
        if peak_out > 0 and peak_in > 0:
            out *= np.float32(peak_in / peak_out)
    else:
        raise ValueError(
            f"Unknown saturation mode {mode!r}, valid: 'soft', 'hard', 'tape'"
        )
    return AudioBuffer(
        out.astype(np.float32),
        sample_rate=buf.sample_rate,
        channel_layout=buf.channel_layout,
        label=buf.label,
    )


# ---------------------------------------------------------------------------
# Composed effects
# ---------------------------------------------------------------------------


def exciter(
    buf: AudioBuffer,
    freq: float = 3000.0,
    amount: float = 0.3,
) -> AudioBuffer:
    """Add harmonics above *freq* via saturation.

    Highpass-filters, saturates to generate harmonics, highpasses again
    to clean up, and blends back into the original.
    """
    highs = highpass(buf, freq)
    saturated = saturate(highs, drive=0.7, mode="soft")
    harmonics = highpass(saturated, freq)
    # Blend: output = original + amount * harmonics
    out = buf.data + np.float32(amount) * harmonics.data
    return AudioBuffer(
        out.astype(np.float32),
        sample_rate=buf.sample_rate,
        channel_layout=buf.channel_layout,
        label=buf.label,
    )


def de_esser(
    buf: AudioBuffer,
    freq: float = 6000.0,
    threshold_db: float = -20.0,
    ratio: float = 4.0,
    bandwidth: float = 2.0,
) -> AudioBuffer:
    """Reduce sibilance around *freq* Hz.

    Extracts the sibilant band, compresses it, and replaces the
    original band with the compressed version.
    """
    bp = bandpass(buf, freq, octaves=bandwidth)
    compressed_bp = compress(
        bp, ratio=ratio, threshold=threshold_db, attack=0.001, release=0.05
    )
    # Replace original band with compressed version
    out = buf.data - bp.data + compressed_bp.data
    return AudioBuffer(
        out.astype(np.float32),
        sample_rate=buf.sample_rate,
        channel_layout=buf.channel_layout,
        label=buf.label,
    )


def parallel_compress(
    buf: AudioBuffer,
    mix: float = 0.5,
    ratio: float = 8.0,
    threshold_db: float = -30.0,
    attack: float = 0.001,
    release: float = 0.05,
) -> AudioBuffer:
    """Blend heavily compressed signal with dry signal (New York compression)."""
    compressed = compress(
        buf, ratio=ratio, threshold=threshold_db, attack=attack, release=release
    )
    out = (1.0 - mix) * buf.data + mix * compressed.data
    return AudioBuffer(
        out.astype(np.float32),
        sample_rate=buf.sample_rate,
        channel_layout=buf.channel_layout,
        label=buf.label,
    )


# ---------------------------------------------------------------------------
# FDN Reverb
# ---------------------------------------------------------------------------

_REVERB_PRESETS: dict[str, dict] = {
    "room": {
        "delays": [197, 251, 337, 433, 521, 617, 743, 859],
        "base_cutoff": 0.35,
    },
    "hall": {
        "delays": [487, 631, 809, 997, 1151, 1327, 1493, 1657],
        "base_cutoff": 0.25,
    },
    "plate": {
        "delays": [149, 211, 307, 401, 491, 587, 677, 769],
        "base_cutoff": 0.45,
    },
    "chamber": {
        "delays": [317, 409, 523, 641, 751, 877, 1009, 1129],
        "base_cutoff": 0.30,
    },
    "cathedral": {
        "delays": [1013, 1259, 1493, 1741, 1997, 2243, 2503, 2749],
        "base_cutoff": 0.15,
    },
}


def reverb(
    buf: AudioBuffer,
    preset: str = "hall",
    mix: float = 0.3,
    decay: float = 0.8,
    damping: float = 0.5,
    pre_delay_ms: float = 0.0,
) -> AudioBuffer:
    """FDN reverb with presets.

    Parameters
    ----------
    preset : str
        One of 'room', 'hall', 'plate', 'chamber', 'cathedral'.
    mix : float
        Wet/dry blend (0.0 = fully dry, 1.0 = fully wet).
    decay : float
        Feedback gain per delay line (0.0 to <1.0).
    damping : float
        Controls lowpass filtering in feedback (0.0 = bright, 1.0 = dark).
    pre_delay_ms : float
        Pre-delay in milliseconds before reverb onset.
    """
    if preset not in _REVERB_PRESETS:
        raise ValueError(
            f"Unknown reverb preset {preset!r}, valid: {list(_REVERB_PRESETS.keys())}"
        )
    cfg = _REVERB_PRESETS[preset]
    sr = buf.sample_rate

    # Scale delay times for sample rate
    sr_scale = sr / 48000.0
    delay_times = [float(d * sr_scale) for d in cfg["delays"]]

    # Mono-sum input for FDN processing
    if buf.channels > 1:
        mono_data = np.mean(buf.data, axis=0).astype(np.float32)
    else:
        mono_data = buf.data[0].copy()

    # Pre-delay: prepend silence
    if pre_delay_ms > 0:
        pre_samples = int(sr * pre_delay_ms / 1000.0)
        mono_data = np.concatenate(
            [
                np.zeros(pre_samples, dtype=np.float32),
                mono_data,
            ]
        )

    # Pad to multiple of 64 for madronalib DSPVector processing
    remainder = len(mono_data) % 64
    if remainder != 0:
        pad_len = 64 - remainder
        mono_data = np.pad(mono_data, (0, pad_len), mode="constant")

    mono_data = np.ascontiguousarray(mono_data, dtype=np.float32)

    # Create and configure FDN8
    fdn = _madronalib.reverbs.FDN8()
    fdn.set_delays_in_samples(delay_times)
    cutoff = cfg["base_cutoff"] * (1.0 - damping * 0.8)
    fdn.set_filter_cutoffs([cutoff] * 8)
    fdn.set_feedback_gains([decay] * 8)

    # Process: FDN8 returns [2, N] stereo
    wet_stereo = np.asarray(fdn.process(mono_data), dtype=np.float32)

    # Trim back to original length (remove padding and pre-delay extension)
    target_frames = buf.frames
    wet_stereo = wet_stereo[:, :target_frames]
    # If wet is shorter than target (shouldn't happen, but guard)
    if wet_stereo.shape[1] < target_frames:
        wet_stereo = np.pad(
            wet_stereo,
            ((0, 0), (0, target_frames - wet_stereo.shape[1])),
            mode="constant",
        )

    # Prepare dry stereo
    if buf.channels == 1:
        dry_stereo = np.tile(buf.data, (2, 1))
    elif buf.channels == 2:
        dry_stereo = buf.data
    else:
        # Multi-channel: downmix to stereo for blending
        dry_stereo = np.zeros((2, buf.frames), dtype=np.float32)
        dry_stereo[0] = np.mean(buf.data[: buf.channels // 2], axis=0)
        dry_stereo[1] = np.mean(buf.data[buf.channels // 2 :], axis=0)

    # Wet/dry blend
    out = (1.0 - mix) * dry_stereo + mix * wet_stereo
    return AudioBuffer(
        out.astype(np.float32),
        sample_rate=buf.sample_rate,
        channel_layout="stereo",
        label=buf.label,
    )


# ---------------------------------------------------------------------------
# Mastering chain
# ---------------------------------------------------------------------------


def master(
    buf: AudioBuffer,
    target_lufs: float = -14.0,
    eq: dict | None = None,
    compress_on: bool = True,
    limit_on: bool = True,
    dc_block_on: bool = True,
) -> AudioBuffer:
    """Simple mastering chain.

    Chain order: dc_block -> EQ -> compress -> limit -> normalize_lufs.

    Parameters
    ----------
    eq : dict or None
        Optional EQ with keys:
        - ``'low_shelf'``: ``(freq_hz, gain_db)`` or ``(freq_hz, gain_db, octaves)``
        - ``'high_shelf'``: ``(freq_hz, gain_db)`` or ``(freq_hz, gain_db, octaves)``
        - ``'peak'``: single ``(freq_hz, gain_db)`` or ``(freq_hz, gain_db, octaves)``,
          or a list of such tuples for multi-band.
    compress_on : bool
        Enable compression stage.
    limit_on : bool
        Enable limiting stage.
    dc_block_on : bool
        Enable DC blocking stage.
    """
    from nanodsp.analysis import normalize_lufs

    result = buf

    # DC block
    if dc_block_on:
        result = dc_block(result)

    # EQ
    if eq is not None:
        if "low_shelf" in eq:
            params = eq["low_shelf"]
            if len(params) == 2:
                result = low_shelf_db(result, params[0], params[1])
            else:
                result = low_shelf_db(result, params[0], params[1], octaves=params[2])
        if "high_shelf" in eq:
            params = eq["high_shelf"]
            if len(params) == 2:
                result = high_shelf_db(result, params[0], params[1])
            else:
                result = high_shelf_db(result, params[0], params[1], octaves=params[2])
        if "peak" in eq:
            peak_params = eq["peak"]
            # Single band or list of bands
            if isinstance(peak_params[0], (list, tuple)):
                bands = peak_params
            else:
                bands = [peak_params]
            for p in bands:
                if len(p) == 2:
                    result = peak_db(result, p[0], p[1])
                else:
                    result = peak_db(result, p[0], p[1], octaves=p[2])

    # Compress
    if compress_on:
        result = compress(result, ratio=3.0, threshold=-18.0, attack=0.01, release=0.1)

    # Limit
    if limit_on:
        result = limit(result, pre_gain=1.0)

    # Normalize
    result = normalize_lufs(result, target_lufs=target_lufs)

    return result


# ---------------------------------------------------------------------------
# Noise gate
# ---------------------------------------------------------------------------


def noise_gate(
    buf: AudioBuffer,
    threshold_db: float = -40.0,
    attack: float = 0.001,
    release: float = 0.05,
    hold_ms: float = 10.0,
) -> AudioBuffer:
    """Gate signal below *threshold_db*, silencing quiet passages.

    Parameters
    ----------
    threshold_db : float
        Gate threshold in dB. Signal below this is attenuated.
    attack : float
        Gate open time in seconds.
    release : float
        Gate close time in seconds.
    hold_ms : float
        Hold time in milliseconds after signal drops below threshold
        before the gate starts closing.
    """
    sr = buf.sample_rate
    threshold_lin = 10.0 ** (threshold_db / 20.0)
    attack_samples = max(1, int(sr * attack))
    release_samples = max(1, int(sr * release))
    hold_samples = max(0, int(sr * hold_ms / 1000.0))

    # Compute envelope across all channels (max abs at each frame)
    envelope = np.max(np.abs(buf.data), axis=0)

    # Build gain curve: 1.0 when open, 0.0 when closed
    gain = np.zeros(buf.frames, dtype=np.float32)
    gate_open = False
    hold_counter = 0

    for i in range(buf.frames):
        if envelope[i] >= threshold_lin:
            gate_open = True
            hold_counter = hold_samples
        elif hold_counter > 0:
            hold_counter -= 1
        else:
            gate_open = False

        gain[i] = 1.0 if gate_open else 0.0

    # Smooth the gain curve with attack/release
    smoothed = np.zeros_like(gain)
    current = 0.0
    for i in range(buf.frames):
        target = gain[i]
        if target > current:
            # Opening: attack
            coeff = 1.0 / attack_samples
            current = min(current + coeff, target)
        else:
            # Closing: release
            coeff = 1.0 / release_samples
            current = max(current - coeff, target)
        smoothed[i] = current

    out = buf.data * smoothed[np.newaxis, :]
    return AudioBuffer(
        out.astype(np.float32),
        sample_rate=buf.sample_rate,
        channel_layout=buf.channel_layout,
        label=buf.label,
    )


# ---------------------------------------------------------------------------
# Stereo delay
# ---------------------------------------------------------------------------


def stereo_delay(
    buf: AudioBuffer,
    left_ms: float = 250.0,
    right_ms: float = 375.0,
    feedback: float = 0.3,
    mix: float = 0.5,
    ping_pong: bool = False,
) -> AudioBuffer:
    """Stereo delay effect.

    Parameters
    ----------
    left_ms, right_ms : float
        Delay times for left and right channels in milliseconds.
    feedback : float
        Feedback amount (0.0 to <1.0).
    mix : float
        Wet/dry blend (0.0 = dry, 1.0 = fully wet).
    ping_pong : bool
        If True, feedback crosses between L/R channels.
    """
    sr = buf.sample_rate
    left_samples = int(sr * left_ms / 1000.0)
    right_samples = int(sr * right_ms / 1000.0)
    max_delay = max(left_samples, right_samples) + 1

    # Ensure stereo input
    if buf.channels == 1:
        dry = np.tile(buf.data, (2, 1))
    elif buf.channels == 2:
        dry = buf.data.copy()
    else:
        raise ValueError(
            f"stereo_delay requires mono or stereo input, got {buf.channels} channels"
        )

    n_frames = buf.frames
    wet = np.zeros((2, n_frames), dtype=np.float32)

    # Simple delay line buffers
    buf_l = np.zeros(max_delay, dtype=np.float32)
    buf_r = np.zeros(max_delay, dtype=np.float32)
    write_pos = 0

    for i in range(n_frames):
        # Read from delay lines
        read_l = (write_pos - left_samples) % max_delay
        read_r = (write_pos - right_samples) % max_delay
        delayed_l = buf_l[read_l]
        delayed_r = buf_r[read_r]

        wet[0, i] = delayed_l
        wet[1, i] = delayed_r

        # Write to delay lines with feedback
        if ping_pong:
            # Cross-feed: left delay gets right feedback, right gets left
            buf_l[write_pos] = dry[0, i] + feedback * delayed_r
            buf_r[write_pos] = dry[1, i] + feedback * delayed_l
        else:
            buf_l[write_pos] = dry[0, i] + feedback * delayed_l
            buf_r[write_pos] = dry[1, i] + feedback * delayed_r

        write_pos = (write_pos + 1) % max_delay

    out = (1.0 - mix) * dry + mix * wet
    return AudioBuffer(
        out.astype(np.float32),
        sample_rate=buf.sample_rate,
        channel_layout="stereo",
        label=buf.label,
    )


# ---------------------------------------------------------------------------
# Multiband compression
# ---------------------------------------------------------------------------


def multiband_compress(
    buf: AudioBuffer,
    crossover_freqs: list[float] | None = None,
    ratios: list[float] | None = None,
    thresholds: list[float] | None = None,
    attack: float = 0.01,
    release: float = 0.1,
) -> AudioBuffer:
    """Split into frequency bands, compress each independently, and recombine.

    Parameters
    ----------
    crossover_freqs : list[float] or None
        Crossover frequencies in Hz. Defaults to [200, 2000, 8000] (4 bands).
    ratios : list[float] or None
        Compression ratio per band (len = len(crossover_freqs) + 1).
        Defaults to [2.0, 3.0, 3.0, 2.0].
    thresholds : list[float] or None
        Threshold in dB per band. Defaults to [-24, -20, -20, -18].
    attack, release : float
        Attack/release times in seconds, shared across all bands.
    """
    if crossover_freqs is None:
        crossover_freqs = [200.0, 2000.0, 8000.0]
    n_bands = len(crossover_freqs) + 1
    if ratios is None:
        ratios = [2.0] + [3.0] * (n_bands - 2) + [2.0] if n_bands > 2 else [2.0, 2.0]
    if thresholds is None:
        thresholds = (
            [-24.0] + [-20.0] * (n_bands - 2) + [-18.0]
            if n_bands > 2
            else [-24.0, -18.0]
        )
    if len(ratios) != n_bands:
        raise ValueError(
            f"ratios length ({len(ratios)}) must be len(crossover_freqs) + 1 ({n_bands})"
        )
    if len(thresholds) != n_bands:
        raise ValueError(
            f"thresholds length ({len(thresholds)}) must be len(crossover_freqs) + 1 ({n_bands})"
        )

    # Sort crossover freqs
    freqs = sorted(crossover_freqs)

    # Split into bands using Linkwitz-Riley (cascaded biquad LP/HP)
    bands = []
    remainder = buf

    for i, freq in enumerate(freqs):
        # Extract low portion
        band_low = lowpass(remainder, freq)
        bands.append(band_low)
        # Remainder is the high portion
        remainder = highpass(remainder, freq)

    # Last band is whatever remains above the highest crossover
    bands.append(remainder)

    # Compress each band
    compressed_bands = []
    for i, band in enumerate(bands):
        compressed = compress(
            band,
            ratio=ratios[i],
            threshold=thresholds[i],
            attack=attack,
            release=release,
        )
        compressed_bands.append(compressed)

    # Recombine
    out = np.zeros_like(buf.data)
    for band in compressed_bands:
        out += band.data

    return AudioBuffer(
        out.astype(np.float32),
        sample_rate=buf.sample_rate,
        channel_layout=buf.channel_layout,
        label=buf.label,
    )


# ---------------------------------------------------------------------------
# Vocal chain
# ---------------------------------------------------------------------------


def vocal_chain(
    buf: AudioBuffer,
    de_ess: bool = True,
    de_ess_freq: float = 6000.0,
    eq: dict | None = None,
    compress_on: bool = True,
    limit_on: bool = True,
    target_lufs: float | None = None,
) -> AudioBuffer:
    """Vocal processing chain: de-esser -> EQ -> compress -> limit -> normalize.

    Parameters
    ----------
    de_ess : bool
        Enable de-essing stage.
    de_ess_freq : float
        De-esser center frequency in Hz.
    eq : dict or None
        EQ settings (same format as :func:`master`). Defaults to a gentle
        vocal-friendly EQ: highpass at 80 Hz, +2 dB presence at 3 kHz,
        +1 dB air shelf at 12 kHz.
    compress_on : bool
        Enable compression (ratio=4, threshold=-24dB, moderate attack/release).
    limit_on : bool
        Enable limiter.
    target_lufs : float or None
        If set, normalize to this loudness. Requires signal >= 400ms.
    """
    from nanodsp.analysis import normalize_lufs

    result = buf

    # De-ess
    if de_ess:
        result = de_esser(result, freq=de_ess_freq, threshold_db=-20.0, ratio=4.0)

    # Highpass to remove rumble
    result = highpass(result, 80.0)

    # EQ
    if eq is not None:
        # Use the same EQ dict parsing as master()
        if "low_shelf" in eq:
            params = eq["low_shelf"]
            if len(params) == 2:
                result = low_shelf_db(result, params[0], params[1])
            else:
                result = low_shelf_db(result, params[0], params[1], octaves=params[2])
        if "high_shelf" in eq:
            params = eq["high_shelf"]
            if len(params) == 2:
                result = high_shelf_db(result, params[0], params[1])
            else:
                result = high_shelf_db(result, params[0], params[1], octaves=params[2])
        if "peak" in eq:
            peak_params = eq["peak"]
            if isinstance(peak_params[0], (list, tuple)):
                peaks = peak_params
            else:
                peaks = [peak_params]
            for p in peaks:
                if len(p) == 2:
                    result = peak_db(result, p[0], p[1])
                else:
                    result = peak_db(result, p[0], p[1], octaves=p[2])
    else:
        # Default vocal EQ: presence boost + air
        result = peak_db(result, 3000.0, 2.0, octaves=1.5)
        result = high_shelf_db(result, 12000.0, 1.0)

    # Compress
    if compress_on:
        result = compress(
            result, ratio=4.0, threshold=-24.0, attack=0.005, release=0.08
        )

    # Limit
    if limit_on:
        result = limit(result, pre_gain=1.0)

    # Normalize
    if target_lufs is not None:
        result = normalize_lufs(result, target_lufs=target_lufs)

    return result


# ---------------------------------------------------------------------------
# STK Effects
# ---------------------------------------------------------------------------


def stk_reverb(
    buf: AudioBuffer,
    algorithm: str = "freeverb",
    mix: float = 0.3,
    room_size: float = 0.5,
    damping: float = 0.5,
    t60: float = 1.0,
) -> AudioBuffer:
    """Apply an STK reverb algorithm.

    Parameters
    ----------
    algorithm : str
        One of 'freeverb', 'jcrev', 'nrev', 'prcrev'.
    mix : float
        Wet/dry mix (0.0 = dry, 1.0 = fully wet).
    room_size : float
        Room size (FreeVerb only, 0.0-1.0).
    damping : float
        Damping (FreeVerb only, 0.0-1.0).
    t60 : float
        Reverberation time in seconds (JCRev, NRev, PRCRev).
    """
    _stk.set_sample_rate(buf.sample_rate)

    algo = algorithm.lower()
    rv: _stk_fx.FreeVerb | _stk_fx.JCRev | _stk_fx.NRev | _stk_fx.PRCRev
    if algo == "freeverb":
        rv = _stk_fx.FreeVerb()
        rv.set_room_size(room_size)
        rv.set_damping(damping)
        rv.set_effect_mix(mix)
    elif algo == "jcrev":
        rv = _stk_fx.JCRev(t60)
        rv.set_effect_mix(mix)
    elif algo == "nrev":
        rv = _stk_fx.NRev(t60)
        rv.set_effect_mix(mix)
    elif algo == "prcrev":
        rv = _stk_fx.PRCRev(t60)
        rv.set_effect_mix(mix)
    else:
        raise ValueError(
            f"Unknown STK reverb algorithm {algorithm!r}, "
            "valid: 'freeverb', 'jcrev', 'nrev', 'prcrev'"
        )

    # Process mono input (sum to mono if stereo)
    if buf.channels > 1:
        mono = np.mean(buf.data, axis=0).astype(np.float32)
    else:
        mono = buf.data[0].copy()
    mono = np.ascontiguousarray(mono, dtype=np.float32)

    if algo == "freeverb":
        # FreeVerb process takes [2, N] and returns [2, N]
        stereo_in = np.stack([mono, mono])
        out = np.asarray(rv.process(stereo_in), dtype=np.float32)
    else:
        # JCRev, NRev, PRCRev take mono, return [2, N]
        out = np.asarray(rv.process(mono), dtype=np.float32)

    if out.ndim == 1:
        out = np.stack([out, out])

    return AudioBuffer(
        out,
        sample_rate=buf.sample_rate,
        channel_layout="stereo",
        label=buf.label,
    )


def stk_chorus(
    buf: AudioBuffer,
    mod_depth: float = 0.05,
    mod_freq: float = 0.25,
    mix: float = 0.5,
) -> AudioBuffer:
    """Apply STK Chorus effect.

    Returns stereo output from mono or stereo input.
    """
    _stk.set_sample_rate(buf.sample_rate)

    ch = _stk_fx.Chorus()
    ch.set_mod_depth(mod_depth)
    ch.set_mod_frequency(mod_freq)
    ch.set_effect_mix(mix)

    if buf.channels > 1:
        mono = np.mean(buf.data, axis=0).astype(np.float32)
    else:
        mono = buf.data[0].copy()
    mono = np.ascontiguousarray(mono, dtype=np.float32)

    # STK Chorus.process returns [2, N]
    out = np.asarray(ch.process(mono), dtype=np.float32)
    if out.ndim == 1:
        out = np.stack([out, out])

    return AudioBuffer(
        out,
        sample_rate=buf.sample_rate,
        channel_layout="stereo",
        label=buf.label,
    )


def stk_echo(
    buf: AudioBuffer,
    delay_ms: float = 250.0,
    mix: float = 0.5,
) -> AudioBuffer:
    """Apply STK Echo effect per channel."""
    _stk.set_sample_rate(buf.sample_rate)
    delay_samples = int(buf.sample_rate * delay_ms / 1000.0)

    def _process(x):
        e = _stk_fx.Echo(delay_samples + 1)
        e.set_delay(delay_samples)
        e.set_effect_mix(mix)
        return e.process(np.ascontiguousarray(x, dtype=np.float32))

    return _process_per_channel(buf, _process)


# ---------------------------------------------------------------------------
# Automatic Gain Control
# ---------------------------------------------------------------------------


def agc(
    buf: AudioBuffer,
    target_level: float = 1.0,
    max_gain_db: float = 60.0,
    average_len: int = 100,
    attack: float = 0.01,
    release: float = 0.01,
) -> AudioBuffer:
    """Automatic Gain Control.

    Parameters
    ----------
    target_level : float
        Desired RMS output level (linear).
    max_gain_db : float
        Maximum gain in dB to prevent boosting silence to infinity.
    average_len : int
        Number of samples for the moving-average power estimator.
    attack : float
        Attack time constant in seconds (fast gain reduction).
    release : float
        Release time constant in seconds (slow gain increase).
    """
    sr = buf.sample_rate
    max_gain_lin = 10.0 ** (max_gain_db / 20.0)
    attack_coeff = 1.0 - np.exp(-1.0 / (sr * attack)) if attack > 0 else 1.0
    release_coeff = 1.0 - np.exp(-1.0 / (sr * release)) if release > 0 else 1.0

    def _process(x):
        n = len(x)
        x64 = x.astype(np.float64)
        out = np.empty(n, dtype=np.float64)
        eps = 1e-10

        # Moving-average power estimate
        power_est = 0.0
        current_gain = 1.0

        for i in range(n):
            # Update running power estimate (exponential moving average)
            power_est += (x64[i] ** 2 - power_est) / average_len

            # Desired gain from power estimate
            rms = np.sqrt(max(power_est, eps))
            desired_gain = min(target_level / rms, max_gain_lin)

            # Asymmetric smoothing
            if desired_gain < current_gain:
                current_gain += attack_coeff * (desired_gain - current_gain)
            else:
                current_gain += release_coeff * (desired_gain - current_gain)

            out[i] = x64[i] * current_gain

        return out.astype(np.float32)

    return _process_per_channel(buf, _process)


# ---------------------------------------------------------------------------
# Virtual analog filters (Faust-generated, MIT-style STK-4.3 license)
# ---------------------------------------------------------------------------

_VA_FILTER_CLASSES = {
    "moog_ladder": _va.MoogLadder,
    "moog_half_ladder": _va.MoogHalfLadder,
    "diode_ladder": _va.DiodeLadder,
    "korg35_lpf": _va.Korg35LPF,
    "korg35_hpf": _va.Korg35HPF,
}


def _va_filter(buf: AudioBuffer, cls, cutoff_hz: float, q: float) -> AudioBuffer:
    def _process(x):
        f = cls()
        f.init(float(buf.sample_rate))
        f.cutoff = cutoff_hz
        f.q = q
        return f.process(x)

    return _process_per_channel(buf, _process)


def va_moog_ladder(
    buf: AudioBuffer, cutoff_hz: float = 1000.0, q: float = 1.0
) -> AudioBuffer:
    """Moog Ladder 24 dB/oct lowpass filter (virtual analog)."""
    return _va_filter(buf, _va.MoogLadder, cutoff_hz, q)


def va_moog_half_ladder(
    buf: AudioBuffer, cutoff_hz: float = 1000.0, q: float = 1.0
) -> AudioBuffer:
    """Moog Half-Ladder 12 dB/oct lowpass filter (virtual analog)."""
    return _va_filter(buf, _va.MoogHalfLadder, cutoff_hz, q)


def va_diode_ladder(
    buf: AudioBuffer, cutoff_hz: float = 1000.0, q: float = 1.0
) -> AudioBuffer:
    """Diode Ladder 24 dB/oct lowpass filter (virtual analog)."""
    return _va_filter(buf, _va.DiodeLadder, cutoff_hz, q)


def va_korg35_lpf(
    buf: AudioBuffer, cutoff_hz: float = 1000.0, q: float = 1.0
) -> AudioBuffer:
    """Korg 35 24 dB/oct lowpass filter (virtual analog)."""
    return _va_filter(buf, _va.Korg35LPF, cutoff_hz, q)


def va_korg35_hpf(
    buf: AudioBuffer, cutoff_hz: float = 1000.0, q: float = 1.0
) -> AudioBuffer:
    """Korg 35 24 dB/oct highpass filter (virtual analog)."""
    return _va_filter(buf, _va.Korg35HPF, cutoff_hz, q)


def va_oberheim(
    buf: AudioBuffer,
    cutoff_hz: float = 1000.0,
    q: float = 1.0,
    mode: str = "lpf",
) -> AudioBuffer:
    """Oberheim multi-mode state-variable filter (virtual analog).

    Args:
        mode: One of 'lpf', 'hpf', 'bpf', 'bsf' (notch).
    """
    mode_map = {"lpf": 0, "hpf": 1, "bpf": 2, "bsf": 3}
    mode_int = mode_map.get(mode)
    if mode_int is None:
        raise ValueError(f"Unknown mode '{mode}', expected one of {list(mode_map)}")

    def _process(x):
        f = _va.OberheimSVF()
        f.init(float(buf.sample_rate))
        f.cutoff = cutoff_hz
        f.q = q
        return f.process(x, mode_int)

    return _process_per_channel(buf, _process)


# ---------------------------------------------------------------------------
# Antialiased Waveshaping
# ---------------------------------------------------------------------------


def aa_hard_clip(buf: AudioBuffer, drive: float = 1.0) -> AudioBuffer:
    """Antialiased hard clipper using 1st-order antiderivative method.

    Parameters
    ----------
    drive : float
        Input gain multiplier before clipping.
    """

    def _process(x):
        if drive != 1.0:
            x = x * drive
        c = _fxdsp.HardClipper()
        return c.process(x)

    return _process_per_channel(buf, _process)


def aa_soft_clip(buf: AudioBuffer, drive: float = 1.0) -> AudioBuffer:
    """Antialiased soft clipper (sin-based saturation) with 1st-order AA.

    Parameters
    ----------
    drive : float
        Input gain multiplier before saturation.
    """

    def _process(x):
        if drive != 1.0:
            x = x * drive
        c = _fxdsp.SoftClipper()
        return c.process(x)

    return _process_per_channel(buf, _process)


def aa_wavefold(buf: AudioBuffer, drive: float = 1.0) -> AudioBuffer:
    """Antialiased wavefolder (Buchla 259 style) with 2nd-order AA.

    Parameters
    ----------
    drive : float
        Input gain multiplier before folding.
    """

    def _process(x):
        if drive != 1.0:
            x = x * drive
        c = _fxdsp.Wavefolder()
        return c.process(x)

    return _process_per_channel(buf, _process)


# ---------------------------------------------------------------------------
# Classic Reverbs (Schroeder, Moorer)
# ---------------------------------------------------------------------------


def schroeder_reverb(
    buf: AudioBuffer,
    feedback: float = 0.7,
    diffusion: float = 0.5,
    mod_depth: float = 0.0,
) -> AudioBuffer:
    """Schroeder reverberator (4 parallel combs + 2 series allpasses)."""

    def _process(x):
        rev = _fxdsp.SchroederReverb()
        rev.init(float(buf.sample_rate))
        rev.feedback = feedback
        rev.diffusion = diffusion
        rev.set_mod_depth(mod_depth)
        return rev.process(x)

    return _process_per_channel(buf, _process)


def moorer_reverb(
    buf: AudioBuffer,
    feedback: float = 0.7,
    diffusion: float = 0.7,
    mod_depth: float = 0.1,
) -> AudioBuffer:
    """Moorer reverberator (early reflections + 4 combs + 2 allpasses)."""

    def _process(x):
        rev = _fxdsp.MoorerReverb()
        rev.init(float(buf.sample_rate))
        rev.feedback = feedback
        rev.diffusion = diffusion
        rev.set_mod_depth(mod_depth)
        return rev.process(x)

    return _process_per_channel(buf, _process)


# ---------------------------------------------------------------------------
# Formant Filter
# ---------------------------------------------------------------------------

_VOWEL_MAP = {"a": 0, "e": 1, "i": 2, "o": 3, "u": 4}


def formant_filter(
    buf: AudioBuffer,
    vowel: int | str = "a",
) -> AudioBuffer:
    """Apply vowel formant filter using cascaded bandpass biquads.

    Parameters
    ----------
    vowel : int or str
        Vowel index (0-4) or name ('a', 'e', 'i', 'o', 'u').
    """
    if isinstance(vowel, str):
        v = _VOWEL_MAP.get(vowel.lower())
        if v is None:
            raise ValueError(
                f"Unknown vowel '{vowel}', expected one of {list(_VOWEL_MAP)}"
            )
    else:
        v = int(vowel)

    def _process(x):
        ff = _fxdsp.FormantFilter()
        ff.init(float(buf.sample_rate))
        ff.vowel = v
        return ff.process(x)

    return _process_per_channel(buf, _process)


# ---------------------------------------------------------------------------
# PSOLA Pitch Shifting
# ---------------------------------------------------------------------------


def psola_pitch_shift(
    buf: AudioBuffer,
    semitones: float = 0.0,
) -> AudioBuffer:
    """Pitch shift using PSOLA (Pitch-Synchronous Overlap-Add).

    Parameters
    ----------
    semitones : float
        Pitch shift in semitones (positive = up, negative = down).
    """

    def _process(x):
        return _fxdsp.psola_pitch_shift(x, float(buf.sample_rate), semitones)

    return _process_per_channel(buf, _process)


# ---------------------------------------------------------------------------
# IIR filter design (DspFilters -- Butterworth, Chebyshev, Elliptic, Bessel)
# ---------------------------------------------------------------------------

_IIR_FAMILIES = {
    "butterworth": 0,
    "butter": 0,
    "chebyshev1": 1,
    "cheby1": 1,
    "chebyshev2": 2,
    "cheby2": 2,
    "elliptic": 3,
    "ellip": 3,
    "bessel": 4,
}

_IIR_TYPES = {
    "lowpass": 0,
    "lp": 0,
    "highpass": 1,
    "hp": 1,
    "bandpass": 2,
    "bp": 2,
    "bandstop": 3,
    "bs": 3,
    "notch": 3,
}


def iir_design(
    family: str,
    filter_type: str,
    order: int,
    sample_rate: float,
    freq: float,
    width: float = 0.0,
    ripple_db: float = 0.0,
    rolloff: float = 0.0,
) -> np.ndarray:
    """Design an IIR filter and return SOS coefficients.

    Returns an array of shape ``[n_sections, 6]`` where each row is
    ``[b0, b1, b2, a0, a1, a2]`` with ``a0 = 1.0``.

    Parameters
    ----------
    family : str
        Filter family: butterworth/butter, chebyshev1/cheby1,
        chebyshev2/cheby2, elliptic/ellip, bessel.
    filter_type : str
        Filter type: lowpass/lp, highpass/hp, bandpass/bp, bandstop/bs/notch.
    order : int
        Filter order (1-16).
    sample_rate : float
        Sample rate in Hz.
    freq : float
        Cutoff (LP/HP) or center (BP/BS) frequency in Hz.
    width : float
        Bandwidth in Hz (required for bandpass/bandstop).
    ripple_db : float
        Passband ripple for Chebyshev I / Elliptic (dB).
        Stopband attenuation for Chebyshev II (dB).
    rolloff : float
        Transition width for Elliptic filters (range approx -16 to 4).
    """
    fam = _IIR_FAMILIES.get(family.lower())
    if fam is None:
        raise ValueError(
            f"Unknown family {family!r}, valid: {list(_IIR_FAMILIES.keys())}"
        )
    typ = _IIR_TYPES.get(filter_type.lower())
    if typ is None:
        raise ValueError(
            f"Unknown type {filter_type!r}, valid: {list(_IIR_TYPES.keys())}"
        )
    return np.asarray(
        _iir.design(fam, typ, order, sample_rate, freq, width, ripple_db, rolloff)
    )


def iir_filter(
    buf: AudioBuffer,
    family: str = "butterworth",
    filter_type: str = "lowpass",
    order: int = 4,
    freq: float = 1000.0,
    width: float = 0.0,
    ripple_db: float = 0.0,
    rolloff: float = 0.0,
) -> AudioBuffer:
    """Apply a multi-order IIR filter.

    Supports Butterworth, Chebyshev I/II, Elliptic, and Bessel filter
    families with orders up to 16, in lowpass, highpass, bandpass, and
    bandstop configurations.

    Parameters
    ----------
    family : str
        butterworth/butter, chebyshev1/cheby1, chebyshev2/cheby2,
        elliptic/ellip, bessel.
    filter_type : str
        lowpass/lp, highpass/hp, bandpass/bp, bandstop/bs/notch.
    order : int
        Filter order (1-16).
    freq : float
        Cutoff or center frequency in Hz.
    width : float
        Bandwidth in Hz (required for bandpass/bandstop).
    ripple_db : float
        Passband ripple (Chebyshev I, Elliptic) or stopband attenuation
        (Chebyshev II) in dB.
    rolloff : float
        Transition width for Elliptic filters.
    """
    fam = _IIR_FAMILIES.get(family.lower())
    if fam is None:
        raise ValueError(
            f"Unknown family {family!r}, valid: {list(_IIR_FAMILIES.keys())}"
        )
    typ = _IIR_TYPES.get(filter_type.lower())
    if typ is None:
        raise ValueError(
            f"Unknown type {filter_type!r}, valid: {list(_IIR_TYPES.keys())}"
        )

    def _process(x: np.ndarray) -> np.ndarray:
        return np.asarray(
            _iir.apply(
                x,
                fam,
                typ,
                order,
                float(buf.sample_rate),
                freq,
                width,
                ripple_db,
                rolloff,
            )
        )

    return _process_per_channel(buf, _process)

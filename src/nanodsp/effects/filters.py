"""Filter functions -- signalsmith biquads, DaisySP, virtual analog, IIR."""

from __future__ import annotations

import numpy as np

from ..buffer import AudioBuffer
from .._helpers import (
    _hz_to_normalized,
    _validate_freq_hz,
    _resolve_biquad_design,
    _process_per_channel,
    _dsy_filt,
    _LADDER_MODE_MAP,
)
from .._core import filters
from .._core import vafilters as _va
from .._core import iirdesign as _iir


# ---------------------------------------------------------------------------
# Filter functions (signalsmith)
# ---------------------------------------------------------------------------


def lowpass(
    buf: AudioBuffer,
    cutoff_hz: float,
    octaves: float | None = None,
    design: str | int = "bilinear",
) -> AudioBuffer:
    """Apply a biquad lowpass filter.

    Parameters
    ----------
    buf : AudioBuffer
        Input audio.
    cutoff_hz : float
        Cutoff frequency in Hz.
    octaves : float or None
        Bandwidth in octaves. If None, uses default Q.
    design : str or int
        Biquad design method (e.g. "bilinear", "one_sided").

    Returns
    -------
    AudioBuffer
        Filtered audio.
    """
    freq = _hz_to_normalized(cutoff_hz, buf.sample_rate)
    design = _resolve_biquad_design(design)

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
    design: str | int = "bilinear",
) -> AudioBuffer:
    """Apply a biquad highpass filter.

    Parameters
    ----------
    buf : AudioBuffer
        Input audio.
    cutoff_hz : float
        Cutoff frequency in Hz.
    octaves : float or None
        Bandwidth in octaves. If None, uses default Q.
    design : str or int
        Biquad design method (e.g. "bilinear", "one_sided").

    Returns
    -------
    AudioBuffer
        Filtered audio.
    """
    freq = _hz_to_normalized(cutoff_hz, buf.sample_rate)
    design = _resolve_biquad_design(design)

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
    design: str | int = "one_sided",
) -> AudioBuffer:
    """Apply a biquad bandpass filter.

    Parameters
    ----------
    buf : AudioBuffer
        Input audio.
    center_hz : float
        Center frequency in Hz.
    octaves : float or None
        Bandwidth in octaves. If None, uses default Q.
    design : str or int
        Biquad design method (e.g. "bilinear", "one_sided").

    Returns
    -------
    AudioBuffer
        Filtered audio.
    """
    freq = _hz_to_normalized(center_hz, buf.sample_rate)
    design = _resolve_biquad_design(design)

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
    design: str | int = "one_sided",
) -> AudioBuffer:
    """Apply a biquad notch (band-reject) filter.

    Parameters
    ----------
    buf : AudioBuffer
        Input audio.
    center_hz : float
        Center frequency in Hz.
    octaves : float or None
        Bandwidth in octaves. If None, uses default Q.
    design : str or int
        Biquad design method (e.g. "bilinear", "one_sided").

    Returns
    -------
    AudioBuffer
        Filtered audio.
    """
    freq = _hz_to_normalized(center_hz, buf.sample_rate)
    design = _resolve_biquad_design(design)

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
    design: str | int = "one_sided",
) -> AudioBuffer:
    """Apply a biquad peak (bell) filter with linear gain.

    Parameters
    ----------
    buf : AudioBuffer
        Input audio.
    center_hz : float
        Center frequency in Hz.
    gain : float
        Linear gain at center frequency.
    octaves : float
        Bandwidth in octaves.
    design : str or int
        Biquad design method.

    Returns
    -------
    AudioBuffer
        Filtered audio.
    """
    freq = _hz_to_normalized(center_hz, buf.sample_rate)
    design = _resolve_biquad_design(design)

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
    design: str | int = "one_sided",
) -> AudioBuffer:
    """Apply a biquad peak (bell) filter with gain in dB.

    Parameters
    ----------
    buf : AudioBuffer
        Input audio.
    center_hz : float
        Center frequency in Hz.
    db : float
        Gain at center frequency in dB.
    octaves : float
        Bandwidth in octaves.
    design : str or int
        Biquad design method.

    Returns
    -------
    AudioBuffer
        Filtered audio.
    """
    freq = _hz_to_normalized(center_hz, buf.sample_rate)
    design = _resolve_biquad_design(design)

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
    design: str | int = "one_sided",
) -> AudioBuffer:
    """Apply a biquad high shelf filter with linear gain.

    Parameters
    ----------
    buf : AudioBuffer
        Input audio.
    cutoff_hz : float
        Shelf corner frequency in Hz.
    gain : float
        Linear gain above the shelf frequency.
    octaves : float or None
        Transition bandwidth in octaves. If None, uses default.
    design : str or int
        Biquad design method.

    Returns
    -------
    AudioBuffer
        Filtered audio.
    """
    freq = _hz_to_normalized(cutoff_hz, buf.sample_rate)
    design = _resolve_biquad_design(design)

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
    design: str | int = "one_sided",
) -> AudioBuffer:
    """Apply a biquad high shelf filter with gain in dB.

    Parameters
    ----------
    buf : AudioBuffer
        Input audio.
    cutoff_hz : float
        Shelf corner frequency in Hz.
    db : float
        Gain above the shelf frequency in dB.
    octaves : float or None
        Transition bandwidth in octaves. If None, uses default.
    design : str or int
        Biquad design method.

    Returns
    -------
    AudioBuffer
        Filtered audio.
    """
    freq = _hz_to_normalized(cutoff_hz, buf.sample_rate)
    design = _resolve_biquad_design(design)

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
    design: str | int = "one_sided",
) -> AudioBuffer:
    """Apply a biquad low shelf filter with linear gain.

    Parameters
    ----------
    buf : AudioBuffer
        Input audio.
    cutoff_hz : float
        Shelf corner frequency in Hz.
    gain : float
        Linear gain below the shelf frequency.
    octaves : float
        Transition bandwidth in octaves.
    design : str or int
        Biquad design method.

    Returns
    -------
    AudioBuffer
        Filtered audio.
    """
    freq = _hz_to_normalized(cutoff_hz, buf.sample_rate)
    design = _resolve_biquad_design(design)

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
    design: str | int = "one_sided",
) -> AudioBuffer:
    """Apply a biquad low shelf filter with gain in dB.

    Parameters
    ----------
    buf : AudioBuffer
        Input audio.
    cutoff_hz : float
        Shelf corner frequency in Hz.
    db : float
        Gain below the shelf frequency in dB.
    octaves : float
        Transition bandwidth in octaves.
    design : str or int
        Biquad design method.

    Returns
    -------
    AudioBuffer
        Filtered audio.
    """
    freq = _hz_to_normalized(cutoff_hz, buf.sample_rate)
    design = _resolve_biquad_design(design)

    def _process(x):
        bq = filters.Biquad()
        bq.low_shelf_db(freq, db, octaves, design)
        return bq.process(x)

    return _process_per_channel(buf, _process)


def allpass(
    buf: AudioBuffer,
    freq_hz: float,
    octaves: float = 1.0,
    design: str | int = "one_sided",
) -> AudioBuffer:
    """Apply a biquad allpass filter.

    Parameters
    ----------
    buf : AudioBuffer
        Input audio.
    freq_hz : float
        Center frequency in Hz.
    octaves : float
        Bandwidth in octaves.
    design : str or int
        Biquad design method.

    Returns
    -------
    AudioBuffer
        Filtered audio (same magnitude, altered phase).
    """
    freq = _hz_to_normalized(freq_hz, buf.sample_rate)
    design = _resolve_biquad_design(design)

    def _process(x):
        bq = filters.Biquad()
        bq.allpass(freq, octaves, design)
        return bq.process(x)

    return _process_per_channel(buf, _process)


def biquad_process(buf: AudioBuffer, biquad) -> AudioBuffer:
    """Process buffer through a pre-configured Biquad, resetting between channels.

    Parameters
    ----------
    buf : AudioBuffer
        Input audio.
    biquad : filters.Biquad
        Pre-configured Biquad instance.

    Returns
    -------
    AudioBuffer
        Filtered audio.
    """

    def _process(x):
        biquad.reset()
        return biquad.process(x)

    return _process_per_channel(buf, _process)


# ---------------------------------------------------------------------------
# DaisySP Filters
# ---------------------------------------------------------------------------


def _make_svf(buf, freq_hz, resonance, drive, process_method):
    """Internal helper for SVF filter variants."""
    _validate_freq_hz(freq_hz, buf.sample_rate)

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
    """State-variable filter lowpass.

    Parameters
    ----------
    buf : AudioBuffer
        Input audio.
    freq_hz : float
        Cutoff frequency in Hz.
    resonance : float
        Resonance amount (0.0 to 1.0).
    drive : float
        Input drive amount.

    Returns
    -------
    AudioBuffer
        Filtered audio.
    """
    return _make_svf(buf, freq_hz, resonance, drive, "process_low")


def svf_highpass(
    buf: AudioBuffer,
    freq_hz: float = 1000.0,
    resonance: float = 0.0,
    drive: float = 0.0,
) -> AudioBuffer:
    """State-variable filter highpass.

    Parameters
    ----------
    buf : AudioBuffer
        Input audio.
    freq_hz : float
        Cutoff frequency in Hz.
    resonance : float
        Resonance amount (0.0 to 1.0).
    drive : float
        Input drive amount.

    Returns
    -------
    AudioBuffer
        Filtered audio.
    """
    return _make_svf(buf, freq_hz, resonance, drive, "process_high")


def svf_bandpass(
    buf: AudioBuffer,
    freq_hz: float = 1000.0,
    resonance: float = 0.0,
    drive: float = 0.0,
) -> AudioBuffer:
    """State-variable filter bandpass.

    Parameters
    ----------
    buf : AudioBuffer
        Input audio.
    freq_hz : float
        Center frequency in Hz.
    resonance : float
        Resonance amount (0.0 to 1.0).
    drive : float
        Input drive amount.

    Returns
    -------
    AudioBuffer
        Filtered audio.
    """
    return _make_svf(buf, freq_hz, resonance, drive, "process_band")


def svf_notch(
    buf: AudioBuffer,
    freq_hz: float = 1000.0,
    resonance: float = 0.0,
    drive: float = 0.0,
) -> AudioBuffer:
    """State-variable filter notch.

    Parameters
    ----------
    buf : AudioBuffer
        Input audio.
    freq_hz : float
        Center frequency in Hz.
    resonance : float
        Resonance amount (0.0 to 1.0).
    drive : float
        Input drive amount.

    Returns
    -------
    AudioBuffer
        Filtered audio.
    """
    return _make_svf(buf, freq_hz, resonance, drive, "process_notch")


def svf_peak(
    buf: AudioBuffer,
    freq_hz: float = 1000.0,
    resonance: float = 0.0,
    drive: float = 0.0,
) -> AudioBuffer:
    """State-variable filter peak.

    Parameters
    ----------
    buf : AudioBuffer
        Input audio.
    freq_hz : float
        Center frequency in Hz.
    resonance : float
        Resonance amount (0.0 to 1.0).
    drive : float
        Input drive amount.

    Returns
    -------
    AudioBuffer
        Filtered audio.
    """
    return _make_svf(buf, freq_hz, resonance, drive, "process_peak")


def ladder_filter(
    buf: AudioBuffer,
    freq_hz: float = 1000.0,
    resonance: float = 0.0,
    mode: str = "lp24",
    drive: float = 1.0,
) -> AudioBuffer:
    """Ladder filter with selectable mode.

    Parameters
    ----------
    mode : str
        One of "lp24", "lp12", "bp24", "bp12", "hp24", "hp12".
    drive : float
        Input drive (multiplier). 1.0 = unity gain (no drive), >1.0 adds saturation.
    """
    mode_key = mode.lower()
    if mode_key not in _LADDER_MODE_MAP:
        raise ValueError(
            f"Unknown ladder mode {mode!r}, valid: {list(_LADDER_MODE_MAP.keys())}"
        )
    mode_val = _LADDER_MODE_MAP[mode_key]
    _validate_freq_hz(freq_hz, buf.sample_rate)

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
    """Moog-style ladder lowpass filter.

    Parameters
    ----------
    buf : AudioBuffer
        Input audio.
    freq_hz : float
        Cutoff frequency in Hz.
    resonance : float
        Resonance amount (0.0 to 1.0).

    Returns
    -------
    AudioBuffer
        Filtered audio.
    """
    _validate_freq_hz(freq_hz, buf.sample_rate)

    def _process(x):
        ml = _dsy_filt.MoogLadder()
        ml.init(buf.sample_rate)
        ml.set_freq(freq_hz)
        ml.set_res(resonance)
        return ml.process(x)

    return _process_per_channel(buf, _process)


def tone_lowpass(buf: AudioBuffer, freq_hz: float = 1000.0) -> AudioBuffer:
    """One-pole lowpass filter (Tone).

    Parameters
    ----------
    buf : AudioBuffer
        Input audio.
    freq_hz : float
        Cutoff frequency in Hz.

    Returns
    -------
    AudioBuffer
        Filtered audio.
    """
    _validate_freq_hz(freq_hz, buf.sample_rate)

    def _process(x):
        t = _dsy_filt.Tone()
        t.init(buf.sample_rate)
        t.set_freq(freq_hz)
        return t.process(x)

    return _process_per_channel(buf, _process)


def tone_highpass(buf: AudioBuffer, freq_hz: float = 1000.0) -> AudioBuffer:
    """One-pole highpass filter (ATone).

    Parameters
    ----------
    buf : AudioBuffer
        Input audio.
    freq_hz : float
        Cutoff frequency in Hz.

    Returns
    -------
    AudioBuffer
        Filtered audio.
    """
    _validate_freq_hz(freq_hz, buf.sample_rate)

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
    """Modal resonator bandpass filter.

    Parameters
    ----------
    buf : AudioBuffer
        Input audio.
    freq_hz : float
        Resonant frequency in Hz.
    q : float
        Quality factor.

    Returns
    -------
    AudioBuffer
        Filtered audio.
    """
    _validate_freq_hz(freq_hz, buf.sample_rate)

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
    """Comb filter.

    Parameters
    ----------
    buf : AudioBuffer
        Input audio.
    freq_hz : float
        Comb frequency in Hz.
    rev_time : float
        Reverb time (feedback decay time in seconds).
    max_size : int
        Maximum delay line size in samples.

    Returns
    -------
    AudioBuffer
        Filtered audio.
    """
    _validate_freq_hz(freq_hz, buf.sample_rate)

    def _process(x):
        c = _dsy_filt.Comb(buf.sample_rate, max_size)
        c.set_freq(freq_hz)
        c.set_rev_time(rev_time)
        return c.process(x)

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
    _validate_freq_hz(cutoff_hz, buf.sample_rate)

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
    """Moog Ladder 24 dB/oct lowpass filter (virtual analog).

    Parameters
    ----------
    buf : AudioBuffer
        Input audio.
    cutoff_hz : float
        Cutoff frequency in Hz.
    q : float
        Resonance / Q factor.

    Returns
    -------
    AudioBuffer
        Filtered audio.
    """
    return _va_filter(buf, _va.MoogLadder, cutoff_hz, q)


def va_moog_half_ladder(
    buf: AudioBuffer, cutoff_hz: float = 1000.0, q: float = 1.0
) -> AudioBuffer:
    """Moog Half-Ladder 12 dB/oct lowpass filter (virtual analog).

    Parameters
    ----------
    buf : AudioBuffer
        Input audio.
    cutoff_hz : float
        Cutoff frequency in Hz.
    q : float
        Resonance / Q factor.

    Returns
    -------
    AudioBuffer
        Filtered audio.
    """
    return _va_filter(buf, _va.MoogHalfLadder, cutoff_hz, q)


def va_diode_ladder(
    buf: AudioBuffer, cutoff_hz: float = 1000.0, q: float = 1.0
) -> AudioBuffer:
    """Diode Ladder 24 dB/oct lowpass filter (virtual analog).

    Parameters
    ----------
    buf : AudioBuffer
        Input audio.
    cutoff_hz : float
        Cutoff frequency in Hz.
    q : float
        Resonance / Q factor.

    Returns
    -------
    AudioBuffer
        Filtered audio.
    """
    return _va_filter(buf, _va.DiodeLadder, cutoff_hz, q)


def va_korg35_lpf(
    buf: AudioBuffer, cutoff_hz: float = 1000.0, q: float = 1.0
) -> AudioBuffer:
    """Korg 35 24 dB/oct lowpass filter (virtual analog).

    Parameters
    ----------
    buf : AudioBuffer
        Input audio.
    cutoff_hz : float
        Cutoff frequency in Hz.
    q : float
        Resonance / Q factor.

    Returns
    -------
    AudioBuffer
        Filtered audio.
    """
    return _va_filter(buf, _va.Korg35LPF, cutoff_hz, q)


def va_korg35_hpf(
    buf: AudioBuffer, cutoff_hz: float = 1000.0, q: float = 1.0
) -> AudioBuffer:
    """Korg 35 24 dB/oct highpass filter (virtual analog).

    Parameters
    ----------
    buf : AudioBuffer
        Input audio.
    cutoff_hz : float
        Cutoff frequency in Hz.
    q : float
        Resonance / Q factor.

    Returns
    -------
    AudioBuffer
        Filtered audio.
    """
    return _va_filter(buf, _va.Korg35HPF, cutoff_hz, q)


def va_oberheim(
    buf: AudioBuffer,
    cutoff_hz: float = 1000.0,
    q: float = 1.0,
    mode: str = "lpf",
) -> AudioBuffer:
    """Oberheim multi-mode state-variable filter (virtual analog).

    Parameters
    ----------
    buf : AudioBuffer
        Input audio.
    cutoff_hz : float
        Cutoff frequency in Hz.
    q : float
        Resonance / Q factor.
    mode : str
        One of 'lpf', 'hpf', 'bpf', 'bsf' (notch).

    Returns
    -------
    AudioBuffer
        Filtered audio.
    """
    _validate_freq_hz(cutoff_hz, buf.sample_rate)
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

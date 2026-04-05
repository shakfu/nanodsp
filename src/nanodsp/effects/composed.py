"""Composed effects -- exciter, de-esser, mastering, vocal chain, etc."""

from __future__ import annotations

from typing import Literal

import numpy as np

from ..buffer import AudioBuffer
from .._helpers import _process_per_channel
from .._core import fxdsp as _fxdsp

from .filters import (
    lowpass,
    highpass,
    bandpass,
    low_shelf_db,
    high_shelf_db,
    peak_db,
)
from .dynamics import compress, limit, noise_gate
from .daisysp import dc_block
from .saturation import saturate


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

    Parameters
    ----------
    freq : float
        Crossover frequency in Hz, > 0 and < Nyquist. Typical: 2000--8000.
    amount : float
        Harmonic blend amount, >= 0. 0.0 = no effect, 1.0 = equal blend.
        Typical: 0.1--0.5.
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

    Parameters
    ----------
    freq : float
        Center frequency of the sibilant band in Hz, > 0 and < Nyquist.
        Typical: 4000--10000.
    threshold_db : float
        Compression threshold in dB. Typical: -30 to -10.
    ratio : float
        Compression ratio, >= 1. Typical: 3--10.
    bandwidth : float
        Band width in octaves, > 0. Typical: 1--3.
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
    """Blend heavily compressed signal with dry signal (New York compression).

    Parameters
    ----------
    mix : float
        Wet/dry blend, 0.0--1.0 (0.0 = fully dry, 1.0 = fully compressed).
    ratio : float
        Compression ratio, >= 1. Typical: 4--20.
    threshold_db : float
        Compression threshold in dB. Typical: -40 to -10.
    attack : float
        Attack time in seconds, > 0. Typical: 0.001--0.01.
    release : float
        Release time in seconds, > 0. Typical: 0.01--0.2.
    """
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
# Ping-Pong Delay (fxdsp C++ backend)
# ---------------------------------------------------------------------------


def ping_pong_delay(
    buf: AudioBuffer,
    delay_ms: float = 375.0,
    feedback: float = 0.5,
    mix: float = 0.5,
) -> AudioBuffer:
    """Stereo ping-pong delay with crossed feedback.

    The delayed signal bounces between left and right channels.
    Mono input is duplicated to stereo before processing.

    Parameters
    ----------
    buf : AudioBuffer
        Input audio (mono or stereo).
    delay_ms : float
        Delay time in milliseconds (same for both channels).
    feedback : float
        Feedback amount (-0.99 to 0.99). Negative values invert phase.
    mix : float
        Dry/wet blend (0.0 = dry, 1.0 = fully wet).

    Returns
    -------
    AudioBuffer
        Stereo ping-pong delayed audio.
    """
    if buf.channels > 2:
        raise ValueError(
            f"ping_pong_delay requires mono or stereo input, got {buf.channels} channels"
        )
    ppd = _fxdsp.PingPongDelay()
    ppd.init(float(buf.sample_rate))
    ppd.delay_ms = delay_ms
    ppd.feedback = feedback
    ppd.mix = mix
    if buf.channels == 1:
        stereo_in = np.vstack([buf.data[0], buf.data[0]])
    else:
        stereo_in = buf.data
    out = ppd.process(np.ascontiguousarray(stereo_in, dtype=np.float32))
    return AudioBuffer(
        out,
        sample_rate=buf.sample_rate,
        channel_layout="stereo",
        label=buf.label,
    )


# ---------------------------------------------------------------------------
# Frequency Shifter (fxdsp C++ backend)
# ---------------------------------------------------------------------------


def freq_shift(
    buf: AudioBuffer,
    shift_hz: float = 100.0,
) -> AudioBuffer:
    """Shift all frequencies by a fixed amount in Hz.

    Unlike pitch shifting, frequency shifting does not preserve harmonic
    relationships.  A 440 Hz tone shifted +100 Hz becomes 540 Hz (not the
    musical interval you would get from pitch shifting).

    Parameters
    ----------
    buf : AudioBuffer
        Input audio.
    shift_hz : float
        Shift amount in Hz.  Positive = up, negative = down.

    Returns
    -------
    AudioBuffer
        Frequency-shifted audio.
    """

    def _process(x):
        fs = _fxdsp.FreqShifter()
        fs.init(float(buf.sample_rate))
        fs.shift_hz = shift_hz
        return fs.process(x)

    return _process_per_channel(buf, _process)


# ---------------------------------------------------------------------------
# Ring Modulator (fxdsp C++ backend)
# ---------------------------------------------------------------------------


def ring_mod(
    buf: AudioBuffer,
    carrier_freq: float = 440.0,
    mix: float = 1.0,
    lfo_freq: float = 0.0,
    lfo_width: float = 0.0,
) -> AudioBuffer:
    """Ring modulation -- multiply input by a carrier sine wave.

    Parameters
    ----------
    buf : AudioBuffer
        Input audio.
    carrier_freq : float
        Carrier oscillator frequency in Hz.
    mix : float
        Dry/wet blend (0.0 = dry, 1.0 = fully modulated).
    lfo_freq : float
        LFO rate in Hz that modulates the carrier frequency.
    lfo_width : float
        LFO modulation depth in Hz.

    Returns
    -------
    AudioBuffer
        Ring-modulated audio.
    """

    def _process(x):
        rm = _fxdsp.RingMod()
        rm.init(float(buf.sample_rate))
        rm.carrier_freq = carrier_freq
        rm.mix = mix
        rm.lfo_freq = lfo_freq
        rm.lfo_width = lfo_width
        return rm.process(x)

    return _process_per_channel(buf, _process)


# ---------------------------------------------------------------------------
# Shimmer Reverb
# ---------------------------------------------------------------------------


def shimmer_reverb(
    buf: AudioBuffer,
    mix: float = 0.4,
    decay: float = 0.8,
    shimmer: float = 0.3,
    shift_semitones: float = 12.0,
    preset: Literal["room", "hall", "plate", "chamber", "cathedral"] = "hall",
) -> AudioBuffer:
    """Reverb with a pitch-shifted shimmer layer.

    Applies reverb, then pitch-shifts the reverb tail and blends the
    shifted layer back in.  Creates the ethereal, rising-tone reverb
    popular in ambient and post-rock.

    Parameters
    ----------
    buf : AudioBuffer
        Input audio.
    mix : float
        Overall wet/dry blend (0.0 = dry, 1.0 = fully wet).
    decay : float
        Reverb decay time (0.0 to 1.0).
    shimmer : float
        Blend of pitched layer within the wet signal (0.0 to 1.0).
    shift_semitones : float
        Pitch shift for shimmer layer in semitones (default +12 = octave up).
    preset : str
        Reverb preset ('room', 'hall', 'plate', 'chamber', 'cathedral').
    """
    from .reverb import reverb as _reverb

    wet = _reverb(buf, preset=preset, mix=1.0, decay=decay)
    shifted = psola_pitch_shift(wet, semitones=shift_semitones)
    wet_blend = (1.0 - shimmer) * wet.data + shimmer * shifted.data
    # FDN reverb always returns stereo; match dry to stereo
    if buf.channels == 1:
        dry = np.tile(buf.data, (2, 1))
    else:
        dry = buf.data
    out = (1.0 - mix) * dry + mix * wet_blend
    return AudioBuffer(
        out.astype(np.float32),
        sample_rate=buf.sample_rate,
        channel_layout="stereo",
        label=buf.label,
    )


# ---------------------------------------------------------------------------
# Tape Echo
# ---------------------------------------------------------------------------


def tape_echo(
    buf: AudioBuffer,
    delay_ms: float = 300.0,
    feedback: float = 0.5,
    repeats: int = 6,
    tone: float = 3000.0,
    drive: float = 0.3,
    mix: float = 0.5,
) -> AudioBuffer:
    """Multi-tap delay with progressive darkening and tape saturation.

    Each repeat passes through a lowpass filter and tape-style saturation,
    so later echoes are progressively darker and warmer -- like a real
    analog tape delay unit.

    Parameters
    ----------
    buf : AudioBuffer
        Input audio.
    delay_ms : float
        Delay time per repeat in milliseconds.
    feedback : float
        Gain decay per repeat (0.0 to <1.0).
    repeats : int
        Number of echo taps to generate.
    tone : float
        Lowpass cutoff in Hz applied per repeat (lower = darker tails).
    drive : float
        Tape saturation amount per repeat (0.0 = clean).
    mix : float
        Wet/dry blend (0.0 = dry, 1.0 = only echoes).
    """
    sr = buf.sample_rate
    delay_samples = int(sr * delay_ms / 1000.0)
    n = buf.frames
    wet = np.zeros_like(buf.data)

    tap = buf
    for i in range(repeats):
        tap = lowpass(tap, tone)
        tap = saturate(tap, drive=drive, mode="tape")
        offset = (i + 1) * delay_samples
        gain = feedback ** (i + 1)
        if offset < n:
            frames_avail = min(tap.frames, n - offset)
            wet[:, offset : offset + frames_avail] += (
                np.float32(gain) * tap.data[:, :frames_avail]
            )

    out = (1.0 - mix) * buf.data + mix * wet
    return AudioBuffer(
        out.astype(np.float32),
        sample_rate=buf.sample_rate,
        channel_layout=buf.channel_layout,
        label=buf.label,
    )


# ---------------------------------------------------------------------------
# Lo-Fi
# ---------------------------------------------------------------------------


def lo_fi(
    buf: AudioBuffer,
    bit_depth: int = 8,
    reduce: float = 0.5,
    drive: float = 0.3,
    tone: float = 4000.0,
) -> AudioBuffer:
    """Lo-fi degradation: bitcrush, sample-rate reduction, saturation, lowpass.

    Parameters
    ----------
    buf : AudioBuffer
        Input audio.
    bit_depth : int
        Bit depth for quantization (lower = crunchier).
    reduce : float
        Sample-rate reduction amount (0.0 = none, 1.0 = maximum).
    drive : float
        Tape saturation amount.
    tone : float
        Lowpass cutoff in Hz (simulates bandwidth reduction).
    """
    from .daisysp import bitcrush, sample_rate_reduce

    result = bitcrush(buf, bit_depth=bit_depth)
    result = sample_rate_reduce(result, freq=1.0 - reduce)
    result = saturate(result, drive=drive, mode="tape")
    result = lowpass(result, tone)
    return result


# ---------------------------------------------------------------------------
# Telephone Filter
# ---------------------------------------------------------------------------


def telephone(
    buf: AudioBuffer,
    low_cut: float = 300.0,
    high_cut: float = 3400.0,
    drive: float = 0.4,
) -> AudioBuffer:
    """Telephone/radio filter: tight bandpass with saturation.

    Simulates the limited bandwidth and nonlinearity of a telephone
    codec or AM radio transmission.

    Parameters
    ----------
    buf : AudioBuffer
        Input audio.
    low_cut : float
        Highpass cutoff in Hz (default 300 = telephone standard).
    high_cut : float
        Lowpass cutoff in Hz (default 3400 = telephone standard).
    drive : float
        Saturation amount (adds harmonic grit).
    """
    result = highpass(buf, low_cut)
    result = lowpass(result, high_cut)
    result = saturate(result, drive=drive, mode="hard")
    return result


# ---------------------------------------------------------------------------
# Gated Reverb
# ---------------------------------------------------------------------------


def gated_reverb(
    buf: AudioBuffer,
    preset: Literal["room", "hall", "plate", "chamber", "cathedral"] = "plate",
    decay: float = 0.7,
    gate_threshold_db: float = -30.0,
    gate_hold_ms: float = 50.0,
    gate_release: float = 0.02,
    mix: float = 0.5,
) -> AudioBuffer:
    """Reverb followed by a noise gate for truncated, punchy tails.

    Classic 80s production technique: a dense reverb is abruptly cut
    by a gate, producing a powerful burst that stops dead.

    Parameters
    ----------
    buf : AudioBuffer
        Input audio.
    preset : str
        Reverb preset ('room', 'hall', 'plate', 'chamber', 'cathedral').
    decay : float
        Reverb decay time (0.0 to 1.0).
    gate_threshold_db : float
        Gate threshold in dB (below this the reverb is silenced).
    gate_hold_ms : float
        Gate hold time in ms before release begins.
    gate_release : float
        Gate release time in seconds.
    mix : float
        Wet/dry blend (0.0 = dry, 1.0 = fully wet).
    """
    from .reverb import reverb as _reverb

    wet = _reverb(buf, preset=preset, mix=1.0, decay=decay)
    gated = noise_gate(
        wet,
        threshold_db=gate_threshold_db,
        hold_ms=gate_hold_ms,
        release=gate_release,
    )
    # FDN reverb always returns stereo; match dry to stereo
    if buf.channels == 1:
        dry = np.tile(buf.data, (2, 1))
    else:
        dry = buf.data
    out = (1.0 - mix) * dry + mix * gated.data
    return AudioBuffer(
        out.astype(np.float32),
        sample_rate=buf.sample_rate,
        channel_layout="stereo",
        label=buf.label,
    )


# ---------------------------------------------------------------------------
# Auto-Pan
# ---------------------------------------------------------------------------


def auto_pan(
    buf: AudioBuffer,
    rate: float = 2.0,
    depth: float = 1.0,
    center: float = 0.0,
) -> AudioBuffer:
    """LFO-driven stereo panning.

    A sine LFO sweeps the signal between left and right channels using
    equal-power panning.  Mono and stereo inputs are both supported;
    stereo inputs are summed to mono before panning.

    Parameters
    ----------
    buf : AudioBuffer
        Input audio (mono or stereo).
    rate : float
        LFO frequency in Hz.
    depth : float
        Panning depth (0.0 = no movement, 1.0 = full L/R sweep).
    center : float
        Pan center position (-1.0 = left, 0.0 = center, 1.0 = right).

    Returns
    -------
    AudioBuffer
        Stereo auto-panned audio.
    """
    sr = buf.sample_rate
    n = buf.frames

    # Sum to mono
    if buf.channels == 1:
        mono = buf.data[0]
    else:
        mono = np.mean(buf.data, axis=0)

    # Sine LFO -> pan position in [-1, 1]
    t = np.arange(n, dtype=np.float32) / sr
    pan_pos = np.clip(
        center + depth * np.sin(np.float32(2.0 * np.pi) * rate * t),
        -1.0,
        1.0,
    )

    # Equal-power panning
    angle = (pan_pos + 1.0) * (np.float32(np.pi) / 4.0)
    gain_l = np.cos(angle).astype(np.float32)
    gain_r = np.sin(angle).astype(np.float32)

    out = np.stack([mono * gain_l, mono * gain_r])
    return AudioBuffer(
        out,
        sample_rate=sr,
        channel_layout="stereo",
        label=buf.label,
    )

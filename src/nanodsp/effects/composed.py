"""Composed effects -- exciter, de-esser, mastering, vocal chain, etc."""

from __future__ import annotations

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
from .dynamics import compress, limit
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

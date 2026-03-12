"""Core DSP building blocks: delays, envelopes, FFT, convolution, rates, mix, LFO, numpy utils."""

from __future__ import annotations

import numpy as np

from nanodsp.buffer import AudioBuffer
from nanodsp._helpers import _process_per_channel
from nanodsp._core import fft, delay as _delay, envelopes, rates, mix
from nanodsp._core import madronalib as _madronalib


# ---------------------------------------------------------------------------
# Delay functions
# ---------------------------------------------------------------------------


def delay(
    buf: AudioBuffer,
    delay_samples: float,
    capacity: int | None = None,
    interpolation: str = "linear",
) -> AudioBuffer:
    """Apply a fixed delay (in samples) per channel.

    Parameters
    ----------
    buf : AudioBuffer
        Input audio.
    delay_samples : float
        Delay amount in samples (fractional for interpolated delay).
    capacity : int or None
        Delay line capacity. If None, auto-sized from delay_samples.
    interpolation : str
        Interpolation mode: 'linear' or 'cubic'.

    Returns
    -------
    AudioBuffer
        Delayed audio.
    """
    cap = capacity if capacity is not None else int(delay_samples) + 64

    def _process(x):
        if interpolation == "cubic":
            d = _delay.DelayCubic(cap)
        else:
            d = _delay.Delay(cap)
        return d.process(x, delay_samples)

    return _process_per_channel(buf, _process)


def delay_varying(
    buf: AudioBuffer,
    delays,
    interpolation: str = "linear",
) -> AudioBuffer:
    """Apply time-varying delay per channel.

    Parameters
    ----------
    buf : AudioBuffer
        Input audio.
    delays : ndarray
        1D (broadcast to all channels) or 2D [channels, frames].
    interpolation : str
        Interpolation mode: 'linear' or 'cubic'.

    Returns
    -------
    AudioBuffer
        Delayed audio.
    """
    delays = np.asarray(delays, dtype=np.float32)
    if delays.ndim == 1:
        delays_2d = np.tile(delays, (buf.channels, 1))
    elif delays.ndim == 2:
        if delays.shape[0] != buf.channels:
            raise ValueError(
                f"delays has {delays.shape[0]} channels, buffer has {buf.channels}"
            )
        delays_2d = delays
    else:
        raise ValueError(f"delays must be 1D or 2D, got {delays.ndim}D")

    if delays_2d.shape[1] != buf.frames:
        raise ValueError(
            f"delays has {delays_2d.shape[1]} frames, buffer has {buf.frames}"
        )

    max_delay = int(np.max(delays_2d)) + 64
    out = np.zeros_like(buf.data)
    for ch in range(buf.channels):
        d: _delay.Delay | _delay.DelayCubic
        if interpolation == "cubic":
            d = _delay.DelayCubic(max_delay)
        else:
            d = _delay.Delay(max_delay)
        ch_delays = np.ascontiguousarray(delays_2d[ch])
        out[ch] = d.process_varying(buf.ensure_1d(ch), ch_delays)

    return AudioBuffer(
        out,
        sample_rate=buf.sample_rate,
        channel_layout=buf.channel_layout,
        label=buf.label,
    )


# ---------------------------------------------------------------------------
# Envelope functions
# ---------------------------------------------------------------------------


def box_filter(buf: AudioBuffer, length: int) -> AudioBuffer:
    """Apply a BoxFilter (moving average) per channel.

    Parameters
    ----------
    buf : AudioBuffer
        Input audio.
    length : int
        Window size in samples.

    Returns
    -------
    AudioBuffer
        Smoothed audio.
    """

    def _process(x):
        bf = envelopes.BoxFilter(length)
        bf.set(length)
        return bf.process(x)

    return _process_per_channel(buf, _process)


def box_stack_filter(buf: AudioBuffer, size: int, layers: int = 4) -> AudioBuffer:
    """Apply a BoxStackFilter (stacked moving average) per channel.

    Parameters
    ----------
    buf : AudioBuffer
        Input audio.
    size : int
        Window size in samples per layer.
    layers : int
        Number of stacked box filter layers.

    Returns
    -------
    AudioBuffer
        Smoothed audio.
    """

    def _process(x):
        bs = envelopes.BoxStackFilter(size, layers)
        bs.set(size)
        return bs.process(x)

    return _process_per_channel(buf, _process)


def peak_hold(buf: AudioBuffer, length: int) -> AudioBuffer:
    """Apply PeakHold per channel.

    Parameters
    ----------
    buf : AudioBuffer
        Input audio.
    length : int
        Hold window size in samples.

    Returns
    -------
    AudioBuffer
        Peak-held envelope.
    """

    def _process(x):
        ph = envelopes.PeakHold(length)
        ph.set(length)
        return ph.process(x)

    return _process_per_channel(buf, _process)


def peak_decay(buf: AudioBuffer, length: int) -> AudioBuffer:
    """Apply PeakDecayLinear per channel.

    Parameters
    ----------
    buf : AudioBuffer
        Input audio.
    length : int
        Decay window size in samples.

    Returns
    -------
    AudioBuffer
        Peak-decayed envelope.
    """

    def _process(x):
        pd = envelopes.PeakDecayLinear(length)
        pd.set(length)
        return pd.process(x)

    return _process_per_channel(buf, _process)


# ---------------------------------------------------------------------------
# FFT functions
# ---------------------------------------------------------------------------


def rfft(buf: AudioBuffer) -> list[np.ndarray]:
    """Forward real FFT per channel.

    Returns a list of complex64 arrays (one per channel, N/2 bins each).
    Uses RealFFT.fast_size_above for efficient FFT size, zero-pads if needed.
    """
    fft_size = fft.RealFFT.fast_size_above(buf.frames)
    rfft_obj = fft.RealFFT(fft_size)
    result = []
    for ch in range(buf.channels):
        x = buf.ensure_1d(ch)
        if len(x) < fft_size:
            x = np.pad(x, (0, fft_size - len(x)), mode="constant")
        result.append(rfft_obj.fft(x))
    return result


def irfft(
    spectra: list[np.ndarray],
    size: int,
    sample_rate: float = 48000.0,
) -> AudioBuffer:
    """Inverse real FFT from list of spectra to AudioBuffer.

    Returns unscaled output (matches C++ convention). Divide by N if needed.
    """
    channels = len(spectra)
    bins = spectra[0].shape[0]
    fft_size = bins * 2
    rfft_obj = fft.RealFFT(fft_size)

    out = np.zeros((channels, size), dtype=np.float32)
    for ch in range(channels):
        full = rfft_obj.ifft(spectra[ch])
        out[ch] = full[:size]

    return AudioBuffer(out, sample_rate=sample_rate)


# ---------------------------------------------------------------------------
# Convolution
# ---------------------------------------------------------------------------


def convolve(
    buf: AudioBuffer,
    ir: AudioBuffer,
    normalize: bool = False,
    trim: bool = True,
) -> AudioBuffer:
    """FFT-based overlap-add convolution.

    Parameters
    ----------
    buf : AudioBuffer
        Input signal.
    ir : AudioBuffer
        Impulse response.
    normalize : bool
        If True, scale IR to unit energy before convolving.
    trim : bool
        If True (default), output has the same length as *buf*.
        If False, output is the full convolution (buf.frames + ir.frames - 1).

    Raises
    ------
    ValueError
        If sample rates differ or channel counts are incompatible.
    """
    if buf.sample_rate != ir.sample_rate:
        raise ValueError(
            f"Sample rate mismatch: buf={buf.sample_rate}, ir={ir.sample_rate}"
        )

    # Channel matching
    if ir.channels == 1 and buf.channels > 1:
        ir_data = np.tile(ir.data, (buf.channels, 1))
    elif ir.channels == buf.channels:
        ir_data = ir.data
    else:
        raise ValueError(
            f"Channel mismatch: buf has {buf.channels}, ir has {ir.channels}. "
            "IR must be mono (broadcasts) or match buf channel count."
        )

    if normalize:
        for ch in range(ir_data.shape[0]):
            energy = np.sqrt(np.sum(ir_data[ch] ** 2))
            if energy > 0:
                ir_data = ir_data.copy()
                ir_data[ch] /= energy

    sig_len = buf.frames
    ir_len = ir.frames
    full_len = sig_len + ir_len - 1
    block_size = ir_len
    fft_size = fft.RealFFT.fast_size_above(2 * block_size)

    n_blocks = (sig_len + block_size - 1) // block_size
    out = np.zeros((buf.channels, full_len), dtype=np.float32)

    for ch in range(buf.channels):
        # FFT the IR once
        ir_padded = np.zeros(fft_size, dtype=np.float32)
        ir_padded[:ir_len] = ir_data[ch]
        IR_freq = np.fft.rfft(ir_padded)

        # Pre-slice all signal blocks into [n_blocks, fft_size]
        blocks = np.zeros((n_blocks, fft_size), dtype=np.float32)
        for b in range(n_blocks):
            start = b * block_size
            end = min(start + block_size, sig_len)
            blocks[b, : end - start] = buf.data[ch, start:end]

        # Batch FFT, multiply, IFFT
        block_freqs = np.fft.rfft(blocks, n=fft_size, axis=1)
        block_freqs *= IR_freq[np.newaxis, :]
        block_results = np.fft.irfft(block_freqs, n=fft_size, axis=1).astype(np.float32)

        # Overlap-add
        for b in range(n_blocks):
            pos = b * block_size
            out_end = min(pos + fft_size, full_len)
            out[ch, pos:out_end] += block_results[b, : out_end - pos]

    if trim:
        out = out[:, :sig_len]

    return AudioBuffer(
        out,
        sample_rate=buf.sample_rate,
        channel_layout=buf.channel_layout,
        label=buf.label,
    )


# ---------------------------------------------------------------------------
# Rates functions
# ---------------------------------------------------------------------------


def upsample_2x(
    buf: AudioBuffer,
    max_block: int | None = None,
    half_latency: int = 16,
    pass_freq: float = 0.43,
) -> AudioBuffer:
    """Upsample by 2x. Returns AudioBuffer with 2x frames and 2x sample rate.

    Parameters
    ----------
    buf : AudioBuffer
        Input audio.
    max_block : int or None
        Maximum block size. If None, uses buf.frames.
    half_latency : int
        Half-band filter latency in samples.
    pass_freq : float
        Normalized passband edge frequency.

    Returns
    -------
    AudioBuffer
        Upsampled audio at 2x sample rate.
    """
    block = max_block if max_block is not None else buf.frames
    os = rates.Oversampler2x(buf.channels, block, half_latency, pass_freq)
    upsampled = os.up(buf.data)
    return AudioBuffer(
        upsampled,
        sample_rate=buf.sample_rate * 2,
        channel_layout=buf.channel_layout,
        label=buf.label,
    )


def oversample_roundtrip(
    buf: AudioBuffer,
    max_block: int | None = None,
    half_latency: int = 16,
    pass_freq: float = 0.43,
) -> AudioBuffer:
    """Upsample then downsample (roundtrip). Same shape and sample rate as input.

    Parameters
    ----------
    buf : AudioBuffer
        Input audio.
    max_block : int or None
        Maximum block size. If None, uses buf.frames.
    half_latency : int
        Half-band filter latency in samples.
    pass_freq : float
        Normalized passband edge frequency.

    Returns
    -------
    AudioBuffer
        Roundtripped audio (same shape as input).
    """
    block = max_block if max_block is not None else buf.frames
    os = rates.Oversampler2x(buf.channels, block, half_latency, pass_freq)
    processed = os.process(buf.data)
    return AudioBuffer(
        processed,
        sample_rate=buf.sample_rate,
        channel_layout=buf.channel_layout,
        label=buf.label,
    )


# ---------------------------------------------------------------------------
# Mix functions
# ---------------------------------------------------------------------------


def hadamard(buf: AudioBuffer) -> AudioBuffer:
    """Apply Hadamard mixing across channels at each frame.

    Requires power-of-2 channel count.

    Parameters
    ----------
    buf : AudioBuffer
        Input audio (must have power-of-2 channel count).

    Returns
    -------
    AudioBuffer
        Hadamard-mixed audio.
    """
    ch = buf.channels
    if ch == 0 or (ch & (ch - 1)) != 0:
        raise ValueError(f"Hadamard requires power-of-2 channel count, got {ch}")
    h = mix.Hadamard(ch)
    out = np.zeros_like(buf.data)
    for i in range(buf.frames):
        frame = np.ascontiguousarray(buf.data[:, i].copy())
        out[:, i] = h.in_place(frame)
    return AudioBuffer(
        out,
        sample_rate=buf.sample_rate,
        channel_layout=buf.channel_layout,
        label=buf.label,
    )


def householder(buf: AudioBuffer) -> AudioBuffer:
    """Apply Householder reflection across channels at each frame.

    Parameters
    ----------
    buf : AudioBuffer
        Input audio.

    Returns
    -------
    AudioBuffer
        Householder-mixed audio.
    """
    ch = buf.channels
    h = mix.Householder(ch)
    out = np.zeros_like(buf.data)
    for i in range(buf.frames):
        frame = np.ascontiguousarray(buf.data[:, i].copy())
        out[:, i] = h.in_place(frame)
    return AudioBuffer(
        out,
        sample_rate=buf.sample_rate,
        channel_layout=buf.channel_layout,
        label=buf.label,
    )


def crossfade(buf_a: AudioBuffer, buf_b: AudioBuffer, x: float) -> AudioBuffer:
    """Crossfade between two buffers using cheap_energy_crossfade coefficients.

    Parameters
    ----------
    buf_a : AudioBuffer
        First audio buffer (returned when x=0).
    buf_b : AudioBuffer
        Second audio buffer (returned when x=1).
    x : float
        Crossfade position (0.0 to 1.0).

    Returns
    -------
    AudioBuffer
        Crossfaded audio.
    """
    if buf_a.sample_rate != buf_b.sample_rate:
        raise ValueError(
            f"Sample rate mismatch: {buf_a.sample_rate} vs {buf_b.sample_rate}"
        )
    if buf_a.channels != buf_b.channels:
        raise ValueError(
            f"Channel count mismatch: {buf_a.channels} vs {buf_b.channels}"
        )
    if buf_a.frames != buf_b.frames:
        raise ValueError(f"Frame count mismatch: {buf_a.frames} vs {buf_b.frames}")
    to_c, from_c = mix.cheap_energy_crossfade(x)
    out = buf_a.data * from_c + buf_b.data * to_c
    return AudioBuffer(
        out.astype(np.float32),
        sample_rate=buf_a.sample_rate,
        channel_layout=buf_a.channel_layout,
        label=buf_a.label,
    )


# ---------------------------------------------------------------------------
# LFO function
# ---------------------------------------------------------------------------


def lfo(
    frames: int,
    low: float,
    high: float,
    rate: float,
    sample_rate: float = 48000.0,
    rate_variation: float = 0.0,
    depth_variation: float = 0.0,
    seed: int | None = None,
) -> AudioBuffer:
    """Generate an LFO signal using CubicLfo.

    Parameters
    ----------
    frames : int
        Number of output samples.
    low, high : float
        Output value range.
    rate : float
        Base rate (cycles per sample).
    sample_rate : float
        Sample rate for the returned AudioBuffer metadata.
    rate_variation, depth_variation : float
        Randomization parameters (0 = deterministic).
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    AudioBuffer
        Mono buffer containing the LFO waveform.
    """
    if seed is not None:
        osc = envelopes.CubicLfo(seed)
    else:
        osc = envelopes.CubicLfo()
    osc.set(low, high, rate, rate_variation, depth_variation)
    data = osc.process(frames)
    return AudioBuffer(np.asarray(data).reshape(1, -1), sample_rate=sample_rate)


# ---------------------------------------------------------------------------
# Pure numpy utilities
# ---------------------------------------------------------------------------


def normalize_peak(buf: AudioBuffer, target_db: float = 0.0) -> AudioBuffer:
    """Normalize peak amplitude to *target_db* dBFS.

    Parameters
    ----------
    buf : AudioBuffer
        Input audio.
    target_db : float
        Target peak level in dBFS.

    Returns
    -------
    AudioBuffer
        Peak-normalized audio.
    """
    peak = np.max(np.abs(buf.data))
    if peak == 0.0:
        return buf.copy()
    target_linear = 10.0 ** (target_db / 20.0)
    scale = np.float32(target_linear / peak)
    return AudioBuffer(
        buf.data * scale,
        sample_rate=buf.sample_rate,
        channel_layout=buf.channel_layout,
        label=buf.label,
    )


def trim_silence(
    buf: AudioBuffer,
    threshold_db: float = -60.0,
    pad_frames: int = 0,
) -> AudioBuffer:
    """Trim leading and trailing silence below *threshold_db*.

    Parameters
    ----------
    buf : AudioBuffer
        Input audio.
    threshold_db : float
        Silence threshold in dB.
    pad_frames : int
        Extra frames to keep around non-silent regions.

    Returns
    -------
    AudioBuffer
        Trimmed audio.
    """
    threshold_linear = 10.0 ** (threshold_db / 20.0)
    # Max across channels at each frame
    frame_peaks = np.max(np.abs(buf.data), axis=0)
    above = np.nonzero(frame_peaks > threshold_linear)[0]
    if len(above) == 0:
        # All silence -- return empty buffer with same metadata
        return AudioBuffer(
            np.zeros((buf.channels, 0), dtype=np.float32),
            sample_rate=buf.sample_rate,
            channel_layout=buf.channel_layout,
            label=buf.label,
        )
    first = max(0, int(above[0]) - pad_frames)
    last = min(buf.frames, int(above[-1]) + 1 + pad_frames)
    return buf.slice(first, last)


def fade_in(
    buf: AudioBuffer,
    duration_ms: float = 10.0,
    curve: str = "linear",
) -> AudioBuffer:
    """Apply a fade-in over *duration_ms* milliseconds.

    Parameters
    ----------
    buf : AudioBuffer
        Input audio.
    duration_ms : float
        Fade duration in milliseconds.
    curve : str
        Fade shape: 'linear', 'ease_in', 'ease_out', or 'smoothstep'.

    Returns
    -------
    AudioBuffer
        Audio with fade-in applied.
    """
    n_samples = max(1, int(buf.sample_rate * duration_ms / 1000.0))
    n_samples = min(n_samples, buf.frames)
    ramp = np.linspace(0.0, 1.0, n_samples, dtype=np.float32)
    ramp = _apply_fade_curve(ramp, curve)
    out = buf.data.copy()
    out[:, :n_samples] *= ramp[np.newaxis, :]
    return AudioBuffer(
        out,
        sample_rate=buf.sample_rate,
        channel_layout=buf.channel_layout,
        label=buf.label,
    )


def fade_out(
    buf: AudioBuffer,
    duration_ms: float = 10.0,
    curve: str = "linear",
) -> AudioBuffer:
    """Apply a fade-out over *duration_ms* milliseconds.

    Parameters
    ----------
    buf : AudioBuffer
        Input audio.
    duration_ms : float
        Fade duration in milliseconds.
    curve : str
        Fade shape: 'linear', 'ease_in', 'ease_out', or 'smoothstep'.

    Returns
    -------
    AudioBuffer
        Audio with fade-out applied.
    """
    n_samples = max(1, int(buf.sample_rate * duration_ms / 1000.0))
    n_samples = min(n_samples, buf.frames)
    ramp = np.linspace(1.0, 0.0, n_samples, dtype=np.float32)
    ramp = _apply_fade_curve(ramp, curve, inverse=True)
    out = buf.data.copy()
    out[:, -n_samples:] *= ramp[np.newaxis, :]
    return AudioBuffer(
        out,
        sample_rate=buf.sample_rate,
        channel_layout=buf.channel_layout,
        label=buf.label,
    )


def _apply_fade_curve(
    ramp: np.ndarray,
    curve: str,
    inverse: bool = False,
) -> np.ndarray:
    """Map a [0,1] ramp through a fade curve.

    For fade_in: ramp goes 0->1, curve shapes the rise.
    For fade_out (inverse=True): ramp goes 1->0, we apply the
    curve to the underlying 0->1 parameter then flip.
    """
    proj = _madronalib.projections
    if curve == "linear":
        return ramp
    # Normalize to 0->1 parameter for curve application
    if inverse:
        t = 1.0 - ramp  # 0->1
    else:
        t = ramp
    t = np.ascontiguousarray(t, dtype=np.float32)
    if curve == "ease_in":
        shaped = np.asarray(proj.ease_in(t), dtype=np.float32)
    elif curve == "ease_out":
        shaped = np.asarray(proj.ease_out(t), dtype=np.float32)
    elif curve == "smoothstep":
        shaped = np.asarray(proj.smoothstep(t), dtype=np.float32)
    else:
        raise ValueError(
            f"Unknown fade curve {curve!r}, valid: 'linear', 'ease_in', "
            "'ease_out', 'smoothstep'"
        )
    if inverse:
        return 1.0 - shaped
    return shaped


def pan(buf: AudioBuffer, position: float = 0.0) -> AudioBuffer:
    """Pan a signal using equal-power panning.

    Mono input produces stereo output. Stereo input scales L/R gains.

    Parameters
    ----------
    buf : AudioBuffer
        Input audio.
    position : float
        Pan position: -1.0 = hard left, 0.0 = center, 1.0 = hard right.

    Returns
    -------
    AudioBuffer
        Panned audio (stereo if mono input).
    """
    theta = (position + 1.0) / 2.0 * (np.pi / 2.0)
    left_gain = np.float32(np.cos(theta))
    right_gain = np.float32(np.sin(theta))

    if buf.channels == 1:
        out = np.zeros((2, buf.frames), dtype=np.float32)
        out[0] = buf.data[0] * left_gain
        out[1] = buf.data[0] * right_gain
        return AudioBuffer(
            out,
            sample_rate=buf.sample_rate,
            channel_layout="stereo",
            label=buf.label,
        )

    # Stereo or multi-channel: scale first two channels
    out = buf.data.copy()
    out[0] *= left_gain
    if buf.channels >= 2:
        out[1] *= right_gain
    return AudioBuffer(
        out,
        sample_rate=buf.sample_rate,
        channel_layout=buf.channel_layout,
        label=buf.label,
    )


def mix_buffers(*buffers: AudioBuffer, gains: list[float] | None = None) -> AudioBuffer:
    """Sum multiple AudioBuffers with optional per-buffer gains.

    All buffers must share the same sample_rate. Shorter buffers are
    zero-padded to the length of the longest.

    Parameters
    ----------
    *buffers : AudioBuffer
        Audio buffers to sum.
    gains : list of float or None
        Per-buffer gain multipliers. If None, all gains are 1.0.

    Returns
    -------
    AudioBuffer
        Summed audio.
    """
    if not buffers:
        raise ValueError("At least one buffer required")
    # Flatten if called with a single list
    if len(buffers) == 1 and isinstance(buffers[0], (list, tuple)):
        buffers = tuple(buffers[0])
    if gains is None:
        gains = [1.0] * len(buffers)
    if len(gains) != len(buffers):
        raise ValueError(
            f"gains length ({len(gains)}) must match number of buffers ({len(buffers)})"
        )
    sr = buffers[0].sample_rate
    for b in buffers[1:]:
        if b.sample_rate != sr:
            raise ValueError(f"Sample rate mismatch: {sr} vs {b.sample_rate}")
    max_channels = max(b.channels for b in buffers)
    max_frames = max(b.frames for b in buffers)
    out = np.zeros((max_channels, max_frames), dtype=np.float32)
    for b, g in zip(buffers, gains):
        data = b.data
        # Broadcast mono to multi-channel if needed
        if data.shape[0] == 1 and max_channels > 1:
            data = np.tile(data, (max_channels, 1))
        out[: data.shape[0], : data.shape[1]] += data * np.float32(g)
    return AudioBuffer(out, sample_rate=sr, label=buffers[0].label)


# ---------------------------------------------------------------------------
# Mid-side processing
# ---------------------------------------------------------------------------


def mid_side_encode(buf: AudioBuffer) -> AudioBuffer:
    """Encode stereo [L, R] to mid-side [M, S].

    M = (L + R) / 2, S = (L - R) / 2.

    Parameters
    ----------
    buf : AudioBuffer
        Stereo input audio.

    Returns
    -------
    AudioBuffer
        Mid-side encoded audio [M, S].
    """
    if buf.channels != 2:
        raise ValueError(
            f"mid_side_encode requires stereo input, got {buf.channels} channels"
        )
    mid = (buf.data[0] + buf.data[1]) * 0.5
    side = (buf.data[0] - buf.data[1]) * 0.5
    out = np.stack([mid, side]).astype(np.float32)
    return AudioBuffer(
        out,
        sample_rate=buf.sample_rate,
        channel_layout="stereo",
        label=buf.label,
    )


def mid_side_decode(buf: AudioBuffer) -> AudioBuffer:
    """Decode mid-side [M, S] back to stereo [L, R].

    L = M + S, R = M - S.

    Parameters
    ----------
    buf : AudioBuffer
        Mid-side input audio [M, S].

    Returns
    -------
    AudioBuffer
        Stereo audio [L, R].
    """
    if buf.channels != 2:
        raise ValueError(
            f"mid_side_decode requires 2-channel [M, S] input, got {buf.channels} channels"
        )
    left = buf.data[0] + buf.data[1]
    right = buf.data[0] - buf.data[1]
    out = np.stack([left, right]).astype(np.float32)
    return AudioBuffer(
        out,
        sample_rate=buf.sample_rate,
        channel_layout="stereo",
        label=buf.label,
    )


# ---------------------------------------------------------------------------
# Cross-correlation
# ---------------------------------------------------------------------------


def xcorr(buf_a: AudioBuffer, buf_b: AudioBuffer | None = None) -> np.ndarray:
    """FFT-based cross-correlation (or autocorrelation).

    Parameters
    ----------
    buf_a : AudioBuffer
        First signal (mono). Multi-channel buffers are mixed to mono.
    buf_b : AudioBuffer or None
        Second signal. If None, computes autocorrelation of *buf_a*.

    Returns
    -------
    np.ndarray
        1D cross-correlation array of length ``len_a + len_b - 1``
        (or ``2 * len_a - 1`` for autocorrelation).
    """
    a = (
        np.mean(buf_a.data, axis=0).astype(np.float64)
        if buf_a.channels > 1
        else buf_a.data[0].astype(np.float64)
    )
    if buf_b is None:
        b = a
    else:
        b = (
            np.mean(buf_b.data, axis=0).astype(np.float64)
            if buf_b.channels > 1
            else buf_b.data[0].astype(np.float64)
        )

    full_len = len(a) + len(b) - 1
    # Next power of 2 for efficient FFT
    fft_size = 1
    while fft_size < full_len:
        fft_size *= 2

    A = np.fft.rfft(a, n=fft_size)
    B = np.fft.rfft(b, n=fft_size)
    corr = np.fft.irfft(A * np.conj(B), n=fft_size)[:full_len]
    return corr.astype(np.float32)


# ---------------------------------------------------------------------------
# Hilbert transform / analytic signal envelope
# ---------------------------------------------------------------------------


def hilbert(buf: AudioBuffer) -> AudioBuffer:
    """Compute the envelope (magnitude of analytic signal) per channel.

    Uses the FFT-based method: zero negative frequencies, IFFT,
    then take the absolute value.

    Returns
    -------
    AudioBuffer
        Envelope of the analytic signal (real-valued).
    """

    def _process(x):
        n = len(x)
        X = np.fft.fft(x.astype(np.float64))
        # Build the analytic signal multiplier
        h = np.zeros(n, dtype=np.float64)
        h[0] = 1.0
        if n % 2 == 0:
            h[n // 2] = 1.0
            h[1 : n // 2] = 2.0
        else:
            h[1 : (n + 1) // 2] = 2.0
        analytic = np.fft.ifft(X * h)
        return np.abs(analytic).astype(np.float32)

    return _process_per_channel(buf, _process)


def envelope(buf: AudioBuffer) -> AudioBuffer:
    """Compute the amplitude envelope (magnitude of analytic signal).

    Alias for :func:`hilbert`.
    """
    return hilbert(buf)


# ---------------------------------------------------------------------------
# Median filter
# ---------------------------------------------------------------------------


def median_filter(buf: AudioBuffer, kernel_size: int = 3) -> AudioBuffer:
    """Apply a median filter per channel.

    Parameters
    ----------
    kernel_size : int
        Window size for the median (must be odd and >= 1).
    """
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be odd and >= 1, got {kernel_size}")

    half = kernel_size // 2

    def _process(x):
        n = len(x)
        padded = np.pad(x, (half, half), mode="edge")
        out = np.empty(n, dtype=np.float32)
        # Use stride_tricks for a sliding window view
        shape = (n, kernel_size)
        strides = (padded.strides[0], padded.strides[0])
        windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
        out[:] = np.median(windows, axis=1).astype(np.float32)
        return out

    return _process_per_channel(buf, _process)


# ---------------------------------------------------------------------------
# LMS adaptive filter
# ---------------------------------------------------------------------------


def lms_filter(
    buf: AudioBuffer,
    ref: AudioBuffer,
    filter_len: int = 32,
    step_size: float = 0.01,
    normalized: bool = True,
) -> tuple[AudioBuffer, AudioBuffer]:
    """LMS (Least Mean Squares) adaptive filter.

    Parameters
    ----------
    buf : AudioBuffer
        Input (desired) signal.
    ref : AudioBuffer
        Reference (noise) signal to be adaptively filtered and subtracted.
    filter_len : int
        Number of filter taps.
    step_size : float
        Adaptation step size (mu).
    normalized : bool
        If True, use Normalized LMS (step_size normalized by input power).

    Returns
    -------
    tuple[AudioBuffer, AudioBuffer]
        (output, error) — output is the filtered reference, error is buf - output.
    """
    if buf.sample_rate != ref.sample_rate:
        raise ValueError(
            f"Sample rate mismatch: buf={buf.sample_rate}, ref={ref.sample_rate}"
        )
    if buf.frames != ref.frames:
        raise ValueError(f"Frame count mismatch: buf={buf.frames}, ref={ref.frames}")

    n_frames = buf.frames
    n_ch = max(buf.channels, ref.channels)

    out_data = np.zeros((n_ch, n_frames), dtype=np.float32)
    err_data = np.zeros((n_ch, n_frames), dtype=np.float32)

    for ch in range(n_ch):
        d = buf.data[min(ch, buf.channels - 1)].astype(np.float64)
        x = ref.data[min(ch, ref.channels - 1)].astype(np.float64)
        w = np.zeros(filter_len, dtype=np.float64)
        x_buf = np.zeros(filter_len, dtype=np.float64)
        eps = 1e-8

        for i in range(n_frames):
            # Shift delay line
            x_buf[1:] = x_buf[:-1]
            x_buf[0] = x[i]

            # Filter output
            y = np.dot(w, x_buf)
            e = d[i] - y

            # Weight update
            if normalized:
                norm = np.dot(x_buf, x_buf) + eps
                w += (step_size / norm) * e * x_buf
            else:
                w += step_size * e * x_buf

            out_data[ch, i] = np.float32(y)
            err_data[ch, i] = np.float32(e)

    output = AudioBuffer(
        out_data,
        sample_rate=buf.sample_rate,
        channel_layout=buf.channel_layout,
        label=buf.label,
    )
    error = AudioBuffer(
        err_data,
        sample_rate=buf.sample_rate,
        channel_layout=buf.channel_layout,
        label=buf.label,
    )
    return output, error


def stereo_widen(buf: AudioBuffer, width: float = 1.5) -> AudioBuffer:
    """Adjust stereo width via mid-side processing.

    Parameters
    ----------
    buf : AudioBuffer
        Stereo input audio.
    width : float
        Width factor: 0.0 = mono, 1.0 = unchanged, >1.0 = wider.

    Returns
    -------
    AudioBuffer
        Width-adjusted stereo audio.
    """
    if buf.channels != 2:
        raise ValueError(
            f"stereo_widen requires stereo input, got {buf.channels} channels"
        )
    ms = mid_side_encode(buf)
    ms.data[1] *= np.float32(width)
    return mid_side_decode(ms)

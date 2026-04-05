"""Audio analysis: loudness, spectral features, pitch/onset detection, resampling."""

from __future__ import annotations

import numpy as np

from nanodsp.buffer import AudioBuffer
from nanodsp.spectral import stft
from nanodsp._core import filters, madronalib as _madronalib

# Small constants to prevent log(0) and division-by-zero in numerical computations.
# _LOG_EPS is intentionally tiny (1e-20) because it guards log10() in LUFS loudness
# metering where even 1e-10 would bias quiet-signal measurements.  _DIV_EPS is
# larger (1e-10) because it guards ordinary division and a tighter value would
# amplify floating-point noise without improving accuracy.
_LOG_EPS: float = 1e-20
_DIV_EPS: float = 1e-10


# ---------------------------------------------------------------------------
# Loudness metering (ITU-R BS.1770-4)
# ---------------------------------------------------------------------------


def _k_weight(x: np.ndarray, sample_rate: float) -> np.ndarray:
    """Apply two-stage K-weighting to a 1D signal via C++ Biquad.

    Stage 1: high shelf ~+4 dB at 1681 Hz (head/ear acoustic model).
    Stage 2: highpass at 38 Hz (revised low-frequency B-weighting).
    """
    freq_pre = 1681.0 / sample_rate
    bq_pre = filters.Biquad()
    bq_pre.high_shelf_db(freq_pre, 4.0)
    stage1 = bq_pre.process(x)
    freq_hp = 38.0 / sample_rate
    bq_hp = filters.Biquad()
    bq_hp.highpass(freq_hp)
    return bq_hp.process(stage1)


def loudness_lufs(buf: AudioBuffer) -> float:
    """Measure integrated loudness per ITU-R BS.1770-4.

    Implements the gated loudness measurement algorithm defined in
    ITU-R BS.1770-4 (10/2015), "Algorithms to measure audio programme
    loudness and true-peak audio level."

    Returns
    -------
    float
        Integrated loudness in LUFS. Returns ``-inf`` for silence or
        signals shorter than 400 ms.

    References
    ----------
    .. [1] ITU-R BS.1770-4, "Algorithms to measure audio programme loudness
       and true-peak audio level," International Telecommunication Union, 2015.
       https://www.itu.int/rec/R-REC-BS.1770
    """
    sr = buf.sample_rate
    block_samples = int(sr * 0.4)  # 400 ms
    hop_samples = int(sr * 0.1)  # 100 ms (75% overlap)

    if buf.frames < block_samples:
        return float("-inf")

    # K-weight each channel
    weighted = []
    for ch in range(buf.channels):
        weighted.append(_k_weight(buf.ensure_1d(ch), sr))

    # Channel weights per ITU-R BS.1770-4
    # 5.1 (6 ch): L=1.0, R=1.0, C=1.0, LFE=0.0, Ls=1.41, Rs=1.41
    # All other layouts: uniform 1.0
    if buf.channels == 6:
        ch_weights = np.array([1.0, 1.0, 1.0, 0.0, 1.41, 1.41], dtype=np.float64)
    else:
        ch_weights = np.ones(buf.channels, dtype=np.float64)

    # Compute per-block loudness
    n_blocks = (buf.frames - block_samples) // hop_samples + 1
    block_power = np.zeros(n_blocks, dtype=np.float64)

    for i in range(n_blocks):
        start = i * hop_samples
        end = start + block_samples
        power = 0.0
        for ch in range(buf.channels):
            segment = weighted[ch][start:end].astype(np.float64)
            power += ch_weights[ch] * np.mean(segment**2)
        block_power[i] = power

    # Convert to LUFS
    block_lufs = -0.691 + 10.0 * np.log10(block_power + _LOG_EPS)

    # Absolute gate: -70 LUFS
    abs_gate_mask = block_lufs >= -70.0
    if not np.any(abs_gate_mask):
        return float("-inf")

    # Relative gate: mean of surviving blocks - 10 dB
    mean_power_abs = np.mean(block_power[abs_gate_mask])
    rel_gate_lufs = -0.691 + 10.0 * np.log10(mean_power_abs + _LOG_EPS) - 10.0
    rel_gate_mask = abs_gate_mask & (block_lufs >= rel_gate_lufs)

    if not np.any(rel_gate_mask):
        return float("-inf")

    # Integrated loudness
    mean_power = np.mean(block_power[rel_gate_mask])
    return float(-0.691 + 10.0 * np.log10(mean_power + _LOG_EPS))


def normalize_lufs(
    buf: AudioBuffer,
    target_lufs: float = -14.0,
) -> AudioBuffer:
    """Normalize loudness to *target_lufs*.

    Parameters
    ----------
    target_lufs : float
        Target integrated loudness in LUFS. Typical: -23 (broadcast) to -14 (streaming).

    Raises
    ------
    ValueError
        If the input is silent or too short to measure.
    """
    current = loudness_lufs(buf)
    if np.isinf(current):
        raise ValueError("Cannot normalize: input is silent or too short to measure")
    delta = target_lufs - current
    return buf.gain_db(delta)


# ---------------------------------------------------------------------------
# Spectral Feature Extraction
# ---------------------------------------------------------------------------


def _stft_magnitudes(
    buf: AudioBuffer, window_size: int, hop_size: int | None
) -> tuple[np.ndarray, np.ndarray, int]:
    """Compute STFT magnitudes and frequency bins (Hz).

    Returns (mag, freqs, hop_size) where:
    - mag: [channels, num_frames, bins] float32
    - freqs: [bins] float64 frequency array in Hz
    """
    if hop_size is None:
        hop_size = window_size // 4
    spec = stft(buf, window_size=window_size, hop_size=hop_size)
    mag = np.abs(spec.data).astype(np.float32)
    freqs = np.arange(spec.bins, dtype=np.float64) * spec.sample_rate / spec.fft_size
    return mag, freqs, hop_size


def spectral_centroid(
    buf: AudioBuffer,
    window_size: int = 2048,
    hop_size: int | None = None,
) -> np.ndarray:
    """Weighted mean frequency per STFT frame.

    Returns Hz values shaped [num_frames] (mono) or [channels, num_frames].
    """
    mag, freqs, _ = _stft_magnitudes(buf, window_size, hop_size)
    # mag: [ch, frames, bins], freqs: [bins]
    weighted = np.sum(mag * freqs[np.newaxis, np.newaxis, :], axis=2)
    total = np.sum(mag, axis=2)
    safe_total = np.where(total > 0, total, 1.0)
    centroid = np.where(total > 0, weighted / safe_total, 0.0).astype(np.float32)
    if centroid.shape[0] == 1:
        return centroid[0]
    return centroid


def spectral_bandwidth(
    buf: AudioBuffer,
    window_size: int = 2048,
    hop_size: int | None = None,
) -> np.ndarray:
    """Weighted standard deviation around spectral centroid per frame.

    Returns Hz values shaped [num_frames] (mono) or [channels, num_frames].
    """
    mag, freqs, _ = _stft_magnitudes(buf, window_size, hop_size)
    total = np.sum(mag, axis=2)
    safe_total = np.where(total > 0, total, 1.0)
    weighted = np.sum(mag * freqs[np.newaxis, np.newaxis, :], axis=2)
    cent = np.where(total > 0, weighted / safe_total, 0.0)
    # Variance: sum(mag * (freq - centroid)^2) / sum(mag)
    diff_sq = (freqs[np.newaxis, np.newaxis, :] - cent[:, :, np.newaxis]) ** 2
    variance = np.sum(mag * diff_sq, axis=2)
    bw = np.where(total > 0, np.sqrt(variance / safe_total), 0.0).astype(np.float32)
    if bw.shape[0] == 1:
        return bw[0]
    return bw


def spectral_rolloff(
    buf: AudioBuffer,
    window_size: int = 2048,
    hop_size: int | None = None,
    percentile: float = 0.85,
) -> np.ndarray:
    """Frequency below which *percentile* of spectral energy lies.

    Parameters
    ----------
    percentile : float
        Energy fraction, 0.0--1.0 (e.g. 0.85 = 85th percentile).

    Returns
    -------
    np.ndarray
        Hz values shaped ``[num_frames]`` (mono) or ``[channels, num_frames]``.
    """
    mag, freqs, _ = _stft_magnitudes(buf, window_size, hop_size)
    energy = mag**2
    cumulative = np.cumsum(energy, axis=2)
    total = cumulative[:, :, -1:]
    threshold = percentile * total
    # For each frame, find the first bin where cumulative >= threshold
    above = cumulative >= threshold
    n_ch, n_frames, n_bins = mag.shape
    rolloff = np.zeros((n_ch, n_frames), dtype=np.float32)
    for ch in range(n_ch):
        for t in range(n_frames):
            idx = np.argmax(above[ch, t])
            rolloff[ch, t] = freqs[idx]
    if rolloff.shape[0] == 1:
        return rolloff[0]
    return rolloff


def spectral_flux(
    buf: AudioBuffer,
    window_size: int = 2048,
    hop_size: int | None = None,
    rectify: bool = False,
) -> np.ndarray:
    """L2 norm of frame-to-frame magnitude difference.

    If *rectify* is True, only positive changes are counted (half-wave rectification).
    Returns [num_frames] (mono) or [channels, num_frames].
    """
    mag, _, _ = _stft_magnitudes(buf, window_size, hop_size)
    n_ch, n_frames, _ = mag.shape
    flux = np.zeros((n_ch, n_frames), dtype=np.float32)
    for t in range(1, n_frames):
        diff = mag[:, t, :] - mag[:, t - 1, :]
        if rectify:
            diff = np.maximum(diff, 0.0)
        flux[:, t] = np.sqrt(np.sum(diff**2, axis=1))
    if flux.shape[0] == 1:
        return flux[0]
    return flux


def spectral_flatness_curve(
    buf: AudioBuffer,
    window_size: int = 2048,
    hop_size: int | None = None,
) -> np.ndarray:
    """Geometric/arithmetic mean ratio per frame (Wiener entropy).

    Range [0, 1]: 0=tonal, 1=noise-like.
    Returns [num_frames] (mono) or [channels, num_frames].
    """
    mag, _, _ = _stft_magnitudes(buf, window_size, hop_size)
    log_mag = np.log(mag + _DIV_EPS)
    geo_mean = np.exp(np.mean(log_mag, axis=2))
    arith_mean = np.mean(mag, axis=2)
    flatness = np.where(arith_mean > _DIV_EPS, geo_mean / arith_mean, 0.0).astype(
        np.float32
    )
    if flatness.shape[0] == 1:
        return flatness[0]
    return flatness


def chromagram(
    buf: AudioBuffer,
    window_size: int = 4096,
    hop_size: int | None = None,
    n_chroma: int = 12,
    tuning_hz: float = 440.0,
) -> np.ndarray:
    """Pitch class energy distribution.

    Maps FFT bins to chroma classes and sums magnitudes.
    Returns [n_chroma, num_frames] (mono) or [channels, n_chroma, num_frames].
    """
    mag, freqs, _ = _stft_magnitudes(buf, window_size, hop_size)
    n_ch, n_frames, n_bins = mag.shape

    # Build bin-to-chroma mapping
    chroma_map = np.full(n_bins, -1, dtype=np.int32)
    for b in range(1, n_bins):  # skip DC
        f = freqs[b]
        if f > 0:
            chroma_map[b] = int(round(n_chroma * np.log2(f / tuning_hz))) % n_chroma

    chroma_result = np.zeros((n_ch, n_chroma, n_frames), dtype=np.float32)
    for b in range(1, n_bins):
        pc = chroma_map[b]
        if pc >= 0:
            # mag[:, :, b] has shape [n_ch, n_frames]
            chroma_result[:, pc, :] += mag[:, :, b]

    if chroma_result.shape[0] == 1:
        return chroma_result[0]
    return chroma_result


# ---------------------------------------------------------------------------
# Pitch Detection (YIN algorithm)
# ---------------------------------------------------------------------------


def pitch_detect(
    buf: AudioBuffer,
    method: str = "yin",
    window_size: int = 2048,
    hop_size: int | None = None,
    fmin: float = 50.0,
    fmax: float = 2000.0,
    threshold: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect fundamental frequency using the YIN algorithm.

    Implements the YIN autocorrelation-based F0 estimator with cumulative
    mean normalized difference function and parabolic interpolation.

    Parameters
    ----------
    fmin : float
        Minimum detectable frequency in Hz, > 0. Typical: 50--200.
    fmax : float
        Maximum detectable frequency in Hz, > fmin. Typical: 2000--4000.
    threshold : float
        YIN aperiodicity threshold, 0.0--1.0. Lower = stricter voicing
        detection. Typical: 0.1--0.3.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (frequencies, confidences) where frequencies are F0 in Hz (0.0 where
        unvoiced) and confidences are 0.0--1.0.  Shape is ``[num_frames]``
        for mono or ``[channels, num_frames]`` for multi-channel.

    References
    ----------
    .. [1] A. de Cheveigne and H. Kawahara, "YIN, a fundamental frequency
       estimator for speech and music," J. Acoust. Soc. Am., vol. 111,
       no. 4, pp. 1917--1930, 2002.
    """
    if method != "yin":
        raise ValueError(f"Unknown pitch detection method: {method!r}")
    if hop_size is None:
        hop_size = window_size // 4

    sr = buf.sample_rate
    tau_min = max(1, int(sr / fmax))
    tau_max = min(window_size // 2, int(sr / fmin))

    n_frames_total = max(0, (buf.frames - window_size) // hop_size + 1)
    all_freqs = np.zeros((buf.channels, n_frames_total), dtype=np.float32)
    all_confs = np.zeros((buf.channels, n_frames_total), dtype=np.float32)

    for ch in range(buf.channels):
        x = buf.ensure_1d(ch)
        for t in range(n_frames_total):
            start = t * hop_size
            frame = x[start : start + window_size].astype(np.float64)

            # Difference function via autocorrelation
            W = len(frame)
            max_tau = min(tau_max + 1, W // 2)
            fft_size = 1
            while fft_size < 2 * W:
                fft_size *= 2
            X = np.fft.rfft(frame, n=fft_size)
            acf = np.fft.irfft(X * np.conj(X), n=fft_size)[:max_tau]
            r0 = acf[0]

            # Build shifted energy: sum(x[tau:W]^2)
            sq = frame**2
            cumsum_sq = np.cumsum(sq)
            energy_shifted = np.empty(max_tau, dtype=np.float64)
            energy_shifted[0] = cumsum_sq[W - 1]
            for tau in range(1, max_tau):
                energy_shifted[tau] = cumsum_sq[W - 1] - cumsum_sq[tau - 1]

            d = np.zeros(max_tau, dtype=np.float64)
            for tau in range(1, max_tau):
                d[tau] = r0 + energy_shifted[tau] - 2.0 * acf[tau]

            # Cumulative mean normalization
            d_prime = np.ones(max_tau, dtype=np.float64)
            running_sum = 0.0
            for tau in range(1, max_tau):
                running_sum += d[tau]
                if running_sum > 0:
                    d_prime[tau] = d[tau] * tau / running_sum
                else:
                    d_prime[tau] = 1.0

            # Find first tau in [tau_min, tau_max) where d'[tau] < threshold
            best_tau = 0
            best_val = 1.0
            search_start = max(tau_min, 1)
            search_end = min(tau_max + 1, max_tau)
            for tau in range(search_start, search_end):
                if d_prime[tau] < threshold:
                    best_tau = tau
                    best_val = d_prime[tau]
                    while tau + 1 < search_end and d_prime[tau + 1] < d_prime[tau]:
                        tau += 1
                        best_tau = tau
                        best_val = d_prime[tau]
                    break

            if best_tau == 0:
                if search_end > search_start:
                    best_tau = search_start + int(
                        np.argmin(d_prime[search_start:search_end])
                    )
                    best_val = d_prime[best_tau]
                    if best_val >= threshold:
                        all_freqs[ch, t] = 0.0
                        all_confs[ch, t] = 0.0
                        continue

            # Parabolic interpolation
            if 1 < best_tau < max_tau - 1:
                a = d_prime[best_tau - 1]
                b = d_prime[best_tau]
                c = d_prime[best_tau + 1]
                denom = 2.0 * (2.0 * b - a - c)
                if abs(denom) > 1e-10:
                    shift = (a - c) / denom
                    refined_tau = best_tau + shift
                else:
                    refined_tau = float(best_tau)
            else:
                refined_tau = float(best_tau)

            if refined_tau > 0:
                f0 = sr / refined_tau
                if fmin <= f0 <= fmax:
                    all_freqs[ch, t] = f0
                    all_confs[ch, t] = max(0.0, 1.0 - best_val)
                else:
                    all_freqs[ch, t] = 0.0
                    all_confs[ch, t] = 0.0
            else:
                all_freqs[ch, t] = 0.0
                all_confs[ch, t] = 0.0

    if buf.channels == 1:
        return all_freqs[0], all_confs[0]
    return all_freqs, all_confs


# ---------------------------------------------------------------------------
# Onset Detection
# ---------------------------------------------------------------------------


def onset_detect(
    buf: AudioBuffer,
    method: str = "spectral_flux",
    window_size: int = 2048,
    hop_size: int | None = None,
    threshold: float | None = None,
    backtrack: bool = False,
    pre_max: int = 3,
    post_max: int = 3,
    pre_avg: int = 3,
    post_avg: int = 3,
    wait: int = 5,
) -> np.ndarray:
    """Detect onsets in audio.

    Returns sample indices (int64) of detected onsets.
    Multi-channel input is mixed to mono first.
    """
    if method != "spectral_flux":
        raise ValueError(f"Unknown onset detection method: {method!r}")
    if hop_size is None:
        hop_size = window_size // 4

    # Mix to mono for detection
    if buf.channels > 1:
        mono_data = np.mean(buf.data, axis=0, keepdims=True).astype(np.float32)
        mono_buf = AudioBuffer(mono_data, sample_rate=buf.sample_rate)
    else:
        mono_buf = buf

    # Compute onset detection function (half-wave rectified spectral flux)
    odf = spectral_flux(
        mono_buf, window_size=window_size, hop_size=hop_size, rectify=True
    )

    if len(odf) == 0:
        return np.array([], dtype=np.int64)

    # Auto threshold
    if threshold is None:
        threshold = float(np.mean(odf) + 0.5 * np.std(odf))

    # Adaptive peak picking
    n = len(odf)
    onsets = []
    last_onset = -wait - 1

    for i in range(n):
        # Must have nonzero energy to be an onset
        if odf[i] <= 0:
            continue
        if i - last_onset < wait:
            continue
        win_start = max(0, i - pre_max)
        win_end = min(n, i + post_max + 1)
        if odf[i] < np.max(odf[win_start:win_end]):
            continue
        avg_start = max(0, i - pre_avg)
        avg_end = min(n, i + post_avg + 1)
        local_avg = np.mean(odf[avg_start:avg_end])
        if odf[i] < local_avg + threshold:
            continue
        onsets.append(i)
        last_onset = i

    # Convert STFT frame indices to sample indices
    onset_samples = np.array([i * hop_size for i in onsets], dtype=np.int64)

    # Optional backtrack: search backward for nearest energy minimum
    if backtrack and len(onset_samples) > 0:
        energy = mono_buf.data[0] ** 2
        kernel_size = min(hop_size, 64)
        if kernel_size > 1:
            kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
            smooth_energy = np.convolve(energy, kernel, mode="same")
        else:
            smooth_energy = energy
        for idx in range(len(onset_samples)):
            pos = onset_samples[idx]
            search_start = max(0, pos - hop_size)
            if search_start < pos:
                min_pos = search_start + int(np.argmin(smooth_energy[search_start:pos]))
                onset_samples[idx] = min_pos

    return onset_samples


# ---------------------------------------------------------------------------
# Resample (madronalib-based, power-of-2 preferred)
# ---------------------------------------------------------------------------


def resample(buf: AudioBuffer, target_sr: float) -> AudioBuffer:
    """Resample audio to a different sample rate.

    Uses madronalib Downsampler/Upsampler for power-of-2 ratios
    (higher quality), and linear interpolation for arbitrary ratios.
    """
    if target_sr == buf.sample_rate:
        return buf.copy()

    ratio = target_sr / buf.sample_rate

    # Check if ratio is a power of 2 (up or down)
    octaves = None
    if ratio > 1.0:
        r = ratio
        o = 0
        while r > 1.0 and r == int(r) and int(r) & (int(r) - 1) == 0:
            r /= 2.0
            o += 1
        if r == 1.0 and o > 0:
            octaves = o
    elif ratio < 1.0:
        r = 1.0 / ratio
        o = 0
        while r > 1.0 and r == int(r) and int(r) & (int(r) - 1) == 0:
            r /= 2.0
            o += 1
        if r == 1.0 and o > 0:
            octaves = -o

    out_channels = []
    for ch in range(buf.channels):
        ch_data = np.ascontiguousarray(buf.data[ch], dtype=np.float32)

        if octaves is not None and octaves > 0:
            # Upsample by power of 2
            remainder = len(ch_data) % 64
            if remainder != 0:
                ch_data = np.pad(ch_data, (0, 64 - remainder), mode="constant")
            up = _madronalib.resampling.Upsampler(octaves)
            resampled = np.asarray(up.process(ch_data, octaves), dtype=np.float32)
            # Trim to expected length
            expected_len = buf.frames * (1 << octaves)
            resampled = resampled[:expected_len]
            out_channels.append(resampled)
        elif octaves is not None and octaves < 0:
            # Downsample by power of 2
            abs_oct = -octaves
            remainder = len(ch_data) % 64
            if remainder != 0:
                ch_data = np.pad(ch_data, (0, 64 - remainder), mode="constant")
            down = _madronalib.resampling.Downsampler(abs_oct)
            resampled = np.asarray(down.process(ch_data), dtype=np.float32)
            # Trim to expected length
            expected_len = buf.frames // (1 << abs_oct)
            resampled = resampled[:expected_len]
            out_channels.append(resampled)
        else:
            # Arbitrary ratio: linear interpolation
            target_frames = max(1, round(buf.frames * ratio))
            old_x = np.linspace(0.0, 1.0, buf.frames, dtype=np.float64)
            new_x = np.linspace(0.0, 1.0, target_frames, dtype=np.float64)
            interped = np.interp(new_x, old_x, ch_data.astype(np.float64))
            out_channels.append(interped.astype(np.float32))

    out = np.stack(out_channels)
    return AudioBuffer(
        out.astype(np.float32),
        sample_rate=target_sr,
        channel_layout=buf.channel_layout,
        label=buf.label,
    )


# ---------------------------------------------------------------------------
# Resampling (FFT-based)
# ---------------------------------------------------------------------------


def resample_fft(buf: AudioBuffer, target_sr: float) -> AudioBuffer:
    """Resample audio to a different sample rate using FFT-based method.

    Parameters
    ----------
    buf : AudioBuffer
        Input audio.
    target_sr : float
        Target sample rate in Hz.

    Returns
    -------
    AudioBuffer
        Resampled audio with sample_rate set to *target_sr*.
    """
    if target_sr == buf.sample_rate:
        return buf.copy()
    if target_sr <= 0:
        raise ValueError(f"target_sr must be positive, got {target_sr}")

    original_len = buf.frames
    target_len = max(1, round(original_len * target_sr / buf.sample_rate))

    out = np.zeros((buf.channels, target_len), dtype=np.float32)
    for ch in range(buf.channels):
        x = buf.ensure_1d(ch).astype(np.float64)
        X = np.fft.rfft(x)

        new_n_bins = target_len // 2 + 1
        old_n_bins = len(X)

        if new_n_bins > old_n_bins:
            # Upsampling: zero-pad spectrum
            new_X = np.zeros(new_n_bins, dtype=np.complex128)
            new_X[:old_n_bins] = X
        else:
            # Downsampling: truncate spectrum
            new_X = X[:new_n_bins].copy()

        result = np.fft.irfft(new_X, n=target_len)
        result *= target_len / original_len
        out[ch] = result.astype(np.float32)

    return AudioBuffer(
        out,
        sample_rate=target_sr,
        channel_layout=buf.channel_layout,
        label=buf.label,
    )


# ---------------------------------------------------------------------------
# GCC-PHAT delay estimation
# ---------------------------------------------------------------------------


def gcc_phat(
    buf: AudioBuffer,
    ref: AudioBuffer,
    sample_rate: float | None = None,
) -> tuple[float, np.ndarray]:
    """Estimate time delay between two signals using GCC-PHAT.

    Implements the Generalized Cross-Correlation with Phase Transform
    (GCC-PHAT) method for robust time-delay estimation.

    Parameters
    ----------
    buf : AudioBuffer
        Signal of interest (mono or mixed to mono).
    ref : AudioBuffer
        Reference signal (mono or mixed to mono).
    sample_rate : float or None
        Override sample rate for delay computation. Defaults to ``buf.sample_rate``.

    Returns
    -------
    tuple[float, np.ndarray]
        (delay_seconds, correlation) -- delay in seconds (positive means *buf*
        is delayed relative to *ref*), and the full GCC-PHAT correlation array.

    References
    ----------
    .. [1] C. Knapp and G. Carter, "The generalized correlation method for
       estimation of time delay," IEEE Trans. Acoust., Speech, Signal Process.,
       vol. 24, no. 4, pp. 320--327, 1976.
    """
    sr = sample_rate if sample_rate is not None else buf.sample_rate

    a = (
        np.mean(buf.data, axis=0).astype(np.float64)
        if buf.channels > 1
        else buf.data[0].astype(np.float64)
    )
    b = (
        np.mean(ref.data, axis=0).astype(np.float64)
        if ref.channels > 1
        else ref.data[0].astype(np.float64)
    )

    n = len(a) + len(b) - 1
    # Next power of 2
    fft_size = 1
    while fft_size < n:
        fft_size *= 2

    A = np.fft.rfft(a, n=fft_size)
    B = np.fft.rfft(b, n=fft_size)

    # Cross-spectrum with phase transform (PHAT) weighting
    cross = A * np.conj(B)
    magnitude = np.abs(cross)
    cross_phat = cross / (magnitude + _DIV_EPS)

    corr = np.fft.irfft(cross_phat, n=fft_size)

    # Find the peak — consider both positive and negative delays
    # Positive lags: corr[0:len_a], negative lags: corr[fft_size-len_b+1:]
    max_lag = min(len(a), len(b))
    # Build candidate region: negative lags (buf leads) and positive lags (buf lags)
    candidates = np.concatenate([corr[:max_lag], corr[fft_size - max_lag + 1 :]])
    peak_idx = np.argmax(candidates)

    if peak_idx < max_lag:
        delay_samples = peak_idx
    else:
        delay_samples = peak_idx - len(candidates)

    delay_seconds = float(delay_samples) / sr
    return delay_seconds, corr[:n].astype(np.float32)

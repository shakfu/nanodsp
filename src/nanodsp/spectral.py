"""STFT, spectral transforms, and EQ matching."""

from __future__ import annotations

from typing import Callable

import numpy as np

from nanodsp.buffer import AudioBuffer
from nanodsp._helpers import Spectrogram
from nanodsp._core import fft


# ---------------------------------------------------------------------------
# STFT functions
# ---------------------------------------------------------------------------


_WINDOW_FUNCTIONS: dict[str, Callable] = {
    "hann": np.hanning,
    "hamming": np.hamming,
    "blackman": np.blackman,
    "bartlett": np.bartlett,
    "ones": np.ones,
    "rectangular": np.ones,
}


def stft(
    buf: AudioBuffer,
    window_size: int = 2048,
    hop_size: int | None = None,
    window: str = "hann",
) -> Spectrogram:
    """Short-time Fourier transform using windowed RealFFT + overlap.

    Parameters
    ----------
    buf : AudioBuffer
        Input audio.
    window_size : int
        Analysis window length in samples.
    hop_size : int or None
        Hop between successive windows.  Defaults to ``window_size // 4``.
    window : str
        Window function name. One of ``"hann"`` (default), ``"hamming"``,
        ``"blackman"``, ``"bartlett"``, ``"rectangular"``/``"ones"``.

    Returns
    -------
    Spectrogram
        Complex64 data shaped ``[channels, num_stft_frames, fft_size // 2]``.
    """
    if hop_size is None:
        hop_size = window_size // 4

    fft_size = fft.RealFFT.fast_size_above(window_size)
    rfft_obj = fft.RealFFT(fft_size)
    bins = fft_size // 2

    win_fn = _WINDOW_FUNCTIONS.get(window.lower())
    if win_fn is None:
        raise ValueError(
            f"Unknown window {window!r}, valid: {list(_WINDOW_FUNCTIONS.keys())}"
        )
    win = win_fn(window_size).astype(np.float32)

    n_frames = buf.frames
    num_stft_frames = max(0, (n_frames - window_size) // hop_size + 1)

    out = np.zeros((buf.channels, num_stft_frames, bins), dtype=np.complex64)

    for ch in range(buf.channels):
        channel_data = buf.ensure_1d(ch)
        for t in range(num_stft_frames):
            start = t * hop_size
            segment = channel_data[start : start + window_size] * win
            if window_size < fft_size:
                padded = np.zeros(fft_size, dtype=np.float32)
                padded[:window_size] = segment
                segment = padded
            out[ch, t, :] = rfft_obj.fft(segment)

    return Spectrogram(
        data=out,
        window_size=window_size,
        hop_size=hop_size,
        fft_size=fft_size,
        sample_rate=buf.sample_rate,
        original_frames=n_frames,
    )


def istft(spec: Spectrogram, window: str = "hann") -> AudioBuffer:
    """Inverse STFT via overlap-add with COLA normalization.

    Parameters
    ----------
    spec : Spectrogram
        Output from :func:`stft`.
    window : str
        Window function name (must match the window used in :func:`stft`).

    Returns
    -------
    AudioBuffer
        Reconstructed audio, trimmed to the original frame count.
    """
    window_size = spec.window_size
    hop_size = spec.hop_size
    fft_size = spec.fft_size

    rfft_obj = fft.RealFFT(fft_size)
    win_fn = _WINDOW_FUNCTIONS.get(window.lower())
    if win_fn is None:
        raise ValueError(
            f"Unknown window {window!r}, valid: {list(_WINDOW_FUNCTIONS.keys())}"
        )
    win = win_fn(window_size).astype(np.float32)

    out_len = (spec.num_frames - 1) * hop_size + window_size
    out = np.zeros((spec.channels, out_len), dtype=np.float32)
    win_sum = np.zeros(out_len, dtype=np.float32)

    for ch in range(spec.channels):
        for t in range(spec.num_frames):
            full = rfft_obj.ifft(np.ascontiguousarray(spec.data[ch, t, :]))
            # ifft is unscaled -- divide by fft_size
            frame = (np.asarray(full[:window_size], dtype=np.float32) / fft_size) * win
            start = t * hop_size
            out[ch, start : start + window_size] += frame

    # Window normalization (sum of squared windows at each position)
    for t in range(spec.num_frames):
        start = t * hop_size
        win_sum[start : start + window_size] += win**2

    # Avoid division by zero at edges
    win_sum = np.maximum(win_sum, 1e-8)
    out /= win_sum[np.newaxis, :]

    # Trim to original length
    trim = min(spec.original_frames, out.shape[1])
    out = out[:, :trim]

    return AudioBuffer(out, sample_rate=spec.sample_rate)


# ---------------------------------------------------------------------------
# Spectral utility functions
# ---------------------------------------------------------------------------


def magnitude(spec: Spectrogram) -> np.ndarray:
    """Return magnitude of spectral data.

    Parameters
    ----------
    spec : Spectrogram
        Input spectrogram.

    Returns
    -------
    np.ndarray
        float32 array shaped ``[channels, num_frames, bins]``.
    """
    return np.abs(spec.data).astype(np.float32)


def phase(spec: Spectrogram) -> np.ndarray:
    """Return phase angle of spectral data.

    Parameters
    ----------
    spec : Spectrogram
        Input spectrogram.

    Returns
    -------
    np.ndarray
        float32 array shaped ``[channels, num_frames, bins]`` in radians.
    """
    return np.angle(spec.data).astype(np.float32)


def from_polar(mag: np.ndarray, ph: np.ndarray, spec: Spectrogram) -> Spectrogram:
    """Reconstruct a Spectrogram from magnitude and phase arrays.

    Parameters
    ----------
    mag : np.ndarray
        Magnitude array, broadcastable to ``spec.data.shape``.
    ph : np.ndarray
        Phase array in radians, broadcastable to ``spec.data.shape``.
    spec : Spectrogram
        Reference spectrogram whose metadata is copied.

    Returns
    -------
    Spectrogram
        New spectrogram with ``mag * exp(j * ph)`` as data.
    """
    data = (mag * np.exp(1j * ph)).astype(np.complex64)
    return Spectrogram(
        data=data,
        window_size=spec.window_size,
        hop_size=spec.hop_size,
        fft_size=spec.fft_size,
        sample_rate=spec.sample_rate,
        original_frames=spec.original_frames,
    )


def apply_mask(spec: Spectrogram, mask: np.ndarray) -> Spectrogram:
    """Multiply spectral data by a real-valued mask.

    Parameters
    ----------
    spec : Spectrogram
        Input spectrogram.
    mask : np.ndarray
        Real-valued mask broadcastable to ``[channels, num_frames, bins]``.

    Returns
    -------
    Spectrogram
        New spectrogram with masked data.

    Raises
    ------
    ValueError
        If *mask* cannot be broadcast to the spectrogram shape.
    """
    try:
        result = spec.data * mask
    except ValueError:
        raise ValueError(
            f"Mask shape {mask.shape} is not broadcastable to "
            f"spectrogram shape {spec.data.shape}"
        )
    return Spectrogram(
        data=result.astype(np.complex64),
        window_size=spec.window_size,
        hop_size=spec.hop_size,
        fft_size=spec.fft_size,
        sample_rate=spec.sample_rate,
        original_frames=spec.original_frames,
    )


def spectral_gate(
    spec: Spectrogram,
    threshold_db: float = -40.0,
    noise_floor_db: float = -80.0,
) -> Spectrogram:
    """Gate spectral bins below a dB threshold.

    Bins whose magnitude falls below *threshold_db* are attenuated to
    *noise_floor_db* rather than zeroed, reducing musical noise artifacts.

    Parameters
    ----------
    spec : Spectrogram
        Input spectrogram.
    threshold_db : float
        Magnitude threshold in dB.  Bins at or above this pass through.
    noise_floor_db : float
        Attenuation applied to bins below the threshold, in dB relative to
        the threshold.

    Returns
    -------
    Spectrogram
        Gated spectrogram.
    """
    eps = 1e-10
    mag = np.abs(spec.data)
    mag_db = 20.0 * np.log10(mag + eps)
    attenuation = 10.0 ** ((noise_floor_db - threshold_db) / 20.0)
    mask = np.where(mag_db >= threshold_db, 1.0, attenuation).astype(np.float32)
    return apply_mask(spec, mask)


def spectral_emphasis(
    spec: Spectrogram,
    low_db: float = 0.0,
    high_db: float = 0.0,
) -> Spectrogram:
    """Apply a linear dB tilt across frequency bins.

    Gain varies linearly from *low_db* at DC to *high_db* at Nyquist.

    Parameters
    ----------
    spec : Spectrogram
        Input spectrogram.
    low_db : float
        Gain at DC in dB.
    high_db : float
        Gain at Nyquist in dB.

    Returns
    -------
    Spectrogram
        Emphasized spectrogram.
    """
    n_bins = spec.bins
    db_ramp = np.linspace(low_db, high_db, n_bins, dtype=np.float32)
    mask = 10.0 ** (db_ramp / 20.0)
    return apply_mask(spec, mask)


def bin_freq(spec: Spectrogram, bin_index: int) -> float:
    """Return the center frequency in Hz of a given FFT bin.

    Parameters
    ----------
    spec : Spectrogram
        Reference spectrogram.
    bin_index : int
        Bin index (0 = DC).

    Returns
    -------
    float
        Frequency in Hz.
    """
    return float(bin_index * spec.sample_rate / spec.fft_size)


def freq_to_bin(spec: Spectrogram, freq_hz: float) -> int:
    """Return the nearest FFT bin for a given frequency.

    Parameters
    ----------
    spec : Spectrogram
        Reference spectrogram.
    freq_hz : float
        Frequency in Hz.

    Returns
    -------
    int
        Nearest bin index, clamped to ``[0, bins - 1]``.

    Raises
    ------
    ValueError
        If *freq_hz* is negative or >= Nyquist.
    """
    nyquist = spec.sample_rate / 2.0
    if freq_hz < 0:
        raise ValueError(f"Frequency must be non-negative, got {freq_hz}")
    if freq_hz >= nyquist:
        raise ValueError(f"Frequency {freq_hz} Hz >= Nyquist ({nyquist} Hz)")
    exact = freq_hz * spec.fft_size / spec.sample_rate
    return int(np.clip(round(exact), 0, spec.bins - 1))


# ---------------------------------------------------------------------------
# Spectral transforms
# ---------------------------------------------------------------------------


def time_stretch(spec: Spectrogram, rate: float) -> Spectrogram:
    """Phase-vocoder time stretch.

    Parameters
    ----------
    spec : Spectrogram
        Input spectrogram.
    rate : float
        Stretch rate.  ``rate > 1`` makes audio shorter (faster),
        ``rate < 1`` makes audio longer (slower).

    Returns
    -------
    Spectrogram
        Time-stretched spectrogram with updated ``original_frames``.

    Raises
    ------
    ValueError
        If *rate* <= 0.
    """
    if rate <= 0:
        raise ValueError(f"Rate must be > 0, got {rate}")

    n_ch, n_frames, n_bins = spec.data.shape
    new_frames = max(1, round(n_frames / rate))
    hop = spec.hop_size

    # Expected phase advance per hop for each bin
    omega = 2.0 * np.pi * np.arange(n_bins) * hop / spec.fft_size

    out = np.zeros((n_ch, new_frames, n_bins), dtype=np.complex64)

    input_mag = np.abs(spec.data)
    input_phase = np.angle(spec.data)

    for ch in range(n_ch):
        # Initialize phase accumulator from first frame
        phase_acc = input_phase[ch, 0].copy()

        for t_out in range(new_frames):
            t_in = t_out * rate
            t_floor = int(t_in)
            frac = t_in - t_floor

            # Clamp to valid input range
            t0 = min(t_floor, n_frames - 1)
            t1 = min(t_floor + 1, n_frames - 1)

            # Interpolate magnitude
            mag = (1.0 - frac) * input_mag[ch, t0] + frac * input_mag[ch, t1]

            if t_out == 0:
                phase_acc = input_phase[ch, t0].copy()
            else:
                # Instantaneous frequency from input phase difference
                if t0 < n_frames - 1:
                    dphi = input_phase[ch, t0 + 1] - input_phase[ch, t0]
                else:
                    dphi = np.zeros(n_bins)
                # Deviation from expected phase advance
                deviation = dphi - omega
                # Wrap to [-pi, pi]
                deviation = deviation - 2.0 * np.pi * np.round(
                    deviation / (2.0 * np.pi)
                )
                # True instantaneous frequency
                inst_freq = omega + deviation
                # Accumulate phase at output hop rate
                phase_acc += inst_freq

            out[ch, t_out] = mag * np.exp(1j * phase_acc)

    new_original = max(1, round(spec.original_frames / rate))
    return Spectrogram(
        data=out,
        window_size=spec.window_size,
        hop_size=spec.hop_size,
        fft_size=spec.fft_size,
        sample_rate=spec.sample_rate,
        original_frames=new_original,
    )


def phase_lock(spec: Spectrogram) -> Spectrogram:
    """Identity phase-locking (Laroche & Dolson 1999).

    Finds spectral peaks in each frame and propagates their phase to
    neighboring bins, reducing phasiness.

    Parameters
    ----------
    spec : Spectrogram
        Input spectrogram.

    Returns
    -------
    Spectrogram
        Phase-locked spectrogram with identical magnitudes.
    """
    n_ch, n_frames, n_bins = spec.data.shape
    hop = spec.hop_size
    fft_size = spec.fft_size

    input_mag = np.abs(spec.data)
    input_phase = np.angle(spec.data)
    out_phase = input_phase.copy()

    all_bins = np.arange(n_bins)
    phase_scale = 2.0 * np.pi * hop / fft_size

    for ch in range(n_ch):
        for t in range(n_frames):
            mag = input_mag[ch, t]

            # Vectorized peak detection: mag >= left neighbor AND mag >= right neighbor
            left = np.empty(n_bins, dtype=np.float32)
            left[0] = -1.0
            left[1:] = mag[:-1]
            right = np.empty(n_bins, dtype=np.float32)
            right[-1] = -1.0
            right[:-1] = mag[1:]
            peak_mask = (mag >= left) & (mag >= right)
            peaks = np.nonzero(peak_mask)[0]

            if len(peaks) == 0:
                continue

            # Nearest-peak assignment via searchsorted + left/right comparison
            idx = np.searchsorted(peaks, all_bins, side="left")
            idx = np.clip(idx, 0, len(peaks) - 1)
            # Compare candidate on the right with candidate on the left
            nearest = peaks[idx]
            left_idx = np.clip(idx - 1, 0, len(peaks) - 1)
            left_candidate = peaks[left_idx]
            use_left = np.abs(all_bins - left_candidate) < np.abs(all_bins - nearest)
            nearest = np.where(use_left, left_candidate, nearest)

            # Vectorized phase propagation
            offset = all_bins - nearest
            out_phase[ch, t, :] = input_phase[ch, t, nearest] + offset * (
                phase_scale * nearest
            )

    out_data = (input_mag * np.exp(1j * out_phase)).astype(np.complex64)
    return Spectrogram(
        data=out_data,
        window_size=spec.window_size,
        hop_size=spec.hop_size,
        fft_size=spec.fft_size,
        sample_rate=spec.sample_rate,
        original_frames=spec.original_frames,
    )


def spectral_freeze(
    spec: Spectrogram,
    frame_index: int = 0,
    num_frames: int | None = None,
) -> Spectrogram:
    """Repeat a single STFT frame to produce a static ("frozen") texture.

    Parameters
    ----------
    spec : Spectrogram
        Input spectrogram.
    frame_index : int
        Index of the frame to freeze.  Negative indices are supported.
    num_frames : int or None
        Number of output STFT frames.  Defaults to ``spec.num_frames``.

    Returns
    -------
    Spectrogram
        Spectrogram with the chosen frame repeated *num_frames* times.

    Raises
    ------
    IndexError
        If *frame_index* is out of range.
    """
    if num_frames is None:
        num_frames = spec.num_frames
    if frame_index < -spec.num_frames or frame_index >= spec.num_frames:
        raise IndexError(
            f"frame_index {frame_index} out of range for {spec.num_frames} frames"
        )
    frame = spec.data[:, frame_index, :]  # [n_ch, n_bins]
    data = np.tile(frame[:, None, :], (1, num_frames, 1))
    original_frames = (num_frames - 1) * spec.hop_size + spec.window_size
    return Spectrogram(
        data=data.astype(np.complex64),
        window_size=spec.window_size,
        hop_size=spec.hop_size,
        fft_size=spec.fft_size,
        sample_rate=spec.sample_rate,
        original_frames=original_frames,
    )


def spectral_morph(
    spec_a: Spectrogram,
    spec_b: Spectrogram,
    mix: float | np.ndarray = 0.5,
) -> Spectrogram:
    """Interpolate between two spectrograms in the polar domain.

    Magnitudes are interpolated linearly; phases use shortest-arc circular
    interpolation, avoiding the cancellation artefacts of complex-valued
    lerp.

    Parameters
    ----------
    spec_a, spec_b : Spectrogram
        Input spectrograms.  Must share ``fft_size``, ``window_size``,
        ``hop_size``, and channel count.  If frame counts differ the
        shorter length is used.
    mix : float or np.ndarray
        Blend factor.  ``0.0`` returns *spec_a*, ``1.0`` returns *spec_b*.
        May be a scalar or an array broadcastable to
        ``[channels, num_frames, bins]`` for time-varying morphing.

    Returns
    -------
    Spectrogram

    Raises
    ------
    ValueError
        If the two spectrograms have incompatible parameters.
    """
    if spec_a.fft_size != spec_b.fft_size:
        raise ValueError(f"fft_size mismatch: {spec_a.fft_size} vs {spec_b.fft_size}")
    if spec_a.window_size != spec_b.window_size:
        raise ValueError(
            f"window_size mismatch: {spec_a.window_size} vs {spec_b.window_size}"
        )
    if spec_a.hop_size != spec_b.hop_size:
        raise ValueError(f"hop_size mismatch: {spec_a.hop_size} vs {spec_b.hop_size}")
    if spec_a.channels != spec_b.channels:
        raise ValueError(
            f"channel count mismatch: {spec_a.channels} vs {spec_b.channels}"
        )

    n_frames = min(spec_a.num_frames, spec_b.num_frames)
    data_a = spec_a.data[:, :n_frames, :]
    data_b = spec_b.data[:, :n_frames, :]

    mag_a = np.abs(data_a)
    mag_b = np.abs(data_b)
    phase_a = np.angle(data_a)
    phase_b = np.angle(data_b)

    mix = np.asarray(mix, dtype=np.float32)

    # Magnitude: linear interpolation
    mag = (1.0 - mix) * mag_a + mix * mag_b

    # Phase: shortest-arc circular interpolation
    phase_diff = phase_b - phase_a
    phase_diff = phase_diff - 2.0 * np.pi * np.round(phase_diff / (2.0 * np.pi))
    ph = phase_a + mix * phase_diff

    data = (mag * np.exp(1j * ph)).astype(np.complex64)
    original_frames = min(spec_a.original_frames, spec_b.original_frames)
    return Spectrogram(
        data=data,
        window_size=spec_a.window_size,
        hop_size=spec_a.hop_size,
        fft_size=spec_a.fft_size,
        sample_rate=spec_a.sample_rate,
        original_frames=original_frames,
    )


def pitch_shift_spectral(
    buf: AudioBuffer,
    semitones: float,
    window_size: int = 2048,
    hop_size: int | None = None,
) -> AudioBuffer:
    """Pitch-shift audio via phase vocoder + resampling.

    Combines :func:`time_stretch` with linear resampling so that pitch
    changes without altering duration.

    Parameters
    ----------
    buf : AudioBuffer
        Input audio.
    semitones : float
        Pitch shift in semitones.  Positive = higher, negative = lower.
    window_size : int
        STFT analysis window size.
    hop_size : int or None
        STFT hop size.  Defaults to ``window_size // 4``.

    Returns
    -------
    AudioBuffer
        Pitch-shifted audio with the same duration and sample rate.
    """
    if semitones == 0.0:
        return AudioBuffer(
            buf.data.copy(),
            sample_rate=buf.sample_rate,
            channel_layout=buf.channel_layout,
            label=buf.label,
        )

    alpha = 2.0 ** (semitones / 12.0)
    # Time-stretch to compensate for the resampling that follows
    stretch_rate = 1.0 / alpha

    spec = stft(buf, window_size=window_size, hop_size=hop_size)
    stretched = time_stretch(spec, stretch_rate)
    audio = istft(stretched)

    # Resample to original length using linear interpolation
    target_frames = buf.frames
    if audio.frames == target_frames:
        resampled = audio.data
    else:
        old_x = np.linspace(0.0, 1.0, audio.frames, dtype=np.float64)
        new_x = np.linspace(0.0, 1.0, target_frames, dtype=np.float64)
        resampled = np.empty((audio.channels, target_frames), dtype=np.float32)
        for ch in range(audio.channels):
            resampled[ch] = np.interp(new_x, old_x, audio.data[ch]).astype(np.float32)

    return AudioBuffer(
        resampled,
        sample_rate=buf.sample_rate,
        channel_layout=buf.channel_layout,
        label=buf.label,
    )


def spectral_denoise(
    spec: Spectrogram,
    noise_frames: int = 10,
    reduction_db: float = -20.0,
    smoothing: int = 0,
) -> Spectrogram:
    """Spectral noise reduction using a profile estimated from leading frames.

    Computes the mean magnitude of the first *noise_frames* STFT frames
    per bin, then attenuates bins whose magnitude falls at or below that
    noise floor.  The leading frames should ideally contain only noise.

    Parameters
    ----------
    spec : Spectrogram
        Input spectrogram.
    noise_frames : int
        Number of leading STFT frames used to build the noise profile.
    reduction_db : float
        Attenuation in dB applied to bins at or below the noise floor.
        More negative = more aggressive reduction.
    smoothing : int
        If > 0, apply a moving-average of this width (in bins) to the
        noise profile, reducing musical-noise artefacts.

    Returns
    -------
    Spectrogram
        Denoised spectrogram.

    Raises
    ------
    ValueError
        If *noise_frames* < 1 or exceeds the number of available frames.
    """
    if noise_frames < 1:
        raise ValueError(f"noise_frames must be >= 1, got {noise_frames}")
    if noise_frames > spec.num_frames:
        raise ValueError(
            f"noise_frames ({noise_frames}) exceeds available frames "
            f"({spec.num_frames})"
        )

    # Mean magnitude across noise frames, per channel per bin: [ch, bins]
    noise_mag = np.mean(np.abs(spec.data[:, :noise_frames, :]), axis=1)

    if smoothing > 0:
        kernel = np.ones(smoothing, dtype=np.float32) / smoothing
        smoothed = np.empty_like(noise_mag)
        for ch in range(spec.channels):
            smoothed[ch] = np.convolve(noise_mag[ch], kernel, mode="same")
        noise_mag = smoothed

    # Gate: pass bins above noise floor, attenuate the rest
    sig_mag = np.abs(spec.data)
    noise_threshold = noise_mag[:, None, :]  # broadcast [ch, 1, bins]
    attenuation = 10.0 ** (reduction_db / 20.0)
    mask = np.where(sig_mag > noise_threshold, 1.0, attenuation).astype(np.float32)
    return apply_mask(spec, mask)


# ---------------------------------------------------------------------------
# EQ matching
# ---------------------------------------------------------------------------


def eq_match(
    buf: AudioBuffer,
    target: AudioBuffer,
    window_size: int = 4096,
    smoothing: int = 0,
) -> AudioBuffer:
    """Match the spectral envelope of *buf* to *target*.

    Parameters
    ----------
    buf : AudioBuffer
        Source audio to be adjusted.
    target : AudioBuffer
        Reference audio whose spectral envelope is matched.
    window_size : int
        STFT window size.
    smoothing : int
        If > 0, apply a moving-average of this width (in bins) to the
        correction curve.

    Raises
    ------
    ValueError
        If sample rates or channel counts differ.
    """
    if buf.sample_rate != target.sample_rate:
        raise ValueError(
            f"Sample rate mismatch: buf={buf.sample_rate}, target={target.sample_rate}"
        )
    if buf.channels != target.channels:
        raise ValueError(
            f"Channel count mismatch: buf has {buf.channels}, target has "
            f"{target.channels}. Convert to matching layout first "
            f"(e.g. to_mono())."
        )

    src_spec = stft(buf, window_size=window_size)
    tgt_spec = stft(target, window_size=window_size)

    # Mean magnitude across all channels and frames -> [bins]
    src_avg = np.mean(np.abs(src_spec.data), axis=(0, 1))
    tgt_avg = np.mean(np.abs(tgt_spec.data), axis=(0, 1))

    eps = 1e-10
    correction = tgt_avg / (src_avg + eps)
    correction = np.clip(correction, 0.0, 100.0).astype(np.float32)

    if smoothing > 0:
        kernel = np.ones(smoothing, dtype=np.float32) / smoothing
        correction = np.convolve(correction, kernel, mode="same").astype(np.float32)

    # Apply correction as a 1D mask [bins] -- broadcasts across channels/frames
    corrected = apply_mask(src_spec, correction)
    result = istft(corrected)

    # Trim to original length
    if result.frames > buf.frames:
        result = AudioBuffer(
            result.data[:, : buf.frames],
            sample_rate=buf.sample_rate,
            channel_layout=buf.channel_layout,
            label=buf.label,
        )

    return AudioBuffer(
        result.data,
        sample_rate=buf.sample_rate,
        channel_layout=buf.channel_layout,
        label=buf.label,
    )

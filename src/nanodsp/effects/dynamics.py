"""Dynamics -- compression, limiting, noise gate, AGC."""

from __future__ import annotations

import numpy as np

from ..buffer import AudioBuffer
from .._helpers import _process_per_channel, _dsy_dyn


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
    """Apply compression per channel.

    Parameters
    ----------
    buf : AudioBuffer
        Input audio.
    ratio : float
        Compression ratio (e.g. 4.0 = 4:1).
    threshold : float
        Threshold in dB.
    attack : float
        Attack time in seconds.
    release : float
        Release time in seconds.
    makeup : float
        Makeup gain in dB.
    auto_makeup : bool
        If True, automatically compensate for gain reduction.

    Returns
    -------
    AudioBuffer
        Compressed audio.
    """

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
    """Apply limiter per channel.

    Parameters
    ----------
    buf : AudioBuffer
        Input audio.
    pre_gain : float
        Gain applied before limiting.

    Returns
    -------
    AudioBuffer
        Limited audio.
    """

    def _process(x):
        lm = _dsy_dyn.Limiter()
        lm.init()
        return lm.process(x, pre_gain)

    return _process_per_channel(buf, _process)


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

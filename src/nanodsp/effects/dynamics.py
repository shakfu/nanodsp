"""Dynamics -- compression, limiting, noise gate, AGC, sidechain, transient shaper, lookahead limiter."""

from __future__ import annotations

import numpy as np

from ..buffer import AudioBuffer
from .._helpers import _process_per_channel, _dsy_dyn
from .._core import fxdsp as _fxdsp


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
        Compression ratio, >= 1.0 (e.g. 4.0 = 4:1). Typical range: 2--20.
    threshold : float
        Threshold in dB, typically -60 to 0.
    attack : float
        Attack time in seconds, > 0. Typical range: 0.001--0.1.
    release : float
        Release time in seconds, > 0. Typical range: 0.01--1.0.
    makeup : float
        Makeup gain in dB. Typical range: 0--30.
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
        Linear gain applied before limiting, > 0. 1.0 = unity.

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
# Sidechain compression
# ---------------------------------------------------------------------------


def sidechain_compress(
    buf: AudioBuffer,
    sidechain: AudioBuffer,
    ratio: float = 4.0,
    threshold: float = -20.0,
    attack: float = 0.01,
    release: float = 0.1,
) -> AudioBuffer:
    """Compress *buf* using the envelope of *sidechain* as the detector.

    The gain reduction is computed from the sidechain signal but applied
    to *buf*.  Common use: duck a bass synth under a kick drum.

    Parameters
    ----------
    sidechain : AudioBuffer
        Signal whose level drives the compressor.  Must have the same
        frame count as *buf*.
    ratio : float
        Compression ratio, >= 1.0. Typical: 2--20.
    threshold : float
        Threshold in dB, typically -60 to 0.
    attack : float
        Attack time in seconds, > 0. Typical: 0.001--0.05.
    release : float
        Release time in seconds, > 0. Typical: 0.01--0.5.
    """
    if buf.frames != sidechain.frames:
        raise ValueError(
            f"Frame count mismatch: buf={buf.frames}, sidechain={sidechain.frames}"
        )

    # Mono envelope from sidechain (max abs across channels)
    sc_env = np.ascontiguousarray(
        np.max(np.abs(sidechain.data), axis=0), dtype=np.float32
    )

    out = np.zeros_like(buf.data)
    for ch in range(buf.channels):
        sc = _fxdsp.SidechainCompressor()
        sc.init(buf.sample_rate)
        sc.set_ratio(ratio)
        sc.set_threshold(threshold)
        sc.set_attack(attack)
        sc.set_release(release)
        out[ch] = sc.process(buf.ensure_1d(ch), sc_env)

    return AudioBuffer(
        out,
        sample_rate=buf.sample_rate,
        channel_layout=buf.channel_layout,
        label=buf.label,
    )


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
        Gate threshold in dB. Signal below this is attenuated. Typical: -60 to -20.
    attack : float
        Gate open time in seconds, > 0. Typical: 0.001--0.01.
    release : float
        Gate close time in seconds, > 0. Typical: 0.01--0.1.
    hold_ms : float
        Hold time in milliseconds (>= 0) after signal drops below threshold
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
# Transient shaper
# ---------------------------------------------------------------------------


def transient_shape(
    buf: AudioBuffer,
    attack_gain: float = 1.0,
    sustain_gain: float = 1.0,
    fast_attack: float = 0.005,
    fast_release: float = 0.02,
    slow_attack: float = 0.05,
    slow_release: float = 0.2,
) -> AudioBuffer:
    """Shape transients by independently scaling attack and sustain components.

    Uses two envelope followers at different speeds.  The fast envelope
    tracks transients; the slow envelope tracks the sustained level.
    When ``attack_gain > 1`` transients are emphasized; when
    ``sustain_gain < 1`` the body between transients is reduced.

    Parameters
    ----------
    attack_gain : float
        Gain multiplier for transient (attack) component, >= 0. 1.0 = unchanged.
    sustain_gain : float
        Gain multiplier for sustain component, >= 0. 1.0 = unchanged.
    fast_attack, fast_release : float
        Fast envelope follower times in seconds. Typical: 0.001--0.01 / 0.01--0.05.
    slow_attack, slow_release : float
        Slow envelope follower times in seconds. Typical: 0.02--0.1 / 0.1--0.5.
    """

    def _process(x):
        ts = _fxdsp.TransientShaper()
        ts.init(buf.sample_rate)
        ts.set_attack_gain(attack_gain)
        ts.set_sustain_gain(sustain_gain)
        ts.set_fast_attack(fast_attack)
        ts.set_fast_release(fast_release)
        ts.set_slow_attack(slow_attack)
        ts.set_slow_release(slow_release)
        return ts.process(x)

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
        Desired RMS output level (linear), > 0. Typical: 0.1--1.0.
    max_gain_db : float
        Maximum gain in dB to prevent boosting silence to infinity. Typical: 20--60.
    average_len : int
        Number of samples for the moving-average power estimator, >= 1. Typical: 50--500.
    attack : float
        Attack time constant in seconds (fast gain reduction), >= 0. Typical: 0.001--0.05.
    release : float
        Release time constant in seconds (slow gain increase), >= 0. Typical: 0.01--0.1.
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
# Lookahead limiter
# ---------------------------------------------------------------------------


def lookahead_limit(
    buf: AudioBuffer,
    threshold_db: float = -1.0,
    lookahead_ms: float = 5.0,
    release: float = 0.1,
) -> AudioBuffer:
    """Brick-wall limiter with lookahead for transparent peak control.

    Delays the audio by *lookahead_ms* so the gain curve can begin
    reducing *before* a peak arrives, avoiding distortion on transients.
    The output should never exceed *threshold_db*.

    Parameters
    ----------
    threshold_db : float
        Ceiling in dBFS, <= 0. Typical: -1 to 0.
    lookahead_ms : float
        Lookahead time in milliseconds, > 0. Typical: 1--10.
    release : float
        Gain recovery time in seconds, > 0. Typical: 0.05--0.5.
    """

    def _process(x):
        lim = _fxdsp.LookaheadLimiter()
        lim.init(buf.sample_rate)
        lim.set_threshold_db(threshold_db)
        lim.set_lookahead_ms(lookahead_ms)
        lim.set_release(release)
        return lim.process(x)

    return _process_per_channel(buf, _process)

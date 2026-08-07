"""Dynamics -- compression, limiting, noise gate, AGC, sidechain, transient shaper, lookahead limiter."""

from __future__ import annotations

import numpy as np

from ..buffer import AudioBuffer
from .._helpers import _process_per_channel, _dsy_dyn
from .._core import fxdsp as _fxdsp


# ---------------------------------------------------------------------------
# Channel linking
# ---------------------------------------------------------------------------


def _linked_gain(buf: AudioBuffer, process_fn) -> np.ndarray:
    """Derive one gain curve for all channels from a linked detector.

    *process_fn* must be a rectified, gain-only processor: ``out = x * g`` where
    ``g`` depends on ``|x|`` alone.  Both the DaisySP ``Compressor`` and
    ``Limiter`` satisfy this, so running the detector signal through the
    processor and dividing recovers the gain curve exactly.

    The detector is the per-frame maximum absolute value across channels, which
    is the standard peak-linked design.  Where the detector is zero every
    channel is zero at that frame, so the gain there is arbitrary and set to 1.
    """
    detector = np.ascontiguousarray(np.max(np.abs(buf.data), axis=0), dtype=np.float32)
    processed = np.asarray(process_fn(detector), dtype=np.float32)
    return np.divide(
        processed,
        detector,
        out=np.ones_like(detector),
        where=detector > 0.0,
    )


def _apply_linked(buf: AudioBuffer, process_fn) -> AudioBuffer:
    """Apply *process_fn* with one gain curve shared across all channels."""
    gain = _linked_gain(buf, process_fn)
    return AudioBuffer(
        (buf.data * gain[np.newaxis, :]).astype(np.float32),
        sample_rate=buf.sample_rate,
        channel_layout=buf.channel_layout,
        label=buf.label,
        copy=False,  # freshly allocated by the multiply
    )


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
    link: bool = True,
) -> AudioBuffer:
    """Apply compression.

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
    link : bool
        Link the channel detectors (default). One gain curve is computed from
        the per-frame maximum across channels and applied to all of them, so
        the stereo image is preserved. With ``link=False`` each channel gets an
        independent detector, which shifts the image whenever channel content is
        asymmetric -- a loud transient in one channel then ducks only that
        channel. Unlinked is occasionally wanted for per-channel utility work;
        for stereo programme material linked is almost always correct.

        No-op for mono input, which takes the direct path unchanged.

    Returns
    -------
    AudioBuffer
        Compressed audio.

    Examples
    --------
    >>> import numpy as np
    >>> from nanodsp import AudioBuffer
    >>> from nanodsp.effects.dynamics import compress

    A sustained loud signal is pulled down once the attack has engaged, so
    compare the settled tail rather than the peak, which still contains the
    un-attacked onset:

    >>> loud = AudioBuffer(np.full((1, 24000), 0.9, dtype=np.float32))
    >>> out = compress(loud, ratio=8.0, threshold=-30.0)
    >>> bool(np.max(np.abs(out.data[:, -1000:])) < 0.9)
    True

    Stereo detectors are linked by default, so both channels get the same gain
    and the image does not shift:

    >>> stereo = AudioBuffer.sine(220.0, channels=2, frames=24000)
    >>> compress(stereo).channels
    2
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

    if link and buf.channels > 1:
        return _apply_linked(buf, _process)
    return _process_per_channel(buf, _process)


def limit(buf: AudioBuffer, pre_gain: float = 1.0, link: bool = True) -> AudioBuffer:
    """Apply a peak limiter.

    Parameters
    ----------
    buf : AudioBuffer
        Input audio.
    pre_gain : float
        Linear gain applied before limiting, > 0. 1.0 = unity.
    link : bool
        Link the channel detectors (default); see :func:`compress`. No-op for
        mono input.

    Returns
    -------
    AudioBuffer
        Limited audio.
    """

    def _process(x):
        lm = _dsy_dyn.Limiter()
        lm.init()
        return lm.process(x, pre_gain)

    if link and buf.channels > 1:
        return _apply_linked(buf, _process)
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

    Notes
    -----
    The detector is always channel-linked: the envelope is the per-frame maximum
    across channels and the resulting gain curve is applied to every channel, so
    the gate opens and closes on all channels together and the stereo image is
    preserved. There is no unlinked mode -- an unlinked gate would chatter one
    channel independently of the other, which is essentially never wanted.
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

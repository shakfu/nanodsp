"""Reverb algorithms -- FDN, Schroeder, Moorer, STK reverbs, STK chorus/echo.
Examples
--------
>>> from nanodsp import AudioBuffer
>>> from nanodsp.effects import reverb
>>> buf = AudioBuffer.sine(440.0, frames=9600, sample_rate=48000.0)

The FDN reverb is mono-in/stereo-out, so it always returns two channels:

>>> reverb.reverb(buf, preset="hall", mix=0.3).channels
2

The classic reverbs are per-channel and preserve the channel count, which also
means they can be streamed (see :mod:`nanodsp.stream`):

>>> reverb.schroeder_reverb(buf).channels
1
>>> reverb.moorer_reverb(buf).frames
9600
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from ..buffer import AudioBuffer
from .._helpers import _process_per_channel, _stk_fx
from .._core import madronalib as _madronalib
from .._core import stk as _stk
from .._core import fxdsp as _fxdsp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_mono(buf: AudioBuffer) -> np.ndarray:
    """Downmix to mono float32 1D array."""
    if buf.channels > 1:
        return np.mean(buf.data, axis=0).astype(np.float32)
    return buf.data[0].copy()


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
    preset: Literal["room", "hall", "plate", "chamber", "cathedral"] = "hall",
    mix: float = 0.3,
    decay: float = 0.8,
    damping: float = 0.5,
    pre_delay_ms: float = 0.0,
) -> AudioBuffer:
    """FDN reverb with presets.

    Parameters
    ----------
    preset : str
        One of ``'room'``, ``'hall'``, ``'plate'``, ``'chamber'``, ``'cathedral'``.
    mix : float
        Wet/dry blend, 0.0--1.0 (0.0 = fully dry, 1.0 = fully wet).
    decay : float
        Feedback gain per delay line, 0.0--<1.0 (values >= 1.0 are unstable).
    damping : float
        Lowpass filtering in feedback, 0.0--1.0 (0.0 = bright, 1.0 = dark).
    pre_delay_ms : float
        Pre-delay in milliseconds before reverb onset, >= 0.

    Returns
    -------
    AudioBuffer
        Always 2-channel, whatever the input channel count, with
        ``channel_layout='stereo'``.

    Notes
    -----
    **This function does not preserve the channel count**, unlike the rest of
    the effects API.  Mono input is widened to stereo and anything above two
    channels is folded down to a stereo pair (first half of the channels to the
    left, second half to the right).  A chain that assumes channel count is
    invariant will change shape here.

    The wet path is also mono: the input is summed to mono before entering the
    FDN, which then emits a decorrelated stereo pair.  Only the dry path
    carries the input's stereo image, so at ``mix=1.0`` the original stereo
    content is gone.  This follows the underlying madronalib ``FDN8``, which is
    a mono-in/stereo-out design; true-stereo reverb would need two decorrelated
    FDN instances.

    Frame count and sample rate are preserved.
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
    mono_data = _to_mono(buf)

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
# Classic Reverbs (Schroeder, Moorer)
# ---------------------------------------------------------------------------


def schroeder_reverb(
    buf: AudioBuffer,
    feedback: float = 0.7,
    diffusion: float = 0.5,
    mod_depth: float = 0.0,
) -> AudioBuffer:
    """Schroeder reverberator (4 parallel combs + 2 series allpasses).

    Parameters
    ----------
    feedback : float
        Comb filter feedback, 0.0--<1.0. Higher = longer tail.
    diffusion : float
        Allpass diffusion, 0.0--1.0. Higher = smoother.
    mod_depth : float
        LFO modulation depth, >= 0. 0.0 = no modulation.
    """

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
    """Moorer reverberator (early reflections + 4 combs + 2 allpasses).

    Parameters
    ----------
    feedback : float
        Comb filter feedback, 0.0--<1.0. Higher = longer tail.
    diffusion : float
        Allpass diffusion, 0.0--1.0. Higher = smoother.
    mod_depth : float
        LFO modulation depth, >= 0. 0.0 = no modulation.
    """

    def _process(x):
        rev = _fxdsp.MoorerReverb()
        rev.init(float(buf.sample_rate))
        rev.feedback = feedback
        rev.diffusion = diffusion
        rev.set_mod_depth(mod_depth)
        return rev.process(x)

    return _process_per_channel(buf, _process)


# ---------------------------------------------------------------------------
# STK Effects
# ---------------------------------------------------------------------------


def stk_reverb(
    buf: AudioBuffer,
    algorithm: Literal["freeverb", "jcrev", "nrev", "prcrev"] = "freeverb",
    mix: float = 0.3,
    room_size: float = 0.5,
    damping: float = 0.5,
    t60: float = 1.0,
) -> AudioBuffer:
    """Apply an STK reverb algorithm.

    Parameters
    ----------
    algorithm : str
        One of ``'freeverb'``, ``'jcrev'``, ``'nrev'``, ``'prcrev'``.
    mix : float
        Wet/dry mix, 0.0--1.0 (0.0 = dry, 1.0 = fully wet).
    room_size : float
        Room size (FreeVerb only), 0.0--1.0.
    damping : float
        Damping (FreeVerb only), 0.0--1.0.
    t60 : float
        Reverberation time in seconds (JCRev, NRev, PRCRev), > 0. Typical: 0.1--10.

    Returns
    -------
    AudioBuffer
        Always 2-channel: the STK reverbs are mono-in/stereo-out, so mono input
        is widened and more than two channels are folded to a stereo pair.
        Frame count and sample rate are preserved.
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
    mono = np.ascontiguousarray(_to_mono(buf), dtype=np.float32)

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

    Always returns 2-channel output: the STK chorus is mono-in/stereo-out, so
    mono input is widened and more than two channels are folded to a stereo
    pair. Frame count and sample rate are preserved.

    Parameters
    ----------
    mod_depth : float
        Modulation depth, >= 0. Typical: 0.01--0.1.
    mod_freq : float
        Modulation frequency in Hz, > 0. Typical: 0.1--5.0.
    mix : float
        Wet/dry mix, 0.0--1.0.
    """
    _stk.set_sample_rate(buf.sample_rate)

    ch = _stk_fx.Chorus()
    ch.set_mod_depth(mod_depth)
    ch.set_mod_frequency(mod_freq)
    ch.set_effect_mix(mix)

    mono = np.ascontiguousarray(_to_mono(buf), dtype=np.float32)

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
# Convolution reverb
# ---------------------------------------------------------------------------


def convolution_reverb(
    buf: AudioBuffer,
    ir: AudioBuffer,
    mix: float = 0.3,
    pre_delay_ms: float = 0.0,
    tail: bool = False,
    normalize: bool = True,
) -> AudioBuffer:
    """Convolution reverb using a recorded impulse response.

    Unlike the algorithmic reverbs in this module, the character comes entirely
    from *ir* -- a recording of a real space (or of another reverb) played back
    through a impulse. This preserves the channel count of *buf*.

    Parameters
    ----------
    buf : AudioBuffer
        Input audio.
    ir : AudioBuffer
        Impulse response, at the same sample rate as *buf*. A mono IR is applied
        to every channel; otherwise the channel counts must match.
    mix : float
        Wet/dry blend, 0.0--1.0 (0.0 = fully dry, 1.0 = fully wet).
    pre_delay_ms : float
        Delay in milliseconds before the wet signal starts, >= 0. Pushes the
        onset of the reverb back without moving the dry signal, which reads as a
        larger space.
    tail : bool
        If True the output is extended by ``ir.frames - 1`` so the reverb tail
        decays naturally past the end of the input. The default trims to the
        input length, which is what a chain expects.
    normalize : bool
        Scale the IR to unit energy first (the default). Recorded IRs vary in
        level by orders of magnitude, so without this *mix* would mean something
        different for every file.

    Returns
    -------
    AudioBuffer
        Same channel count and sample rate as *buf*. Length is ``buf.frames``
        unless *tail* is set.

    Raises
    ------
    ValueError
        If the sample rates differ, the channel counts are incompatible, *mix*
        is outside [0, 1], or *pre_delay_ms* is negative.

    Examples
    --------
    >>> import numpy as np
    >>> from nanodsp import AudioBuffer
    >>> from nanodsp.effects.reverb import convolution_reverb
    >>> rng = np.random.default_rng(0)
    >>> ir = AudioBuffer(
    ...     (np.exp(-np.linspace(0, 6, 4000)) * rng.standard_normal(4000)).astype(
    ...         np.float32
    ...     )
    ... )
    >>> dry = AudioBuffer.sine(220.0, channels=2, frames=8000)
    >>> wet = convolution_reverb(dry, ir, mix=0.4, pre_delay_ms=20.0)
    >>> wet.channels, wet.frames
    (2, 8000)

    ``tail=True`` lets the reverb decay past the end of the input:

    >>> convolution_reverb(dry, ir, tail=True).frames
    11999
    """
    from ..ops import convolve

    if not 0.0 <= mix <= 1.0:
        raise ValueError(f"mix must be in [0, 1], got {mix}")
    if pre_delay_ms < 0:
        raise ValueError(f"pre_delay_ms must be >= 0, got {pre_delay_ms}")
    if buf.sample_rate != ir.sample_rate:
        raise ValueError(
            f"Sample rate mismatch: buf={buf.sample_rate}, ir={ir.sample_rate}. "
            "Resample the IR first (nanodsp.analysis.resample)."
        )
    if ir.frames == 0:
        raise ValueError("impulse response is empty")

    pre_samples = int(round(buf.sample_rate * pre_delay_ms / 1000.0))
    out_frames = buf.frames + (ir.frames - 1 if tail else 0)

    # Pre-delay is applied to the IR rather than the wet signal: prepending
    # silence to the IR delays the whole response, including the tail, which is
    # what a real pre-delay does. Delaying the wet output afterwards would clip
    # the tail by the same amount.
    if pre_samples:
        ir = AudioBuffer(
            np.pad(ir.data, ((0, 0), (pre_samples, 0))),
            sample_rate=ir.sample_rate,
            copy=False,
        )

    # Convolve untrimmed so the tail exists, then fit to the requested length.
    wet = convolve(buf, ir, normalize=normalize, trim=False)
    wet_data = wet.data[:, :out_frames]
    if wet_data.shape[1] < out_frames:
        wet_data = np.pad(wet_data, ((0, 0), (0, out_frames - wet_data.shape[1])))

    dry_data = buf.data
    if out_frames > buf.frames:
        dry_data = np.pad(dry_data, ((0, 0), (0, out_frames - buf.frames)))

    out = (1.0 - mix) * dry_data + mix * wet_data
    return AudioBuffer(
        out.astype(np.float32),
        sample_rate=buf.sample_rate,
        channel_layout=buf.channel_layout,
        label=buf.label,
        copy=False,
    )

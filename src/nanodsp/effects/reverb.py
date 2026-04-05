"""Reverb algorithms -- FDN, Schroeder, Moorer, STK reverbs, STK chorus/echo."""

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

    Returns stereo output from mono or stereo input.

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

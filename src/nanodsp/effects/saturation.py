"""Saturation and antialiased waveshaping."""

from __future__ import annotations

import numpy as np

from ..buffer import AudioBuffer
from .._helpers import _process_per_channel
from .._core import fxdsp as _fxdsp


# ---------------------------------------------------------------------------
# Saturation
# ---------------------------------------------------------------------------


def saturate(
    buf: AudioBuffer,
    drive: float = 0.5,
    mode: str = "soft",
) -> AudioBuffer:
    """Apply saturation/distortion.

    Modes:
    - ``'soft'``: tanh soft clipping, normalized to preserve peak.
    - ``'hard'``: hard clipping to [-1, 1].
    - ``'tape'``: asymmetric soft clip ``x - x^3/3``.

    *drive*: 0.0 to 1.0 controls intensity (maps to gain 1x-10x).
    """
    drive_scaled = np.float32(1.0 + drive * 9.0)
    data = buf.data * drive_scaled

    if mode == "soft":
        out = np.tanh(data)
        # Normalize to preserve original peak
        peak_in = np.max(np.abs(buf.data))
        peak_out = np.max(np.abs(out))
        if peak_out > 0 and peak_in > 0:
            out *= np.float32(peak_in / peak_out)
    elif mode == "hard":
        out = np.clip(data, -1.0, 1.0)
    elif mode == "tape":
        # Asymmetric soft clip: x - x^3/3 for |x| < 1, clamped otherwise
        out = np.where(
            np.abs(data) < 1.0,
            data - (data**3) / 3.0,
            np.sign(data) * 2.0 / 3.0,
        )
        # Normalize to preserve original peak
        peak_in = np.max(np.abs(buf.data))
        peak_out = np.max(np.abs(out))
        if peak_out > 0 and peak_in > 0:
            out *= np.float32(peak_in / peak_out)
    else:
        raise ValueError(
            f"Unknown saturation mode {mode!r}, valid: 'soft', 'hard', 'tape'"
        )
    return AudioBuffer(
        out.astype(np.float32),
        sample_rate=buf.sample_rate,
        channel_layout=buf.channel_layout,
        label=buf.label,
    )


# ---------------------------------------------------------------------------
# Antialiased Waveshaping
# ---------------------------------------------------------------------------


def aa_hard_clip(buf: AudioBuffer, drive: float = 1.0) -> AudioBuffer:
    """Antialiased hard clipper using 1st-order antiderivative method.

    Parameters
    ----------
    drive : float
        Input gain multiplier before clipping.
    """

    def _process(x):
        if drive != 1.0:
            x = x * drive
        c = _fxdsp.HardClipper()
        return c.process(x)

    return _process_per_channel(buf, _process)


def aa_soft_clip(buf: AudioBuffer, drive: float = 1.0) -> AudioBuffer:
    """Antialiased soft clipper (sin-based saturation) with 1st-order AA.

    Parameters
    ----------
    drive : float
        Input gain multiplier before saturation.
    """

    def _process(x):
        if drive != 1.0:
            x = x * drive
        c = _fxdsp.SoftClipper()
        return c.process(x)

    return _process_per_channel(buf, _process)


def aa_wavefold(buf: AudioBuffer, drive: float = 1.0) -> AudioBuffer:
    """Antialiased wavefolder (Buchla 259 style) with 2nd-order AA.

    Parameters
    ----------
    drive : float
        Input gain multiplier before folding.
    """

    def _process(x):
        if drive != 1.0:
            x = x * drive
        c = _fxdsp.Wavefolder()
        return c.process(x)

    return _process_per_channel(buf, _process)

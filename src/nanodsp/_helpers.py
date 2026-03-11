"""Shared private utilities for nanodsp modules."""

from __future__ import annotations

import numpy as np

from nanodsp.buffer import AudioBuffer
from nanodsp._core import daisysp as _daisysp
from nanodsp._core import stk as _stk


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _hz_to_normalized(freq_hz: float, sample_rate: float) -> float:
    """Convert Hz to normalized frequency [0, 0.5).

    Raises ValueError if freq_hz is negative or >= Nyquist.
    """
    if freq_hz < 0:
        raise ValueError(f"Frequency must be non-negative, got {freq_hz}")
    nyquist = sample_rate / 2.0
    if freq_hz >= nyquist:
        raise ValueError(f"Frequency {freq_hz} Hz >= Nyquist ({nyquist} Hz)")
    return freq_hz / sample_rate


def _process_per_channel(buf: AudioBuffer, process_fn) -> AudioBuffer:
    """Apply process_fn(1d_array) -> 1d_array per channel, return new AudioBuffer."""
    out = np.zeros_like(buf.data)
    for ch in range(buf.channels):
        out[ch] = process_fn(buf.ensure_1d(ch))
    return AudioBuffer(
        out,
        sample_rate=buf.sample_rate,
        channel_layout=buf.channel_layout,
        label=buf.label,
    )


# ---------------------------------------------------------------------------
# DaisySP submodule aliases
# ---------------------------------------------------------------------------

_dsy_osc = _daisysp.oscillators
_dsy_filt = _daisysp.filters
_dsy_fx = _daisysp.effects
_dsy_dyn = _daisysp.dynamics
_dsy_noise = _daisysp.noise
_dsy_drums = _daisysp.drums
_dsy_pm = _daisysp.physical_modeling
_dsy_util = _daisysp.utility

_WAVEFORM_MAP: dict[str, int] = {
    "sine": _dsy_osc.WAVE_SIN,
    "sin": _dsy_osc.WAVE_SIN,
    "tri": _dsy_osc.WAVE_TRI,
    "triangle": _dsy_osc.WAVE_TRI,
    "saw": _dsy_osc.WAVE_SAW,
    "ramp": _dsy_osc.WAVE_RAMP,
    "square": _dsy_osc.WAVE_SQUARE,
    "polyblep_tri": _dsy_osc.WAVE_POLYBLEP_TRI,
    "polyblep_saw": _dsy_osc.WAVE_POLYBLEP_SAW,
    "polyblep_square": _dsy_osc.WAVE_POLYBLEP_SQUARE,
}

_BLOSC_WAVEFORM_MAP: dict[str, int] = {
    "triangle": _dsy_osc.BLOSC_WAVE_TRIANGLE,
    "tri": _dsy_osc.BLOSC_WAVE_TRIANGLE,
    "saw": _dsy_osc.BLOSC_WAVE_SAW,
    "square": _dsy_osc.BLOSC_WAVE_SQUARE,
    "off": _dsy_osc.BLOSC_WAVE_OFF,
}

_LADDER_MODE_MAP: dict[str, int] = {
    "lp24": _dsy_filt.LadderFilterMode.LP24,
    "lp12": _dsy_filt.LadderFilterMode.LP12,
    "bp24": _dsy_filt.LadderFilterMode.BP24,
    "bp12": _dsy_filt.LadderFilterMode.BP12,
    "hp24": _dsy_filt.LadderFilterMode.HP24,
    "hp12": _dsy_filt.LadderFilterMode.HP12,
}


def _resolve_waveform(waveform: int | str, mapping: dict[str, int]) -> int:
    """Resolve a waveform name or int constant to an int."""
    if isinstance(waveform, int):
        return waveform
    key = waveform.lower()
    if key not in mapping:
        raise ValueError(
            f"Unknown waveform {waveform!r}, valid names: {list(mapping.keys())}"
        )
    return mapping[key]


# ---------------------------------------------------------------------------
# STK submodule aliases
# ---------------------------------------------------------------------------

_stk_inst = _stk.instruments
_stk_gen = _stk.generators
_stk_fx = _stk.effects

# Map of instrument names to (class, constructor_style)
# constructor_style: "freq" = takes lowest_frequency, "void" = no args,
#                    "freq_required" = lowest_frequency without default
_STK_INSTRUMENTS: dict[str, tuple[type, str]] = {
    "clarinet": (_stk_inst.Clarinet, "freq"),
    "flute": (_stk_inst.Flute, "freq"),
    "brass": (_stk_inst.Brass, "freq"),
    "bowed": (_stk_inst.Bowed, "freq"),
    "plucked": (_stk_inst.Plucked, "freq"),
    "sitar": (_stk_inst.Sitar, "freq"),
    "stifkarp": (_stk_inst.StifKarp, "freq"),
    "saxofony": (_stk_inst.Saxofony, "freq"),
    "recorder": (_stk_inst.Recorder, "void"),
    "blowbotl": (_stk_inst.BlowBotl, "void"),
    "blowhole": (_stk_inst.BlowHole, "freq_required"),
    "whistle": (_stk_inst.Whistle, "void"),
}


# ---------------------------------------------------------------------------
# Spectrogram container
# ---------------------------------------------------------------------------


class Spectrogram:
    """Lightweight container for STFT output, consumed by ``istft``."""

    __slots__ = (
        "data",
        "window_size",
        "hop_size",
        "fft_size",
        "sample_rate",
        "original_frames",
    )

    def __init__(
        self,
        data: np.ndarray,
        window_size: int,
        hop_size: int,
        fft_size: int,
        sample_rate: float,
        original_frames: int,
    ):
        self.data = data  # [channels, num_stft_frames, bins] complex64
        self.window_size = window_size
        self.hop_size = hop_size
        self.fft_size = fft_size
        self.sample_rate = sample_rate
        self.original_frames = original_frames

    @property
    def channels(self) -> int:
        return self.data.shape[0]

    @property
    def num_frames(self) -> int:
        return self.data.shape[1]

    @property
    def bins(self) -> int:
        return self.data.shape[2]

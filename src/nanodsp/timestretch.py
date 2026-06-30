"""Time-stretching effects.

PaulStretch extreme time-stretching via phase-randomized spectral resynthesis.
The algorithm is by Nasca Octavian Paul (public domain); this is an original
implementation on top of the signalsmith RealFFT and does not use the GPLv3
paulxstretch application sources.
"""

from __future__ import annotations

import numpy as np

from nanodsp.buffer import AudioBuffer
from nanodsp._core import paulstretch as _ps


def paulstretch(
    buf: AudioBuffer,
    stretch: float = 8.0,
    window_size: int = 4096,
    onset: float = 0.0,
    pitch_semitones: float = 0.0,
    harmonics: int = 0,
    spread: float = 0.0,
    lowpass_hz: float = 0.0,
    highpass_hz: float = 0.0,
    seed: int = 42,
) -> AudioBuffer:
    """Extreme time-stretch using the PaulStretch algorithm.

    Stretches audio by a factor of ``stretch`` (e.g. 8 = eight times longer)
    by resynthesizing overlapping FFT frames with randomized phases. The result
    is the smeared, pad-like texture PaulStretch is known for. The pitch is
    preserved unless ``pitch_semitones`` is set.

    Parameters
    ----------
    buf : AudioBuffer
        Input audio.
    stretch : float
        Time-stretch factor, must be > 0. Values > 1 lengthen the audio
        (the typical use); values < 1 shorten it. Typical: 4--50.
    window_size : int
        FFT window length in samples, must be >= 16. Larger windows give a
        smoother, more diffuse sound; smaller windows keep more detail.
        Typical: 2048--16384. Rounded up to an even number.
    onset : float
        Transient preservation in ``(0, 1]``; 0 (default) disables it. Higher
        values preserve sharper attacks by keeping the original phase on
        detected onsets instead of randomizing it.
    pitch_semitones : float
        Spectral pitch shift in semitones (+12 = up one octave). Shifts the
        whole magnitude spectrum, so formants move with the pitch.
    harmonics : int
        Number of added harmonic copies (0 = off). Each adds an integer
        multiple of the spectrum with geometric decay, thickening the tone.
    spread : float
        Spectral blur radius in bins (0 = off). Smears energy across
        neighbouring frequency bins for a more diffuse, noisy texture.
    lowpass_hz : float
        Spectral low-pass cutoff in Hz (<= 0 disables). Zeroes bins above
        this frequency before resynthesis.
    highpass_hz : float
        Spectral high-pass cutoff in Hz (<= 0 disables). Zeroes bins below
        this frequency before resynthesis.
    seed : int
        Base seed for phase randomization. Output is reproducible for a given
        seed. Each channel uses ``seed + channel_index`` so stereo material is
        decorrelated (wider) rather than identical across channels.

    Returns
    -------
    AudioBuffer
        Stretched audio. Length is approximately ``frames * stretch``; all
        channels share the same length. Sample rate and channel layout are
        preserved.

    Raises
    ------
    ValueError
        If ``stretch`` is not positive or ``window_size`` is too small.
    """
    if stretch <= 0:
        raise ValueError(f"stretch must be positive, got {stretch}")
    if window_size < 16:
        raise ValueError(f"window_size must be >= 16, got {window_size}")

    proc = _ps.PaulStretch(int(window_size), float(buf.sample_rate))
    proc.onset_sensitivity = float(onset)
    proc.pitch_semitones = float(pitch_semitones)
    proc.harmonics = int(harmonics)
    proc.spread = float(spread)
    proc.lowpass_hz = float(lowpass_hz)
    proc.highpass_hz = float(highpass_hz)

    channels = []
    for ch in range(buf.channels):
        proc.reset()
        proc.set_seed(int(seed) + ch)
        channels.append(proc.process(buf.ensure_1d(ch), float(stretch)))

    out = np.stack(channels) if len(channels) > 1 else channels[0].reshape(1, -1)
    return AudioBuffer(
        out,
        sample_rate=buf.sample_rate,
        channel_layout=buf.channel_layout,
        label=buf.label,
    )

"""Time-stretching and pitch-shifting effects.

Two complementary backends:

- :func:`paulstretch` -- PaulStretch *extreme* time-stretching via
  phase-randomized spectral resynthesis (Nasca Octavian Paul, public domain;
  an original implementation on the signalsmith RealFFT, not the GPLv3
  paulxstretch sources). Built for very large factors and ambient textures.
- :func:`signalsmith_stretch` -- the MIT-licensed signalsmith-stretch library
  (Geraint Luff / Signalsmith Audio), a transient-aware, phase-vocoder-derived
  stretcher with independent pitch-shifting. Built to stay musical at modest
  ratios.

Examples
--------
>>> from nanodsp import AudioBuffer, timestretch
>>> buf = AudioBuffer.sine(440.0, frames=48000, sample_rate=48000.0)

Signalsmith stays musical at modest ratios and decouples time from pitch:

>>> timestretch.signalsmith_stretch(buf, stretch=2.0).frames
96000
>>> timestretch.signalsmith_stretch(buf, stretch=1.0, semitones=-7.0).frames
48000

PaulStretch is for extreme factors, where a phase vocoder breaks down:

>>> bool(timestretch.paulstretch(buf, stretch=8.0).frames > 8 * 40000)
True
"""

from __future__ import annotations

import numpy as np

from nanodsp.buffer import AudioBuffer
from nanodsp._helpers import _process_per_channel
from nanodsp._core import paulstretch as _ps
from nanodsp._core import signalsmith_stretch as _ss
from nanodsp._core import keyframe as _kf


def paulstretch(
    buf: AudioBuffer,
    stretch: float = 8.0,
    window_size: int = 4096,
    onset: float = 0.0,
    pitch_semitones: float = 0.0,
    harmonics: int = 0,
    spread: float = 0.0,
    spread_octaves: float = 0.0,
    tonal_vs_noise: float = 0.0,
    tonal_noise_octaves: float = 0.2,
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
        neighbouring frequency bins for a more diffuse, noisy texture. Because
        bin spacing is linear in frequency, this smears low partials much more
        (musically) than high ones; prefer ``spread_octaves`` unless that
        asymmetry is what you want.
    spread_octaves : float
        Constant-Q spectral spread width in octaves (0 = off). Smears each
        partial across a fixed fraction of its own frequency, so the effect is
        musically even across the range and independent of ``window_size`` and
        sample rate. Typical: 0.05--0.5.
    tonal_vs_noise : float
        Tonal/noise balance in ``[-1, 1]``; 0 (default) leaves the spectrum
        untouched. The spectrum is split into peaks that stand above their own
        local envelope and the noise floor beneath them, and this blends the
        result toward one part or the other. ``+1`` keeps only the peaks, for
        a cleaner and more pitched drone; ``-1`` keeps only the floor, for a
        breathy, unpitched wash. The whole range is usable, and the effect
        increases monotonically with the setting.
    tonal_noise_octaves : float
        Width in octaves of the spectral envelope used to decide what counts
        as a peak for ``tonal_vs_noise``. Narrow settings are a higher bar and
        keep only sharp partials; wider settings let more of the spectrum
        through as tonal. Typical: 0.1--0.5.
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
        If ``stretch`` is not positive, ``window_size`` is too small, or
        ``tonal_vs_noise`` is outside ``[-1, 1]``.
    """
    if stretch <= 0:
        raise ValueError(f"stretch must be positive, got {stretch}")
    if window_size < 16:
        raise ValueError(f"window_size must be >= 16, got {window_size}")
    if not -1.0 <= tonal_vs_noise <= 1.0:
        raise ValueError(f"tonal_vs_noise must be in [-1, 1], got {tonal_vs_noise}")

    proc = _ps.PaulStretch(int(window_size), float(buf.sample_rate))
    proc.onset_sensitivity = float(onset)
    proc.pitch_semitones = float(pitch_semitones)
    proc.harmonics = int(harmonics)
    proc.spread = float(spread)
    proc.spread_octaves = float(spread_octaves)
    proc.tonal_vs_noise = float(tonal_vs_noise)
    proc.tonal_noise_octaves = float(tonal_noise_octaves)
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


def signalsmith_stretch(
    buf: AudioBuffer,
    stretch: float = 1.0,
    semitones: float = 0.0,
    tonality_hz: float = 0.0,
    cheaper: bool = False,
    seed: int = 0,
) -> AudioBuffer:
    """High-quality time-stretch and pitch-shift (signalsmith-stretch).

    Changes duration and/or pitch using the MIT-licensed signalsmith-stretch
    library, a transient-aware phase-vocoder-derived algorithm. Unlike
    :func:`paulstretch`, time-stretch and pitch-shift are decoupled and the
    result stays musical at modest ratios rather than smearing into a texture.
    All channels are processed together in a single pass, keeping a stereo
    image coherent.

    Parameters
    ----------
    buf : AudioBuffer
        Input audio.
    stretch : float
        Time-stretch factor, must be > 0. Values > 1 lengthen the audio,
        values < 1 shorten it, and 1.0 (default) leaves duration unchanged --
        useful for pure pitch-shifting. Typical: 0.5--4.
    semitones : float
        Pitch shift in semitones, independent of ``stretch`` (+12 = up one
        octave, -12 = down). Fractional values are allowed. 0 (default) leaves
        pitch unchanged.
    tonality_hz : float
        Tonality limit in Hz (<= 0 disables). Above this frequency the pitch
        shift is rolled back toward the original signal, which preserves
        high-frequency timbre/"air" on large shifts. A common choice is around
        8000 Hz for voice.
    cheaper : bool
        Use the lower-CPU preset (slightly lower quality) instead of the
        default preset.
    seed : int
        Seed for the internal phase randomization (engaged past ~2x stretch).
        Output is reproducible for a given seed.

    Returns
    -------
    AudioBuffer
        Processed audio. Length is approximately ``frames * stretch``; all
        channels share the same length. Sample rate and channel layout are
        preserved.

    Raises
    ------
    ValueError
        If ``stretch`` is not positive.
    """
    if stretch <= 0:
        raise ValueError(f"stretch must be positive, got {stretch}")

    proc = _ss.SignalsmithStretch(
        int(buf.channels), float(buf.sample_rate), bool(cheaper), int(seed)
    )
    proc.transpose_semitones = float(semitones)
    proc.tonality_hz = float(tonality_hz)

    out = proc.process(buf.data, float(stretch))
    return AudioBuffer(
        out,
        sample_rate=buf.sample_rate,
        channel_layout=buf.channel_layout,
        label=buf.label,
    )


def keyframe_sparsify(buf: AudioBuffer, threshold: float = 0.001) -> AudioBuffer:
    """Reduce audio to its local extrema and reconstruct it.

    This is the representation underlying :func:`keyframe_stretch`, exposed on
    its own. The signal is replaced by a sparse set of its local extrema and
    rebuilt by interpolating between them, so the round trip is lossy: the
    result is the input as the stretcher sees it, with no time or pitch change
    applied. Useful for auditing how much the representation costs on given
    material before stretching it, and usable as a mild lo-fi effect in its own
    right.

    Parameters
    ----------
    buf : AudioBuffer
        Input audio.
    threshold : float
        Amplitude difference below which an extremum is discarded, in the same
        units as the samples. The default of 0.001 (-60 dB) reconstructs
        faithfully; raising it discards low-amplitude detail and, because high
        frequencies tend to have lower amplitudes in practice, acts as an
        amplitude-dependent lowpass.

    Returns
    -------
    AudioBuffer
        Same length, channel count and sample rate as the input.

    Raises
    ------
    ValueError
        If *threshold* is negative.

    Examples
    --------
    >>> from nanodsp import AudioBuffer, timestretch
    >>> buf = AudioBuffer.sine(440.0, frames=4800, sample_rate=48000.0)
    >>> timestretch.keyframe_sparsify(buf).frames
    4800
    """
    if threshold < 0:
        raise ValueError(f"threshold must be non-negative, got {threshold}")

    def _process(x):
        return _kf.sparsify(x, float(threshold))

    return _process_per_channel(buf, _process)


def keyframe_stretch(
    buf: AudioBuffer,
    stretch: float = 1.0,
    semitones: float = 0.0,
    threshold: float = 0.001,
    splice_keyframes: int = 16,
    max_splice_ms: float = 200.0,
) -> AudioBuffer:
    """Time-stretch and pitch-shift by splicing in the extrema domain.

    An overlap-add stretcher whose splice duration adapts to the signal instead
    of being fixed. The audio is first reduced to its local extrema; because
    extrema crowd together where the signal is busy and spread out where it is
    sustained, their spacing is a per-sample estimate of local information
    density, available without an FFT or a correlation search. Each crossfade
    is sized in extrema rather than samples, so it shortens automatically at a
    transient and lengthens across a sustained note.

    This is a different trade from the other two stretchers here.
    :func:`signalsmith_stretch` is the one to reach for when the result should
    sound like the input; :func:`paulstretch` is for extreme, deliberately
    smeared textures. This one is cheap and transient-preserving, and colours
    the sound -- the underlying paper reports it as a creative effect rather
    than a transparent one, with measurable spectral contrast loss on tonally
    nuanced material. It is also the only stretcher here that produces output
    sample by sample with no block latency.

    Parameters
    ----------
    buf : AudioBuffer
        Input audio.
    stretch : float
        Output length as a multiple of the input, > 0. 2.0 is twice as long.
    semitones : float
        Pitch shift in semitones, independent of *stretch*.
    threshold : float
        Extrema-discard threshold; see :func:`keyframe_sparsify`.
    splice_keyframes : int
        How far the playhead may drift from where it should be, measured in
        keyframes, before a splice is triggered. Larger values splice less
        often and over longer spans. Must be >= 1.
    max_splice_ms : float
        Ceiling on splice duration in milliseconds. Long silences or otherwise
        sparse passages would otherwise produce a splice lasting seconds, which
        smears everything leading up to the next transient. Pass 0 to disable.

    Returns
    -------
    AudioBuffer
        Stretched audio of ``round(frames * stretch)`` samples; all channels
        share the same length. Sample rate and channel layout are preserved.

    Raises
    ------
    ValueError
        If *stretch* is not positive, *splice_keyframes* is below 1, or
        *threshold* or *max_splice_ms* is negative.

    References
    ----------
    .. [1] M. Nielsen, "Keyframe Time Stretching via Extrema Sampling," Proc.
       29th Int. Conf. Digital Audio Effects (DAFx26), Cambridge, MA, USA,
       Sept. 2026. https://github.com/heavylight-industries/dafx26-paper

    Examples
    --------
    >>> from nanodsp import AudioBuffer, timestretch
    >>> buf = AudioBuffer.sine(440.0, frames=48000, sample_rate=48000.0)
    >>> timestretch.keyframe_stretch(buf, stretch=2.0).frames
    96000

    Pitch shifting is independent of stretching:

    >>> timestretch.keyframe_stretch(buf, semitones=7.0).frames
    48000
    """
    if stretch <= 0:
        raise ValueError(f"stretch must be positive, got {stretch}")
    if splice_keyframes < 1:
        raise ValueError(f"splice_keyframes must be >= 1, got {splice_keyframes}")
    if threshold < 0:
        raise ValueError(f"threshold must be non-negative, got {threshold}")
    if max_splice_ms < 0:
        raise ValueError(f"max_splice_ms must be non-negative, got {max_splice_ms}")

    n_out = int(round(buf.frames * float(stretch)))
    time_rate = 1.0 / float(stretch)
    pitch_rate = 2.0 ** (float(semitones) / 12.0)
    max_splice = float(max_splice_ms) * buf.sample_rate / 1000.0

    channels = [
        _kf.stretch(
            buf.ensure_1d(ch),
            time_rate,
            pitch_rate,
            int(splice_keyframes),
            float(threshold),
            max_splice,
            n_out,
        )
        for ch in range(buf.channels)
    ]

    out = np.stack(channels) if len(channels) > 1 else channels[0].reshape(1, -1)
    return AudioBuffer(
        out,
        sample_rate=buf.sample_rate,
        channel_layout=buf.channel_layout,
        label=buf.label,
        copy=False,
    )

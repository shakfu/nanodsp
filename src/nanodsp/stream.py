"""Streaming / real-time audio processing infrastructure.

Provides ring buffers, block processors, and overlap-add utilities for
processing audio in fixed-size chunks.

Examples
--------
Stateless effects rebuild their DSP object on every call, so feeding them
successive blocks restarts the filter at each boundary.  The ``stateful_*``
constructors keep one object per channel alive instead, so a block-by-block
run matches whole-buffer processing exactly:

>>> import numpy as np
>>> from nanodsp import AudioBuffer
>>> from nanodsp.stream import stateful_lowpass
>>> buf = AudioBuffer.sine(220.0, channels=2, frames=4096, sample_rate=48000.0)
>>> whole = stateful_lowpass(1000.0, channels=2).process(buf)
>>> streamer = stateful_lowpass(1000.0, channels=2)
>>> blocks = [streamer.process(buf.slice(i, i + 512)) for i in range(0, 4096, 512)]
>>> joined = np.concatenate([b.data for b in blocks], axis=1)
>>> bool(np.allclose(whole.data, joined, atol=1e-6))
True

A ring buffer for producer/consumer work:

>>> from nanodsp.stream import RingBuffer
>>> ring = RingBuffer(channels=1, capacity=1024)
>>> ring.write(AudioBuffer.sine(440.0, frames=256))
256
>>> ring.read(100).frames
100
>>> ring.available_read
156
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from nanodsp.buffer import AudioBuffer


# ---------------------------------------------------------------------------
# Ring Buffer
# ---------------------------------------------------------------------------


class RingBuffer:
    """Ring buffer for streaming audio.

    Stores planar float32 audio in a circular buffer with independent
    read and write positions.

    .. warning::

        This class is **not thread-safe**.  Concurrent reads and writes
        from different threads can corrupt internal state.  If you need
        to share a RingBuffer between a producer thread and a consumer
        thread, protect every ``read``/``write``/``peek``/``clear`` call
        with an external lock (e.g. ``threading.Lock``).

    Parameters
    ----------
    channels : int
        Number of audio channels.
    capacity : int
        Maximum number of frames the buffer can hold.
    sample_rate : float
        Sample rate metadata for AudioBuffer output.
    """

    __slots__ = (
        "_buf",
        "_read_pos",
        "_write_pos",
        "_size",
        "_capacity",
        "_channels",
        "_sample_rate",
    )

    def __init__(self, channels: int, capacity: int, sample_rate: float = 48000.0):
        if channels < 1:
            raise ValueError(f"channels must be >= 1, got {channels}")
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self._buf = np.zeros((channels, capacity), dtype=np.float32)
        self._read_pos = 0
        self._write_pos = 0
        self._size = 0
        self._capacity = capacity
        self._channels = channels
        self._sample_rate = sample_rate

    @property
    def channels(self) -> int:
        return self._channels

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def sample_rate(self) -> float:
        return self._sample_rate

    @property
    def available_read(self) -> int:
        """Number of frames available to read."""
        return self._size

    @property
    def available_write(self) -> int:
        """Number of frames that can be written before full."""
        return self._capacity - self._size

    def write(self, data: AudioBuffer | np.ndarray) -> int:
        """Write frames into the buffer.

        Returns the number of frames actually written (may be less than
        requested if the buffer is nearly full).
        """
        if isinstance(data, AudioBuffer):
            arr = data.data
        else:
            arr = np.asarray(data, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)

        if arr.shape[0] != self._channels:
            raise ValueError(
                f"Channel mismatch: buffer has {self._channels}, "
                f"data has {arr.shape[0]}"
            )

        frames_to_write = min(arr.shape[1], self.available_write)
        if frames_to_write == 0:
            return 0

        pos = self._write_pos
        cap = self._capacity

        # First segment: from write_pos to end of buffer (or frames_to_write)
        first = min(frames_to_write, cap - pos)
        self._buf[:, pos : pos + first] = arr[:, :first]

        # Second segment: wrap around
        second = frames_to_write - first
        if second > 0:
            self._buf[:, :second] = arr[:, first : first + second]

        self._write_pos = (pos + frames_to_write) % cap
        self._size += frames_to_write
        return frames_to_write

    def read(self, frames: int) -> AudioBuffer:
        """Read and consume frames from the buffer.

        Returns an AudioBuffer that may be shorter than *frames* if
        insufficient data is available.
        """
        frames_to_read = min(frames, self._size)
        if frames_to_read == 0:
            return AudioBuffer(
                np.zeros((self._channels, 0), dtype=np.float32),
                sample_rate=self._sample_rate,
            )

        out = np.zeros((self._channels, frames_to_read), dtype=np.float32)
        pos = self._read_pos
        cap = self._capacity

        first = min(frames_to_read, cap - pos)
        out[:, :first] = self._buf[:, pos : pos + first]

        second = frames_to_read - first
        if second > 0:
            out[:, first : first + second] = self._buf[:, :second]

        self._read_pos = (pos + frames_to_read) % cap
        self._size -= frames_to_read
        return AudioBuffer(out, sample_rate=self._sample_rate)

    def peek(self, frames: int) -> AudioBuffer:
        """Read frames without consuming them."""
        frames_to_read = min(frames, self._size)
        if frames_to_read == 0:
            return AudioBuffer(
                np.zeros((self._channels, 0), dtype=np.float32),
                sample_rate=self._sample_rate,
            )

        out = np.zeros((self._channels, frames_to_read), dtype=np.float32)
        pos = self._read_pos
        cap = self._capacity

        first = min(frames_to_read, cap - pos)
        out[:, :first] = self._buf[:, pos : pos + first]

        second = frames_to_read - first
        if second > 0:
            out[:, first : first + second] = self._buf[:, :second]

        return AudioBuffer(out, sample_rate=self._sample_rate)

    def clear(self) -> None:
        """Reset to empty without reallocating."""
        self._read_pos = 0
        self._write_pos = 0
        self._size = 0


# ---------------------------------------------------------------------------
# Block Processor
# ---------------------------------------------------------------------------


class BlockProcessor:
    """Base class for block-based audio processors.

    Subclass and override :meth:`process_block` to implement custom
    processing.  Call :meth:`process` to run on an arbitrary-length buffer.

    .. note::

        Each block is processed independently -- no state is carried
        between successive blocks.  This means stateful DSP objects
        (IIR filters, reverbs, compressors) that are instantiated inside
        ``process_block`` will be re-created per block, losing their
        internal memory.  To preserve state across blocks, instantiate
        stateful objects in ``__init__`` and reuse them in
        ``process_block``.

    Parameters
    ----------
    block_size : int
        Number of frames per processing block.
    channels : int
        Expected channel count.
    sample_rate : float
        Sample rate metadata.
    """

    def __init__(
        self, block_size: int, channels: int = 1, sample_rate: float = 48000.0
    ):
        if block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {block_size}")
        self.block_size = block_size
        self.channels = channels
        self.sample_rate = sample_rate

    def process_block(self, block: AudioBuffer) -> AudioBuffer:
        """Process exactly block_size frames. Must return same shape.

        Override in subclasses.
        """
        raise NotImplementedError

    def process(self, buf: AudioBuffer) -> AudioBuffer:
        """Process an entire buffer in block_size chunks.

        The last block is zero-padded if needed; output is trimmed to the
        original length.
        """
        n_frames = buf.frames
        n_blocks = (n_frames + self.block_size - 1) // self.block_size
        out_parts = []

        for b in range(n_blocks):
            start = b * self.block_size
            end = min(start + self.block_size, n_frames)
            chunk = buf.data[:, start:end]

            # Zero-pad last block if needed
            if chunk.shape[1] < self.block_size:
                padded = np.zeros((buf.channels, self.block_size), dtype=np.float32)
                padded[:, : chunk.shape[1]] = chunk
                chunk = padded

            block_buf = AudioBuffer(chunk, sample_rate=buf.sample_rate)
            result = self.process_block(block_buf)
            out_parts.append(result.data)

        out = np.concatenate(out_parts, axis=1)[:, :n_frames]
        return AudioBuffer(
            out,
            sample_rate=buf.sample_rate,
            channel_layout=buf.channel_layout,
            label=buf.label,
        )

    def reset(self) -> None:
        """Reset internal state. Override in subclasses if needed."""
        pass


# ---------------------------------------------------------------------------
# Callback Processor
# ---------------------------------------------------------------------------


class CallbackProcessor(BlockProcessor):
    """Wraps a callable as a BlockProcessor.

    Parameters
    ----------
    callback : callable
        Function ``(AudioBuffer) -> AudioBuffer`` that processes one block.
    block_size : int
        Block size in frames.
    channels : int
        Channel count.
    sample_rate : float
        Sample rate metadata.
    """

    def __init__(
        self, callback, block_size: int, channels: int = 1, sample_rate: float = 48000.0
    ):
        super().__init__(block_size, channels, sample_rate)
        self.callback = callback

    def process_block(self, block: AudioBuffer) -> AudioBuffer:
        return self.callback(block)


# ---------------------------------------------------------------------------
# Processor Chain
# ---------------------------------------------------------------------------


class ProcessorChain:
    """Chain multiple BlockProcessors in series.

    Parameters
    ----------
    *processors : BlockProcessor
        Processors to chain. All must share the same block_size.
    """

    def __init__(self, *processors: BlockProcessor):
        if not processors:
            raise ValueError("At least one processor required")
        self.processors = list(processors)

    def process(self, buf: AudioBuffer) -> AudioBuffer:
        """Process through all processors in order."""
        for p in self.processors:
            buf = p.process(buf)
        return buf

    def reset(self) -> None:
        """Reset all processors."""
        for p in self.processors:
            p.reset()


# ---------------------------------------------------------------------------
# Overlap-add block processing utility
# ---------------------------------------------------------------------------


def process_blocks(
    buf: AudioBuffer,
    fn,
    block_size: int,
    hop_size: int | None = None,
) -> AudioBuffer:
    """Process a buffer through *fn* in blocks, optionally with overlap-add.

    Parameters
    ----------
    buf : AudioBuffer
        Input audio.
    fn : callable
        ``(AudioBuffer) -> AudioBuffer`` block processing function.
    block_size : int
        Size of each processing block in frames.
    hop_size : int or None
        Hop between successive blocks.  ``None`` or equal to *block_size*
        means non-overlapping.  Values < *block_size* trigger overlap-add
        with Hann windowing and COLA normalization.

    Returns
    -------
    AudioBuffer
        Processed audio, same length as input.
    """
    if hop_size is None or hop_size >= block_size:
        # Non-overlapping: simple chunking
        n_frames = buf.frames
        n_blocks = (n_frames + block_size - 1) // block_size
        out_parts = []
        for b in range(n_blocks):
            start = b * block_size
            end = min(start + block_size, n_frames)
            chunk = buf.data[:, start:end]
            if chunk.shape[1] < block_size:
                padded = np.zeros((buf.channels, block_size), dtype=np.float32)
                padded[:, : chunk.shape[1]] = chunk
                chunk = padded
            block_buf = AudioBuffer(chunk, sample_rate=buf.sample_rate)
            result = fn(block_buf)
            out_parts.append(result.data)
        out = np.concatenate(out_parts, axis=1)[:, :n_frames]
    else:
        # Overlap-add with Hann window
        n_frames = buf.frames
        window = np.hanning(block_size).astype(np.float32)
        out = np.zeros((buf.channels, n_frames), dtype=np.float32)
        win_sum = np.zeros(n_frames, dtype=np.float32)

        pos = 0
        while pos + block_size <= n_frames:
            chunk = buf.data[:, pos : pos + block_size] * window[np.newaxis, :]
            block_buf = AudioBuffer(chunk, sample_rate=buf.sample_rate)
            result = fn(block_buf)
            out[:, pos : pos + block_size] += result.data * window[np.newaxis, :]
            win_sum[pos : pos + block_size] += window**2
            pos += hop_size

        # Handle final partial block
        if pos < n_frames and pos + block_size > n_frames:
            remaining = n_frames - pos
            padded = np.zeros((buf.channels, block_size), dtype=np.float32)
            padded[:, :remaining] = buf.data[:, pos:] * window[np.newaxis, :remaining]
            block_buf = AudioBuffer(padded, sample_rate=buf.sample_rate)
            result = fn(block_buf)
            out[:, pos:] += result.data[:, :remaining] * window[np.newaxis, :remaining]
            win_sum[pos:] += window[:remaining] ** 2

        # COLA normalization
        win_sum = np.maximum(win_sum, 1e-8)
        out /= win_sum[np.newaxis, :]

    return AudioBuffer(
        out,
        sample_rate=buf.sample_rate,
        channel_layout=buf.channel_layout,
        label=buf.label,
    )


# ---------------------------------------------------------------------------
# Stateful streaming filters
# ---------------------------------------------------------------------------


class StatefulProcessor(BlockProcessor):
    """A DSP object that preserves per-channel state across ``process`` calls.

    Holds one persistent DSP object per channel (built by *factory*) so that
    audio fed in successive blocks is processed continuously, with no per-block
    re-initialization.  This is the key difference from the stateless functions
    in :mod:`nanodsp.effects`, which rebuild their DSP object on every call and
    therefore cannot be streamed without discontinuities at block boundaries.

    Each per-channel object must expose a processing method taking and returning
    a 1-D float32 array and retaining internal state across calls (e.g.
    ``nanodsp._core.filters.Biquad``, the DaisySP filters and effects, or the
    fxdsp reverbs).  Because the object handles arbitrary-length input itself,
    :meth:`process` applies it to the whole buffer with no chunking or
    zero-padding: calling ``process`` repeatedly on consecutive buffers yields
    exactly the same result as processing their concatenation in a single call.

    Parameters
    ----------
    factory : callable
        Zero-argument callable returning a freshly-constructed, configured
        per-channel DSP object.
    channels : int
        Number of channels; one persistent object is built per channel.
    sample_rate : float
        Sample-rate metadata for output buffers.
    apply : callable or None
        ``(obj, x) -> y`` adapter, for objects whose processing method is not a
        plain ``obj.process(x)`` -- the DaisySP ``Limiter``, for instance, takes
        its pre-gain as a second argument.  Defaults to ``obj.process(x)``.

    Notes
    -----
    Not thread-safe.  Feed blocks from a single thread, or guard ``process`` /
    :meth:`reset` with an external lock.

    Only effects whose channels are independent can be streamed this way. Two
    kinds cannot, and have no ``stateful_*`` constructor:

    - **Channel-linked dynamics.** ``compress`` and ``limit`` derive one gain
      curve from all channels at once (see their ``link`` parameter). The
      per-channel constructors here are the *unlinked* equivalents.
    - **Mono-to-stereo effects.** The FDN ``reverb``, ``chorus`` and the STK
      reverbs sum to mono and emit a decorrelated stereo pair, which does not
      fit a per-channel model. :func:`stateful_schroeder_reverb` and
      :func:`stateful_moorer_reverb` are true per-channel reverbs and do stream.
    """

    def __init__(
        self,
        factory: Callable[[], Any],
        channels: int = 1,
        sample_rate: float = 48000.0,
        apply: Callable[[Any, np.ndarray], Any] | None = None,
    ):
        # block_size is nominal: process() is sample-accurate over any length.
        super().__init__(block_size=1, channels=channels, sample_rate=sample_rate)
        self._factory = factory
        self._apply_fn = apply if apply is not None else (lambda obj, x: obj.process(x))
        self._procs = [factory() for _ in range(channels)]

    def _apply(self, buf: AudioBuffer) -> AudioBuffer:
        if buf.channels != self.channels:
            raise ValueError(
                f"{type(self).__name__} configured for {self.channels} channel(s), "
                f"got {buf.channels}"
            )
        out = np.zeros_like(buf.data)
        for ch in range(buf.channels):
            out[ch] = np.asarray(
                self._apply_fn(self._procs[ch], buf.ensure_1d(ch)), dtype=np.float32
            )
        return AudioBuffer(
            out,
            sample_rate=buf.sample_rate,
            channel_layout=buf.channel_layout,
            label=buf.label,
            copy=False,
        )

    def process_block(self, block: AudioBuffer) -> AudioBuffer:
        """Process one block, retaining state for the next call."""
        return self._apply(block)

    def process(self, buf: AudioBuffer) -> AudioBuffer:
        """Process an entire buffer continuously (no chunking or padding)."""
        return self._apply(buf)

    def reset(self) -> None:
        """Rebuild every per-channel object, clearing all state."""
        self._procs = [self._factory() for _ in range(self.channels)]


#: Backwards-compatible alias. The class was named for filters before it grew
#: constructors for dynamics, modulation and reverb.
StatefulFilter = StatefulProcessor


def _biquad_factory(
    method: str, freq: float, octaves: float | None, design: str | int
) -> Callable[[], Any]:
    """Build a factory producing configured signalsmith ``Biquad`` objects.

    Mirrors the configuration used by :mod:`nanodsp.effects.filters` so a
    streamed filter matches its stateless counterpart sample-for-sample.
    """
    from nanodsp._core import filters
    from nanodsp._helpers import _resolve_biquad_design

    resolved = _resolve_biquad_design(design)

    def factory() -> Any:
        bq = filters.Biquad()
        configure = getattr(bq, method)
        if octaves is not None:
            configure(freq, octaves, resolved)
        else:
            configure(freq, design=resolved)
        return bq

    return factory


def stateful_lowpass(
    cutoff_hz: float,
    channels: int = 1,
    sample_rate: float = 48000.0,
    octaves: float | None = None,
    design: str | int = "bilinear",
) -> StatefulFilter:
    """Streaming biquad lowpass (see :func:`nanodsp.effects.filters.lowpass`)."""
    from nanodsp._helpers import _hz_to_normalized

    freq = _hz_to_normalized(cutoff_hz, sample_rate)
    return StatefulFilter(
        _biquad_factory("lowpass", freq, octaves, design),
        channels=channels,
        sample_rate=sample_rate,
    )


def stateful_highpass(
    cutoff_hz: float,
    channels: int = 1,
    sample_rate: float = 48000.0,
    octaves: float | None = None,
    design: str | int = "bilinear",
) -> StatefulFilter:
    """Streaming biquad highpass (see :func:`nanodsp.effects.filters.highpass`)."""
    from nanodsp._helpers import _hz_to_normalized

    freq = _hz_to_normalized(cutoff_hz, sample_rate)
    return StatefulFilter(
        _biquad_factory("highpass", freq, octaves, design),
        channels=channels,
        sample_rate=sample_rate,
    )


def stateful_bandpass(
    center_hz: float,
    channels: int = 1,
    sample_rate: float = 48000.0,
    octaves: float | None = None,
    design: str | int = "one_sided",
) -> StatefulFilter:
    """Streaming biquad bandpass (see :func:`nanodsp.effects.filters.bandpass`)."""
    from nanodsp._helpers import _hz_to_normalized

    freq = _hz_to_normalized(center_hz, sample_rate)
    return StatefulFilter(
        _biquad_factory("bandpass", freq, octaves, design),
        channels=channels,
        sample_rate=sample_rate,
    )


def stateful_notch(
    center_hz: float,
    channels: int = 1,
    sample_rate: float = 48000.0,
    octaves: float | None = None,
    design: str | int = "one_sided",
) -> StatefulFilter:
    """Streaming biquad notch (see :func:`nanodsp.effects.filters.notch`)."""
    from nanodsp._helpers import _hz_to_normalized

    freq = _hz_to_normalized(center_hz, sample_rate)
    return StatefulFilter(
        _biquad_factory("notch", freq, octaves, design),
        channels=channels,
        sample_rate=sample_rate,
    )


def stateful_moog_ladder(
    cutoff_hz: float,
    resonance: float = 0.0,
    channels: int = 1,
    sample_rate: float = 48000.0,
) -> StatefulFilter:
    """Streaming DaisySP Moog ladder lowpass.

    Demonstrates that :class:`StatefulFilter` generalizes beyond the signalsmith
    biquads to any stateful per-channel DSP object.
    """
    from nanodsp._helpers import _dsy_filt

    def factory() -> Any:
        f = _dsy_filt.MoogLadder()
        f.init(sample_rate)
        f.set_freq(cutoff_hz)
        f.set_res(resonance)
        return f

    return StatefulFilter(factory, channels=channels, sample_rate=sample_rate)


# ---------------------------------------------------------------------------
# Stateful dynamics
#
# These are the *unlinked* equivalents of nanodsp.effects.dynamics.compress /
# limit: each channel gets its own detector. Linked detection needs all channels
# at once, which the per-channel model cannot express -- see StatefulProcessor's
# notes.
# ---------------------------------------------------------------------------


def stateful_compress(
    ratio: float = 4.0,
    threshold: float = -20.0,
    attack: float = 0.01,
    release: float = 0.1,
    makeup: float = 0.0,
    auto_makeup: bool = False,
    channels: int = 1,
    sample_rate: float = 48000.0,
) -> StatefulProcessor:
    """Streaming compressor, one detector per channel.

    See :func:`nanodsp.effects.dynamics.compress` for the parameters. Unlike
    that function this is per-channel only; there is no linked mode.
    """
    from nanodsp._helpers import _dsy_dyn

    def factory() -> Any:
        c = _dsy_dyn.Compressor()
        c.init(sample_rate)
        c.set_ratio(ratio)
        c.set_threshold(threshold)
        c.set_attack(attack)
        c.set_release(release)
        c.set_makeup(makeup)
        c.auto_makeup(auto_makeup)
        return c

    return StatefulProcessor(factory, channels=channels, sample_rate=sample_rate)


def stateful_limit(
    pre_gain: float = 1.0,
    channels: int = 1,
    sample_rate: float = 48000.0,
) -> StatefulProcessor:
    """Streaming peak limiter, one detector per channel.

    See :func:`nanodsp.effects.dynamics.limit`. The pre-gain is applied through
    the *apply* adapter, since the underlying object takes it per call rather
    than as configuration.
    """
    from nanodsp._helpers import _dsy_dyn

    def factory() -> Any:
        lm = _dsy_dyn.Limiter()
        lm.init()
        return lm

    return StatefulProcessor(
        factory,
        channels=channels,
        sample_rate=sample_rate,
        apply=lambda obj, x: obj.process(x, pre_gain),
    )


# ---------------------------------------------------------------------------
# Stateful filters (DaisySP)
# ---------------------------------------------------------------------------


def stateful_tone_lowpass(
    cutoff_hz: float,
    channels: int = 1,
    sample_rate: float = 48000.0,
) -> StatefulProcessor:
    """Streaming one-pole lowpass (see :func:`nanodsp.effects.filters.tone_lowpass`)."""
    from nanodsp._helpers import _dsy_filt

    def factory() -> Any:
        f = _dsy_filt.Tone()
        f.init(sample_rate)
        f.set_freq(cutoff_hz)
        return f

    return StatefulProcessor(factory, channels=channels, sample_rate=sample_rate)


def stateful_tone_highpass(
    cutoff_hz: float,
    channels: int = 1,
    sample_rate: float = 48000.0,
) -> StatefulProcessor:
    """Streaming one-pole highpass (see :func:`nanodsp.effects.filters.tone_highpass`)."""
    from nanodsp._helpers import _dsy_filt

    def factory() -> Any:
        f = _dsy_filt.ATone()
        f.init(sample_rate)
        f.set_freq(cutoff_hz)
        return f

    return StatefulProcessor(factory, channels=channels, sample_rate=sample_rate)


def stateful_ladder_filter(
    cutoff_hz: float,
    resonance: float = 0.0,
    mode: str = "lp24",
    channels: int = 1,
    sample_rate: float = 48000.0,
) -> StatefulProcessor:
    """Streaming DaisySP ladder filter (see :func:`nanodsp.effects.filters.ladder_filter`)."""
    from nanodsp._helpers import _LADDER_MODE_MAP, _dsy_filt

    if mode not in _LADDER_MODE_MAP:
        raise ValueError(
            f"Unknown ladder mode {mode!r}, valid: {list(_LADDER_MODE_MAP)}"
        )
    # The map holds ints; the binding is typed to the enum, which accepts them.
    resolved: Any = _LADDER_MODE_MAP[mode]

    def factory() -> Any:
        f = _dsy_filt.LadderFilter()
        f.init(sample_rate)
        f.set_freq(cutoff_hz)
        f.set_res(resonance)
        f.set_filter_mode(resolved)
        return f

    return StatefulProcessor(factory, channels=channels, sample_rate=sample_rate)


# ---------------------------------------------------------------------------
# Stateful effects and utilities
# ---------------------------------------------------------------------------


def stateful_dc_block(
    channels: int = 1,
    sample_rate: float = 48000.0,
) -> StatefulProcessor:
    """Streaming DC blocker (see :func:`nanodsp.effects.composed.dc_block`)."""
    from nanodsp._helpers import _dsy_util

    def factory() -> Any:
        d = _dsy_util.DcBlock()
        d.init(sample_rate)
        return d

    return StatefulProcessor(factory, channels=channels, sample_rate=sample_rate)


def stateful_overdrive(
    drive: float = 0.5,
    channels: int = 1,
    sample_rate: float = 48000.0,
) -> StatefulProcessor:
    """Streaming overdrive (see :func:`nanodsp.effects.daisysp.overdrive`)."""
    from nanodsp._helpers import _dsy_fx

    def factory() -> Any:
        o = _dsy_fx.Overdrive()
        o.init()
        o.set_drive(drive)
        return o

    return StatefulProcessor(factory, channels=channels, sample_rate=sample_rate)


def stateful_tremolo(
    freq: float = 5.0,
    depth: float = 0.5,
    channels: int = 1,
    sample_rate: float = 48000.0,
) -> StatefulProcessor:
    """Streaming tremolo (see :func:`nanodsp.effects.daisysp.tremolo`)."""
    from nanodsp._helpers import _dsy_fx

    def factory() -> Any:
        t = _dsy_fx.Tremolo()
        t.init(sample_rate)
        t.set_freq(freq)
        t.set_depth(depth)
        return t

    return StatefulProcessor(factory, channels=channels, sample_rate=sample_rate)


def stateful_autowah(
    wah: float = 0.5,
    dry_wet: float = 0.5,
    level: float = 0.5,
    channels: int = 1,
    sample_rate: float = 48000.0,
) -> StatefulProcessor:
    """Streaming auto-wah (see :func:`nanodsp.effects.composed.autowah`)."""
    from nanodsp._helpers import _dsy_fx

    def factory() -> Any:
        a = _dsy_fx.Autowah()
        a.init(sample_rate)
        a.set_wah(wah)
        a.set_dry_wet(dry_wet)
        a.set_level(level)
        return a

    return StatefulProcessor(factory, channels=channels, sample_rate=sample_rate)


def stateful_flanger(
    lfo_freq: float = 0.3,
    lfo_depth: float = 0.5,
    delay_ms: float = 5.0,
    feedback: float = 0.2,
    channels: int = 1,
    sample_rate: float = 48000.0,
) -> StatefulProcessor:
    """Streaming flanger (see :func:`nanodsp.effects.daisysp.flanger`)."""
    from nanodsp._helpers import _dsy_fx

    def factory() -> Any:
        f = _dsy_fx.Flanger()
        f.init(sample_rate)
        f.set_lfo_freq(lfo_freq)
        f.set_lfo_depth(lfo_depth)
        f.set_delay_ms(delay_ms)
        f.set_feedback(feedback)
        return f

    return StatefulProcessor(factory, channels=channels, sample_rate=sample_rate)


def stateful_phaser(
    lfo_freq: float = 0.3,
    lfo_depth: float = 0.5,
    freq: float = 400.0,
    feedback: float = 0.2,
    channels: int = 1,
    sample_rate: float = 48000.0,
) -> StatefulProcessor:
    """Streaming phaser (see :func:`nanodsp.effects.daisysp.phaser`)."""
    from nanodsp._helpers import _dsy_fx

    def factory() -> Any:
        p = _dsy_fx.Phaser()
        p.init(sample_rate)
        p.set_lfo_freq(lfo_freq)
        p.set_lfo_depth(lfo_depth)
        p.set_freq(freq)
        p.set_feedback(feedback)
        return p

    return StatefulProcessor(factory, channels=channels, sample_rate=sample_rate)


def stateful_schroeder_reverb(
    feedback: float = 0.7,
    diffusion: float = 0.5,
    mod_depth: float = 0.0,
    channels: int = 1,
    sample_rate: float = 48000.0,
) -> StatefulProcessor:
    """Streaming Schroeder reverb (see :func:`nanodsp.effects.reverb.schroeder_reverb`).

    A true per-channel reverb, unlike the FDN ``reverb``, so it streams.
    """
    from nanodsp._core import fxdsp as _fxdsp

    def factory() -> Any:
        rev = _fxdsp.SchroederReverb()
        rev.init(float(sample_rate))
        rev.feedback = feedback
        rev.diffusion = diffusion
        rev.set_mod_depth(mod_depth)
        return rev

    return StatefulProcessor(factory, channels=channels, sample_rate=sample_rate)


def stateful_moorer_reverb(
    feedback: float = 0.7,
    diffusion: float = 0.7,
    mod_depth: float = 0.1,
    channels: int = 1,
    sample_rate: float = 48000.0,
) -> StatefulProcessor:
    """Streaming Moorer reverb (see :func:`nanodsp.effects.reverb.moorer_reverb`)."""
    from nanodsp._core import fxdsp as _fxdsp

    def factory() -> Any:
        rev = _fxdsp.MoorerReverb()
        rev.init(float(sample_rate))
        rev.feedback = feedback
        rev.diffusion = diffusion
        rev.set_mod_depth(mod_depth)
        return rev

    return StatefulProcessor(factory, channels=channels, sample_rate=sample_rate)


# ---------------------------------------------------------------------------
# CLI effect name -> stateful constructor
# ---------------------------------------------------------------------------

#: Maps a chainable effect name (as accepted by ``nanodsp process -f``) to the
#: constructor that streams it. Only effects whose channels are independent
#: appear here; see :class:`StatefulProcessor` for what cannot stream and why.
#: Parameter names match the stateless function, so coerced CLI parameters pass
#: straight through.
STREAMABLE: dict[str, Callable[..., StatefulProcessor]] = {
    "lowpass": stateful_lowpass,
    "highpass": stateful_highpass,
    "bandpass": stateful_bandpass,
    "notch": stateful_notch,
    "moog_ladder": stateful_moog_ladder,
    "ladder_filter": stateful_ladder_filter,
    "tone_lowpass": stateful_tone_lowpass,
    "tone_highpass": stateful_tone_highpass,
    "compress": stateful_compress,
    "limit": stateful_limit,
    "dc_block": stateful_dc_block,
    "overdrive": stateful_overdrive,
    "tremolo": stateful_tremolo,
    "autowah": stateful_autowah,
    "flanger": stateful_flanger,
    "phaser": stateful_phaser,
    "schroeder_reverb": stateful_schroeder_reverb,
    "moorer_reverb": stateful_moorer_reverb,
}

#: Effects in :data:`STREAMABLE` whose streaming form differs from the
#: whole-buffer form in a way the user should be told about, rather than
#: silently getting different audio.
STREAMING_CAVEATS: dict[str, str] = {
    "compress": "runs unlinked (per-channel detectors); the whole-buffer form links channels by default",
    "limit": "runs unlinked (per-channel detectors); the whole-buffer form links channels by default",
}


def build_streaming_chain(
    steps: "list[dict]", channels: int, sample_rate: float
) -> list[StatefulProcessor]:
    """Build stateful processors for a CLI chain.

    Parameters
    ----------
    steps : list of dict
        Chain steps as produced by the CLI, each with ``name`` and ``params``.
    channels, sample_rate
        Taken from the file being processed.

    Raises
    ------
    ValueError
        If any step has no streaming equivalent, naming all of them at once so
        the user can fix the whole chain in one go.
    """
    missing = [s["name"] for s in steps if s["name"] not in STREAMABLE]
    if missing:
        raise ValueError(
            f"cannot stream: {', '.join(sorted(set(missing)))} "
            f"{'has' if len(set(missing)) == 1 else 'have'} no streaming form. "
            f"Streamable effects: {', '.join(sorted(STREAMABLE))}"
        )
    procs = []
    for step in steps:
        params = dict(step["params"])
        # The whole-buffer compressor takes `link`, which per-channel streaming
        # cannot honour; STREAMING_CAVEATS reports the difference.
        params.pop("link", None)
        procs.append(
            STREAMABLE[step["name"]](
                **params, channels=channels, sample_rate=sample_rate
            )
        )
    return procs

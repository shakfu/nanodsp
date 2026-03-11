"""Streaming / real-time audio processing infrastructure.

Provides ring buffers, block processors, and overlap-add utilities for
processing audio in fixed-size chunks.
"""

from __future__ import annotations

import numpy as np

from nanodsp.buffer import AudioBuffer


# ---------------------------------------------------------------------------
# Ring Buffer
# ---------------------------------------------------------------------------


class RingBuffer:
    """Lock-free-style ring buffer for streaming audio.

    Stores planar float32 audio in a circular buffer with independent
    read and write positions.

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

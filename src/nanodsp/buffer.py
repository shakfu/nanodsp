"""AudioBuffer -- typed, metadata-carrying wrapper around planar float32 numpy arrays.

Provides a uniform Python-level abstraction for audio data that interoperates
transparently with the 1D and 2D C++ bindings in nanodsp._core.
"""

from __future__ import annotations

from typing import Any, Callable, Literal, overload

import numpy as np
from numpy.typing import ArrayLike


class AudioBuffer:
    """A 2D ``[channels, frames]`` float32 audio buffer with metadata.

    Parameters
    ----------
    data : array-like or AudioBuffer
        Audio samples.  1D input is normalised to ``[1, N]``.
    sample_rate : float
        Sample rate in Hz.  Must be positive (> 0).
    channel_layout : str or None
        E.g. ``'mono'``, ``'stereo'``.  Inferred from channel count when *None*.
    label : str or None
        Free-form label carried as metadata.
    copy : bool
        Copy *data* into storage the buffer owns (the default).  Pass ``False``
        only when the caller is handing over an array nothing else references,
        which avoids one full-buffer copy.

        The default is ``True`` because the alternative silently aliases: a
        contiguous float32 array passed in would otherwise be adopted as-is, so
        later writes through the buffer would mutate the caller's array while an
        identical call with a non-contiguous or non-float32 array -- which has
        to be converted -- would not.  Whether a call aliased depended on the
        dtype and memory layout of the argument, which is not a distinction
        callers can reasonably track.

    Raises
    ------
    ValueError
        If *data* is not 1D or 2D, or if *sample_rate* is not positive.
    """

    __slots__ = ("_data", "_sample_rate", "_channel_layout", "_label")

    def __init__(
        self,
        data: ArrayLike | AudioBuffer,
        sample_rate: float = 48000.0,
        channel_layout: str | None = None,
        label: str | None = None,
        copy: bool = True,
    ) -> None:
        if isinstance(data, AudioBuffer):
            arr = data._data
        else:
            arr = np.asarray(data, dtype=np.float32)

        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        elif arr.ndim != 2:
            raise ValueError(f"AudioBuffer requires 1D or 2D data, got {arr.ndim}D")

        # Ensure contiguous float32. Either conversion already produces storage
        # nothing else holds, so an explicit copy afterwards would be redundant.
        converted = False
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
            converted = True
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
            converted = True

        if copy and not converted:
            arr = arr.copy()

        sr = float(sample_rate)
        if sr <= 0:
            raise ValueError(f"sample_rate must be positive, got {sr}")

        self._data: np.ndarray = arr
        self._sample_rate: float = sr
        self._label: str | None = label

        if channel_layout is None:
            ch = arr.shape[0]
            if ch == 1:
                channel_layout = "mono"
            elif ch == 2:
                channel_layout = "stereo"
        self._channel_layout: str | None = channel_layout

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def data(self) -> np.ndarray:
        """Raw 2D ``[channels, frames]`` float32 array."""
        return self._data

    @property
    def sample_rate(self) -> float:
        return self._sample_rate

    @property
    def channels(self) -> int:
        return self._data.shape[0]

    @property
    def frames(self) -> int:
        return self._data.shape[1]

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return self._data.shape[1] / self._sample_rate

    @property
    def channel_layout(self) -> str | None:
        return self._channel_layout

    @property
    def label(self) -> str | None:
        return self._label

    @property
    def dtype(self) -> np.dtype:
        return self._data.dtype

    # ------------------------------------------------------------------
    # Channel access
    # ------------------------------------------------------------------

    def channel(self, i: int) -> np.ndarray:
        """Return a 1D numpy view of channel *i*."""
        if i < 0 or i >= self.channels:
            raise IndexError(
                f"Channel {i} out of range for {self.channels}-channel buffer "
                f"(valid: 0-{self.channels - 1})"
            )
        return self._data[i]

    @property
    def mono(self) -> np.ndarray:
        """1D numpy view -- only valid when ``channels == 1``."""
        if self.channels != 1:
            raise ValueError(f"mono requires 1-channel buffer, got {self.channels}")
        return self._data[0]

    @overload
    def __getitem__(self, key: int) -> np.ndarray: ...
    @overload
    def __getitem__(self, key: slice) -> AudioBuffer: ...
    @overload
    def __getitem__(self, key: tuple[int, slice]) -> np.ndarray: ...

    def __getitem__(
        self, key: int | slice | tuple[int, slice]
    ) -> np.ndarray | AudioBuffer:
        if isinstance(key, int):
            return self.channel(key)
        if isinstance(key, slice):
            sub = self._data[key]
            return AudioBuffer(
                sub,
                sample_rate=self._sample_rate,
                channel_layout=None,
                label=self._label,
            )
        if isinstance(key, tuple):
            ch, frame_slice = key
            return self._data[ch, frame_slice]
        raise TypeError(f"Invalid index type: {type(key)}")

    # ------------------------------------------------------------------
    # Numpy interop
    # ------------------------------------------------------------------

    def __array__(
        self, dtype: np.dtype | None = None, copy: bool | None = None
    ) -> np.ndarray:
        """numpy conversion protocol.

        *copy* is honoured rather than ignored: numpy passes ``True`` when the
        caller asked for a copy (``np.array(buf)``) and ``False`` when a copy is
        forbidden (``np.asarray(buf, copy=False)``), and returning shared
        storage in the first case would defeat the request.
        """
        if dtype is not None and np.dtype(dtype) != self._data.dtype:
            # astype always allocates, satisfying copy=True and copy=None alike.
            if copy is False:
                raise ValueError(
                    "cannot avoid a copy while converting AudioBuffer "
                    f"from {self._data.dtype} to {np.dtype(dtype)}"
                )
            return self._data.astype(dtype)
        if copy:
            return self._data.copy()
        return self._data

    if __import__("sys").version_info >= (3, 12):

        def __buffer__(self, flags: int, /) -> memoryview:
            return self._data.__buffer__(flags)

    def __len__(self) -> int:
        """Number of frames (not channels)."""
        return self.frames

    def __repr__(self) -> str:
        parts = [
            f"channels={self.channels}",
            f"frames={self.frames}",
            f"sr={self.sample_rate}",
        ]
        if self._channel_layout is not None:
            parts.append(f"layout='{self._channel_layout}'")
        if self._label is not None:
            parts.append(f"label='{self._label}'")
        return f"AudioBuffer({', '.join(parts)})"

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def zeros(
        cls,
        channels: int,
        frames: int,
        sample_rate: float = 48000.0,
        **kw,
    ) -> AudioBuffer:
        return cls(
            np.zeros((channels, frames), dtype=np.float32),
            sample_rate=sample_rate,
            **kw,
        )

    @classmethod
    def ones(
        cls,
        channels: int,
        frames: int,
        sample_rate: float = 48000.0,
        **kw,
    ) -> AudioBuffer:
        return cls(
            np.ones((channels, frames), dtype=np.float32),
            sample_rate=sample_rate,
            **kw,
        )

    @classmethod
    def impulse(
        cls,
        channels: int = 1,
        frames: int = 1024,
        sample_rate: float = 48000.0,
        **kw,
    ) -> AudioBuffer:
        """Unit impulse at frame 0 in every channel."""
        arr = np.zeros((channels, frames), dtype=np.float32)
        arr[:, 0] = 1.0
        return cls(arr, sample_rate=sample_rate, **kw)

    @classmethod
    def sine(
        cls,
        freq: float,
        channels: int = 1,
        frames: int = 4096,
        sample_rate: float = 48000.0,
        **kw,
    ) -> AudioBuffer:
        t = np.arange(frames, dtype=np.float32) / sample_rate
        row = np.sin(2.0 * np.pi * freq * t).astype(np.float32)
        arr = np.tile(row, (channels, 1))
        return cls(arr, sample_rate=sample_rate, **kw)

    @classmethod
    def noise(
        cls,
        channels: int = 1,
        frames: int = 4096,
        sample_rate: float = 48000.0,
        seed: int | None = None,
        **kw,
    ) -> AudioBuffer:
        rng = np.random.default_rng(seed)
        arr = rng.standard_normal((channels, frames)).astype(np.float32)
        return cls(arr, sample_rate=sample_rate, **kw)

    @classmethod
    def from_numpy(
        cls,
        arr: np.ndarray,
        sample_rate: float = 48000.0,
        **kw,
    ) -> AudioBuffer:
        """Explicit alias for the constructor."""
        return cls(arr, sample_rate=sample_rate, **kw)

    # ------------------------------------------------------------------
    # Channel operations
    # ------------------------------------------------------------------

    def to_mono(
        self, method: Literal["mean", "left", "right", "sum"] = "mean"
    ) -> AudioBuffer:
        """Mix down to mono.

        *method*: ``'mean'`` (default), ``'left'``, ``'right'``, ``'sum'``.
        """
        if method == "mean":
            mixed = self._data.mean(axis=0, keepdims=True)
        elif method == "left":
            mixed = self._data[0:1]
        elif method == "right":
            mixed = self._data[-1:]
        elif method == "sum":
            mixed = self._data.sum(axis=0, keepdims=True)
        else:
            raise ValueError(f"Unknown mono method: {method!r}")
        return AudioBuffer(
            mixed.astype(np.float32),
            sample_rate=self._sample_rate,
            label=self._label,
        )

    def to_channels(self, n: int) -> AudioBuffer:
        """Upmix mono to *n* channels by copying, or error if incompatible."""
        if self.channels == n:
            return self.copy()
        if self.channels != 1:
            raise ValueError(
                f"Cannot upmix {self.channels}-channel buffer to {n} channels; "
                "source must be mono"
            )
        arr = np.tile(self._data, (n, 1))
        return AudioBuffer(
            arr,
            sample_rate=self._sample_rate,
            label=self._label,
        )

    def split(self) -> list[AudioBuffer]:
        """Split into a list of mono AudioBuffers, one per channel."""
        return [
            AudioBuffer(
                self._data[i : i + 1].copy(),
                sample_rate=self._sample_rate,
                label=self._label,
            )
            for i in range(self.channels)
        ]

    @staticmethod
    def concat_channels(*buffers: AudioBuffer) -> AudioBuffer:
        """Stack channels from multiple AudioBuffers."""
        if not buffers:
            raise ValueError("At least one buffer required")
        # Flatten in case called with a single list argument
        if len(buffers) == 1 and isinstance(buffers[0], (list, tuple)):
            buffers = tuple(buffers[0])
        sr = buffers[0].sample_rate
        for b in buffers[1:]:
            if b.sample_rate != sr:
                raise ValueError(f"sample_rate mismatch: {sr} vs {b.sample_rate}")
        arr = np.concatenate([b.data for b in buffers], axis=0)
        return AudioBuffer(arr, sample_rate=sr, label=buffers[0].label)

    # ------------------------------------------------------------------
    # Time slicing
    # ------------------------------------------------------------------

    def slice(self, start_frame: int, end_frame: int) -> AudioBuffer:
        """Return frames ``[start_frame:end_frame]`` as an independent buffer.

        The result never shares storage with this buffer.  It previously did so
        for mono and not for multi-channel input -- a frame slice of a
        single-row array is still contiguous and was adopted as-is, while the
        same slice of a two-row array is not and had to be copied -- so whether
        writes propagated back depended on the channel count.

        For a genuine view, index the underlying array directly:
        ``buf.data[:, start:end]``.
        """
        return AudioBuffer(
            self._data[:, start_frame:end_frame],
            sample_rate=self._sample_rate,
            channel_layout=self._channel_layout,
            label=self._label,
        )

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def _check_sr(self, other: AudioBuffer) -> None:
        if self._sample_rate != other._sample_rate:
            raise ValueError(
                f"Sample rate mismatch: {self._sample_rate} vs {other._sample_rate}"
            )

    @staticmethod
    def _broadcast(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Handle mono<->multi channel broadcasting."""
        if a.shape[0] == 1 and b.shape[0] != 1:
            a = np.broadcast_to(a, b.shape)
        elif b.shape[0] == 1 and a.shape[0] != 1:
            b = np.broadcast_to(b, a.shape)
        return a, b

    def __add__(self, other: AudioBuffer | float | int) -> AudioBuffer:
        if isinstance(other, AudioBuffer):
            self._check_sr(other)
            a, b = self._broadcast(self._data, other._data)
            return AudioBuffer(
                (a + b).astype(np.float32),
                sample_rate=self._sample_rate,
                label=self._label,
            )
        return AudioBuffer(
            (self._data + np.float32(other)),
            sample_rate=self._sample_rate,
            label=self._label,
        )

    def __radd__(self, other: AudioBuffer | float | int) -> AudioBuffer:
        return self.__add__(other)

    def __sub__(self, other: AudioBuffer | float | int) -> AudioBuffer:
        if isinstance(other, AudioBuffer):
            self._check_sr(other)
            a, b = self._broadcast(self._data, other._data)
            return AudioBuffer(
                (a - b).astype(np.float32),
                sample_rate=self._sample_rate,
                label=self._label,
            )
        return AudioBuffer(
            (self._data - np.float32(other)),
            sample_rate=self._sample_rate,
            label=self._label,
        )

    def __rsub__(self, other: float | int) -> AudioBuffer:
        return AudioBuffer(
            (np.float32(other) - self._data),
            sample_rate=self._sample_rate,
            label=self._label,
        )

    def __mul__(self, other: AudioBuffer | float | int) -> AudioBuffer:
        if isinstance(other, AudioBuffer):
            self._check_sr(other)
            a, b = self._broadcast(self._data, other._data)
            return AudioBuffer(
                (a * b).astype(np.float32),
                sample_rate=self._sample_rate,
                label=self._label,
            )
        return AudioBuffer(
            (self._data * np.float32(other)),
            sample_rate=self._sample_rate,
            label=self._label,
        )

    def __rmul__(self, other: AudioBuffer | float | int) -> AudioBuffer:
        return self.__mul__(other)

    def __truediv__(self, other: AudioBuffer | float | int) -> AudioBuffer:
        if isinstance(other, AudioBuffer):
            self._check_sr(other)
            a, b = self._broadcast(self._data, other._data)
            return AudioBuffer(
                (a / b).astype(np.float32),
                sample_rate=self._sample_rate,
                label=self._label,
            )
        return AudioBuffer(
            (self._data / np.float32(other)),
            sample_rate=self._sample_rate,
            label=self._label,
        )

    def __neg__(self) -> AudioBuffer:
        return AudioBuffer(
            -self._data,
            sample_rate=self._sample_rate,
            channel_layout=self._channel_layout,
            label=self._label,
        )

    def gain_db(self, db: float) -> AudioBuffer:
        """Return a new buffer scaled by ``10**(db/20)``."""
        factor = np.float32(10.0 ** (db / 20.0))
        return AudioBuffer(
            self._data * factor,
            sample_rate=self._sample_rate,
            channel_layout=self._channel_layout,
            label=self._label,
        )

    def pipe(
        self, fn: Callable[..., AudioBuffer], *args: Any, **kwargs: Any
    ) -> AudioBuffer:
        """Chain a DSP function: ``buf.pipe(dsp.lowpass, 5000)``.

        Calls ``fn(self, *args, **kwargs)`` and validates the return type.
        """
        result = fn(self, *args, **kwargs)
        if not isinstance(result, AudioBuffer):
            raise TypeError(
                f"pipe() requires fn to return AudioBuffer, got {type(result).__name__}"
            )
        return result

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    @classmethod
    def from_file(cls, path: str) -> AudioBuffer:
        """Read an audio file (WAV/FLAC, detected by extension)."""
        from nanodsp.io import read as _read

        return _read(path)

    def write(self, path: str, bit_depth: int = 16) -> None:
        """Write this buffer to an audio file (WAV/FLAC, detected by extension)."""
        from nanodsp.io import write as _write

        _write(path, self, bit_depth=bit_depth)

    # ------------------------------------------------------------------
    # Copy
    # ------------------------------------------------------------------

    def copy(self) -> AudioBuffer:
        """Deep copy with independent numpy storage."""
        return AudioBuffer(
            self._data.copy(),
            sample_rate=self._sample_rate,
            channel_layout=self._channel_layout,
            label=self._label,
        )

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def ensure_1d(self, channel: int = 0) -> np.ndarray:
        """Return a contiguous 1D float32 view for 1D C++ bindings."""
        return np.ascontiguousarray(self._data[channel])

    def ensure_2d(self) -> np.ndarray:
        """Return the contiguous 2D float32 array for 2D C++ bindings."""
        return self._data

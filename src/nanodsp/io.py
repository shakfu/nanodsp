"""Audio file I/O for AudioBuffer.

Supported formats (detected by extension):
  .wav  -- 8/16/24/32-bit PCM read, 16/24-bit PCM write (stdlib ``wave``)
  .flac -- 16/24-bit read/write (CHOC FLAC codec, zero external dependencies)
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from nanodsp.buffer import AudioBuffer


def _decode_wav_frames(
    raw_bytes: bytes,
    sampwidth: int,
    n_channels: int,
    n_frames: int,
    sample_rate: int,
    source: str = "<bytes>",
) -> AudioBuffer:
    """Decode raw WAV sample bytes into an AudioBuffer."""
    total_samples = n_frames * n_channels

    if sampwidth == 1:
        samples = np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32)
        samples = (samples - 128.0) / 128.0
    elif sampwidth == 2:
        samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)
        samples = samples / 32768.0
    elif sampwidth == 3:
        raw = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(-1, 3)
        padded = np.zeros((len(raw), 4), dtype=np.uint8)
        padded[:, 0:3] = raw
        padded[:, 3] = np.where(raw[:, 2] & 0x80, 0xFF, 0x00)
        samples = padded.view(np.int32).flatten().astype(np.float32)
        samples = samples / 8388608.0
    elif sampwidth == 4:
        samples = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32)
        samples = samples / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth} bytes in {source}")

    if len(samples) != total_samples:
        raise ValueError(
            f"Expected {total_samples} samples, got {len(samples)} in {source}"
        )

    if n_channels == 1:
        data = samples.reshape(1, -1)
    else:
        data = samples.reshape(-1, n_channels).T

    data = np.ascontiguousarray(data, dtype=np.float32)
    return AudioBuffer(data, sample_rate=float(sample_rate))


def _encode_wav_frames(buf: AudioBuffer, bit_depth: int) -> bytes:
    """Encode an AudioBuffer into raw WAV sample bytes."""
    data = buf.data.copy()
    np.clip(data, -1.0, 1.0, out=data)
    interleaved = data.T.flatten()

    if bit_depth == 16:
        scaled = (interleaved * 32767.0).astype(np.int16)
        return scaled.tobytes()
    else:  # 24
        scaled = np.clip(interleaved * 8388607.0, -8388608.0, 8388607.0).astype(
            np.int32
        )
        bytes_4 = scaled.view(np.uint8).reshape(-1, 4)
        return bytes_4[:, :3].tobytes()


def read_wav(path: str | Path) -> AudioBuffer:
    """Read a WAV file and return an AudioBuffer.

    Supports 8-bit unsigned, 16-bit signed, 24-bit signed, and 32-bit signed PCM.
    Output is float32 normalized to [-1, 1].
    """
    path = Path(path)
    try:
        with wave.open(str(path), "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            raw_bytes = wf.readframes(n_frames)
    except wave.Error as e:
        raise wave.Error(f"{e} (file: '{path}')") from e

    return _decode_wav_frames(
        raw_bytes, sampwidth, n_channels, n_frames, sample_rate, source=f"'{path}'"
    )


def write_wav(
    path: str | Path,
    buf: AudioBuffer,
    bit_depth: int = 16,
) -> None:
    """Write an AudioBuffer to a WAV file.

    Parameters
    ----------
    path : str or Path
        Output file path.
    buf : AudioBuffer
        Audio data to write.
    bit_depth : int
        Output bit depth: 16 or 24.
    """
    if bit_depth not in (16, 24):
        raise ValueError(f"Unsupported bit_depth: {bit_depth} (use 16 or 24)")

    path = Path(path)
    raw_bytes = _encode_wav_frames(buf, bit_depth)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(buf.channels)
        wf.setsampwidth(bit_depth // 8)
        wf.setframerate(int(buf.sample_rate))
        wf.writeframes(raw_bytes)


def read_wav_bytes(data: bytes) -> AudioBuffer:
    """Read WAV data from raw bytes and return an AudioBuffer.

    Supports 8/16/24/32-bit PCM. Output is float32 normalized to [-1, 1].
    """
    import io as _io

    bio = _io.BytesIO(data)
    with wave.open(bio, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw_bytes = wf.readframes(n_frames)

    return _decode_wav_frames(raw_bytes, sampwidth, n_channels, n_frames, sample_rate)


def write_wav_bytes(buf: AudioBuffer, bit_depth: int = 16) -> bytes:
    """Serialize an AudioBuffer to WAV bytes.

    Parameters
    ----------
    buf : AudioBuffer
        Audio data to write.
    bit_depth : int
        Output bit depth: 16 or 24.

    Returns
    -------
    bytes
        WAV file content.
    """
    import io as _io

    if bit_depth not in (16, 24):
        raise ValueError(f"Unsupported bit_depth: {bit_depth} (use 16 or 24)")

    raw_bytes = _encode_wav_frames(buf, bit_depth)

    bio = _io.BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(buf.channels)
        wf.setsampwidth(bit_depth // 8)
        wf.setframerate(int(buf.sample_rate))
        wf.writeframes(raw_bytes)

    return bio.getvalue()


def read_flac(path: str | Path) -> AudioBuffer:
    """Read a FLAC file and return an AudioBuffer.

    Output is float32 normalized to [-1, 1].
    """
    from nanodsp._core import choc

    path = Path(path)
    data, sample_rate = choc.read_flac(str(path))
    return AudioBuffer(data, sample_rate=sample_rate)


def write_flac(
    path: str | Path,
    buf: AudioBuffer,
    bit_depth: int = 16,
) -> None:
    """Write an AudioBuffer to a FLAC file.

    Parameters
    ----------
    path : str or Path
        Output file path.
    buf : AudioBuffer
        Audio data to write.
    bit_depth : int
        Output bit depth: 16 or 24.
    """
    from nanodsp._core import choc

    if bit_depth not in (16, 24):
        raise ValueError(f"Unsupported bit_depth: {bit_depth} (use 16 or 24)")

    path = Path(path)
    data = buf.data.copy()
    np.clip(data, -1.0, 1.0, out=data)
    choc.write_flac(str(path), data, buf.sample_rate, bit_depth)


_FORMAT_READERS = {
    ".wav": read_wav,
    ".flac": read_flac,
}

_FORMAT_WRITERS = {
    ".wav": write_wav,
    ".flac": write_flac,
}


def read(path: str | Path) -> AudioBuffer:
    """Read an audio file and return an AudioBuffer.

    Format is detected by file extension (.wav, .flac).
    """
    path = Path(path)
    ext = path.suffix.lower()
    reader = _FORMAT_READERS.get(ext)
    if reader is None:
        supported = ", ".join(sorted(_FORMAT_READERS))
        raise ValueError(f"Unsupported audio format '{ext}'. Supported: {supported}")
    return reader(path)


def write(
    path: str | Path,
    buf: AudioBuffer,
    bit_depth: int = 16,
) -> None:
    """Write an AudioBuffer to an audio file.

    Format is detected by file extension (.wav, .flac).

    Parameters
    ----------
    path : str or Path
        Output file path.
    buf : AudioBuffer
        Audio data to write.
    bit_depth : int
        Output bit depth: 16 or 24.
    """
    path = Path(path)
    ext = path.suffix.lower()
    writer = _FORMAT_WRITERS.get(ext)
    if writer is None:
        supported = ", ".join(sorted(_FORMAT_WRITERS))
        raise ValueError(f"Unsupported audio format '{ext}'. Supported: {supported}")
    writer(path, buf, bit_depth=bit_depth)

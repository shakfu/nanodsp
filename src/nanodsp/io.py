"""Audio file I/O for AudioBuffer.

Supported formats (detected by extension):
  .wav  -- 8/16/24/32-bit PCM and 32/64-bit IEEE float, read and write
  .flac -- 16/24-bit read/write (CHOC FLAC codec, zero external dependencies)

WAV handling parses RIFF chunks directly rather than going through the stdlib
``wave`` module.  ``wave`` accepts only ``WAVE_FORMAT_PCM`` and raises on
anything else, which rejected two very common cases outright: 32-bit float WAV
(what ffmpeg, Audacity and most DAWs write from a float pipeline) and
``WAVE_FORMAT_EXTENSIBLE``, which is what many writers emit for anything above
two channels or 16 bits.  It also cannot *write* float at all.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from nanodsp.buffer import AudioBuffer

# RIFF format tags.
_WAVE_FORMAT_PCM = 0x0001
_WAVE_FORMAT_IEEE_FLOAT = 0x0003
_WAVE_FORMAT_EXTENSIBLE = 0xFFFE

_FORMAT_NAMES = {
    _WAVE_FORMAT_PCM: "PCM",
    _WAVE_FORMAT_IEEE_FLOAT: "IEEE float",
    0x0006: "A-law",
    0x0007: "mu-law",
    0x0011: "IMA ADPCM",
    0x0055: "MP3",
}

# bit_depth values accepted on write. 16/24 are signed PCM; 32/64 are IEEE
# float, which is what "32-bit WAV" almost always means in an audio workflow
# (32-bit *integer* WAV is rare and gains nothing over 24-bit here, since the
# in-memory representation is float32 either way).
_PCM_WRITE_DEPTHS = (16, 24)
_FLOAT_WRITE_DEPTHS = (32, 64)
_WRITE_DEPTHS = _PCM_WRITE_DEPTHS + _FLOAT_WRITE_DEPTHS


def _parse_riff(data: bytes, source: str = "<bytes>") -> tuple[dict, bytes]:
    """Walk a RIFF/WAVE byte stream, returning ``(fmt_info, sample_bytes)``.

    Chunks other than ``fmt `` and ``data`` (``LIST``, ``fact``, ``cue ``, ...)
    are skipped.  Chunk bodies are word-aligned: an odd-sized body is followed
    by a pad byte that is not counted in the declared size.
    """
    if len(data) < 12:
        raise ValueError(f"Not a RIFF file: too short ({len(data)} bytes) in {source}")

    riff, _riff_size, wave_id = struct.unpack_from("<4sI4s", data, 0)
    if riff in (b"RF64", b"BW64"):
        raise ValueError(
            f"{riff.decode()} (>4 GB) files are not supported in {source}; "
            "convert to WAV or FLAC first"
        )
    if riff != b"RIFF" or wave_id != b"WAVE":
        raise ValueError(f"Not a RIFF/WAVE file in {source}")

    fmt: dict | None = None
    sample_bytes: bytes | None = None
    pos = 12
    while pos + 8 <= len(data):
        chunk_id, chunk_size = struct.unpack_from("<4sI", data, pos)
        body = pos + 8
        end = body + chunk_size
        if end > len(data):
            # Truncated final chunk: salvage what is present rather than
            # failing, which matches how most players treat a cut-off file.
            end = len(data)

        if chunk_id == b"fmt ":
            if chunk_size < 16:
                raise ValueError(f"Malformed fmt chunk in {source}")
            tag, channels, rate, _bps, _align, bits = struct.unpack_from(
                "<HHIIHH", data, body
            )
            if tag == _WAVE_FORMAT_EXTENSIBLE:
                # The real format lives in the first two bytes of the SubFormat
                # GUID, 24 bytes into the chunk body.
                if chunk_size < 40:
                    raise ValueError(
                        f"Malformed WAVE_FORMAT_EXTENSIBLE fmt chunk in {source}"
                    )
                (tag,) = struct.unpack_from("<H", data, body + 24)
            fmt = {
                "tag": tag,
                "channels": channels,
                "sample_rate": rate,
                "bits": bits,
            }
        elif chunk_id == b"data":
            sample_bytes = data[body:end]

        pos = end + (end & 1)  # skip the pad byte on odd-sized chunks

    if fmt is None:
        raise ValueError(f"No fmt chunk found in {source}")
    if sample_bytes is None:
        raise ValueError(f"No data chunk found in {source}")
    if fmt["channels"] < 1:
        raise ValueError(f"Invalid channel count {fmt['channels']} in {source}")
    return fmt, sample_bytes


def _decode_wav_bytes(raw: bytes, source: str = "<bytes>") -> AudioBuffer:
    """Decode a complete RIFF/WAVE byte stream into an AudioBuffer."""
    fmt, sample_bytes = _parse_riff(raw, source)
    tag = fmt["tag"]
    bits = fmt["bits"]
    channels = fmt["channels"]

    if tag == _WAVE_FORMAT_IEEE_FLOAT:
        if bits == 32:
            samples = np.frombuffer(sample_bytes, dtype="<f4").astype(np.float32)
        elif bits == 64:
            samples = np.frombuffer(sample_bytes, dtype="<f8").astype(np.float32)
        else:
            raise ValueError(f"Unsupported float width {bits}-bit in {source}")
    elif tag == _WAVE_FORMAT_PCM:
        sampwidth = bits // 8
        n_frames = len(sample_bytes) // (sampwidth * channels) if sampwidth else 0
        return _decode_wav_frames(
            sample_bytes,
            sampwidth,
            channels,
            n_frames,
            fmt["sample_rate"],
            source=source,
        )
    else:
        name = _FORMAT_NAMES.get(tag, f"0x{tag:04X}")
        raise ValueError(
            f"Unsupported WAV encoding: {name} in {source}. "
            "Supported: PCM (8/16/24/32-bit) and IEEE float (32/64-bit)."
        )

    # Trim any trailing partial frame rather than failing on a ragged file.
    usable = (len(samples) // channels) * channels
    samples = samples[:usable]
    if channels == 1:
        data = samples.reshape(1, -1)
    else:
        data = samples.reshape(-1, channels).T
    return AudioBuffer(
        np.ascontiguousarray(data, dtype=np.float32),
        sample_rate=float(fmt["sample_rate"]),
    )


def _decode_wav_frames(
    raw_bytes: bytes,
    sampwidth: int,
    n_channels: int,
    n_frames: int,
    sample_rate: int,
    source: str = "<bytes>",
) -> AudioBuffer:
    """Decode raw WAV sample bytes into an AudioBuffer.

    Integer dtypes are spelled little-endian explicitly ("<i2" rather than
    ``np.int16``) because WAV is defined as little-endian regardless of host
    byte order; the native-order spelling would silently byte-swap on a
    big-endian machine.
    """
    total_samples = n_frames * n_channels

    if sampwidth == 1:
        samples = np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32)
        samples = (samples - 128.0) / 128.0
    elif sampwidth == 2:
        samples = np.frombuffer(raw_bytes, dtype="<i2").astype(np.float32)
        samples = samples / 32768.0
    elif sampwidth == 3:
        raw = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(-1, 3)
        padded = np.zeros((len(raw), 4), dtype=np.uint8)
        padded[:, 0:3] = raw
        padded[:, 3] = np.where(raw[:, 2] & 0x80, 0xFF, 0x00)
        samples = padded.view("<i4").flatten().astype(np.float32)
        samples = samples / 8388608.0
    elif sampwidth == 4:
        samples = np.frombuffer(raw_bytes, dtype="<i4").astype(np.float32)
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
    """Encode an AudioBuffer into raw WAV sample bytes.

    Samples are scaled, rounded to nearest, and clipped to the target integer
    range.  Rounding matters: casting a scaled float directly with ``astype``
    truncates toward zero, which costs up to a full LSB of error per sample and
    biases the result toward silence.  Rounding halves the worst-case error to
    0.5 LSB and removes the bias.

    Both directions use the same scale factor ``2**(n-1)``, matching
    :func:`_decode_wav_frames` and the convention used by libsndfile and most
    DAWs.  Scaling by ``2**(n-1) - 1`` instead would leave a residual gain error
    of ``x / 2**(n-1)`` -- up to a full LSB near full scale, which swamps the
    rounding it was meant to protect.  With matched scales every exactly
    representable level round-trips bit-exactly, and the only clipped value is
    +1.0, which saturates to ``2**(n-1) - 1`` because the positive range is one
    code shorter than the negative.

    None of that applies to the float formats: they are written verbatim with
    no scaling and no clipping, because the point of a float file is to carry
    values outside [-1, 1] losslessly through a gain stage.
    """
    if bit_depth in _FLOAT_WRITE_DEPTHS:
        interleaved = buf.data.T.flatten()
        dtype = "<f4" if bit_depth == 32 else "<f8"
        return interleaved.astype(dtype).tobytes()

    data = buf.data.copy()
    np.clip(data, -1.0, 1.0, out=data)
    interleaved = data.T.flatten()

    if bit_depth == 16:
        scaled = np.clip(np.rint(interleaved * 32768.0), -32768.0, 32767.0)
        return scaled.astype("<i2").tobytes()
    else:  # 24
        scaled = np.clip(np.rint(interleaved * 8388608.0), -8388608.0, 8388607.0)
        # WAV stores 24-bit samples as three little-endian bytes; take the low
        # three bytes of each int32 in memory order.
        bytes_4 = scaled.astype("<i4").view(np.uint8).reshape(-1, 4)
        return bytes_4[:, :3].tobytes()


def _wav_header(buf: AudioBuffer, bit_depth: int, data_size: int) -> bytes:
    """Build a canonical 44-byte RIFF/WAVE header.

    Written by hand because the stdlib ``wave`` module can only emit
    ``WAVE_FORMAT_PCM``, so it cannot write the float formats.
    """
    is_float = bit_depth in _FLOAT_WRITE_DEPTHS
    tag = _WAVE_FORMAT_IEEE_FLOAT if is_float else _WAVE_FORMAT_PCM
    channels = buf.channels
    rate = int(buf.sample_rate)
    block_align = channels * (bit_depth // 8)
    return (
        b"RIFF"
        + struct.pack("<I", 36 + data_size)
        + b"WAVEfmt "
        + struct.pack(
            "<IHHIIHH",
            16,  # fmt chunk size
            tag,
            channels,
            rate,
            rate * block_align,  # byte rate
            block_align,
            bit_depth,
        )
        + b"data"
        + struct.pack("<I", data_size)
    )


def _validate_write_depth(bit_depth: int) -> None:
    if bit_depth not in _WRITE_DEPTHS:
        raise ValueError(
            f"Unsupported bit_depth: {bit_depth} "
            f"(use 16 or 24 for PCM, 32 or 64 for IEEE float)"
        )


def _encode_wav(buf: AudioBuffer, bit_depth: int) -> bytes:
    """Serialise an AudioBuffer to a complete WAV byte stream."""
    _validate_write_depth(bit_depth)
    raw = _encode_wav_frames(buf, bit_depth)
    if len(raw) + 44 > 0xFFFFFFFF:
        raise ValueError(
            "WAV output exceeds the 4 GB RIFF limit; write FLAC or split the file"
        )
    return _wav_header(buf, bit_depth, len(raw)) + raw


def read_wav(path: str | Path) -> AudioBuffer:
    """Read a WAV file and return an AudioBuffer.

    Supports 8-bit unsigned, 16/24/32-bit signed PCM, and 32/64-bit IEEE float,
    including files that wrap those in ``WAVE_FORMAT_EXTENSIBLE``.  Output is
    float32; PCM input is normalized to [-1, 1], float input is passed through
    unscaled and may therefore exceed that range.
    """
    path = Path(path)
    return _decode_wav_bytes(path.read_bytes(), source=f"'{path}'")


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
        16 or 24 for signed PCM (samples are clipped to [-1, 1] and rounded),
        or 32 or 64 for IEEE float (samples written verbatim, not clipped).
    """
    Path(path).write_bytes(_encode_wav(buf, bit_depth))


def read_wav_bytes(data: bytes) -> AudioBuffer:
    """Read WAV data from raw bytes and return an AudioBuffer.

    Accepts the same encodings as :func:`read_wav`.
    """
    return _decode_wav_bytes(data)


def write_wav_bytes(buf: AudioBuffer, bit_depth: int = 16) -> bytes:
    """Serialize an AudioBuffer to WAV bytes.

    Parameters
    ----------
    buf : AudioBuffer
        Audio data to write.
    bit_depth : int
        16 or 24 for signed PCM, 32 or 64 for IEEE float. See :func:`write_wav`.

    Returns
    -------
    bytes
        WAV file content.
    """
    return _encode_wav(buf, bit_depth)


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
        raise ValueError(
            f"Unsupported bit_depth for FLAC: {bit_depth} (use 16 or 24). "
            "FLAC is an integer format; write WAV for float output."
        )

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

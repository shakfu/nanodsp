"""Tests for nanodsp.io module (WAV and FLAC file I/O)."""

import struct
import wave
from pathlib import Path

import numpy as np
import pytest

from nanodsp.buffer import AudioBuffer
from nanodsp.io import read, read_flac, read_wav, write, write_flac, write_wav


@pytest.fixture
def tmp_wav(tmp_path):
    """Return a factory that creates WAV files with given parameters."""

    def _make(
        data_bytes: bytes,
        n_channels: int,
        sampwidth: int,
        framerate: int,
        n_frames: int,
        filename: str = "test.wav",
    ) -> Path:
        p = tmp_path / filename
        with wave.open(str(p), "wb") as wf:
            wf.setnchannels(n_channels)
            wf.setsampwidth(sampwidth)
            wf.setframerate(framerate)
            wf.writeframes(data_bytes)
        return p

    return _make


# ---------------------------------------------------------------------------
# Read tests
# ---------------------------------------------------------------------------


class TestReadWav:
    def test_read_8bit(self, tmp_wav):
        # 8-bit: unsigned, 128 = silence, 0 = -1, 255 ~ +1
        samples = bytes([128, 0, 255])
        p = tmp_wav(samples, 1, 1, 48000, 3)
        buf = read_wav(p)
        assert buf.channels == 1
        assert buf.frames == 3
        assert buf.sample_rate == 48000.0
        np.testing.assert_allclose(buf.data[0, 0], 0.0, atol=0.01)
        np.testing.assert_allclose(buf.data[0, 1], -1.0, atol=0.01)
        assert buf.data[0, 2] > 0.99

    def test_read_16bit_mono(self, tmp_wav):
        samples = np.array([0, 16384, -16384], dtype=np.int16)
        p = tmp_wav(samples.tobytes(), 1, 2, 44100, 3)
        buf = read_wav(p)
        assert buf.channels == 1
        assert buf.frames == 3
        assert buf.sample_rate == 44100.0
        np.testing.assert_allclose(buf.data[0, 0], 0.0, atol=0.001)
        np.testing.assert_allclose(buf.data[0, 1], 0.5, atol=0.001)
        np.testing.assert_allclose(buf.data[0, 2], -0.5, atol=0.001)

    def test_read_16bit_stereo(self, tmp_wav):
        # Interleaved: L0 R0 L1 R1
        samples = np.array([10000, -10000, 20000, -20000], dtype=np.int16)
        p = tmp_wav(samples.tobytes(), 2, 2, 48000, 2)
        buf = read_wav(p)
        assert buf.channels == 2
        assert buf.frames == 2
        # L channel positive, R channel negative
        assert buf.data[0, 0] > 0
        assert buf.data[1, 0] < 0

    def test_read_24bit_positive(self, tmp_wav):
        # 24-bit: 3 bytes little-endian. +8388607 -> ~1.0
        # Value 4194304 = 0x400000 -> ~0.5
        val = 4194304
        b = struct.pack("<i", val)[:3]
        p = tmp_wav(b, 1, 3, 48000, 1)
        buf = read_wav(p)
        assert buf.channels == 1
        np.testing.assert_allclose(buf.data[0, 0], 0.5, atol=0.001)

    def test_read_24bit_negative(self, tmp_wav):
        val = -4194304
        b = struct.pack("<i", val)[:3]
        p = tmp_wav(b, 1, 3, 48000, 1)
        buf = read_wav(p)
        assert buf.channels == 1
        np.testing.assert_allclose(buf.data[0, 0], -0.5, atol=0.001)

    def test_read_32bit(self, tmp_wav):
        samples = np.array([0, 1073741824, -1073741824], dtype=np.int32)
        p = tmp_wav(samples.tobytes(), 1, 4, 48000, 3)
        buf = read_wav(p)
        assert buf.channels == 1
        assert buf.frames == 3
        np.testing.assert_allclose(buf.data[0, 0], 0.0, atol=0.001)
        np.testing.assert_allclose(buf.data[0, 1], 0.5, atol=0.001)
        np.testing.assert_allclose(buf.data[0, 2], -0.5, atol=0.001)


# ---------------------------------------------------------------------------
# Write tests
# ---------------------------------------------------------------------------


class TestWriteWav:
    def test_16bit_roundtrip(self, tmp_path):
        buf = AudioBuffer.sine(440.0, channels=1, frames=1024, sample_rate=48000.0)
        p = tmp_path / "out16.wav"
        write_wav(p, buf, bit_depth=16)
        recovered = read_wav(p)
        assert recovered.channels == 1
        assert recovered.frames == 1024
        assert recovered.sample_rate == 48000.0
        # 16-bit quantization error: max ~1/32768
        np.testing.assert_allclose(recovered.data, buf.data, atol=1.0 / 32768 + 1e-4)

    def test_24bit_roundtrip(self, tmp_path):
        buf = AudioBuffer.sine(440.0, channels=1, frames=1024, sample_rate=48000.0)
        p = tmp_path / "out24.wav"
        write_wav(p, buf, bit_depth=24)
        recovered = read_wav(p)
        assert recovered.frames == buf.frames
        # 24-bit has much finer resolution
        np.testing.assert_allclose(recovered.data, buf.data, atol=1.0 / 8388608 + 1e-5)

    def test_stereo_roundtrip(self, tmp_path):
        # Use sine waves (bounded in [-1, 1]) to avoid clipping artifacts
        buf = AudioBuffer.sine(440.0, channels=2, frames=512, sample_rate=44100.0)
        p = tmp_path / "stereo.wav"
        write_wav(p, buf, bit_depth=16)
        recovered = read_wav(p)
        assert recovered.channels == 2
        assert recovered.frames == 512
        np.testing.assert_allclose(recovered.data, buf.data, atol=1.0 / 32768 + 1e-4)

    def test_clipping(self, tmp_path):
        # Values outside [-1, 1] should be clipped
        data = np.array([[2.0, -2.0, 0.5]], dtype=np.float32)
        buf = AudioBuffer(data, sample_rate=48000.0)
        p = tmp_path / "clip.wav"
        write_wav(p, buf, bit_depth=16)
        recovered = read_wav(p)
        assert recovered.data[0, 0] > 0.99
        assert recovered.data[0, 1] < -0.99
        np.testing.assert_allclose(recovered.data[0, 2], 0.5, atol=0.001)

    def test_invalid_bit_depth_raises(self, tmp_path):
        buf = AudioBuffer.zeros(1, 64, sample_rate=48000.0)
        with pytest.raises(ValueError, match="bit_depth"):
            write_wav(tmp_path / "bad.wav", buf, bit_depth=8)

    def test_invalid_bit_depth_32_raises(self, tmp_path):
        buf = AudioBuffer.zeros(1, 64, sample_rate=48000.0)
        with pytest.raises(ValueError, match="bit_depth"):
            write_wav(tmp_path / "bad.wav", buf, bit_depth=32)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_buffer(self, tmp_path):
        buf = AudioBuffer.zeros(1, 0, sample_rate=48000.0)
        p = tmp_path / "empty.wav"
        write_wav(p, buf, bit_depth=16)
        recovered = read_wav(p)
        assert recovered.frames == 0

    def test_path_object(self, tmp_path):
        buf = AudioBuffer.sine(440.0, frames=256, sample_rate=48000.0)
        p = tmp_path / "pathobj.wav"
        write_wav(p, buf)
        recovered = read_wav(p)
        assert recovered.frames == 256

    def test_string_path(self, tmp_path):
        buf = AudioBuffer.sine(440.0, frames=256, sample_rate=48000.0)
        p = str(tmp_path / "strpath.wav")
        write_wav(p, buf)
        recovered = read_wav(p)
        assert recovered.frames == 256


# ---------------------------------------------------------------------------
# FLAC tests
# ---------------------------------------------------------------------------


class TestReadFlac:
    def test_16bit_roundtrip(self, tmp_path):
        buf = AudioBuffer.sine(440.0, channels=1, frames=1024, sample_rate=48000.0)
        p = tmp_path / "out16.flac"
        write_flac(p, buf, bit_depth=16)
        recovered = read_flac(p)
        assert recovered.channels == 1
        assert recovered.frames == 1024
        assert recovered.sample_rate == 48000.0
        # 16-bit quantization error
        np.testing.assert_allclose(recovered.data, buf.data, atol=1.0 / 32768 + 1e-4)

    def test_24bit_roundtrip(self, tmp_path):
        buf = AudioBuffer.sine(440.0, channels=1, frames=1024, sample_rate=48000.0)
        p = tmp_path / "out24.flac"
        write_flac(p, buf, bit_depth=24)
        recovered = read_flac(p)
        # 24-bit has finer resolution
        np.testing.assert_allclose(recovered.data, buf.data, atol=1.0 / 8388608 + 1e-5)

    def test_stereo_roundtrip(self, tmp_path):
        buf = AudioBuffer.sine(440.0, channels=2, frames=512, sample_rate=44100.0)
        p = tmp_path / "stereo.flac"
        write_flac(p, buf, bit_depth=16)
        recovered = read_flac(p)
        assert recovered.channels == 2
        assert recovered.frames == 512
        np.testing.assert_allclose(recovered.data, buf.data, atol=1.0 / 32768 + 1e-4)

    def test_clipping(self, tmp_path):
        data = np.array([[2.0, -2.0, 0.5]], dtype=np.float32)
        buf = AudioBuffer(data, sample_rate=48000.0)
        p = tmp_path / "clip.flac"
        write_flac(p, buf, bit_depth=16)
        recovered = read_flac(p)
        assert recovered.data[0, 0] > 0.99
        assert recovered.data[0, 1] < -0.99
        np.testing.assert_allclose(recovered.data[0, 2], 0.5, atol=0.001)

    def test_invalid_bit_depth_raises(self, tmp_path):
        buf = AudioBuffer.zeros(1, 64, sample_rate=48000.0)
        with pytest.raises(ValueError, match="bit_depth"):
            write_flac(tmp_path / "bad.flac", buf, bit_depth=8)

    def test_path_object(self, tmp_path):
        buf = AudioBuffer.sine(440.0, frames=256, sample_rate=48000.0)
        p = tmp_path / "pathobj.flac"
        write_flac(p, buf)
        recovered = read_flac(p)
        assert recovered.frames == 256

    def test_string_path(self, tmp_path):
        buf = AudioBuffer.sine(440.0, frames=256, sample_rate=48000.0)
        p = str(tmp_path / "strpath.flac")
        write_flac(p, buf)
        recovered = read_flac(p)
        assert recovered.frames == 256

    def test_nonexistent_file_raises(self, tmp_path):
        with pytest.raises(RuntimeError):
            read_flac(tmp_path / "nonexistent.flac")

    def test_multichannel(self, tmp_path):
        data = np.random.default_rng(42).uniform(-0.5, 0.5, (4, 256)).astype(np.float32)
        buf = AudioBuffer(data, sample_rate=96000.0)
        p = tmp_path / "multi.flac"
        write_flac(p, buf, bit_depth=24)
        recovered = read_flac(p)
        assert recovered.channels == 4
        assert recovered.frames == 256
        assert recovered.sample_rate == 96000.0
        np.testing.assert_allclose(recovered.data, buf.data, atol=1.0 / 8388608 + 1e-5)


# ---------------------------------------------------------------------------
# Generic read/write dispatch tests
# ---------------------------------------------------------------------------


class TestGenericReadWrite:
    def test_wav_dispatch(self, tmp_path):
        buf = AudioBuffer.sine(440.0, frames=256, sample_rate=48000.0)
        p = tmp_path / "test.wav"
        write(p, buf)
        recovered = read(p)
        assert recovered.frames == 256
        np.testing.assert_allclose(recovered.data, buf.data, atol=1.0 / 32768 + 1e-4)

    def test_flac_dispatch(self, tmp_path):
        buf = AudioBuffer.sine(440.0, frames=256, sample_rate=48000.0)
        p = tmp_path / "test.flac"
        write(p, buf)
        recovered = read(p)
        assert recovered.frames == 256
        np.testing.assert_allclose(recovered.data, buf.data, atol=1.0 / 32768 + 1e-4)

    def test_unsupported_extension_read_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unsupported"):
            read(tmp_path / "test.mp3")

    def test_unsupported_extension_write_raises(self, tmp_path):
        buf = AudioBuffer.zeros(1, 64, sample_rate=48000.0)
        with pytest.raises(ValueError, match="Unsupported"):
            write(tmp_path / "test.ogg", buf)

    def test_audiobuffer_read_write_wav(self, tmp_path):
        buf = AudioBuffer.sine(440.0, frames=256, sample_rate=44100.0)
        p = tmp_path / "ab.wav"
        buf.write(str(p))
        recovered = AudioBuffer.from_file(str(p))
        assert recovered.frames == 256
        assert recovered.sample_rate == 44100.0
        np.testing.assert_allclose(recovered.data, buf.data, atol=1.0 / 32768 + 1e-4)

    def test_audiobuffer_read_write_flac(self, tmp_path):
        buf = AudioBuffer.sine(440.0, channels=2, frames=512, sample_rate=48000.0)
        p = tmp_path / "ab.flac"
        buf.write(str(p), bit_depth=24)
        recovered = AudioBuffer.from_file(str(p))
        assert recovered.channels == 2
        assert recovered.frames == 512
        np.testing.assert_allclose(recovered.data, buf.data, atol=1.0 / 8388608 + 1e-4)

    def test_case_insensitive_extension(self, tmp_path):
        buf = AudioBuffer.sine(440.0, frames=128, sample_rate=48000.0)
        p = tmp_path / "test.FLAC"
        write(p, buf)
        recovered = read(p)
        assert recovered.frames == 128

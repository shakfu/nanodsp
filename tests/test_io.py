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

    def test_bit_depth_32_writes_ieee_float(self, tmp_path):
        """32 used to be rejected; it now means IEEE float, not 32-bit PCM."""
        buf = AudioBuffer.sine(440.0, frames=256, sample_rate=48000.0)
        path = tmp_path / "float.wav"
        write_wav(path, buf, bit_depth=32)
        assert np.allclose(read_wav(path).data, buf.data, atol=1e-7)

    def test_invalid_bit_depth_20_raises(self, tmp_path):
        buf = AudioBuffer.zeros(1, 64, sample_rate=48000.0)
        with pytest.raises(ValueError, match="bit_depth"):
            write_wav(tmp_path / "bad.wav", buf, bit_depth=20)


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


# ---------------------------------------------------------------------------
# Quantisation quality (regression for the truncating encoder)
# ---------------------------------------------------------------------------


class TestQuantisation:
    """PCM encoding must round to nearest with matched encode/decode scales.

    The encoder previously truncated toward zero and scaled by 2**(n-1) - 1
    while the decoder divided by 2**(n-1). Together those cost up to 1.9 LSB of
    round-trip error at 16-bit and introduced a small DC bias.
    """

    @staticmethod
    def _roundtrip(x, bit_depth):
        from nanodsp.io import read_wav_bytes, write_wav_bytes

        buf = AudioBuffer(np.asarray(x, dtype=np.float32).copy())
        return read_wav_bytes(write_wav_bytes(buf, bit_depth=bit_depth)).mono

    @pytest.mark.parametrize("bit_depth", [16, 24])
    def test_error_within_half_lsb(self, bit_depth):
        lsb = 1.0 / (1 << (bit_depth - 1))
        rng = np.random.default_rng(0)
        x = rng.uniform(-0.99, 0.99, 20000).astype(np.float32)
        err = np.abs(self._roundtrip(x, bit_depth) - x) / lsb
        # Rounding to nearest bounds the error at exactly half an LSB; a small
        # float32 tolerance covers the scaling multiply.
        assert err.max() <= 0.5 + 1e-3, f"max error {err.max():.4f} LSB"

    @pytest.mark.parametrize("bit_depth", [16, 24])
    def test_no_dc_bias(self, bit_depth):
        lsb = 1.0 / (1 << (bit_depth - 1))
        rng = np.random.default_rng(1)
        x = rng.uniform(-0.9, 0.9, 50000).astype(np.float32)
        err = (self._roundtrip(x, bit_depth) - x) / lsb
        # Truncation toward zero biased this to roughly -0.008 LSB.
        assert abs(err.mean()) < 0.005, f"mean error {err.mean():+.5f} LSB"

    @pytest.mark.parametrize("bit_depth", [16, 24])
    def test_representable_levels_are_bit_exact(self, bit_depth):
        levels = np.array([0.0, 0.5, -0.5, 0.25, -0.25, -1.0], dtype=np.float32)
        assert np.array_equal(self._roundtrip(levels, bit_depth), levels)

    @pytest.mark.parametrize("bit_depth", [16, 24])
    def test_positive_full_scale_saturates_one_code_short(self, bit_depth):
        # Two's complement has one fewer positive code, so +1.0 cannot be
        # represented exactly; it must saturate rather than wrap to negative.
        full = 1 << (bit_depth - 1)
        got = float(self._roundtrip(np.array([1.0], dtype=np.float32), bit_depth)[0])
        assert got == pytest.approx((full - 1) / full, abs=1e-9)
        assert got > 0.0

    @pytest.mark.parametrize("bit_depth", [16, 24])
    def test_out_of_range_input_clips_without_wrapping(self, bit_depth):
        x = np.array([2.0, -2.0, 1.5, -1.5], dtype=np.float32)
        got = self._roundtrip(x, bit_depth)
        assert np.all(got[[0, 2]] > 0.9), "positive overload wrapped negative"
        assert np.all(got[[1, 3]] < -0.9), "negative overload wrapped positive"


# ---------------------------------------------------------------------------
# Float and extensible WAV support
# ---------------------------------------------------------------------------


def _handmade_wav(samples, fmt_tag, bits, channels=1, rate=48000, extensible=False):
    """Build a WAV byte stream directly, bypassing nanodsp's own writer."""
    import struct

    if fmt_tag == 3:
        dtype = "<f4" if bits == 32 else "<f8"
        raw = np.asarray(samples, dtype=dtype).tobytes()
    else:
        scale = float(1 << (bits - 1))
        ints = np.clip(np.rint(np.asarray(samples) * scale), -scale, scale - 1)
        raw = ints.astype({16: "<i2", 32: "<i4"}[bits]).tobytes()

    if extensible:
        # WAVE_FORMAT_EXTENSIBLE: cbSize=22, then the SubFormat GUID whose
        # first two bytes carry the real format tag.
        fmt_body = (
            struct.pack(
                "<HHIIHH",
                0xFFFE,
                channels,
                rate,
                rate * channels * bits // 8,
                channels * bits // 8,
                bits,
            )
            + struct.pack("<HHI", 22, bits, 0)
            + struct.pack("<H", fmt_tag)
            + b"\x00" * 14
        )
    else:
        fmt_body = struct.pack(
            "<HHIIHH",
            fmt_tag,
            channels,
            rate,
            rate * channels * bits // 8,
            channels * bits // 8,
            bits,
        )

    return (
        b"RIFF"
        + struct.pack("<I", 4 + 8 + len(fmt_body) + 8 + len(raw))
        + b"WAVE"
        + b"fmt "
        + struct.pack("<I", len(fmt_body))
        + fmt_body
        + b"data"
        + struct.pack("<I", len(raw))
        + raw
    )


class TestFloatWav:
    """32/64-bit IEEE float WAV, the format stdlib `wave` rejects outright."""

    def test_read_float32_written_externally(self):
        from nanodsp.io import read_wav_bytes

        x = (np.sin(2 * np.pi * 440 * np.arange(480) / 48000) * 0.5).astype(np.float32)
        buf = read_wav_bytes(_handmade_wav(x, fmt_tag=3, bits=32))
        assert buf.channels == 1
        assert buf.sample_rate == 48000.0
        assert np.allclose(buf.mono, x, atol=1e-7)

    def test_read_float64_written_externally(self):
        from nanodsp.io import read_wav_bytes

        x = np.linspace(-0.9, 0.9, 480)
        buf = read_wav_bytes(_handmade_wav(x, fmt_tag=3, bits=64))
        assert np.allclose(buf.mono, x.astype(np.float32), atol=1e-6)

    @pytest.mark.parametrize("bit_depth", [32, 64])
    def test_float_roundtrip_is_lossless(self, bit_depth, tmp_path):
        from nanodsp.io import read_wav, write_wav

        rng = np.random.default_rng(0)
        x = rng.uniform(-0.9, 0.9, 5000).astype(np.float32)
        buf = AudioBuffer(x.copy(), sample_rate=44100.0)
        path = tmp_path / f"f{bit_depth}.wav"
        write_wav(path, buf, bit_depth=bit_depth)
        got = read_wav(path)
        assert got.sample_rate == 44100.0
        # float32 in, float32 or float64 out: exactly recoverable either way.
        assert np.array_equal(got.mono, x)

    def test_float_preserves_values_beyond_unity(self, tmp_path):
        """PCM clips to [-1, 1]; float must not, or gain staging is destroyed."""
        from nanodsp.io import read_wav, write_wav

        x = np.array([2.5, -3.75, 0.5], dtype=np.float32)
        path = tmp_path / "hot.wav"
        write_wav(path, AudioBuffer(x.copy()), bit_depth=32)
        assert np.array_equal(read_wav(path).mono, x)

    def test_pcm_still_clips(self, tmp_path):
        from nanodsp.io import read_wav, write_wav

        x = np.array([2.5, -3.75, 0.5], dtype=np.float32)
        path = tmp_path / "clipped.wav"
        write_wav(path, AudioBuffer(x.copy()), bit_depth=16)
        got = read_wav(path).mono
        assert got[0] == pytest.approx(1.0, abs=1e-4)
        assert got[1] == pytest.approx(-1.0, abs=1e-4)

    def test_stereo_float_roundtrip(self, tmp_path):
        from nanodsp.io import read_wav, write_wav

        rng = np.random.default_rng(2)
        data = rng.uniform(-0.8, 0.8, (2, 1000)).astype(np.float32)
        path = tmp_path / "st.wav"
        write_wav(path, AudioBuffer(data.copy(), sample_rate=48000.0), bit_depth=32)
        got = read_wav(path)
        assert got.channels == 2
        assert np.array_equal(got.data, data)


class TestExtensibleWav:
    """WAVE_FORMAT_EXTENSIBLE, common for >2 channels or >16 bits."""

    def test_extensible_pcm16(self):
        from nanodsp.io import read_wav_bytes

        x = np.linspace(-0.8, 0.8, 400)
        buf = read_wav_bytes(_handmade_wav(x, fmt_tag=1, bits=16, extensible=True))
        assert np.allclose(buf.mono, x, atol=1e-4)

    def test_extensible_float32(self):
        from nanodsp.io import read_wav_bytes

        x = np.linspace(-0.8, 0.8, 400).astype(np.float32)
        buf = read_wav_bytes(_handmade_wav(x, fmt_tag=3, bits=32, extensible=True))
        assert np.allclose(buf.mono, x, atol=1e-7)


class TestRiffParsing:
    def test_unknown_chunks_are_skipped(self):
        """A LIST chunk before data must not confuse the parser."""
        import struct
        from nanodsp.io import read_wav_bytes

        base = _handmade_wav(np.zeros(64), fmt_tag=1, bits=16)
        head, rest = base[:12], base[12:]
        listing = b"LIST" + struct.pack("<I", 10) + b"INFOhello?"  # even-sized body
        doctored = (
            head[:4]
            + struct.pack("<I", len(rest) + len(listing) + 4)
            + b"WAVE"
            + listing
            + rest
        )
        assert read_wav_bytes(doctored).frames == 64

    def test_odd_sized_chunk_pad_byte_handled(self):
        import struct
        from nanodsp.io import read_wav_bytes

        base = _handmade_wav(np.zeros(64), fmt_tag=1, bits=16)
        head, rest = base[:12], base[12:]
        # 9-byte body plus one pad byte that is not counted in the size field.
        odd = b"LIST" + struct.pack("<I", 9) + b"INFOhello" + b"\x00"
        doctored = (
            head[:4]
            + struct.pack("<I", len(rest) + len(odd) + 4)
            + b"WAVE"
            + odd
            + rest
        )
        assert read_wav_bytes(doctored).frames == 64

    def test_unsupported_codec_names_itself(self):
        from nanodsp.io import read_wav_bytes

        data = _handmade_wav(np.zeros(64), fmt_tag=1, bits=16)
        doctored = bytearray(data)
        doctored[20:22] = (0x0011).to_bytes(2, "little")  # IMA ADPCM
        with pytest.raises(ValueError, match="IMA ADPCM"):
            read_wav_bytes(bytes(doctored))

    def test_rf64_gives_a_clear_error(self):
        from nanodsp.io import read_wav_bytes

        data = bytearray(_handmade_wav(np.zeros(16), fmt_tag=1, bits=16))
        data[0:4] = b"RF64"
        with pytest.raises(ValueError, match="RF64"):
            read_wav_bytes(bytes(data))

    def test_non_riff_rejected(self):
        from nanodsp.io import read_wav_bytes

        with pytest.raises(ValueError, match="RIFF"):
            read_wav_bytes(b"OggS" + b"\x00" * 64)

    def test_truncated_header_rejected(self):
        from nanodsp.io import read_wav_bytes

        with pytest.raises(ValueError, match="too short"):
            read_wav_bytes(b"RIF")

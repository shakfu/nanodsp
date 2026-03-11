"""Tests for nanodsp.buffer.AudioBuffer."""

import sys

import numpy as np
import numpy.testing as npt
import pytest

from nanodsp.buffer import AudioBuffer


# =========================================================================
# Construction
# =========================================================================


class TestConstruction:
    def test_from_1d_numpy(self):
        arr = np.ones(100, dtype=np.float32)
        buf = AudioBuffer(arr)
        assert buf.data.shape == (1, 100)
        assert buf.channels == 1
        assert buf.frames == 100

    def test_from_2d_numpy(self):
        arr = np.zeros((3, 200), dtype=np.float32)
        buf = AudioBuffer(arr)
        assert buf.data.shape == (3, 200)
        assert buf.channels == 3

    def test_from_list(self):
        buf = AudioBuffer([1.0, 2.0, 3.0])
        assert buf.dtype == np.float32
        npt.assert_array_equal(buf[0], [1.0, 2.0, 3.0])

    def test_from_audiobuffer(self):
        orig = AudioBuffer.sine(440, label="orig")
        copy = AudioBuffer(orig)
        npt.assert_array_equal(copy.data, orig.data)
        # independent storage
        copy.data[0, 0] = 999.0
        assert orig.data[0, 0] != 999.0

    def test_invalid_dtype_converts(self):
        arr = np.ones(10, dtype=np.float64)
        buf = AudioBuffer(arr)
        assert buf.dtype == np.float32

    def test_empty_array(self):
        arr = np.zeros((1, 0), dtype=np.float32)
        buf = AudioBuffer(arr)
        assert buf.frames == 0
        assert buf.channels == 1

    def test_3d_raises(self):
        with pytest.raises(ValueError, match="1D or 2D"):
            AudioBuffer(np.zeros((2, 3, 4)))


# =========================================================================
# Properties
# =========================================================================


class TestProperties:
    def test_channels_frames_duration(self):
        buf = AudioBuffer.zeros(2, 48000, sample_rate=48000.0)
        assert buf.channels == 2
        assert buf.frames == 48000
        assert buf.duration == pytest.approx(1.0)

    def test_sample_rate(self):
        buf = AudioBuffer.zeros(1, 100, sample_rate=44100.0)
        assert buf.sample_rate == 44100.0

    def test_layout_mono(self):
        buf = AudioBuffer.zeros(1, 10)
        assert buf.channel_layout == "mono"

    def test_layout_stereo(self):
        buf = AudioBuffer.zeros(2, 10)
        assert buf.channel_layout == "stereo"

    def test_layout_6ch_none(self):
        buf = AudioBuffer.zeros(6, 10)
        assert buf.channel_layout is None

    def test_label(self):
        buf = AudioBuffer.zeros(1, 10, label="test")
        assert buf.label == "test"

    def test_dtype_always_float32(self):
        buf = AudioBuffer(np.zeros(10, dtype=np.int16))
        assert buf.dtype == np.float32


# =========================================================================
# Channel access
# =========================================================================


class TestChannelAccess:
    def test_getitem_int(self):
        buf = AudioBuffer(np.arange(6, dtype=np.float32).reshape(2, 3))
        ch0 = buf[0]
        assert ch0.shape == (3,)
        npt.assert_array_equal(ch0, [0, 1, 2])

    def test_getitem_slice(self):
        buf = AudioBuffer.zeros(4, 100)
        sub = buf[1:3]
        assert isinstance(sub, AudioBuffer)
        assert sub.channels == 2

    def test_channel_method(self):
        buf = AudioBuffer.zeros(3, 10)
        npt.assert_array_equal(buf.channel(1), buf[1])

    def test_mono_property(self):
        buf = AudioBuffer.sine(100, channels=1, frames=256)
        m = buf.mono
        assert m.shape == (256,)

    def test_mono_raises_multi(self):
        buf = AudioBuffer.zeros(2, 10)
        with pytest.raises(ValueError, match="mono requires 1-channel"):
            _ = buf.mono

    def test_out_of_bounds(self):
        buf = AudioBuffer.zeros(2, 10)
        with pytest.raises(IndexError):
            buf[5]

    def test_tuple_index(self):
        buf = AudioBuffer(np.arange(20, dtype=np.float32).reshape(2, 10))
        view = buf[0, 2:5]
        npt.assert_array_equal(view, [2, 3, 4])


# =========================================================================
# Numpy interop
# =========================================================================


class TestNumpyInterop:
    def test_asarray(self):
        buf = AudioBuffer.zeros(2, 10)
        a = np.asarray(buf)
        assert a.shape == (2, 10)

    def test_len_returns_frames(self):
        buf = AudioBuffer.zeros(2, 512)
        assert len(buf) == 512

    def test_repr(self):
        buf = AudioBuffer.zeros(2, 1024, sample_rate=48000.0, label="input")
        r = repr(buf)
        assert "channels=2" in r
        assert "frames=1024" in r
        assert "sr=48000.0" in r
        assert "layout='stereo'" in r
        assert "label='input'" in r

    @pytest.mark.skipif(
        sys.version_info < (3, 12), reason="__buffer__ requires Python 3.12+"
    )
    def test_buffer_protocol(self):
        buf = AudioBuffer.sine(440.0, channels=2, frames=128, sample_rate=48000.0)
        mv = memoryview(buf)
        assert mv.shape == (2, 128)
        assert mv.format == "f"  # float32
        assert not mv.readonly

    @pytest.mark.skipif(
        sys.version_info < (3, 12), reason="__buffer__ requires Python 3.12+"
    )
    def test_memoryview_shares_memory(self):
        buf = AudioBuffer.zeros(1, 64)
        mv = memoryview(buf)
        mv[0, 0] = 1.0
        assert buf.data[0, 0] == 1.0


# =========================================================================
# Factory methods
# =========================================================================


class TestFactoryMethods:
    def test_zeros(self):
        buf = AudioBuffer.zeros(2, 128)
        assert buf.data.shape == (2, 128)
        assert np.all(buf.data == 0)

    def test_ones(self):
        buf = AudioBuffer.ones(1, 64)
        assert np.all(buf.data == 1.0)

    def test_impulse(self):
        buf = AudioBuffer.impulse(channels=2, frames=32)
        for ch in range(2):
            assert buf[ch][0] == 1.0
            assert np.all(buf[ch][1:] == 0.0)

    def test_sine_frequency(self):
        sr = 48000.0
        freq = 1000.0
        frames = 4096
        buf = AudioBuffer.sine(freq, frames=frames, sample_rate=sr)
        spectrum = np.abs(np.fft.rfft(buf[0]))
        peak_bin = np.argmax(spectrum)
        detected_freq = peak_bin * sr / frames
        assert abs(detected_freq - freq) < sr / frames

    def test_noise(self):
        buf = AudioBuffer.noise(channels=2, frames=1024, seed=42)
        assert not np.allclose(buf[0], 0)
        # different channels should differ
        assert not np.array_equal(buf[0], buf[1])

    def test_noise_seed_reproducible(self):
        a = AudioBuffer.noise(seed=7)
        b = AudioBuffer.noise(seed=7)
        npt.assert_array_equal(a.data, b.data)

    def test_from_numpy(self):
        arr = np.ones(10, dtype=np.float32)
        buf = AudioBuffer.from_numpy(arr, sample_rate=44100.0)
        assert buf.sample_rate == 44100.0
        assert buf.frames == 10


# =========================================================================
# Channel operations
# =========================================================================


class TestChannelOps:
    def test_to_mono_mean(self):
        arr = np.array([[2.0, 4.0], [4.0, 8.0]], dtype=np.float32)
        buf = AudioBuffer(arr)
        m = buf.to_mono("mean")
        assert m.channels == 1
        npt.assert_allclose(m[0], [3.0, 6.0])

    def test_to_mono_left(self):
        arr = np.array([[1.0], [2.0]], dtype=np.float32)
        m = AudioBuffer(arr).to_mono("left")
        npt.assert_allclose(m[0], [1.0])

    def test_to_mono_right(self):
        arr = np.array([[1.0], [2.0]], dtype=np.float32)
        m = AudioBuffer(arr).to_mono("right")
        npt.assert_allclose(m[0], [2.0])

    def test_to_mono_sum(self):
        arr = np.array([[1.0], [2.0]], dtype=np.float32)
        m = AudioBuffer(arr).to_mono("sum")
        npt.assert_allclose(m[0], [3.0])

    def test_to_channels_from_mono(self):
        buf = AudioBuffer.ones(1, 10)
        s = buf.to_channels(2)
        assert s.channels == 2
        npt.assert_array_equal(s[0], s[1])

    def test_to_channels_same(self):
        buf = AudioBuffer.zeros(3, 10)
        c = buf.to_channels(3)
        assert c.channels == 3

    def test_to_channels_error(self):
        buf = AudioBuffer.zeros(2, 10)
        with pytest.raises(ValueError, match="source must be mono"):
            buf.to_channels(4)

    def test_split_concat_roundtrip(self):
        buf = AudioBuffer(np.arange(6, dtype=np.float32).reshape(3, 2))
        parts = buf.split()
        assert len(parts) == 3
        assert all(p.channels == 1 for p in parts)
        rejoined = AudioBuffer.concat_channels(*parts)
        npt.assert_array_equal(rejoined.data, buf.data)

    def test_concat_sr_mismatch(self):
        a = AudioBuffer.zeros(1, 10, sample_rate=44100)
        b = AudioBuffer.zeros(1, 10, sample_rate=48000)
        with pytest.raises(ValueError, match="sample_rate"):
            AudioBuffer.concat_channels(a, b)


# =========================================================================
# Arithmetic
# =========================================================================


class TestArithmetic:
    def test_scalar_add(self):
        buf = AudioBuffer.ones(1, 4)
        result = buf + 2.0
        npt.assert_allclose(result[0], [3.0, 3.0, 3.0, 3.0])

    def test_scalar_sub(self):
        buf = AudioBuffer.ones(1, 4)
        result = buf - 0.5
        npt.assert_allclose(result[0], [0.5, 0.5, 0.5, 0.5])

    def test_scalar_mul(self):
        buf = AudioBuffer.ones(1, 4)
        result = buf * 3.0
        npt.assert_allclose(result[0], [3.0, 3.0, 3.0, 3.0])

    def test_scalar_div(self):
        buf = AudioBuffer.ones(1, 4) * 6.0
        result = buf / 2.0
        npt.assert_allclose(result[0], [3.0, 3.0, 3.0, 3.0])

    def test_buffer_add(self):
        a = AudioBuffer.ones(2, 4)
        b = AudioBuffer.ones(2, 4) * 2.0
        result = a + b
        npt.assert_allclose(result.data, 3.0)

    def test_mono_stereo_broadcast(self):
        mono = AudioBuffer.ones(1, 4) * 10.0
        stereo = AudioBuffer.ones(2, 4)
        result = mono + stereo
        assert result.channels == 2
        npt.assert_allclose(result.data, 11.0)

    def test_sr_mismatch_raises(self):
        a = AudioBuffer.zeros(1, 10, sample_rate=44100)
        b = AudioBuffer.zeros(1, 10, sample_rate=48000)
        with pytest.raises(ValueError, match="Sample rate mismatch"):
            _ = a + b

    def test_negation(self):
        buf = AudioBuffer.ones(1, 4)
        neg = -buf
        npt.assert_allclose(neg[0], [-1.0, -1.0, -1.0, -1.0])

    def test_gain_db(self):
        buf = AudioBuffer.ones(1, 4)
        boosted = buf.gain_db(20.0)
        npt.assert_allclose(boosted[0], 10.0, rtol=1e-5)
        cut = buf.gain_db(-20.0)
        npt.assert_allclose(cut[0], 0.1, rtol=1e-5)

    def test_radd(self):
        buf = AudioBuffer.ones(1, 4)
        result = 2.0 + buf
        npt.assert_allclose(result[0], [3.0, 3.0, 3.0, 3.0])

    def test_rmul(self):
        buf = AudioBuffer.ones(1, 4)
        result = 3.0 * buf
        npt.assert_allclose(result[0], [3.0, 3.0, 3.0, 3.0])

    def test_rsub(self):
        buf = AudioBuffer.ones(1, 4)
        result = 5.0 - buf
        npt.assert_allclose(result[0], [4.0, 4.0, 4.0, 4.0])


# =========================================================================
# Slicing
# =========================================================================


class TestSlicing:
    def test_slice_frames(self):
        buf = AudioBuffer(np.arange(20, dtype=np.float32).reshape(2, 10))
        s = buf.slice(2, 5)
        assert s.frames == 3
        assert s.channels == 2
        npt.assert_array_equal(s[0], [2, 3, 4])

    def test_slice_preserves_metadata(self):
        buf = AudioBuffer.zeros(2, 100, sample_rate=44100.0, label="x")
        s = buf.slice(10, 20)
        assert s.sample_rate == 44100.0
        assert s.label == "x"
        assert s.channel_layout == "stereo"


# =========================================================================
# Copy
# =========================================================================


class TestCopy:
    def test_copy_independent(self):
        buf = AudioBuffer.ones(2, 10, label="orig")
        c = buf.copy()
        c.data[0, 0] = 999.0
        assert buf.data[0, 0] == 1.0

    def test_copy_preserves_metadata(self):
        buf = AudioBuffer.zeros(2, 10, sample_rate=44100.0, label="t")
        c = buf.copy()
        assert c.sample_rate == 44100.0
        assert c.label == "t"
        assert c.channel_layout == "stereo"


# =========================================================================
# Validation helpers
# =========================================================================


# =========================================================================
# Pipe
# =========================================================================


class TestPipe:
    def test_basic_chaining(self):
        buf = AudioBuffer.ones(1, 64)
        result = buf.pipe(lambda b: b.gain_db(6.0))
        assert isinstance(result, AudioBuffer)
        assert result.frames == 64
        assert result.data[0, 0] > 1.0

    def test_kwargs_forwarded(self):
        def scale(b, factor=1.0):
            return AudioBuffer(
                b.data * factor,
                sample_rate=b.sample_rate,
                channel_layout=b.channel_layout,
                label=b.label,
            )

        buf = AudioBuffer.ones(1, 10)
        result = buf.pipe(scale, factor=3.0)
        npt.assert_allclose(result.data, 3.0)

    def test_args_forwarded(self):
        def add_offset(b, offset):
            return AudioBuffer(
                b.data + offset,
                sample_rate=b.sample_rate,
                channel_layout=b.channel_layout,
                label=b.label,
            )

        buf = AudioBuffer.zeros(1, 10)
        result = buf.pipe(add_offset, 5.0)
        npt.assert_allclose(result.data, 5.0)

    def test_type_error_on_non_audiobuffer_return(self):
        buf = AudioBuffer.ones(1, 10)
        with pytest.raises(TypeError, match="AudioBuffer"):
            buf.pipe(lambda b: b.data)

    def test_metadata_preserved_through_chain(self):
        buf = AudioBuffer.zeros(2, 100, sample_rate=44100.0, label="chain_test")
        result = buf.pipe(lambda b: b.gain_db(0.0))
        assert result.sample_rate == 44100.0
        assert result.channel_layout == "stereo"
        assert result.label == "chain_test"

    def test_multi_step_chain(self):
        buf = AudioBuffer.ones(1, 10)
        result = (
            buf.pipe(lambda b: b.gain_db(6.0))
            .pipe(lambda b: b.gain_db(6.0))
            .pipe(lambda b: b.gain_db(-12.0))
        )
        npt.assert_allclose(result.data, buf.data, rtol=1e-5)

    def test_integration_with_dsp_functions(self):
        from nanodsp import effects

        buf = AudioBuffer.sine(1000.0, frames=4096, sample_rate=48000.0)
        result = buf.pipe(effects.lowpass, 5000.0)
        assert isinstance(result, AudioBuffer)
        assert result.frames == 4096


# =========================================================================
# Validation helpers
# =========================================================================


class TestValidationHelpers:
    def test_ensure_1d(self):
        buf = AudioBuffer.zeros(3, 10)
        ch1 = buf.ensure_1d(1)
        assert ch1.ndim == 1
        assert ch1.dtype == np.float32

    def test_ensure_2d(self):
        buf = AudioBuffer.zeros(2, 10)
        d = buf.ensure_2d()
        assert d.ndim == 2
        assert d is buf.data


# =========================================================================
# Integration with C++ modules
# =========================================================================


class TestIntegrationWithModules:
    def test_filters_biquad(self):
        from nanodsp._core import filters

        buf = AudioBuffer.impulse(channels=1, frames=1024, sample_rate=48000)
        bq = filters.Biquad()
        bq.lowpass(0.1)
        filtered = bq.process(buf[0])
        assert filtered.shape == (1024,)
        assert filtered.dtype == np.float32

    def test_fft_realfft(self):
        from nanodsp._core import fft

        buf = AudioBuffer.sine(1000.0, frames=1024, sample_rate=48000)
        fft_obj = fft.RealFFT(1024)
        spectrum = fft_obj.fft(buf[0])
        assert spectrum.dtype == np.complex64
        assert spectrum.shape == (512,)

    def test_rates_oversampler(self):
        from nanodsp._core import rates

        buf = AudioBuffer.sine(440.0, channels=2, frames=256, sample_rate=48000)
        os = rates.Oversampler2x(2, 256)
        up = os.up(buf.data)
        assert up.shape == (2, 512)

    def test_spectral_stft(self):
        from nanodsp._core import spectral

        ws = 256
        buf = AudioBuffer.sine(1000.0, channels=1, frames=ws, sample_rate=48000)
        stft = spectral.STFT(1, ws, ws)
        stft.analyse(buf.data)
        spec = stft.get_spectrum()
        assert spec.dtype == np.complex64

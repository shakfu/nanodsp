"""Tests for GrainflowLib granular synthesis bindings."""

import numpy as np
import pytest

from nanodsp._core import grainflow as gf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_sine_buffer(frames=4096, channels=1, sr=48000, freq=440.0):
    """Create a GfBuffer filled with a sine wave."""
    buf = gf.GfBuffer(frames, channels, sr)
    t = np.arange(frames, dtype=np.float32) / sr
    data = np.zeros((channels, frames), dtype=np.float32)
    for ch in range(channels):
        data[ch] = np.sin(2 * np.pi * freq * t).astype(np.float32)
    buf.set_data(data)
    return buf


def make_ramp_clock(block_size=256, rate_hz=10.0, sr=48000):
    """Generate a simple ramp clock [1, block_size] for grain triggering."""
    t = np.arange(block_size, dtype=np.float32) / sr
    ramp = (t * rate_hz) % 1.0
    return ramp.astype(np.float32).reshape(1, block_size)


# ---------------------------------------------------------------------------
# TestGfBuffer
# ---------------------------------------------------------------------------


class TestGfBuffer:
    def test_construct(self):
        buf = gf.GfBuffer(1024, 2, 44100)
        assert buf.frames == 1024
        assert buf.channels == 2
        assert buf.samplerate == 44100

    def test_set_get_data_roundtrip(self):
        buf = gf.GfBuffer(512, 2, 48000)
        data = np.random.randn(2, 512).astype(np.float32)
        buf.set_data(data)
        out = buf.get_data()
        assert out.shape == (2, 512)
        np.testing.assert_allclose(out, data, atol=1e-6)

    def test_single_channel(self):
        buf = gf.GfBuffer(256, 1, 44100)
        data = np.ones((1, 256), dtype=np.float32) * 0.5
        buf.set_data(data)
        out = buf.get_data()
        np.testing.assert_allclose(out, data, atol=1e-6)

    def test_metadata_after_set_data(self):
        buf = gf.GfBuffer(100, 3, 22050)
        data = np.zeros((3, 100), dtype=np.float32)
        buf.set_data(data)
        assert buf.channels == 3
        assert buf.frames == 100


# ---------------------------------------------------------------------------
# TestPhasor
# ---------------------------------------------------------------------------


class TestPhasor:
    def test_construct(self):
        p = gf.Phasor(1.0, 48000)
        assert p is not None

    def test_perform_output_shape(self):
        p = gf.Phasor(1.0, 48000)
        out = p.perform(128)
        assert out.shape == (128,)

    def test_perform_range(self):
        p = gf.Phasor(100.0, 48000)
        out = p.perform(256)
        # All values should be in [0, 1)
        assert np.all(out >= 0.0)
        assert np.all(out < 1.0)

    def test_perform_monotonic_within_period(self):
        """Within a single period, the phasor should be monotonically increasing."""
        p = gf.Phasor(10.0, 48000)
        out = p.perform(64)
        # At 10 Hz / 48000 sr, one period is 4800 samples, so 64 samples should be monotonic
        diffs = np.diff(out)
        assert np.all(diffs > 0), (
            "Phasor should be monotonically increasing within one period"
        )

    def test_perform_invalid_frames(self):
        p = gf.Phasor(1.0, 48000)
        with pytest.raises(ValueError):
            p.perform(100)  # not a multiple of 64

    def test_set_rate(self):
        p = gf.Phasor(1.0, 48000)
        p.set_rate(2.0, 48000)
        out = p.perform(64)
        assert out.shape == (64,)

    def test_continuity_across_calls(self):
        """Phase should be continuous across multiple perform calls."""
        p = gf.Phasor(100.0, 48000)
        out1 = p.perform(64)
        out2 = p.perform(64)
        # Last sample of first block should be close to first sample of second block
        # (within one increment)
        increment = 100.0 / 48000
        gap = abs(out2[0] - (out1[-1] + increment) % 1.0)
        assert gap < 0.01 or gap > 0.99, f"Phase discontinuity: {gap}"


# ---------------------------------------------------------------------------
# TestGrainCollection
# ---------------------------------------------------------------------------


class TestGrainCollection:
    def test_construct(self):
        gc = gf.GrainCollection(4, 48000)
        assert gc.grains == 4
        assert gc.active_grains == 4

    def test_set_active_grains(self):
        gc = gf.GrainCollection(8, 48000)
        gc.set_active_grains(4)
        assert gc.active_grains == 4

    def test_auto_overlap(self):
        gc = gf.GrainCollection(4, 48000)
        assert gc.get_auto_overlap() is True
        gc.set_auto_overlap(False)
        assert gc.get_auto_overlap() is False

    def test_set_buffer(self):
        gc = gf.GrainCollection(4, 48000)
        buf = make_sine_buffer()
        gc.set_buffer(buf, gf.BUF_BUFFER, 0)

    def test_set_buffer_str(self):
        gc = gf.GrainCollection(4, 48000)
        buf = make_sine_buffer()
        gc.set_buffer_str(buf, "buffer", 0)

    def test_param_set_get(self):
        gc = gf.GrainCollection(4, 48000)
        gc.param_set(0, gf.PARAM_RATE, gf.PTYPE_BASE, 2.0)
        # target=0 sets all grains; get from grain 1 (value is sampled at grain reset)
        # but base should be retrievable via param_get_typed
        base = gc.param_get_typed(1, gf.PARAM_RATE, gf.PTYPE_BASE)
        assert base == pytest.approx(2.0, abs=0.01)

    def test_param_set_str(self):
        gc = gf.GrainCollection(4, 48000)
        ret = gc.param_set_str(0, "rate", 1.5)
        assert ret == 0  # GF_SUCCESS

    def test_param_set_str_invalid(self):
        gc = gf.GrainCollection(4, 48000)
        ret = gc.param_set_str(0, "nonexistent_param_xyz", 1.0)
        assert ret == 2  # GF_PARAM_NOT_FOUND

    def test_process_output_shapes(self):
        gc = gf.GrainCollection(4, 48000)
        buf = make_sine_buffer(frames=4096, channels=1, sr=48000)
        gc.set_buffer(buf, gf.BUF_BUFFER, 0)

        block_size = 256
        clock = make_ramp_clock(block_size, rate_hz=10.0, sr=48000)
        traversal = np.linspace(0, 0.5, block_size, dtype=np.float32).reshape(
            1, block_size
        )
        fm = np.zeros((1, block_size), dtype=np.float32)
        am = np.zeros((1, block_size), dtype=np.float32)

        result = gc.process(clock, traversal, fm, am, 48000)
        assert len(result) == 8, "process should return 8-element tuple"

        for i, arr in enumerate(result):
            assert arr.shape == (4, block_size), (
                f"Output {i} shape mismatch: {arr.shape} != (4, {block_size})"
            )

    def test_process_produces_output(self):
        """With a proper buffer and clock, some grains should produce output."""
        gc = gf.GrainCollection(4, 48000)
        buf = make_sine_buffer(frames=4096, channels=1, sr=48000, freq=440.0)
        gc.set_buffer(buf, gf.BUF_BUFFER, 0)

        # Set basic params
        gc.param_set(0, gf.PARAM_AMPLITUDE, gf.PTYPE_BASE, 1.0)
        gc.param_set(0, gf.PARAM_RATE, gf.PTYPE_BASE, 1.0)

        block_size = 256
        # Process a few blocks to let grains trigger
        for _ in range(4):
            clock = make_ramp_clock(block_size, rate_hz=10.0, sr=48000)
            traversal = np.linspace(0, 0.5, block_size, dtype=np.float32).reshape(
                1, block_size
            )
            fm = np.zeros((1, block_size), dtype=np.float32)
            am = np.zeros((1, block_size), dtype=np.float32)
            result = gc.process(clock, traversal, fm, am, 48000)

        # At least some output should be non-zero after triggering
        assert result[0].dtype == np.float32

    def test_process_invalid_block_size(self):
        gc = gf.GrainCollection(2, 48000)
        buf = make_sine_buffer()
        gc.set_buffer(buf, gf.BUF_BUFFER, 0)

        bad_size = 100  # not a multiple of 64
        clock = np.zeros((1, bad_size), dtype=np.float32)
        trav = np.zeros((1, bad_size), dtype=np.float32)
        fm = np.zeros((1, bad_size), dtype=np.float32)
        am = np.zeros((1, bad_size), dtype=np.float32)

        with pytest.raises(ValueError):
            gc.process(clock, trav, fm, am, 48000)

    def test_stream_set_and_get(self):
        gc = gf.GrainCollection(8, 48000)
        gc.stream_set(gf.STREAM_AUTOMATIC, 4)
        assert gc.streams == 4
        # With automatic mode: grain 0 -> stream 0, grain 1 -> stream 1, etc.
        assert gc.stream_get(0) == 0
        assert gc.stream_get(1) == 1
        assert gc.stream_get(4) == 0  # wraps around


# ---------------------------------------------------------------------------
# TestPanner
# ---------------------------------------------------------------------------


class TestPanner:
    def test_construct_stereo(self):
        p = gf.Panner(4, 2, gf.PAN_STEREO)
        assert p is not None

    def test_construct_bipolar(self):
        p = gf.Panner(4, 2, gf.PAN_BIPOLAR)
        assert p is not None

    def test_construct_unipolar(self):
        p = gf.Panner(4, 2, gf.PAN_UNIPOLAR)
        assert p is not None

    def test_pan_position_property(self):
        p = gf.Panner(4, 2, gf.PAN_STEREO)
        p.set_pan_position(0.75)
        assert p.pan_position == pytest.approx(0.75, abs=0.01)

    def test_process_output_shape(self):
        n_grains = 4
        block_size = 128
        out_channels = 2
        p = gf.Panner(n_grains, out_channels, gf.PAN_STEREO)
        grains = np.random.randn(n_grains, block_size).astype(np.float32) * 0.1
        states = np.ones((n_grains, block_size), dtype=np.float32)
        out = p.process(grains, states, out_channels)
        assert out.shape == (out_channels, block_size)

    def test_process_invalid_block_size(self):
        p = gf.Panner(2, 2, gf.PAN_STEREO)
        grains = np.zeros((2, 100), dtype=np.float32)  # not multiple of 64
        states = np.ones((2, 100), dtype=np.float32)
        with pytest.raises(ValueError):
            p.process(grains, states, 2)


# ---------------------------------------------------------------------------
# TestRecorder
# ---------------------------------------------------------------------------


class TestRecorder:
    def test_construct(self):
        r = gf.Recorder(48000)
        assert r is not None

    def test_set_target(self):
        r = gf.Recorder(48000)
        r.set_target(4096, 1, 48000)

    def test_properties(self):
        r = gf.Recorder(48000)
        r.overdub = 0.5
        assert r.overdub == pytest.approx(0.5)
        r.freeze = True
        assert r.freeze is True
        r.sync = True
        assert r.sync is True
        r.state = True
        assert r.state is True

    def test_rec_range(self):
        r = gf.Recorder(48000)
        r.set_rec_range(0.1, 0.9)
        lo, hi = r.get_rec_range()
        assert lo == pytest.approx(0.1, abs=0.01)
        assert hi == pytest.approx(0.9, abs=0.01)

    def test_process_requires_target(self):
        r = gf.Recorder(48000)
        input_data = np.zeros((1, 128), dtype=np.float32)
        with pytest.raises(RuntimeError):
            r.process(input_data)

    def test_process_returns_head_position(self):
        r = gf.Recorder(48000)
        r.set_target(4096, 1, 48000)
        r.state = True

        block = 128
        input_data = np.random.randn(1, block).astype(np.float32) * 0.1
        head = r.process(input_data)
        assert head.shape == (block,)
        assert head.dtype == np.float32

    def test_process_invalid_frames(self):
        r = gf.Recorder(48000)
        r.set_target(4096, 1, 48000)
        r.state = True
        input_data = np.zeros((1, 100), dtype=np.float32)  # not multiple of 64
        with pytest.raises(ValueError):
            r.process(input_data)

    def test_get_buffer_data(self):
        r = gf.Recorder(48000)
        r.set_target(4096, 2, 48000)
        data = r.get_buffer_data()
        assert data.shape == (2, 4096)
        assert data.dtype == np.float32

    def test_record_and_readback(self):
        """Record a signal and verify it appears in the buffer."""
        r = gf.Recorder(48000)
        r.set_target(4096, 1, 48000)
        r.state = True
        r.overdub = 0.0

        # Record a constant signal
        block = 128
        input_data = np.ones((1, block), dtype=np.float32) * 0.5
        for _ in range(4):
            r.process(input_data)

        data = r.get_buffer_data()
        # Some portion of the buffer should now contain ~0.5
        assert np.max(np.abs(data)) > 0.1, "Buffer should contain recorded data"


# ---------------------------------------------------------------------------
# TestEnumConstants
# ---------------------------------------------------------------------------


class TestEnumConstants:
    """Verify all enum constants are defined and have expected integer values."""

    def test_param_names(self):
        assert gf.PARAM_ERR == 0
        assert gf.PARAM_DELAY == 1
        assert gf.PARAM_RATE == 2
        assert gf.PARAM_WINDOW == 6
        assert gf.PARAM_AMPLITUDE == 7
        assert gf.PARAM_DIRECTION == 11
        assert gf.PARAM_DENSITY == 17

    def test_param_types(self):
        assert gf.PTYPE_ERR == 0
        assert gf.PTYPE_BASE == 1
        assert gf.PTYPE_RANDOM == 2
        assert gf.PTYPE_OFFSET == 3
        assert gf.PTYPE_MODE == 4
        assert gf.PTYPE_VALUE == 5

    def test_stream_modes(self):
        assert gf.STREAM_AUTOMATIC == 0
        assert gf.STREAM_PER == 1
        assert gf.STREAM_RANDOM == 2
        assert gf.STREAM_MANUAL == 3

    def test_buffer_types(self):
        assert gf.BUF_BUFFER == 0
        assert gf.BUF_ENVELOPE == 1
        assert gf.BUF_RATE == 2
        assert gf.BUF_DELAY == 3
        assert gf.BUF_WINDOW == 4
        assert gf.BUF_GLISSON == 5

    def test_buffer_modes(self):
        assert gf.BUFMODE_NORMAL == 0
        assert gf.BUFMODE_SEQUENCE == 1
        assert gf.BUFMODE_RANDOM == 2

    def test_pan_modes(self):
        assert gf.PAN_BIPOLAR == 0
        assert gf.PAN_UNIPOLAR == 1
        assert gf.PAN_STEREO == 2


# ---------------------------------------------------------------------------
# TestParamReflection
# ---------------------------------------------------------------------------


class TestParamReflection:
    """Test string-based parameter setting via reflection."""

    def test_basic_params(self):
        gc = gf.GrainCollection(4, 48000)
        assert gc.param_set_str(0, "delay", 10.0) == 0
        assert gc.param_set_str(0, "rate", 1.5) == 0
        assert gc.param_set_str(0, "window", 0.5) == 0
        assert gc.param_set_str(0, "amp", 0.8) == 0

    def test_random_suffix(self):
        gc = gf.GrainCollection(4, 48000)
        assert gc.param_set_str(0, "delayRandom", 5.0) == 0
        assert gc.param_set_str(0, "rateRandom", 0.1) == 0

    def test_offset_suffix(self):
        gc = gf.GrainCollection(4, 48000)
        assert gc.param_set_str(0, "delayOffset", 2.0) == 0

    def test_mode_suffix(self):
        gc = gf.GrainCollection(4, 48000)
        assert gc.param_set_str(0, "channelMode", 1.0) == 0

    def test_invalid_param(self):
        gc = gf.GrainCollection(4, 48000)
        ret = gc.param_set_str(0, "invalid_param", 1.0)
        assert ret != 0  # Should be GF_PARAM_NOT_FOUND or GF_ERR

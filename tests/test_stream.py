"""Tests for nanodsp.stream module (streaming / real-time processing)."""

import numpy as np
import pytest

from nanodsp.buffer import AudioBuffer
from nanodsp.stream import (
    BlockProcessor,
    CallbackProcessor,
    ProcessorChain,
    RingBuffer,
    process_blocks,
)


# ---------------------------------------------------------------------------
# RingBuffer
# ---------------------------------------------------------------------------


class TestRingBuffer:
    def test_construction(self):
        rb = RingBuffer(2, 1024, sample_rate=44100.0)
        assert rb.channels == 2
        assert rb.capacity == 1024
        assert rb.sample_rate == 44100.0
        assert rb.available_read == 0
        assert rb.available_write == 1024

    def test_invalid_construction(self):
        with pytest.raises(ValueError):
            RingBuffer(0, 1024)
        with pytest.raises(ValueError):
            RingBuffer(1, 0)

    def test_write_read_roundtrip(self):
        rb = RingBuffer(1, 256)
        data = AudioBuffer.sine(440.0, channels=1, frames=100, sample_rate=48000.0)
        written = rb.write(data)
        assert written == 100
        assert rb.available_read == 100
        assert rb.available_write == 156

        result = rb.read(100)
        assert result.frames == 100
        assert rb.available_read == 0
        np.testing.assert_array_equal(result.data, data.data)

    def test_write_fills_read_empties(self):
        rb = RingBuffer(1, 64)
        data = AudioBuffer.ones(1, 64, sample_rate=48000.0)
        written = rb.write(data)
        assert written == 64
        assert rb.available_write == 0

        result = rb.read(64)
        assert result.frames == 64
        assert rb.available_read == 0
        assert rb.available_write == 64

    def test_partial_write_when_full(self):
        rb = RingBuffer(1, 32)
        data = AudioBuffer.ones(1, 20, sample_rate=48000.0)
        rb.write(data)
        assert rb.available_write == 12

        more = AudioBuffer.ones(1, 20, sample_rate=48000.0)
        written = rb.write(more)
        assert written == 12
        assert rb.available_write == 0

    def test_partial_read_when_insufficient(self):
        rb = RingBuffer(1, 256)
        data = AudioBuffer.ones(1, 10, sample_rate=48000.0)
        rb.write(data)
        result = rb.read(50)
        assert result.frames == 10

    def test_wrap_around(self):
        rb = RingBuffer(1, 32)
        # Write 20, read 20, write 20 again -> wraps around
        data1 = AudioBuffer(np.arange(20, dtype=np.float32).reshape(1, -1))
        rb.write(data1)
        rb.read(20)

        data2 = AudioBuffer(np.arange(20, 40, dtype=np.float32).reshape(1, -1))
        written = rb.write(data2)
        assert written == 20

        result = rb.read(20)
        np.testing.assert_array_equal(result.data[0], data2.data[0])

    def test_peek_does_not_consume(self):
        rb = RingBuffer(1, 128)
        data = AudioBuffer.sine(440.0, channels=1, frames=50, sample_rate=48000.0)
        rb.write(data)

        peeked = rb.peek(50)
        assert rb.available_read == 50  # unchanged
        assert peeked.frames == 50
        np.testing.assert_array_equal(peeked.data, data.data)

        # Can still read the same data
        result = rb.read(50)
        np.testing.assert_array_equal(result.data, data.data)

    def test_clear_resets(self):
        rb = RingBuffer(1, 128)
        rb.write(AudioBuffer.ones(1, 100))
        rb.clear()
        assert rb.available_read == 0
        assert rb.available_write == 128

    def test_multichannel(self):
        rb = RingBuffer(2, 128)
        data = AudioBuffer.noise(2, 64, seed=42)
        rb.write(data)
        result = rb.read(64)
        np.testing.assert_array_equal(result.data, data.data)

    def test_empty_read_returns_zero_frames(self):
        rb = RingBuffer(1, 128)
        result = rb.read(10)
        assert result.frames == 0

    def test_channel_mismatch_raises(self):
        rb = RingBuffer(2, 128)
        with pytest.raises(ValueError, match="Channel mismatch"):
            rb.write(AudioBuffer.ones(1, 10))

    def test_numpy_array_write(self):
        rb = RingBuffer(1, 128)
        arr = np.ones(50, dtype=np.float32)
        written = rb.write(arr)
        assert written == 50
        result = rb.read(50)
        np.testing.assert_array_equal(result.data[0], arr)

    def test_multiple_wrap_arounds(self):
        rb = RingBuffer(1, 16)
        for i in range(10):
            data = AudioBuffer(np.full((1, 8), float(i), dtype=np.float32))
            rb.write(data)
            result = rb.read(8)
            assert result.frames == 8
            np.testing.assert_array_equal(result.data[0], np.full(8, float(i)))

    def test_available_read_write_consistency(self):
        rb = RingBuffer(1, 100)
        for _ in range(5):
            rb.write(AudioBuffer.ones(1, 30))
            assert rb.available_read + rb.available_write == 100
            rb.read(30)
            assert rb.available_read + rb.available_write == 100

    def test_peek_partial(self):
        rb = RingBuffer(1, 128)
        rb.write(AudioBuffer.ones(1, 50))
        peeked = rb.peek(20)
        assert peeked.frames == 20
        assert rb.available_read == 50  # unchanged

    def test_peek_more_than_available(self):
        rb = RingBuffer(1, 128)
        rb.write(AudioBuffer.ones(1, 10))
        peeked = rb.peek(50)
        assert peeked.frames == 10

    def test_write_zero_length(self):
        rb = RingBuffer(1, 128)
        written = rb.write(AudioBuffer(np.zeros((1, 0), dtype=np.float32)))
        assert written == 0
        assert rb.available_read == 0


# ---------------------------------------------------------------------------
# BlockProcessor
# ---------------------------------------------------------------------------


class _IdentityProcessor(BlockProcessor):
    def process_block(self, block):
        return block


class _GainProcessor(BlockProcessor):
    def __init__(self, gain, block_size, **kw):
        super().__init__(block_size, **kw)
        self.gain = gain
        self._reset_count = 0

    def process_block(self, block):
        return AudioBuffer(
            block.data * self.gain,
            sample_rate=block.sample_rate,
        )

    def reset(self):
        self._reset_count += 1


class TestBlockProcessor:
    def test_passthrough(self):
        proc = _IdentityProcessor(block_size=256)
        buf = AudioBuffer.noise(1, 1000, seed=0)
        result = proc.process(buf)
        np.testing.assert_array_equal(result.data, buf.data)

    def test_correct_chunking(self):
        proc = _GainProcessor(2.0, block_size=256)
        buf = AudioBuffer.ones(1, 1000)
        result = proc.process(buf)
        np.testing.assert_allclose(result.data, 2.0, atol=1e-6)

    def test_last_block_padding_trimming(self):
        proc = _IdentityProcessor(block_size=256)
        buf = AudioBuffer.noise(1, 300, seed=0)
        result = proc.process(buf)
        assert result.frames == 300
        np.testing.assert_array_equal(result.data[:, :300], buf.data[:, :300])

    def test_reset_called(self):
        proc = _GainProcessor(1.0, block_size=64)
        proc.reset()
        assert proc._reset_count == 1

    def test_not_implemented_error(self):
        proc = BlockProcessor(block_size=64)
        buf = AudioBuffer.ones(1, 64)
        with pytest.raises(NotImplementedError):
            proc.process(buf)

    def test_metadata_preserved(self):
        proc = _IdentityProcessor(block_size=256)
        buf = AudioBuffer.noise(2, 500, sample_rate=44100.0, seed=0, label="test")
        result = proc.process(buf)
        assert result.sample_rate == 44100.0
        assert result.channels == 2
        assert result.label == "test"

    def test_invalid_block_size(self):
        with pytest.raises(ValueError):
            BlockProcessor(block_size=0)

    def test_multichannel(self):
        proc = _GainProcessor(0.5, block_size=128, channels=2)
        buf = AudioBuffer.ones(2, 300)
        result = proc.process(buf)
        assert result.channels == 2
        np.testing.assert_allclose(result.data, 0.5, atol=1e-6)

    def test_exact_block_multiple(self):
        proc = _IdentityProcessor(block_size=128)
        buf = AudioBuffer.noise(1, 512, seed=0)
        result = proc.process(buf)
        assert result.frames == 512
        np.testing.assert_array_equal(result.data, buf.data)

    def test_single_sample_block(self):
        proc = _GainProcessor(3.0, block_size=1)
        buf = AudioBuffer.ones(1, 10)
        result = proc.process(buf)
        np.testing.assert_allclose(result.data, 3.0, atol=1e-6)


# ---------------------------------------------------------------------------
# CallbackProcessor
# ---------------------------------------------------------------------------


class TestCallbackProcessor:
    def test_wraps_function(self):
        def double(block):
            return AudioBuffer(block.data * 2, sample_rate=block.sample_rate)

        proc = CallbackProcessor(double, block_size=128)
        buf = AudioBuffer.ones(1, 300)
        result = proc.process(buf)
        np.testing.assert_allclose(result.data, 2.0, atol=1e-6)

    def test_callback_lambda(self):
        proc = CallbackProcessor(
            lambda b: AudioBuffer(b.data + 1.0, sample_rate=b.sample_rate),
            block_size=64,
        )
        buf = AudioBuffer.zeros(1, 200)
        result = proc.process(buf)
        np.testing.assert_allclose(result.data, 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# ProcessorChain
# ---------------------------------------------------------------------------


class TestProcessorChain:
    def test_chain_of_identity(self):
        chain = ProcessorChain(
            _IdentityProcessor(256),
            _IdentityProcessor(256),
        )
        buf = AudioBuffer.noise(1, 500, seed=0)
        result = chain.process(buf)
        np.testing.assert_array_equal(result.data, buf.data)

    def test_chain_applies_in_order(self):
        # gain 2 then gain 3 = gain 6
        chain = ProcessorChain(
            _GainProcessor(2.0, 128),
            _GainProcessor(3.0, 128),
        )
        buf = AudioBuffer.ones(1, 300)
        result = chain.process(buf)
        np.testing.assert_allclose(result.data, 6.0, atol=1e-5)

    def test_reset_propagates(self):
        p1 = _GainProcessor(1.0, 64)
        p2 = _GainProcessor(1.0, 64)
        chain = ProcessorChain(p1, p2)
        chain.reset()
        assert p1._reset_count == 1
        assert p2._reset_count == 1

    def test_empty_chain_raises(self):
        with pytest.raises(ValueError):
            ProcessorChain()

    def test_three_processors(self):
        chain = ProcessorChain(
            _GainProcessor(2.0, 128),
            _GainProcessor(2.0, 128),
            _GainProcessor(2.0, 128),
        )
        buf = AudioBuffer.ones(1, 200)
        result = chain.process(buf)
        np.testing.assert_allclose(result.data, 8.0, atol=1e-4)

    def test_chain_multichannel(self):
        chain = ProcessorChain(
            _GainProcessor(0.5, 64, channels=2),
        )
        buf = AudioBuffer.ones(2, 100)
        result = chain.process(buf)
        assert result.channels == 2
        np.testing.assert_allclose(result.data, 0.5, atol=1e-6)


# ---------------------------------------------------------------------------
# process_blocks
# ---------------------------------------------------------------------------


class TestProcessBlocks:
    def test_non_overlapping_identity(self):
        buf = AudioBuffer.noise(1, 500, seed=0)
        result = process_blocks(buf, lambda b: b, block_size=128)
        np.testing.assert_array_equal(result.data, buf.data)

    def test_non_overlapping_gain(self):
        buf = AudioBuffer.ones(1, 300)

        def double(b):
            return AudioBuffer(b.data * 2, sample_rate=b.sample_rate)

        result = process_blocks(buf, double, block_size=128)
        np.testing.assert_allclose(result.data, 2.0, atol=1e-6)

    def test_overlap_add_reconstructs(self):
        buf = AudioBuffer.noise(1, 4096, seed=42)
        result = process_blocks(buf, lambda b: b, block_size=512, hop_size=128)
        # With identity processing and COLA, interior should reconstruct well
        # (edges may differ due to incomplete window coverage)
        margin = 512  # skip edges equal to one block
        np.testing.assert_allclose(
            result.data[:, margin:-margin],
            buf.data[:, margin:-margin],
            atol=0.02,
        )

    def test_block_larger_than_buf(self):
        buf = AudioBuffer.ones(1, 50)
        result = process_blocks(buf, lambda b: b, block_size=256)
        assert result.frames == 50

    def test_stereo_support(self):
        buf = AudioBuffer.noise(2, 300, seed=0)
        result = process_blocks(buf, lambda b: b, block_size=128)
        assert result.channels == 2
        np.testing.assert_array_equal(result.data, buf.data)

    def test_metadata_preserved(self):
        buf = AudioBuffer.noise(1, 300, sample_rate=44100.0, seed=0, label="x")
        result = process_blocks(buf, lambda b: b, block_size=128)
        assert result.sample_rate == 44100.0
        assert result.label == "x"

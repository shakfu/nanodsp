"""Chunked I/O, the stateful processor layer, and CLI file operands.

The claim these tests exist to defend is that streaming is not an approximation:
processing a file block by block through a stateful chain must produce exactly
what whole-file processing produces. A stateless effect rebuilt per block would
restart its filter memory at every boundary and fail here.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanodsp import AudioBuffer, stream
from nanodsp.io import BlockWriter, read, read_blocks, write

SR = 48000.0


@pytest.fixture
def noise_file(tmp_path):
    def _make(channels=2, frames=50000, bit_depth=16, name="in.wav", sr=SR):
        rng = np.random.default_rng(0)
        data = rng.uniform(-0.8, 0.8, (channels, frames)).astype(np.float32)
        path = tmp_path / name
        write(path, AudioBuffer(data, sample_rate=sr), bit_depth=bit_depth)
        return path

    return _make


class TestReadBlocks:
    @pytest.mark.parametrize("bit_depth", [16, 24, 32])
    @pytest.mark.parametrize("channels", [1, 2])
    def test_blocks_rejoin_to_the_whole_file(self, noise_file, bit_depth, channels):
        path = noise_file(channels=channels, bit_depth=bit_depth)
        whole = read(path)
        joined = np.concatenate(
            [b.data for b in read_blocks(path, block_size=4096)], axis=1
        )
        assert np.array_equal(joined, whole.data)

    def test_block_metadata_matches_the_file(self, noise_file):
        path = noise_file(channels=2, sr=44100.0)
        for block in read_blocks(path, block_size=1000):
            assert block.sample_rate == 44100.0
            assert block.channels == 2

    def test_final_block_is_short_not_padded(self, noise_file):
        path = noise_file(channels=1, frames=7777)
        sizes = [b.frames for b in read_blocks(path, block_size=1000)]
        assert sizes == [1000] * 7 + [777]

    def test_block_larger_than_file(self, noise_file):
        path = noise_file(channels=1, frames=100)
        blocks = list(read_blocks(path, block_size=1 << 20))
        assert len(blocks) == 1 and blocks[0].frames == 100

    def test_rejects_bad_block_size(self, noise_file):
        with pytest.raises(ValueError, match="block_size"):
            list(read_blocks(noise_file(), block_size=0))

    def test_reports_unsupported_encoding(self, tmp_path):
        # Build a minimal WAV, then doctor its format tag.
        path = tmp_path / "adpcm.wav"
        write(path, AudioBuffer.sine(440.0, frames=64), bit_depth=16)
        raw = bytearray(path.read_bytes())
        raw[20:22] = (0x0011).to_bytes(2, "little")
        path.write_bytes(bytes(raw))
        with pytest.raises(ValueError, match="IMA ADPCM"):
            list(read_blocks(path))


class TestBlockWriter:
    def test_streamed_copy_is_identical(self, noise_file, tmp_path):
        src_path = noise_file(channels=2)
        src = read(src_path)
        out = tmp_path / "out.wav"
        with BlockWriter(out, src.sample_rate, src.channels, 16) as w:
            for block in read_blocks(src_path, block_size=1000):
                w.write(block)
            assert w.frames == src.frames
        assert np.array_equal(read(out).data, src.data)

    @pytest.mark.parametrize("bit_depth", [16, 24, 32])
    def test_header_is_valid_for_each_depth(self, tmp_path, bit_depth):
        out = tmp_path / f"o{bit_depth}.wav"
        buf = AudioBuffer.sine(440.0, channels=2, frames=1000, sample_rate=SR)
        with BlockWriter(out, SR, 2, bit_depth) as w:
            w.write(buf)
        got = read(out)
        assert got.channels == 2 and got.frames == 1000
        assert got.sample_rate == SR

    def test_channel_mismatch_rejected(self, tmp_path):
        with BlockWriter(tmp_path / "o.wav", SR, 2, 16) as w:
            with pytest.raises(ValueError, match="channel"):
                w.write(AudioBuffer.sine(440.0, channels=1, frames=64))

    def test_write_after_close_rejected(self, tmp_path):
        w = BlockWriter(tmp_path / "o.wav", SR, 1, 16)
        w.close()
        with pytest.raises(ValueError, match="closed"):
            w.write(AudioBuffer.sine(440.0, frames=64))

    def test_empty_output_is_readable(self, tmp_path):
        out = tmp_path / "empty.wav"
        BlockWriter(out, SR, 1, 16).close()
        assert read(out).frames == 0

    def test_close_is_idempotent(self, tmp_path):
        w = BlockWriter(tmp_path / "o.wav", SR, 1, 16)
        w.close()
        w.close()


class TestStatefulProcessors:
    """Every streaming form must equal its whole-buffer form, block-for-block."""

    @pytest.mark.parametrize("name", sorted(stream.STREAMABLE))
    @pytest.mark.parametrize("block_size", [128, 997])
    def test_block_continuity(self, name, block_size):
        import inspect

        ctor = stream.STREAMABLE[name]
        params = inspect.signature(ctor).parameters
        args = [1000.0] if "cutoff_hz" in params or "center_hz" in params else []
        kw = {"channels": 2, "sample_rate": SR}
        buf = AudioBuffer.sine(220.0, channels=2, frames=8000, sample_rate=SR)

        whole = ctor(*args, **kw).process(buf)
        streamer = ctor(*args, **kw)
        parts = [
            streamer.process(buf.slice(i, min(i + block_size, buf.frames)))
            for i in range(0, buf.frames, block_size)
        ]
        streamed = np.concatenate([p.data for p in parts], axis=1)
        assert np.allclose(whole.data, streamed, atol=1e-6), (
            f"{name} is not continuous across {block_size}-frame blocks"
        )

    def test_reset_returns_to_the_initial_state(self):
        p = stream.stateful_lowpass(1000.0, channels=1, sample_rate=SR)
        buf = AudioBuffer.sine(220.0, frames=2000, sample_rate=SR)
        first = p.process(buf).data.copy()
        p.process(buf)
        p.reset()
        assert np.array_equal(p.process(buf).data, first)

    def test_channel_mismatch_rejected(self):
        p = stream.stateful_lowpass(1000.0, channels=2, sample_rate=SR)
        with pytest.raises(ValueError, match="channel"):
            p.process(AudioBuffer.sine(220.0, channels=1, frames=64, sample_rate=SR))

    def test_stateful_filter_alias_preserved(self):
        assert stream.StatefulFilter is stream.StatefulProcessor

    def test_apply_adapter_used_for_limiter(self):
        """stateful_limit needs the adapter: its process() takes a second arg."""
        loud = AudioBuffer(np.full((1, 4000), 0.9, dtype=np.float32), sample_rate=SR)
        out = stream.stateful_limit(pre_gain=4.0, sample_rate=SR).process(loud)
        assert np.max(np.abs(out.data)) < 4.0 * 0.9


class TestStreamingChain:
    def test_rejects_non_streamable_effect(self):
        with pytest.raises(ValueError, match="no streaming form"):
            stream.build_streaming_chain([{"name": "reverb", "params": {}}], 2, SR)

    def test_error_names_every_offender(self):
        steps = [
            {"name": "reverb", "params": {}},
            {"name": "shimmer_reverb", "params": {}},
        ]
        with pytest.raises(ValueError) as exc:
            stream.build_streaming_chain(steps, 2, SR)
        assert "reverb" in str(exc.value) and "shimmer_reverb" in str(exc.value)

    def test_link_parameter_dropped(self):
        """`link` has no per-channel meaning; it must not reach the constructor."""
        procs = stream.build_streaming_chain(
            [{"name": "compress", "params": {"ratio": 4.0, "link": True}}], 2, SR
        )
        assert len(procs) == 1

    def test_caveats_cover_the_linked_dynamics(self):
        assert set(stream.STREAMING_CAVEATS) == {"compress", "limit"}

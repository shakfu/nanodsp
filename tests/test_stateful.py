"""Tests for stateful streaming filters (nanodsp.stream.StatefulFilter et al.).

The defining property: feeding a signal through a stateful filter in arbitrary
chunks yields exactly the same result as processing the whole signal at once,
because per-channel filter state is retained across calls. This is what the
stateless nanodsp.effects.filters functions cannot do.
"""

import numpy as np
import pytest

from nanodsp import AudioBuffer
from nanodsp import stream
from nanodsp.effects import filters as F

SR = 48000.0

# Irregular chunk sizes (including a single-sample chunk) summing to 10000.
_CHUNKS = [1000, 1, 4096, 333, 2570, 2000]


def _stream_in_chunks(proc, buf, sizes):
    parts, pos = [], 0
    for size in sizes:
        end = min(pos + size, buf.frames)
        parts.append(proc.process(buf.slice(pos, end)).data)
        pos = end
    return AudioBuffer(np.concatenate(parts, axis=1), sample_rate=buf.sample_rate)


_STATEFUL_VS_STATELESS = [
    (
        "lowpass",
        lambda sr: stream.stateful_lowpass(1000.0, channels=2, sample_rate=sr),
        lambda b: F.lowpass(b, cutoff_hz=1000.0),
    ),
    (
        "highpass",
        lambda sr: stream.stateful_highpass(1000.0, channels=2, sample_rate=sr),
        lambda b: F.highpass(b, cutoff_hz=1000.0),
    ),
    (
        "bandpass",
        lambda sr: stream.stateful_bandpass(1000.0, channels=2, sample_rate=sr),
        lambda b: F.bandpass(b, center_hz=1000.0),
    ),
    (
        "notch",
        lambda sr: stream.stateful_notch(1000.0, channels=2, sample_rate=sr),
        lambda b: F.notch(b, center_hz=1000.0),
    ),
]


class TestStateContinuity:
    @pytest.mark.parametrize(
        "make_sf, stateless",
        [(c[1], c[2]) for c in _STATEFUL_VS_STATELESS],
        ids=[c[0] for c in _STATEFUL_VS_STATELESS],
    )
    def test_chunked_equals_whole(self, make_sf, stateless):
        sig = AudioBuffer.noise(channels=2, frames=10000, seed=7, sample_rate=SR)
        reference = stateless(sig)  # stateless whole-buffer result
        streamed = _stream_in_chunks(make_sf(SR), sig, _CHUNKS)
        # Stateful chunked streaming reproduces the whole-buffer result exactly.
        np.testing.assert_allclose(streamed.data, reference.data, atol=1e-6)

    def test_naive_stateless_chunking_is_discontinuous(self):
        # Control: the stateless API rebuilds the filter each call, so chunking
        # it does NOT match whole-buffer processing -- the bug stateful fixes.
        sig = AudioBuffer.noise(channels=2, frames=10000, seed=7, sample_rate=SR)
        reference = F.lowpass(sig, cutoff_hz=1000.0)
        pos, parts = 0, []
        for size in _CHUNKS:
            end = min(pos + size, sig.frames)
            parts.append(F.lowpass(sig.slice(pos, end), cutoff_hz=1000.0).data)
            pos = end
        naive = np.concatenate(parts, axis=1)
        assert np.max(np.abs(naive - reference.data)) > 1e-3

    def test_process_block_matches_process(self):
        sig = AudioBuffer.noise(frames=4096, seed=1, sample_rate=SR)
        a = stream.stateful_lowpass(800.0, sample_rate=SR).process(sig)
        b = stream.stateful_lowpass(800.0, sample_rate=SR).process_block(sig)
        np.testing.assert_array_equal(a.data, b.data)


class TestResetAndMetadata:
    def test_reset_restores_initial_state(self):
        sig = AudioBuffer.noise(frames=8000, seed=2, sample_rate=SR)
        sf = stream.stateful_lowpass(1200.0, sample_rate=SR)
        first = sf.process(sig)
        sf.process(sig)  # advance state further
        sf.reset()
        after = sf.process(sig)
        np.testing.assert_array_equal(first.data, after.data)

    def test_output_metadata_and_dtype_preserved(self):
        sig = AudioBuffer.noise(channels=2, frames=512, seed=4, sample_rate=44100.0)
        out = stream.stateful_lowpass(1000.0, channels=2, sample_rate=44100.0).process(
            sig
        )
        assert out.channels == 2
        assert out.frames == 512
        assert out.sample_rate == 44100.0
        assert out.data.dtype == np.float32

    def test_channel_mismatch_raises(self):
        sf = stream.stateful_lowpass(1000.0, channels=1, sample_rate=SR)
        stereo = AudioBuffer.noise(channels=2, frames=256, sample_rate=SR)
        with pytest.raises(ValueError):
            sf.process(stereo)


class TestComposition:
    def test_in_processor_chain(self):
        sig = AudioBuffer.noise(frames=6000, seed=5, sample_rate=SR)
        # Reference: cascade applied to the whole buffer.
        chain_whole = stream.ProcessorChain(
            stream.stateful_highpass(200.0, sample_rate=SR),
            stream.stateful_lowpass(4000.0, sample_rate=SR),
        )
        reference = chain_whole.process(sig)

        # Same cascade, fed in chunks -- must match the whole-buffer cascade.
        chain_stream = stream.ProcessorChain(
            stream.stateful_highpass(200.0, sample_rate=SR),
            stream.stateful_lowpass(4000.0, sample_rate=SR),
        )
        streamed = _stream_in_chunks(chain_stream, sig, [2000, 1500, 2500])
        np.testing.assert_allclose(streamed.data, reference.data, atol=1e-6)

    def test_chain_reset(self):
        sig = AudioBuffer.noise(frames=4000, seed=6, sample_rate=SR)
        chain = stream.ProcessorChain(
            stream.stateful_highpass(200.0, sample_rate=SR),
            stream.stateful_lowpass(4000.0, sample_rate=SR),
        )
        first = chain.process(sig)
        chain.process(sig)
        chain.reset()
        after = chain.process(sig)
        np.testing.assert_array_equal(first.data, after.data)


class TestGenericAndDaisySP:
    def test_moog_ladder_streams_continuously(self):
        sig = AudioBuffer.noise(frames=8000, seed=3, sample_rate=SR)
        whole = stream.stateful_moog_ladder(
            1000.0, resonance=0.3, sample_rate=SR
        ).process(sig)
        streamed = _stream_in_chunks(
            stream.stateful_moog_ladder(1000.0, resonance=0.3, sample_rate=SR),
            sig,
            [3000, 5000],
        )
        np.testing.assert_allclose(streamed.data, whole.data, atol=1e-6)

    def test_custom_factory(self):
        # The generic StatefulFilter wraps any object exposing process(1d)->1d.
        from nanodsp._core import filters

        def factory():
            bq = filters.Biquad()
            bq.lowpass(0.05)
            return bq

        sf = stream.StatefulFilter(factory, channels=1, sample_rate=SR)
        sig = AudioBuffer.noise(frames=2048, seed=8, sample_rate=SR)
        streamed = _stream_in_chunks(sf, sig, [1000, 1048])
        whole = stream.StatefulFilter(factory, channels=1, sample_rate=SR).process(sig)
        np.testing.assert_allclose(streamed.data, whole.data, atol=1e-6)

    def test_is_block_processor(self):
        sf = stream.stateful_lowpass(1000.0, sample_rate=SR)
        assert isinstance(sf, stream.BlockProcessor)

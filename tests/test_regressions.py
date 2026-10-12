"""Regression tests for the defects fixed in the third project review (see CHANGELOG.md).

Each test asserts on the failure as observed -- a crash, a hang, a wrong pitch,
a wrong level -- rather than on shape, which is what let most of them ship.
"""

from __future__ import annotations

import inspect
import json
import struct
import subprocess
import sys
from math import factorial

import numpy as np
import pytest

from nanodsp import AudioBuffer, io, ops, spectral, synthesis
from nanodsp._cli import get_processor
from nanodsp.__main__ import main
from nanodsp.analysis import _k_weight_coeffs, loudness_lufs, pitch_detect
from nanodsp.effects import composed, daisysp, filters
from nanodsp.stream import STREAMABLE

SR = 48000.0


def _peak_hz(x: np.ndarray, sr: float = SR) -> float:
    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return float(np.argmax(spectrum)) * sr / len(x)


def _bessel_j(n: int, x: float) -> float:
    return sum(
        (-1) ** m / (factorial(m) * factorial(m + n)) * (x / 2) ** (2 * m + n)
        for m in range(30)
    )


# ---------------------------------------------------------------------------
# Vendored C++ and bindings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sr", [88200.0, 96000.0, 192000.0])
def test_reverb_sc_high_sample_rates(sr):
    """Delay lines were offset by bytes into a float array: heap overflow above 48 kHz."""
    out = daisysp.reverb_sc(
        AudioBuffer.sine(440.0, frames=int(sr // 4), sample_rate=sr)
    )
    assert np.isfinite(out.data).all()


def test_reverb_sc_rejects_rate_beyond_buffer():
    with pytest.raises(ValueError, match="sample_rate"):
        daisysp.reverb_sc(AudioBuffer.sine(440.0, frames=1000, sample_rate=384000.0))


def _run_isolated(code: str, tmp_path) -> subprocess.CompletedProcess:
    """Run in a child so a hang or segfault fails the test instead of the suite."""
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_flac_empty_and_truncated_do_not_hang_or_crash(tmp_path):
    code = """
import numpy as np
from nanodsp import io, AudioBuffer
io.write("e.flac", AudioBuffer(np.zeros((2, 0), dtype=np.float32), sample_rate=48000))
assert io.read("e.flac").frames == 0
rng = np.random.default_rng(0)
io.write("t.flac", AudioBuffer((rng.standard_normal((2, 48000)) * 0.2).astype(np.float32)))
raw = open("t.flac", "rb").read()
for n in (4, 42, 1000, len(raw) // 2):
    open("x.flac", "wb").write(raw[:n])
    try:
        io.read("x.flac")
    except RuntimeError:
        pass
    else:
        raise AssertionError(f"{n}-byte prefix read without error")
print("ok")
"""
    result = _run_isolated(code, tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_minblep_saw_harmonics():
    """The table had 1424 values in 2048 slots; the zero tail added +1 after every edge."""
    x = synthesis.minblep(480000, freq=100.0).data[0]
    mags = np.abs(np.fft.rfft(x)) / len(x) * 2
    for k in (1, 2, 3):
        assert mags[1000 * k] == pytest.approx(2 / (np.pi * k), rel=0.02)


def test_fm2_default_carrier_is_freq():
    """Init never tuned the oscillators, so the default 440 Hz played at 100 Hz."""
    x = synthesis.fm2(48000, index=0.0).data[0]
    assert _peak_hz(x) == pytest.approx(440.0, abs=1.0)


@pytest.mark.parametrize("beta", [0.5, 1.0, 2.4])
def test_fm2_index_is_classical(beta):
    x = synthesis.fm2(48000, freq=2000.0, ratio=0.1, index=beta).data[0]
    mags = np.abs(np.fft.rfft(x)) / len(x) * 2
    for k in range(3):
        assert mags[2000 + 200 * k] == pytest.approx(abs(_bessel_j(k, beta)), abs=0.01)


@pytest.mark.parametrize(
    "kwargs", [{"decay": 4.0}, {"decay": -0.5}, {"freq": 2.4e6}, {"freq": -440.0}]
)
def test_pluck_rejects_out_of_range(kwargs):
    with pytest.raises(ValueError):
        synthesis.pluck(4800, **kwargs)


def test_pluck_decay_one():
    assert np.isfinite(synthesis.pluck(4800, decay=1.0).data).all()


@pytest.mark.parametrize("max_size", [0, 100])
def test_comb_rejects_buffer_shorter_than_10ms(max_size):
    with pytest.raises(ValueError, match="max_size"):
        filters.comb_filter(AudioBuffer.sine(440.0, frames=4800), max_size=max_size)


@pytest.mark.parametrize(
    "fn",
    [
        lambda b: ops.box_filter(b, 0),
        lambda b: ops.box_filter(b, -1),
        lambda b: ops.box_stack_filter(b, -1000),
        lambda b: ops.peak_hold(b, 0),
        lambda b: ops.peak_decay(b, 0),
    ],
)
def test_envelope_lengths_validated(fn):
    with pytest.raises(ValueError):
        fn(AudioBuffer.sine(440.0, frames=4800))


def test_oversampler_chunks_by_max_block():
    """Frames beyond max_block used to write past the internal buffer."""
    rng = np.random.default_rng(0)
    buf = AudioBuffer(rng.standard_normal((2, 20000)).astype(np.float32))
    np.testing.assert_array_equal(
        ops.upsample_2x(buf).data, ops.upsample_2x(buf, max_block=16).data
    )
    np.testing.assert_array_equal(
        ops.oversample_roundtrip(buf).data,
        ops.oversample_roundtrip(buf, max_block=17).data,
    )


def test_ping_pong_delay_beyond_two_seconds():
    """Delays over the fixed 2 s line were silently clamped."""
    x = np.zeros((1, 48000 * 4), dtype=np.float32)
    x[0, 0] = 1.0
    out = composed.ping_pong_delay(
        AudioBuffer(x), delay_ms=3000.0, feedback=0.0, mix=1.0
    )
    assert int(np.argmax(np.abs(out.data[0]))) == pytest.approx(144000, abs=2)


@pytest.mark.parametrize("delay_ms", [float("inf"), float("nan"), -1.0, 1e9])
def test_ping_pong_delay_rejects_bad_delay(delay_ms):
    with pytest.raises(ValueError):
        composed.ping_pong_delay(AudioBuffer.sine(440.0, frames=480), delay_ms=delay_ms)


@pytest.mark.parametrize("semitones", [12.0, 7.0, 0.5, -7.0, -12.0, 24.0])
def test_daisysp_pitch_shift_pitch(semitones):
    """Downward shifts mapped to 2 - 2^(s/12); octaves multiplied; fractions truncated.

    Measured as periodicity: the crossfade's amplitude modulation can put the
    largest spectral peak on a sideband.
    """
    t = np.arange(96000) / SR
    tone = sum(np.sin(2 * np.pi * 220.0 * k * t) / k for k in range(1, 6)) * 0.3
    out = daisysp.pitch_shift(
        AudioBuffer(tone.astype(np.float32)[None]), semitones=semitones
    )
    freqs, _ = pitch_detect(out, fmin=40.0, window_size=4096)
    measured = float(np.median(freqs[freqs > 0]))
    cents = 1200 * np.log2(measured / (220.0 * 2 ** (semitones / 12)))
    # Measured worst case is 22 cents (+24 st). The old defects were >= 50.
    assert abs(cents) < 30


# ---------------------------------------------------------------------------
# Python DSP
# ---------------------------------------------------------------------------


def test_multiband_compress_is_flat_at_unity():
    """A single Butterworth LP/HP pair nulled each crossover by 13-26 dB."""
    for f in (200.0, 632.0, 2000.0, 8000.0):
        x = AudioBuffer.sine(f, frames=48000)
        y = composed.multiband_compress(x, ratios=[1.0] * 4, thresholds=[0.0] * 4)
        gain = np.sqrt(
            np.mean(y.data[0, 24000:] ** 2) / np.mean(x.data[0, 24000:] ** 2)
        )
        assert 20 * np.log10(gain) == pytest.approx(0.0, abs=0.1)


def test_yin_low_fundamental():
    """The difference function used whole-frame energy; 55 Hz was never voiced."""
    freqs, _ = pitch_detect(AudioBuffer.sine(55.0, frames=48000))
    voiced = freqs[freqs > 0]
    assert len(voiced) == len(freqs)
    assert float(np.median(voiced)) == pytest.approx(55.0, rel=0.01)


def test_k_weighting_matches_bs1770_table():
    shelf, hp = _k_weight_coeffs(48000.0)
    np.testing.assert_allclose(
        shelf,
        [
            1.53512485958697,
            -2.69169618940638,
            1.19839281085285,
            -1.69065929318241,
            0.73248077421585,
        ],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        hp, [1.0, -2.0, 1.0, -1.99004745483398, 0.99007225036621], atol=1e-12
    )


def test_loudness_reference_tone():
    """BS.1770: a full-scale 997 Hz sine reads -3.01 LUFS."""
    assert loudness_lufs(AudioBuffer.sine(997.0, frames=96000)) == pytest.approx(
        -3.01, abs=0.02
    )


@pytest.mark.parametrize("frames", [1000, 8000, 8191])
def test_istft_keeps_every_frame(frames):
    buf = AudioBuffer.noise(channels=1, frames=frames, seed=0)
    out = spectral.istft(spectral.stft(buf, window_size=1024, hop_size=256))
    assert out.frames == frames
    # The tail used to be dropped; it must now reconstruct. The first ~65
    # samples are tapered where under 0.1% of the full window overlap remains.
    np.testing.assert_allclose(out.data[0, 128:], buf.data[0, 128:], atol=1e-4)


def test_phase_lock_preserves_unmodified_signal():
    x = AudioBuffer.sine(440.0, frames=48000)
    y = spectral.istft(spectral.phase_lock(spectral.stft(x, 2048, 512)))
    a, b = x.data[0, 2048:-2048], y.data[0, 2048:-2048]
    snr = 10 * np.log10(np.sum(a**2) / np.sum((a - b) ** 2))
    assert snr > 30


def test_master_never_exceeds_full_scale():
    """normalize_lufs ran after the limiter and pushed peaks to +8 dBFS."""
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(48000 * 4) * 10 ** (-50 / 20)).astype(np.float32)
    x[::12000] = 0.9
    out = composed.master(AudioBuffer(x[None]))
    assert float(np.max(np.abs(out.data))) <= 1.0 + 1e-6


def test_master_hits_target_when_headroom_allows():
    out = composed.master(AudioBuffer.sine(220.0, frames=96000).gain_db(-20.0))
    assert loudness_lufs(out) == pytest.approx(-14.0, abs=0.1)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def _pcm_wav(bits: int, block_align: int, payload: bytes) -> bytes:
    fmt = struct.pack("<HHIIHH", 1, 1, 48000, 48000 * block_align, block_align, bits)
    body = b"WAVE" + b"fmt " + struct.pack("<I", 16) + fmt
    body += b"data" + struct.pack("<I", len(payload)) + payload
    return b"RIFF" + struct.pack("<I", len(body)) + body


def test_wav_12_and_20_bit(tmp_path):
    """Container width came from bits // 8, so 12-bit decoded as 8-bit noise."""
    v = np.array([0.5, -0.25, 0.125, -1.0])
    p12 = (np.round(v * 32768).astype("<i2") & ~0xF).astype("<i2").tobytes()
    (tmp_path / "a.wav").write_bytes(_pcm_wav(12, 2, p12))
    np.testing.assert_allclose(io.read(tmp_path / "a.wav").data[0], v)

    ints = np.round(v * 8388608).astype(np.int64) & ~0xF
    p20 = b"".join(int(i).to_bytes(4, "little", signed=True)[:3] for i in ints)
    (tmp_path / "b.wav").write_bytes(_pcm_wav(20, 3, p20))
    np.testing.assert_allclose(io.read(tmp_path / "b.wav").data[0], v)
    blocks = list(io.read_blocks(tmp_path / "b.wav", block_size=3))
    np.testing.assert_allclose(np.concatenate([b.data[0] for b in blocks]), v)


def test_wav_truncated_mid_frame_is_salvaged(tmp_path):
    payload = np.round(np.array([0.5, -0.5, 0.25]) * 32767).astype("<i2").tobytes()
    (tmp_path / "t.wav").write_bytes(_pcm_wav(16, 2, payload)[:-1])
    assert io.read(tmp_path / "t.wav").frames == 2


# ---------------------------------------------------------------------------
# Streaming and CLI
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(STREAMABLE))
def test_streaming_signature_matches_stateless(name):
    """Seven constructors had drifted in names or defaults from the whole-file form."""
    fn, _ = get_processor(name)
    whole = {
        k: p.default
        for k, p in list(inspect.signature(fn).parameters.items())[1:]
        if k != "link"
    }
    streamed = {
        k: p.default
        for k, p in inspect.signature(STREAMABLE[name]).parameters.items()
        if k not in ("channels", "sample_rate")
    }
    assert streamed == whole


def _write(path, frames=48000, channels=1):
    io.write(
        path, AudioBuffer.noise(channels=channels, frames=frames, seed=1).gain_db(-6.0)
    )


def test_stream_refuses_to_overwrite_its_input(tmp_path):
    src = tmp_path / "tone.wav"
    _write(src)
    before = src.read_bytes()
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "process",
                str(src),
                "-o",
                str(src),
                "--stream",
                "-f",
                "lowpass:cutoff_hz=1000",
            ]
        )
    assert exc.value.code == 1
    assert src.read_bytes() == before


def test_batch_refuses_duplicate_outputs(tmp_path):
    for d in ("a", "b"):
        (tmp_path / d).mkdir()
        _write(tmp_path / d / "x.wav")
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "process",
                str(tmp_path / "a" / "x.wav"),
                str(tmp_path / "b" / "x.wav"),
                "-O",
                str(tmp_path / "out"),
                "-f",
                "lowpass:cutoff_hz=1000",
            ]
        )
    assert exc.value.code == 1


def test_whole_file_in_place_still_allowed(tmp_path):
    src = tmp_path / "tone.wav"
    _write(src)
    main(["process", str(src), "-o", str(src), "-f", "lowpass:cutoff_hz=1000"])
    assert io.read(src).frames == 48000


def test_analyze_json_is_strict_for_silence(tmp_path, capsys):
    path = tmp_path / "sil.wav"
    io.write(path, AudioBuffer.zeros(1, 48000))

    def _reject(token):
        raise ValueError(token)

    for sub in ("info", "loudness"):
        main(["analyze", str(path), sub, "--json"])
        json.loads(capsys.readouterr().out, parse_constant=_reject)


@pytest.mark.parametrize("stretch", [float("inf"), float("nan")])
@pytest.mark.parametrize(
    "fn", ["paulstretch", "signalsmith_stretch", "keyframe_stretch"]
)
def test_timestretch_rejects_non_finite(fn, stretch):
    """paulstretch(inf) hung with the GIL released; signalsmith returned 0 frames."""
    from nanodsp import timestretch

    with pytest.raises(ValueError, match="finite"):
        getattr(timestretch, fn)(AudioBuffer.sine(440.0, frames=4800), stretch=stretch)


# ---------------------------------------------------------------------------
# Medium findings
# ---------------------------------------------------------------------------


def _impulse(frames: int = 400, at: int = 0) -> AudioBuffer:
    x = np.zeros((1, frames), dtype=np.float32)
    x[0, at] = 1.0
    return AudioBuffer(x)


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
@pytest.mark.parametrize("d", [1, 5, 63])
def test_delay_lands_where_asked(interpolation, d):
    """The cubic interpolator's 1-sample latency was added to every delay."""
    out = ops.delay(_impulse(), d, interpolation=interpolation)
    assert int(np.argmax(np.abs(out.data[0]))) == d


@pytest.mark.parametrize("d, cap", [(100, 10), (-5, None), (float("inf"), None)])
def test_delay_rejects_values_that_wrap(d, cap):
    with pytest.raises(ValueError, match="delay"):
        ops.delay(_impulse(), d, capacity=cap)


def test_stereo_pan_is_unity_at_centre():
    out = ops.pan(AudioBuffer.ones(2, 16), 0.0)
    np.testing.assert_allclose(out.data, 1.0)
    with pytest.raises(ValueError):
        ops.pan(AudioBuffer.ones(2, 16), 1.5)


def test_sine_phase_is_accurate_over_long_buffers():
    sr, frames = 48000.0, 48000 * 60
    x = AudioBuffer.sine(1000.0, frames=frames, sample_rate=sr).data[0, -48000:]
    t = np.arange(frames - 48000, frames, dtype=np.float64) / sr
    assert np.max(np.abs(x - np.sin(2 * np.pi * 1000.0 * t))) < 1e-6


def test_convolution_reverb_tail_includes_pre_delay():
    from nanodsp.effects.reverb import convolution_reverb

    ir = _impulse(2000, at=1500)
    out = convolution_reverb(
        _impulse(100), ir, mix=1.0, pre_delay_ms=10.0, tail=True, normalize=False
    )
    assert out.frames == 100 + 1999 + 480
    assert int(np.argmax(np.abs(out.data[0]))) == 1500 + 480


@pytest.mark.parametrize(
    "fn",
    [
        lambda b: filters.modal_bandpass(b, q=0.0),
        lambda b: filters.va_oberheim(b, q=0.0),
        lambda b: filters.va_moog_ladder(b, q=-1.0),
    ],
)
def test_filter_q_must_be_positive(fn):
    with pytest.raises(ValueError, match="q must be"):
        fn(AudioBuffer.sine(440.0, frames=480))


def test_onsets_are_not_reported_early():
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(48000) * 0.001).astype(np.float32)
    tone = np.sin(2 * np.pi * 440 * np.arange(2000) / SR).astype(np.float32)
    for s in (10000, 30000):
        x[s : s + 2000] += tone
    from nanodsp.analysis import onset_detect

    onsets = onset_detect(AudioBuffer(x[None]))
    # Within one hop (512) of the truth; the frame start was ~800 early.
    np.testing.assert_allclose(onsets, [10000, 30000], atol=256)


def test_pitch_detect_reaches_fmin_at_high_rates():
    freqs, _ = pitch_detect(AudioBuffer.sine(55.0, frames=96000, sample_rate=96000.0))
    assert (freqs > 0).all()
    assert float(np.median(freqs)) == pytest.approx(55.0, rel=0.01)


@pytest.mark.parametrize("hop", [0, -1])
def test_process_blocks_rejects_non_positive_hop(hop):
    from nanodsp.stream import process_blocks

    with pytest.raises(ValueError, match="hop_size"):
        process_blocks(AudioBuffer.sine(440.0, frames=100), lambda b: b, 10, hop)


def test_unclosed_block_writer_is_readable(tmp_path):
    path = tmp_path / "u.wav"
    w = io.BlockWriter(path, 48000.0)
    w.write(AudioBuffer.sine(440.0, frames=1000))
    w._fh.flush()  # type: ignore[union-attr]
    assert io.read(path).frames == 1000
    w.close()
    assert io.read(path).frames == 1000


def test_block_writer_refuses_to_pass_4gb(tmp_path):
    w = io.BlockWriter(tmp_path / "big.wav", 48000.0, channels=2)
    w._frames = (0xFFFFFFFF - 44) // 4
    with pytest.raises(ValueError, match="4 GB"):
        w.write(AudioBuffer.sine(440.0, channels=2, frames=10))
    w._frames = 0
    w.close()


@pytest.mark.parametrize("cls", ["SchroederReverb", "MoorerReverb"])
def test_fxdsp_reverbs_at_192k(cls):
    from nanodsp._core import fxdsp

    rev = getattr(fxdsp, cls)()
    rev.init(192000.0)
    out = np.asarray(rev.process(np.ones(19200, dtype=np.float32) * 0.1))
    assert np.isfinite(out).all()


def _raises_from_core(fn) -> None:
    with pytest.raises((ValueError, IndexError, RuntimeError)):
        fn()


@pytest.mark.parametrize(
    "fn",
    [
        lambda c: c.spectral.STFT(1, 256, 128).get_spectrum_channel(100000),
        lambda c: c.spectral.STFT(2, 256, 128).analyse(np.zeros((1, 256), np.float32)),
        lambda c: c.mix.Hadamard(1 << 20).in_place(np.zeros(2, np.float32)),
        lambda c: c.mix.Householder(1 << 22).in_place(np.zeros(2, np.float32)),
        lambda c: c.mix.Hadamard(3),
        lambda c: c.madronalib.delays.PitchbendableDelay().process(
            np.zeros(64, np.float32), np.zeros(64, np.float32)
        ),
        lambda c: c.madronalib.reverbs.FDN4().process(np.zeros(64, np.float32)),
        lambda c: c.madronalib.resampling.Upsampler(1).process(
            np.zeros(64, np.float32), 12
        ),
        lambda c: c.madronalib.resampling.Downsampler(0),
        lambda c: c.grainflow.Panner(64, 8, 2).process(
            np.zeros((64, 64), np.float32), np.zeros((1, 64), np.float32), 1
        ),
        lambda c: c.grainflow.GrainCollection(4, 48000).stream_get(10**6),
        lambda c: c.hisstools.analysis.PartialTracker(4, 4).get_track(10**8),
        lambda c: c.hisstools.spectral.SpectralProcessor(1024).change_phase(
            np.zeros(4096, np.float32), 0.5
        ),
        lambda c: c.daisysp.oscillators.HarmonicOscillator().set_amplitudes(
            np.zeros(2, np.float32)
        ),
        lambda c: c.stk.effects.FreeVerb().process(np.zeros((1, 64), np.float32)),
    ],
)
def test_core_rejects_out_of_range_shapes_and_indices(fn):
    """Each of these crashed the interpreter or read out of bounds."""
    from nanodsp import _core

    _raises_from_core(lambda: fn(_core))


def test_stk_functions_hold_the_sample_rate_lock():
    from nanodsp.effects import reverb

    for fn in (synthesis.synth_note, synthesis.synth_sequence, reverb.stk_reverb):
        assert hasattr(fn, "__wrapped__"), fn.__name__


def test_bool_parameter_typo_is_an_error():
    from nanodsp._cli import coerce_value

    assert coerce_value("off", bool) is False
    with pytest.raises(ValueError, match="boolean"):
        coerce_value("ture", bool)


def test_malformed_user_preset_is_reported(tmp_path, monkeypatch):
    from nanodsp import _cli

    path = tmp_path / "presets.json"
    path.write_text(json.dumps({"foo": ["oops"]}))
    monkeypatch.setenv("NANODSP_PRESETS", str(path))
    with pytest.raises(ValueError, match="'foo'"):
        _cli.load_user_presets()


@pytest.mark.parametrize(
    "argv",
    [
        ["process", "IN", "-o", "OUT", "-f", "lowpass:1000"],
        ["synth", "OUT", "sine", "--freq", "abc"],
        ["synth", "OUT", "drum", "--type", "synth_note"],
        ["convert", "IN", "OUT", "--sample-rate", "-5"],
        ["convert", "IN", "OUT", "--channels", "0"],
    ],
)
def test_cli_errors_exit_cleanly(argv, tmp_path, capsys):
    src = tmp_path / "in.wav"
    _write(src, frames=4800)
    argv = [
        str(src) if a == "IN" else str(tmp_path / "out.wav") if a == "OUT" else a
        for a in argv
    ]
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 1
    assert "Traceback" not in capsys.readouterr().err


def test_info_on_empty_file(tmp_path, capsys):
    path = tmp_path / "empty.wav"
    io.write(path, AudioBuffer.zeros(1, 0))
    main(["info", str(path), "--json"])
    assert json.loads(capsys.readouterr().out)["frames"] == 0

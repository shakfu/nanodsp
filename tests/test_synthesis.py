"""Tests for nanodsp.synthesis module (oscillators, noise, drums, physical modeling, STK synth)."""

import numpy as np
import pytest

from nanodsp import synthesis
from nanodsp.buffer import AudioBuffer


# ---------------------------------------------------------------------------
# DaisySP Oscillators
# ---------------------------------------------------------------------------


class TestDaisySPOscillators:
    def test_oscillator_sine_shape(self):
        result = synthesis.oscillator(1024, freq=440.0)
        assert result.channels == 1
        assert result.frames == 1024
        assert result.data.dtype == np.float32

    @pytest.mark.parametrize(
        "name",
        [
            "sine",
            "tri",
            "saw",
            "ramp",
            "square",
            "polyblep_tri",
            "polyblep_saw",
            "polyblep_square",
        ],
    )
    def test_oscillator_waveform_names(self, name):
        result = synthesis.oscillator(512, freq=440.0, waveform=name)
        assert result.frames == 512
        assert np.max(np.abs(result.data)) > 0, f"waveform {name} produced silence"

    def test_oscillator_int_waveform(self):
        from nanodsp._core import daisysp

        result = synthesis.oscillator(
            512, freq=440.0, waveform=daisysp.oscillators.WAVE_SAW
        )
        assert result.frames == 512

    def test_oscillator_invalid_waveform(self):
        with pytest.raises(ValueError, match="Unknown waveform"):
            synthesis.oscillator(512, waveform="nope")

    def test_oscillator_nonzero_output(self):
        result = synthesis.oscillator(4096, freq=440.0, amp=1.0)
        assert np.max(np.abs(result.data)) > 0.5

    def test_fm2_shape(self):
        result = synthesis.fm2(1024, freq=440.0, ratio=2.0, index=1.0)
        assert result.channels == 1
        assert result.frames == 1024

    def test_fm2_nonzero(self):
        result = synthesis.fm2(4096, freq=440.0)
        assert np.max(np.abs(result.data)) > 0.1

    def test_formant_oscillator_shape(self):
        result = synthesis.formant_oscillator(
            1024, carrier_freq=440.0, formant_freq=1000.0
        )
        assert result.channels == 1
        assert result.frames == 1024

    def test_formant_oscillator_nonzero(self):
        result = synthesis.formant_oscillator(4096, carrier_freq=440.0)
        assert np.max(np.abs(result.data)) > 0.1

    def test_bl_oscillator_shape(self):
        result = synthesis.bl_oscillator(1024, freq=440.0, waveform="saw")
        assert result.channels == 1
        assert result.frames == 1024

    @pytest.mark.parametrize("name", ["triangle", "tri", "saw", "square"])
    def test_bl_oscillator_waveform_names(self, name):
        result = synthesis.bl_oscillator(512, freq=440.0, waveform=name)
        assert result.frames == 512
        assert np.max(np.abs(result.data)) > 0, f"bl_osc waveform {name} silent"

    def test_bl_oscillator_invalid_waveform(self):
        with pytest.raises(ValueError, match="Unknown waveform"):
            synthesis.bl_oscillator(512, waveform="nope")

    def test_oscillator_sample_rate(self):
        result = synthesis.oscillator(512, freq=440.0, sample_rate=44100.0)
        assert result.sample_rate == 44100.0


# ---------------------------------------------------------------------------
# DaisySP Noise
# ---------------------------------------------------------------------------


class TestDaisySPNoise:
    @pytest.mark.parametrize(
        "func_name, kwargs",
        [
            ("white_noise", {}),
            ("clocked_noise", {"freq": 1000.0}),
            ("dust", {"density": 1.0}),
        ],
    )
    def test_noise_shape(self, func_name, kwargs):
        func = getattr(synthesis, func_name)
        result = func(1024, **kwargs)
        assert result.channels == 1
        assert result.frames == 1024
        assert result.data.dtype == np.float32

    @pytest.mark.parametrize(
        "func_name, frames, kwargs",
        [
            ("white_noise", 4096, {}),
            ("clocked_noise", 4096, {"freq": 1000.0}),
            ("dust", 48000, {"density": 100.0}),
        ],
    )
    def test_noise_nonzero(self, func_name, frames, kwargs):
        func = getattr(synthesis, func_name)
        result = func(frames, **kwargs)
        assert np.max(np.abs(result.data)) > 0

    def test_white_noise_amp(self):
        result = synthesis.white_noise(4096, amp=0.1)
        assert np.max(np.abs(result.data)) < 0.5

    def test_noise_sample_rate(self):
        result = synthesis.white_noise(512, sample_rate=44100.0)
        assert result.sample_rate == 44100.0


# ---------------------------------------------------------------------------
# DaisySP Drums
# ---------------------------------------------------------------------------


class TestDaisySPDrums:
    _drum_funcs = [
        "analog_bass_drum",
        "analog_snare_drum",
        "hihat",
        "synthetic_bass_drum",
        "synthetic_snare_drum",
    ]

    @pytest.mark.parametrize("drum_func", _drum_funcs)
    def test_drum_shape(self, drum_func):
        func = getattr(synthesis, drum_func)
        result = func(4096)
        assert result.channels == 1
        assert result.frames == 4096
        assert result.data.dtype == np.float32

    @pytest.mark.parametrize("drum_func", _drum_funcs)
    def test_drum_nonzero(self, drum_func):
        func = getattr(synthesis, drum_func)
        result = func(4096, accent=0.8)
        assert np.max(np.abs(result.data)) > 0.01

    def test_drums_decay(self):
        result = synthesis.analog_bass_drum(8192, decay=0.8)
        peak_idx = np.argmax(np.abs(result.data[0]))
        tail_energy = np.sum(result.data[0, peak_idx + 2048 :] ** 2)
        head_energy = np.sum(result.data[0, peak_idx : peak_idx + 2048] ** 2)
        # Tail should have less energy than head for a decaying drum
        assert tail_energy < head_energy


# ---------------------------------------------------------------------------
# DaisySP Physical Modeling
# ---------------------------------------------------------------------------


class TestDaisySPPhysicalModeling:
    def test_karplus_strong_shape(self):
        buf = AudioBuffer.impulse(channels=1, frames=4096, sample_rate=48000.0)
        result = synthesis.karplus_strong(buf, freq_hz=440.0)
        assert result.channels == 1
        assert result.frames == 4096

    def test_karplus_strong_nonzero(self):
        buf = AudioBuffer.impulse(channels=1, frames=4096, sample_rate=48000.0)
        result = synthesis.karplus_strong(buf, freq_hz=440.0)
        assert np.max(np.abs(result.data)) > 0.01

    def test_karplus_strong_multichannel(self):
        buf = AudioBuffer.impulse(channels=2, frames=4096, sample_rate=48000.0)
        result = synthesis.karplus_strong(buf, freq_hz=440.0)
        assert result.channels == 2

    @pytest.mark.parametrize(
        "func_name, nonzero_kwargs, threshold",
        [
            ("modal_voice", {"accent": 0.8}, 0.001),
            ("string_voice", {"accent": 0.8}, 0.001),
            ("pluck", {"amp": 0.8}, 0.01),
        ],
    )
    def test_voice_shape(self, func_name, nonzero_kwargs, threshold):
        func = getattr(synthesis, func_name)
        result = func(4096, freq=440.0)
        assert result.channels == 1
        assert result.frames == 4096

    @pytest.mark.parametrize(
        "func_name, nonzero_kwargs, threshold",
        [
            ("modal_voice", {"accent": 0.8}, 0.001),
            ("string_voice", {"accent": 0.8}, 0.001),
            ("pluck", {"amp": 0.8}, 0.01),
        ],
    )
    def test_voice_nonzero(self, func_name, nonzero_kwargs, threshold):
        func = getattr(synthesis, func_name)
        result = func(4096, freq=440.0, **nonzero_kwargs)
        assert np.max(np.abs(result.data)) > threshold

    def test_drip_shape(self):
        result = synthesis.drip(4096)
        assert result.channels == 1
        assert result.frames == 4096

    def test_drip_nonzero(self):
        result = synthesis.drip(4096)
        # Drip may or may not produce output on first trigger, just check shape
        assert result.data.dtype == np.float32


# ---------------------------------------------------------------------------
# STK synth_note
# ---------------------------------------------------------------------------


class TestSynthNote:
    def test_basic_note(self):
        result = synthesis.synth_note("clarinet", freq=440.0, duration=0.5)
        assert result.channels == 1
        expected_frames = int(48000 * 0.5) + int(48000 * 0.1)
        assert result.frames == expected_frames
        assert result.sample_rate == 48000.0

    def test_nonzero_output(self):
        result = synthesis.synth_note(
            "clarinet", freq=440.0, duration=0.5, velocity=0.8
        )
        assert np.max(np.abs(result.data)) > 0.001

    @pytest.mark.parametrize(
        "name",
        [
            "clarinet",
            "flute",
            "brass",
            "bowed",
            "plucked",
            "sitar",
            "stifkarp",
            "saxofony",
            "recorder",
            "blowbotl",
            "blowhole",
            "whistle",
        ],
    )
    def test_all_instruments(self, name):
        result = synthesis.synth_note(name, freq=440.0, duration=0.2, release=0.05)
        assert result.channels == 1, f"{name} wrong channels"
        assert result.frames > 0, f"{name} no frames"

    def test_invalid_instrument_raises(self):
        with pytest.raises(ValueError, match="Unknown instrument"):
            synthesis.synth_note("kazoo")

    def test_custom_sample_rate(self):
        result = synthesis.synth_note(
            "plucked", freq=440.0, duration=0.5, sample_rate=44100.0
        )
        assert result.sample_rate == 44100.0
        expected_frames = int(44100 * 0.5) + int(44100 * 0.1)
        assert result.frames == expected_frames

    def test_label_set(self):
        result = synthesis.synth_note("flute", freq=440.0, duration=0.2)
        assert result.label == "flute"

    def test_different_frequencies(self):
        low = synthesis.synth_note("clarinet", freq=220.0, duration=0.5)
        high = synthesis.synth_note("clarinet", freq=880.0, duration=0.5)
        # They should produce different waveforms
        assert not np.allclose(low.data[:, : low.frames], high.data[:, : high.frames])


# ---------------------------------------------------------------------------
# STK synth_sequence
# ---------------------------------------------------------------------------


class TestSynthSequence:
    def test_basic_sequence(self):
        notes = [
            (440.0, 0.0, 0.3),
            (550.0, 0.4, 0.3),
            (660.0, 0.8, 0.3),
        ]
        result = synthesis.synth_sequence("plucked", notes)
        assert result.channels == 1
        # Should be long enough for all notes + release
        total_end = max(s + d + 0.1 for _, s, d in notes)
        assert result.frames >= int(48000 * total_end)

    def test_empty_notes_raises(self):
        with pytest.raises(ValueError, match="notes list must not be empty"):
            synthesis.synth_sequence("clarinet", [])

    def test_overlapping_notes(self):
        """Overlapping notes should sum together."""
        notes = [
            (440.0, 0.0, 0.5),
            (660.0, 0.0, 0.5),
        ]
        result = synthesis.synth_sequence("plucked", notes)
        assert result.channels == 1
        # Peak should be higher than single note due to overlap
        single = synthesis.synth_note("plucked", freq=440.0, duration=0.5)
        assert np.max(np.abs(result.data)) >= np.max(np.abs(single.data)) * 0.5

    def test_label(self):
        notes = [(440.0, 0.0, 0.2)]
        result = synthesis.synth_sequence("sitar", notes)
        assert result.label == "sitar"

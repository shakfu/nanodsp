"""Tests for the nanodsp CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nanodsp.buffer import AudioBuffer
from nanodsp._cli import (
    parse_fx_token,
    coerce_value,
    coerce_params,
    get_registry,
    get_categories,
    get_function,
    format_signature,
    PRESETS,
    apply_preset,
    get_preset_categories,
)
from nanodsp.__main__ import build_parser, main


# ---------------------------------------------------------------------------
# FX token parsing
# ---------------------------------------------------------------------------


class TestParseFxToken:
    def test_name_only(self):
        name, params = parse_fx_token("lowpass")
        assert name == "lowpass"
        assert params == {}

    def test_name_with_params(self):
        name, params = parse_fx_token("lowpass:cutoff_hz=1000,octaves=2")
        assert name == "lowpass"
        assert params == {"cutoff_hz": "1000", "octaves": "2"}

    def test_single_param(self):
        name, params = parse_fx_token("compress:ratio=4")
        assert name == "compress"
        assert params == {"ratio": "4"}

    def test_whitespace_stripped(self):
        name, params = parse_fx_token(" highpass : freq = 80 ")
        assert name == "highpass"
        assert params == {"freq": "80"}

    def test_invalid_param_no_equals(self):
        with pytest.raises(ValueError, match="Invalid parameter"):
            parse_fx_token("lowpass:bad_param")

    def test_string_param_value(self):
        name, params = parse_fx_token("saturate:drive=0.5,mode=tape")
        assert name == "saturate"
        assert params == {"drive": "0.5", "mode": "tape"}


# ---------------------------------------------------------------------------
# Type coercion
# ---------------------------------------------------------------------------


class TestCoerceValue:
    def test_bool_true(self):
        assert coerce_value("true", bool) is True
        assert coerce_value("1", bool) is True
        assert coerce_value("yes", bool) is True

    def test_bool_false(self):
        assert coerce_value("false", bool) is False
        assert coerce_value("0", bool) is False
        assert coerce_value("no", bool) is False

    def test_int(self):
        assert coerce_value("42", int) == 42

    def test_float(self):
        assert coerce_value("3.14", float) == pytest.approx(3.14)

    def test_str(self):
        assert coerce_value("hello", str) == "hello"

    def test_none_guess_float(self):
        result = coerce_value("3.14", None)
        assert result == pytest.approx(3.14)

    def test_none_guess_int(self):
        result = coerce_value("42", None)
        assert result == 42

    def test_none_guess_str(self):
        result = coerce_value("hello", None)
        assert result == "hello"


class TestCoerceParams:
    def test_with_signature(self):
        from nanodsp.effects import lowpass

        params = coerce_params(lowpass, {"cutoff_hz": "1000"})
        assert params["cutoff_hz"] == pytest.approx(1000.0)

    def test_with_int_default(self):
        from nanodsp.effects import decimator

        params = coerce_params(decimator, {"bits_to_crush": "12"})
        assert params["bits_to_crush"] == 12

    def test_with_bool_default(self):
        from nanodsp.effects import decimator

        params = coerce_params(decimator, {"smooth": "true"})
        assert params["smooth"] is True

    def test_unknown_param_guess(self):
        from nanodsp.effects import lowpass

        params = coerce_params(lowpass, {"unknown_param": "3.5"})
        assert params["unknown_param"] == pytest.approx(3.5)


# ---------------------------------------------------------------------------
# Function registry
# ---------------------------------------------------------------------------


class TestFunctionRegistry:
    def test_registry_not_empty(self):
        reg = get_registry()
        assert len(reg) > 50  # We know there are ~125 functions

    def test_known_functions_present(self):
        reg = get_registry()
        for name in ["lowpass", "highpass", "compress", "reverb", "oscillator"]:
            assert name in reg, f"{name} missing from registry"

    def test_registry_tuple_structure(self):
        fn, module_name = get_function("lowpass")
        assert callable(fn)
        assert module_name == "effects"

    def test_unknown_function_raises(self):
        with pytest.raises(KeyError):
            get_function("nonexistent_function")

    def test_categories_not_empty(self):
        cats = get_categories()
        assert len(cats) > 0
        for cat, names in cats.items():
            if names:
                for name in names:
                    assert name in get_registry()

    def test_filters_category(self):
        cats = get_categories()
        assert "lowpass" in cats["filters"]
        assert "highpass" in cats["filters"]
        assert "bandpass" in cats["filters"]

    def test_effects_category(self):
        cats = get_categories()
        assert "chorus" in cats["effects"]
        assert "reverb" in cats["effects"]

    def test_dynamics_category(self):
        cats = get_categories()
        assert "compress" in cats["dynamics"]
        assert "limit" in cats["dynamics"]

    def test_synthesis_category(self):
        cats = get_categories()
        assert "oscillator" in cats["synthesis"]

    def test_analysis_category(self):
        cats = get_categories()
        assert "loudness_lufs" in cats["analysis"]


class TestFormatSignature:
    def test_with_defaults(self):
        from nanodsp.effects import lowpass

        sig = format_signature(lowpass)
        assert "cutoff_hz" in sig

    def test_skips_buf_param(self):
        from nanodsp.effects import lowpass

        sig = format_signature(lowpass)
        assert "buf" not in sig

    def test_synthesis_fn(self):
        from nanodsp.synthesis import oscillator

        sig = format_signature(oscillator)
        assert "frames" in sig
        assert "freq" in sig


# ---------------------------------------------------------------------------
# Preset registry
# ---------------------------------------------------------------------------


class TestPresetRegistry:
    def test_presets_not_empty(self):
        assert len(PRESETS) >= 15

    def test_all_presets_have_category(self):
        for name, info in PRESETS.items():
            assert "category" in info, f"Preset {name!r} missing category"

    def test_all_presets_have_description(self):
        for name, info in PRESETS.items():
            assert "description" in info, f"Preset {name!r} missing description"

    def test_all_presets_have_fn_or_chain(self):
        for name, info in PRESETS.items():
            assert "fn" in info or "chain" in info, (
                f"Preset {name!r} needs 'fn' or 'chain'"
            )

    def test_preset_categories(self):
        cats = get_preset_categories()
        assert "mastering" in cats
        assert "spatial" in cats
        assert "lofi" in cats

    def test_apply_preset_simple(self):
        sr = 48000
        buf = AudioBuffer.sine(440.0, frames=sr, sample_rate=sr)
        result = apply_preset("dc_remove", buf)
        assert isinstance(result, AudioBuffer)
        assert result.frames == buf.frames

    def test_apply_preset_chain(self):
        sr = 48000
        buf = AudioBuffer.sine(440.0, frames=sr, sample_rate=sr)
        result = apply_preset("telephone", buf)
        assert isinstance(result, AudioBuffer)
        assert result.frames == buf.frames

    def test_apply_preset_with_overrides(self):
        sr = 48000
        buf = AudioBuffer.sine(440.0, frames=sr, sample_rate=sr)
        result = apply_preset("gentle_compress", buf, {"ratio": 3.0})
        assert isinstance(result, AudioBuffer)

    def test_apply_unknown_preset_raises(self):
        buf = AudioBuffer.sine(440.0, frames=48000, sample_rate=48000)
        with pytest.raises(KeyError, match="Unknown preset"):
            apply_preset("nonexistent", buf)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class TestParser:
    def test_version(self):
        parser = build_parser()
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(["--version"])
        assert exc.value.code == 0

    def test_info_command(self):
        parser = build_parser()
        args = parser.parse_args(["info", "test.wav"])
        assert args.command == "info"
        assert args.file == "test.wav"

    def test_process_command(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "process",
                "in.wav",
                "-o",
                "out.wav",
                "--fx",
                "lowpass:cutoff_hz=1000",
                "--fx",
                "compress:ratio=4",
            ]
        )
        assert args.command == "process"
        assert args.input == ["in.wav"]
        assert args.output == "out.wav"
        assert args.fx == ["lowpass:cutoff_hz=1000", "compress:ratio=4"]

    def test_process_batch_args(self):
        parser = build_parser()
        args = parser.parse_args(
            ["process", "a.wav", "b.wav", "-O", "outdir", "-f", "lowpass:cutoff_hz=500"]
        )
        assert args.input == ["a.wav", "b.wav"]
        assert args.output_dir == "outdir"
        assert args.output is None

    def test_process_dry_run_flag(self):
        parser = build_parser()
        args = parser.parse_args(
            ["process", "in.wav", "-n", "-f", "lowpass:cutoff_hz=500"]
        )
        assert args.dry_run is True

    def test_verbose_flag(self):
        parser = build_parser()
        args = parser.parse_args(["-v", "list"])
        assert args.verbose is True
        assert args.quiet is False

    def test_quiet_flag(self):
        parser = build_parser()
        args = parser.parse_args(["-q", "list"])
        assert args.quiet is True
        assert args.verbose is False

    def test_verbose_quiet_mutually_exclusive(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["-v", "-q", "list"])

    def test_analyze_command(self):
        parser = build_parser()
        args = parser.parse_args(["analyze", "in.wav", "loudness"])
        assert args.command == "analyze"
        assert args.analysis_type == "loudness"

    def test_synth_command(self):
        parser = build_parser()
        args = parser.parse_args(["synth", "out.wav", "sine", "--freq=440"])
        assert args.command == "synth"
        assert args.synth_type == "sine"
        assert args.freq == "440"

    def test_convert_command(self):
        parser = build_parser()
        args = parser.parse_args(["convert", "in.wav", "out.flac"])
        assert args.command == "convert"

    def test_list_command(self):
        parser = build_parser()
        args = parser.parse_args(["list"])
        assert args.command == "list"

    def test_list_with_category(self):
        parser = build_parser()
        args = parser.parse_args(["list", "filters"])
        assert args.command == "list"
        assert args.category == "filters"

    def test_preset_list(self):
        parser = build_parser()
        args = parser.parse_args(["preset", "list"])
        assert args.command == "preset"
        assert args.preset_action == "list"

    def test_preset_info(self):
        parser = build_parser()
        args = parser.parse_args(["preset", "info", "master"])
        assert args.command == "preset"
        assert args.preset_action == "info"
        assert args.name == "master"


# ---------------------------------------------------------------------------
# End-to-end CLI tests (using tmp files)
# ---------------------------------------------------------------------------


class TestCLIEndToEnd:
    @pytest.fixture
    def wav_file(self, tmp_path):
        """Create a test WAV file."""
        from nanodsp.io import write_wav

        buf = AudioBuffer.sine(440.0, frames=48000, sample_rate=48000.0)
        path = tmp_path / "test.wav"
        write_wav(str(path), buf)
        return str(path)

    def test_info(self, wav_file, capsys):
        main(["info", wav_file])
        captured = capsys.readouterr()
        assert "sample_rate" in captured.out
        assert "48000" in captured.out

    def test_info_json(self, wav_file, capsys):
        main(["info", wav_file, "--json"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["sample_rate"] == 48000

    def test_process_single_fx(self, wav_file, tmp_path):
        out = str(tmp_path / "out.wav")
        main(["process", wav_file, "-o", out, "--fx", "lowpass:cutoff_hz=1000"])
        assert Path(out).exists()

    def test_process_multiple_fx(self, wav_file, tmp_path):
        out = str(tmp_path / "out.wav")
        main(
            [
                "process",
                wav_file,
                "-o",
                out,
                "--fx",
                "highpass:cutoff_hz=80",
                "--fx",
                "compress:ratio=4,threshold=-20",
            ]
        )
        assert Path(out).exists()

    def test_process_with_preset(self, wav_file, tmp_path):
        out = str(tmp_path / "out.wav")
        main(["process", wav_file, "-o", out, "--preset", "telephone"])
        assert Path(out).exists()

    def test_synth_sine(self, tmp_path):
        out = str(tmp_path / "tone.wav")
        main(["synth", out, "sine", "--freq=440", "--duration=0.5"])
        assert Path(out).exists()
        from nanodsp.io import read_wav

        buf = read_wav(out)
        assert buf.frames == 24000
        assert buf.sample_rate == 48000.0

    def test_synth_noise(self, tmp_path):
        out = str(tmp_path / "noise.wav")
        main(["synth", out, "noise", "--duration=0.5"])
        assert Path(out).exists()

    def test_synth_drum(self, tmp_path):
        out = str(tmp_path / "kick.wav")
        main(
            [
                "synth",
                out,
                "drum",
                "--type=analog_bass_drum",
                "--freq=60",
                "--duration=0.5",
            ]
        )
        assert Path(out).exists()

    def test_synth_oscillator(self, tmp_path):
        out = str(tmp_path / "osc.wav")
        main(
            [
                "synth",
                out,
                "oscillator",
                "--freq=220",
                "--waveform=saw",
                "--duration=0.5",
            ]
        )
        assert Path(out).exists()

    def test_synth_fm(self, tmp_path):
        out = str(tmp_path / "fm.wav")
        main(
            [
                "synth",
                out,
                "fm",
                "--freq=440",
                "--ratio=2.0",
                "--index=1.0",
                "--duration=0.5",
            ]
        )
        assert Path(out).exists()

    def test_synth_note(self, tmp_path):
        out = str(tmp_path / "note.wav")
        main(
            [
                "synth",
                out,
                "note",
                "--instrument=clarinet",
                "--freq=440",
                "--duration=0.5",
            ]
        )
        assert Path(out).exists()

    def test_synth_sequence(self, tmp_path):
        out = str(tmp_path / "seq.wav")
        notes = json.dumps(
            [
                {"freq": 440, "start": 0.0, "dur": 0.3},
                {"freq": 550, "start": 0.3, "dur": 0.3},
            ]
        )
        main(["synth", out, "sequence", "--instrument=flute", f"--notes={notes}"])
        assert Path(out).exists()

    def test_analyze_loudness(self, wav_file, capsys):
        main(["analyze", wav_file, "loudness"])
        captured = capsys.readouterr()
        assert "LUFS" in captured.out

    def test_analyze_centroid(self, wav_file, capsys):
        main(["analyze", wav_file, "centroid"])
        captured = capsys.readouterr()
        assert "shape" in captured.out

    def test_analyze_info(self, wav_file, capsys):
        main(["analyze", wav_file, "info"])
        captured = capsys.readouterr()
        assert "sample_rate" in captured.out

    def test_analyze_json(self, wav_file, capsys):
        main(["analyze", wav_file, "loudness", "--json"])
        captured = capsys.readouterr()
        result = json.loads(captured.out)
        assert isinstance(result, float)

    def test_convert_wav_to_flac(self, wav_file, tmp_path):
        out = str(tmp_path / "out.flac")
        main(["convert", wav_file, out])
        assert Path(out).exists()

    def test_convert_with_sample_rate(self, wav_file, tmp_path):
        out = str(tmp_path / "out.wav")
        main(["convert", wav_file, out, "--sample-rate=24000"])
        assert Path(out).exists()
        from nanodsp.io import read_wav

        buf = read_wav(out)
        assert buf.sample_rate == 24000.0

    def test_convert_to_mono(self, tmp_path):
        from nanodsp.io import write_wav, read_wav

        buf = AudioBuffer.sine(440.0, channels=2, frames=48000, sample_rate=48000.0)
        inp = str(tmp_path / "stereo.wav")
        write_wav(inp, buf)
        out = str(tmp_path / "mono.wav")
        main(["convert", inp, out, "--channels=1"])
        result = read_wav(out)
        assert result.channels == 1

    def test_list_all(self, capsys):
        main(["list"])
        captured = capsys.readouterr()
        assert "filters" in captured.out
        assert "lowpass" in captured.out

    def test_list_category(self, capsys):
        main(["list", "filters"])
        captured = capsys.readouterr()
        assert "lowpass" in captured.out
        assert "highpass" in captured.out

    def test_preset_list(self, capsys):
        main(["preset", "list"])
        captured = capsys.readouterr()
        assert "master" in captured.out

    def test_preset_list_category(self, capsys):
        main(["preset", "list", "spatial"])
        captured = capsys.readouterr()
        assert "room" in captured.out
        assert "hall" in captured.out

    def test_preset_info(self, capsys):
        main(["preset", "info", "master"])
        captured = capsys.readouterr()
        assert "Mastering chain" in captured.out

    def test_preset_apply(self, wav_file, tmp_path):
        out = str(tmp_path / "out.wav")
        main(["preset", "apply", "telephone", wav_file, out])
        assert Path(out).exists()

    def test_no_command_shows_help(self, capsys):
        with pytest.raises(SystemExit) as exc:
            main([])
        assert exc.value.code == 0

    def test_synth_24bit(self, tmp_path):
        out = str(tmp_path / "tone24.wav")
        main(["synth", out, "sine", "--freq=440", "--duration=0.5", "-b", "24"])
        assert Path(out).exists()

    def test_process_unknown_fx(self, wav_file, tmp_path):
        out = str(tmp_path / "out.wav")
        with pytest.raises(SystemExit):
            main(["process", wav_file, "-o", out, "--fx", "nonexistent_effect"])

    def test_process_unknown_preset(self, wav_file, tmp_path):
        out = str(tmp_path / "out.wav")
        with pytest.raises(SystemExit):
            main(["process", wav_file, "-o", out, "--preset", "nonexistent"])

    # --- Dry-run tests ---

    def test_process_dry_run(self, wav_file, capsys):
        """Dry run prints chain info without reading/writing files."""
        main(
            [
                "process",
                wav_file,
                "-n",
                "-f",
                "lowpass:cutoff_hz=1000",
                "-f",
                "compress:ratio=4",
            ]
        )
        captured = capsys.readouterr()
        assert "Chain" in captured.out
        assert "lowpass" in captured.out
        assert "compress" in captured.out
        assert "2 steps" in captured.out

    def test_process_dry_run_no_output_required(self, capsys):
        """Dry run does not require -o or -O."""
        main(["process", "fake_file.wav", "-n", "-f", "lowpass:cutoff_hz=1000"])
        captured = capsys.readouterr()
        assert "Chain" in captured.out

    def test_process_dry_run_with_preset(self, wav_file, capsys):
        main(["process", wav_file, "-n", "-p", "telephone"])
        captured = capsys.readouterr()
        assert "telephone" in captured.out
        assert "preset" in captured.out.lower()

    def test_process_dry_run_empty_chain(self, capsys):
        main(["process", "fake.wav", "-n"])
        captured = capsys.readouterr()
        assert "empty" in captured.out.lower()

    # --- Batch mode tests ---

    def test_process_batch_output_dir(self, tmp_path):
        """Batch mode writes files to --output-dir."""
        from nanodsp.io import write_wav

        # Create two input files
        buf = AudioBuffer.sine(440.0, frames=4800, sample_rate=48000.0)
        in1 = str(tmp_path / "a.wav")
        in2 = str(tmp_path / "b.wav")
        write_wav(in1, buf)
        write_wav(in2, buf)

        out_dir = str(tmp_path / "batch_out")
        main(
            [
                "process",
                in1,
                in2,
                "-O",
                out_dir,
                "-f",
                "lowpass:cutoff_hz=2000",
            ]
        )
        assert (Path(out_dir) / "a.wav").exists()
        assert (Path(out_dir) / "b.wav").exists()

    def test_process_no_output_error(self, wav_file):
        """process without -o or -O exits with error."""
        with pytest.raises(SystemExit):
            main(["process", wav_file, "-f", "lowpass:cutoff_hz=1000"])

    def test_process_both_output_flags_error(self, wav_file, tmp_path):
        """process with both -o and -O exits with error."""
        out = str(tmp_path / "out.wav")
        out_dir = str(tmp_path / "outdir")
        with pytest.raises(SystemExit):
            main(
                [
                    "process",
                    wav_file,
                    "-o",
                    out,
                    "-O",
                    out_dir,
                    "-f",
                    "lowpass:cutoff_hz=1000",
                ]
            )

    def test_process_multi_input_with_single_output_error(self, tmp_path):
        """Multiple inputs with -o (not -O) exits with error."""
        from nanodsp.io import write_wav

        buf = AudioBuffer.sine(440.0, frames=4800, sample_rate=48000.0)
        in1 = str(tmp_path / "a.wav")
        in2 = str(tmp_path / "b.wav")
        write_wav(in1, buf)
        write_wav(in2, buf)
        out = str(tmp_path / "out.wav")
        with pytest.raises(SystemExit):
            main(["process", in1, in2, "-o", out, "-f", "lowpass:cutoff_hz=1000"])

    # --- Verbose / quiet tests ---

    def test_verbose_output(self, wav_file, tmp_path, capsys):
        """Verbose mode prints extra details."""
        out = str(tmp_path / "out.wav")
        main(["-v", "process", wav_file, "-o", out, "-f", "lowpass:cutoff_hz=1000"])
        captured = capsys.readouterr()
        assert "Reading" in captured.out
        assert "Writing" in captured.out

    def test_quiet_suppresses_output(self, wav_file, tmp_path, capsys):
        """Quiet mode suppresses normal output."""
        out = str(tmp_path / "out.wav")
        main(["-q", "process", wav_file, "-o", out, "-f", "lowpass:cutoff_hz=1000"])
        captured = capsys.readouterr()
        # "Wrote ..." should be suppressed in quiet mode
        assert "Wrote" not in captured.out

    # --- Short flags ---

    def test_short_flags_f_and_p(self, wav_file, tmp_path):
        """Short flags -f and -p work for --fx and --preset."""
        out = str(tmp_path / "out.wav")
        main(["process", wav_file, "-o", out, "-f", "lowpass:cutoff_hz=1000"])
        assert Path(out).exists()

    def test_short_flag_p_preset(self, wav_file, tmp_path):
        out = str(tmp_path / "out.wav")
        main(["process", wav_file, "-o", out, "-p", "telephone"])
        assert Path(out).exists()

    # --- New preset tests ---

    def test_new_presets_exist(self):
        new_names = [
            "master_pop",
            "master_hiphop",
            "master_classical",
            "master_edm",
            "master_podcast",
            "radio",
            "underwater",
            "megaphone",
            "tape_warmth",
            "shimmer",
            "vaporwave",
            "walkie_talkie",
            "8bit",
        ]
        for name in new_names:
            assert name in PRESETS, f"Preset {name!r} missing from PRESETS"

    def test_preset_count_at_least_30(self):
        assert len(PRESETS) >= 30

    def test_new_preset_categories(self):
        cats = get_preset_categories()
        assert "creative" in cats
        creative_presets = cats["creative"]
        for name in [
            "radio",
            "underwater",
            "megaphone",
            "tape_warmth",
            "shimmer",
            "vaporwave",
            "walkie_talkie",
        ]:
            assert name in creative_presets, f"{name!r} not in creative category"

    def test_apply_preset_radio(self):
        buf = AudioBuffer.sine(440.0, frames=4800, sample_rate=48000.0)
        result = apply_preset("radio", buf)
        assert isinstance(result, AudioBuffer)
        assert result.frames == buf.frames

    def test_apply_preset_tape_warmth(self):
        buf = AudioBuffer.sine(440.0, frames=4800, sample_rate=48000.0)
        result = apply_preset("tape_warmth", buf)
        assert isinstance(result, AudioBuffer)
        assert result.frames == buf.frames

    def test_apply_preset_8bit(self):
        buf = AudioBuffer.sine(440.0, frames=4800, sample_rate=48000.0)
        result = apply_preset("8bit", buf)
        assert isinstance(result, AudioBuffer)

    def test_apply_preset_walkie_talkie(self):
        buf = AudioBuffer.sine(440.0, frames=4800, sample_rate=48000.0)
        result = apply_preset("walkie_talkie", buf)
        assert isinstance(result, AudioBuffer)
        assert result.frames == buf.frames

    def test_apply_preset_megaphone(self):
        buf = AudioBuffer.sine(440.0, frames=4800, sample_rate=48000.0)
        result = apply_preset("megaphone", buf)
        assert isinstance(result, AudioBuffer)
        assert result.frames == buf.frames

    def test_apply_preset_master_pop(self):
        buf = AudioBuffer.sine(440.0, frames=48000, sample_rate=48000.0)
        result = apply_preset("master_pop", buf)
        assert isinstance(result, AudioBuffer)

    def test_apply_preset_master_podcast(self):
        buf = AudioBuffer.sine(440.0, frames=48000, sample_rate=48000.0)
        result = apply_preset("master_podcast", buf)
        assert isinstance(result, AudioBuffer)

    # --- Pipe tests ---

    def test_pipe_basic(self, tmp_path, monkeypatch):
        """Pipe reads WAV from stdin, writes WAV to stdout."""
        import io as _io
        from nanodsp.io import write_wav_bytes, read_wav_bytes

        buf = AudioBuffer.sine(440.0, frames=4800, sample_rate=48000.0)
        wav_data = write_wav_bytes(buf)

        # Mock stdin/stdout
        mock_stdin = _io.BytesIO(wav_data)
        mock_stdout = _io.BytesIO()

        monkeypatch.setattr(
            "sys.stdin", type("FakeStdin", (), {"buffer": mock_stdin})()
        )
        monkeypatch.setattr(
            "sys.stdout", type("FakeStdout", (), {"buffer": mock_stdout})()
        )

        main(["pipe", "-f", "lowpass:cutoff_hz=2000"])

        mock_stdout.seek(0)
        result = read_wav_bytes(mock_stdout.read())
        assert result.channels == 1
        assert result.frames == 4800
        assert result.sample_rate == 48000.0

    def test_pipe_with_preset(self, tmp_path, monkeypatch):
        """Pipe works with preset flag."""
        import io as _io
        from nanodsp.io import write_wav_bytes, read_wav_bytes

        buf = AudioBuffer.sine(440.0, frames=4800, sample_rate=48000.0)
        wav_data = write_wav_bytes(buf)

        mock_stdin = _io.BytesIO(wav_data)
        mock_stdout = _io.BytesIO()

        monkeypatch.setattr(
            "sys.stdin", type("FakeStdin", (), {"buffer": mock_stdin})()
        )
        monkeypatch.setattr(
            "sys.stdout", type("FakeStdout", (), {"buffer": mock_stdout})()
        )

        main(["pipe", "-p", "telephone"])

        mock_stdout.seek(0)
        result = read_wav_bytes(mock_stdout.read())
        assert result.channels == 1
        assert result.frames == 4800

    def test_pipe_empty_stdin(self, monkeypatch):
        """Pipe with empty stdin exits with error."""
        import io as _io

        mock_stdin = _io.BytesIO(b"")
        monkeypatch.setattr(
            "sys.stdin", type("FakeStdin", (), {"buffer": mock_stdin})()
        )

        with pytest.raises(SystemExit):
            main(["pipe", "-f", "lowpass:cutoff_hz=1000"])

    def test_pipe_no_fx_passthrough(self, monkeypatch):
        """Pipe with no fx/preset passes audio through unchanged."""
        import io as _io
        from nanodsp.io import write_wav_bytes, read_wav_bytes

        buf = AudioBuffer.sine(440.0, frames=4800, sample_rate=48000.0)
        wav_data = write_wav_bytes(buf)

        mock_stdin = _io.BytesIO(wav_data)
        mock_stdout = _io.BytesIO()

        monkeypatch.setattr(
            "sys.stdin", type("FakeStdin", (), {"buffer": mock_stdin})()
        )
        monkeypatch.setattr(
            "sys.stdout", type("FakeStdout", (), {"buffer": mock_stdout})()
        )

        main(["pipe"])

        mock_stdout.seek(0)
        result = read_wav_bytes(mock_stdout.read())
        assert result.frames == 4800
        import numpy as np

        # Should be close to original (quantization noise from 16-bit roundtrip)
        assert np.max(np.abs(result.data - buf.data)) < 0.001

    # --- Benchmark tests ---

    def test_benchmark_parser(self):
        parser = build_parser()
        args = parser.parse_args(["benchmark", "lowpass:cutoff_hz=1000"])
        assert args.command == "benchmark"
        assert args.function == "lowpass:cutoff_hz=1000"
        assert args.iterations == 100
        assert args.warmup == 5
        assert args.duration == 1.0

    def test_benchmark_parser_custom(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "benchmark",
                "compress:ratio=4",
                "-n",
                "50",
                "--warmup=3",
                "--duration=0.5",
                "--channels=2",
            ]
        )
        assert args.iterations == 50
        assert args.warmup == 3
        assert args.duration == 0.5
        assert args.channels == 2

    def test_benchmark_runs(self, capsys):
        main(
            [
                "benchmark",
                "lowpass:cutoff_hz=1000",
                "-n",
                "3",
                "--warmup=1",
                "--duration=0.1",
            ]
        )
        captured = capsys.readouterr()
        assert "Function: lowpass(cutoff_hz=1000)" in captured.out
        assert "Throughput:" in captured.out
        assert "mean:" in captured.out

    def test_benchmark_json(self, capsys):
        main(
            [
                "benchmark",
                "lowpass:cutoff_hz=1000",
                "-n",
                "3",
                "--warmup=1",
                "--duration=0.1",
                "--json",
            ]
        )
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "time_mean_ms" in data
        assert "throughput_x" in data
        assert data["iterations"] == 3

    def test_benchmark_unknown_function(self):
        with pytest.raises(SystemExit):
            main(["benchmark", "nonexistent_function"])

    # --- read_wav_bytes / write_wav_bytes tests ---

    def test_wav_bytes_roundtrip(self):
        from nanodsp.io import read_wav_bytes, write_wav_bytes

        buf = AudioBuffer.sine(440.0, frames=4800, sample_rate=48000.0)
        data = write_wav_bytes(buf, bit_depth=16)
        assert isinstance(data, bytes)
        assert len(data) > 0

        result = read_wav_bytes(data)
        assert result.channels == 1
        assert result.frames == 4800
        assert result.sample_rate == 48000.0

    def test_wav_bytes_24bit(self):
        from nanodsp.io import read_wav_bytes, write_wav_bytes

        buf = AudioBuffer.sine(440.0, frames=4800, sample_rate=48000.0)
        data = write_wav_bytes(buf, bit_depth=24)
        result = read_wav_bytes(data)
        assert result.frames == 4800

    def test_wav_bytes_stereo(self):
        from nanodsp.io import read_wav_bytes, write_wav_bytes

        buf = AudioBuffer.sine(440.0, channels=2, frames=4800, sample_rate=48000.0)
        data = write_wav_bytes(buf)
        result = read_wav_bytes(data)
        assert result.channels == 2
        assert result.frames == 4800

    def test_wav_bytes_invalid_bit_depth(self):
        from nanodsp.io import write_wav_bytes

        buf = AudioBuffer.sine(440.0, frames=4800, sample_rate=48000.0)
        with pytest.raises(ValueError, match="Unsupported bit_depth"):
            write_wav_bytes(buf, bit_depth=8)

    # --- Pipe parser tests ---

    def test_pipe_parser(self):
        parser = build_parser()
        args = parser.parse_args(["pipe", "-f", "lowpass:cutoff_hz=1000"])
        assert args.command == "pipe"
        assert args.fx == ["lowpass:cutoff_hz=1000"]

    def test_pipe_parser_preset(self):
        parser = build_parser()
        args = parser.parse_args(["pipe", "-p", "telephone"])
        assert args.command == "pipe"
        assert args.preset == ["telephone"]

    def test_pipe_parser_bit_depth(self):
        parser = build_parser()
        args = parser.parse_args(["pipe", "-b", "24"])
        assert args.bit_depth == 24

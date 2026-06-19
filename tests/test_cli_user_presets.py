"""Tests for CLI tightening: user-defined presets, narrowed errors, completers."""

from __future__ import annotations

import json

import pytest

from nanodsp.buffer import AudioBuffer
from nanodsp import _cli
from nanodsp.__main__ import (
    main,
    _preset_name_completer,
    _preset_category_completer,
    _function_completer,
    _category_completer,
)

_USER_PRESETS = {
    "my_boost": {
        "category": "custom",
        "description": "user low shelf",
        "fn": "effects.low_shelf_db",
        "defaults": {"cutoff_hz": 150.0, "db": 4.0},
    },
    "my_telephone": {
        "category": "custom",
        "description": "user telephone",
        "chain": [
            ["effects", "highpass", {"cutoff_hz": 400.0}],
            ["effects", "lowpass", {"cutoff_hz": 3000.0}],
        ],
    },
}


def _write_presets(tmp_path, data):
    path = tmp_path / "presets.json"
    path.write_text(json.dumps(data))
    return path


@pytest.fixture
def user_presets(tmp_path, monkeypatch):
    path = _write_presets(tmp_path, _USER_PRESETS)
    monkeypatch.setenv("NANODSP_PRESETS", str(path))
    return path


class TestLoadUserPresets:
    def test_absent_file_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setenv("NANODSP_PRESETS", str(tmp_path / "nope.json"))
        assert _cli.load_user_presets() == {}

    def test_loads_definitions(self, user_presets):
        loaded = _cli.load_user_presets()
        assert set(loaded) == {"my_boost", "my_telephone"}

    def test_malformed_json_raises(self, tmp_path, monkeypatch):
        path = tmp_path / "presets.json"
        path.write_text("{ not valid json")
        monkeypatch.setenv("NANODSP_PRESETS", str(path))
        with pytest.raises(ValueError, match="Failed to load user presets"):
            _cli.load_user_presets()

    def test_non_object_top_level_raises(self, tmp_path, monkeypatch):
        path = _write_presets(tmp_path, ["not", "an", "object"])
        monkeypatch.setenv("NANODSP_PRESETS", str(path))
        with pytest.raises(ValueError, match="must be a JSON object"):
            _cli.load_user_presets()


class TestMergeAndApply:
    def test_get_presets_merges(self, user_presets):
        presets = _cli.get_presets()
        assert "my_boost" in presets  # user
        assert "master" in presets  # built-in

    def test_user_overrides_builtin(self, tmp_path, monkeypatch):
        path = _write_presets(
            tmp_path,
            {
                "master": {
                    "description": "OVERRIDDEN",
                    "fn": "effects.dc_block",
                    "defaults": {},
                }
            },
        )
        monkeypatch.setenv("NANODSP_PRESETS", str(path))
        assert _cli.get_presets()["master"]["description"] == "OVERRIDDEN"

    def test_categories_include_user(self, user_presets):
        assert "custom" in _cli.get_preset_categories()

    def test_apply_user_fn_preset(self, user_presets):
        buf = AudioBuffer.sine(440.0, frames=4800, sample_rate=48000.0)
        out = _cli.apply_preset("my_boost", buf)
        assert isinstance(out, AudioBuffer)
        assert out.frames == buf.frames

    def test_apply_user_chain_preset(self, user_presets):
        buf = AudioBuffer.sine(440.0, frames=4800, sample_rate=48000.0)
        out = _cli.apply_preset("my_telephone", buf)
        assert isinstance(out, AudioBuffer)
        assert out.frames == buf.frames

    def test_preset_without_fn_or_chain_raises(self, tmp_path, monkeypatch):
        path = _write_presets(tmp_path, {"broken": {"description": "no body"}})
        monkeypatch.setenv("NANODSP_PRESETS", str(path))
        buf = AudioBuffer.sine(440.0, frames=1000, sample_rate=48000.0)
        with pytest.raises(ValueError, match="must define 'fn' or 'chain'"):
            _cli.apply_preset("broken", buf)


class TestCLIIntegration:
    def test_apply_user_preset_via_cli(self, user_presets, tmp_path):
        inp = tmp_path / "in.wav"
        out = tmp_path / "out.wav"
        AudioBuffer.sine(440.0, frames=4800, sample_rate=48000.0).write(str(inp))
        main(["-q", "preset", "apply", "my_boost", str(inp), str(out)])
        assert out.is_file()

    def test_process_with_user_preset(self, user_presets, tmp_path):
        inp = tmp_path / "in.wav"
        out = tmp_path / "out.wav"
        AudioBuffer.sine(440.0, frames=4800, sample_rate=48000.0).write(str(inp))
        main(["-q", "process", str(inp), "-o", str(out), "-p", "my_telephone"])
        assert out.is_file()

    def test_malformed_user_presets_exits_cleanly(self, tmp_path, monkeypatch, capsys):
        path = tmp_path / "presets.json"
        path.write_text("{ broken")
        monkeypatch.setenv("NANODSP_PRESETS", str(path))
        with pytest.raises(SystemExit) as exc:
            main(["preset", "list"])
        assert exc.value.code == 1
        assert "Failed to load user presets" in capsys.readouterr().err


class TestNarrowedErrors:
    def test_missing_input_file_exits(self, tmp_path):
        with pytest.raises(SystemExit) as exc:
            main(["info", str(tmp_path / "does_not_exist.wav")])
        assert exc.value.code == 1

    def test_bad_fx_value_exits(self, tmp_path, capsys):
        inp = tmp_path / "in.wav"
        out = tmp_path / "out.wav"
        AudioBuffer.sine(440.0, frames=2000, sample_rate=48000.0).write(str(inp))
        # cutoff above Nyquist raises ValueError in the filter -> clean exit.
        with pytest.raises(SystemExit) as exc:
            main(
                [
                    "process",
                    str(inp),
                    "-o",
                    str(out),
                    "--fx",
                    "lowpass:cutoff_hz=999999",
                ]
            )
        assert exc.value.code == 1


class TestCompleters:
    def test_preset_name_completer_includes_user(self, user_presets):
        assert "my_boost" in _preset_name_completer("my_")

    def test_preset_name_completer_builtin(self):
        assert "master" in _preset_name_completer("mast")

    def test_preset_category_completer(self):
        assert "dynamics" in _preset_category_completer("dyn")

    def test_function_completer(self):
        assert set(_function_completer("low")) >= {"lowpass", "low_shelf"}

    def test_category_completer(self):
        assert "filters" in _category_completer("fil")

    def test_completer_handles_malformed_presets(self, tmp_path, monkeypatch):
        path = tmp_path / "presets.json"
        path.write_text("{ broken")
        monkeypatch.setenv("NANODSP_PRESETS", str(path))
        # Completion must never crash the shell, even on a bad user file.
        assert _preset_name_completer("x") == []
        assert _preset_category_completer("x") == []

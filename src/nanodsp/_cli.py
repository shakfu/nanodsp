"""Function registry, preset registry, fx token parser, and type coercion for CLI."""

from __future__ import annotations

import inspect
from typing import Any

from nanodsp import effects, ops, spectral, analysis, synthesis


# ---------------------------------------------------------------------------
# Function registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, tuple[Any, str]] = {}

# Categories group function names for the `list` command
CATEGORIES: dict[str, list[str]] = {
    "filters": [],
    "effects": [],
    "dynamics": [],
    "spectral": [],
    "analysis": [],
    "synthesis": [],
    "ops": [],
}

# Map from category keyword -> module_name for grouping
_CATEGORY_MAP: dict[str, str] = {
    "filters": "effects",  # signalsmith + DaisySP filter fns live in effects
    "effects": "effects",
    "dynamics": "effects",
    "spectral": "spectral",
    "analysis": "analysis",
    "synthesis": "synthesis",
    "ops": "ops",
}

# Sets for sub-categorization within the effects module
_FILTER_NAMES = {
    "lowpass",
    "highpass",
    "bandpass",
    "notch",
    "peak",
    "peak_db",
    "high_shelf",
    "high_shelf_db",
    "low_shelf",
    "low_shelf_db",
    "allpass",
    "svf_lowpass",
    "svf_highpass",
    "svf_bandpass",
    "svf_notch",
    "svf_peak",
    "ladder_filter",
    "moog_ladder",
    "tone_lowpass",
    "tone_highpass",
    "modal_bandpass",
    "comb_filter",
}

_DYNAMICS_NAMES = {
    "compress",
    "limit",
    "noise_gate",
    "multiband_compress",
    "parallel_compress",
}


def _register(module: Any, module_name: str, include: set[str] | None = None) -> None:
    """Register public callables from a module."""
    names = include or {
        n
        for n in dir(module)
        if not n.startswith("_") and callable(getattr(module, n, None))
    }
    for name in sorted(names):
        fn = getattr(module, name, None)
        if fn is None or not callable(fn):
            continue
        _REGISTRY[name] = (fn, module_name)
        # Categorize
        if module_name == "effects":
            if name in _FILTER_NAMES:
                CATEGORIES["filters"].append(name)
            elif name in _DYNAMICS_NAMES:
                CATEGORIES["dynamics"].append(name)
            else:
                CATEGORIES["effects"].append(name)
        elif module_name in CATEGORIES:
            CATEGORIES[module_name].append(name)


def _build_registry() -> None:
    """Build the function registry from all modules."""
    if _REGISTRY:
        return
    _register(effects, "effects")
    _register(ops, "ops")
    _register(spectral, "spectral")
    _register(analysis, "analysis")
    _register(synthesis, "synthesis")


def get_registry() -> dict[str, tuple[Any, str]]:
    """Return the function registry, building it on first call."""
    _build_registry()
    return _REGISTRY


def get_categories() -> dict[str, list[str]]:
    """Return the category map, building registry on first call."""
    _build_registry()
    return CATEGORIES


def get_function(name: str) -> tuple[Any, str]:
    """Look up a function by name. Raises KeyError if not found."""
    reg = get_registry()
    if name not in reg:
        raise KeyError(f"Unknown function: {name!r}")
    return reg[name]


def format_signature(fn: Any) -> str:
    """Return a compact signature string for a callable, skipping 'buf' params."""
    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError):
        return "()"
    parts = []
    for pname, param in sig.parameters.items():
        if pname in ("buf", "self", "cls"):
            continue
        if param.default is inspect.Parameter.empty:
            parts.append(pname)
        else:
            parts.append(f"{pname}={param.default!r}")
    return f"({', '.join(parts)})"


# ---------------------------------------------------------------------------
# Preset registry
# ---------------------------------------------------------------------------

PRESETS: dict[str, dict[str, Any]] = {
    # --- Mastering ---
    "master": {
        "category": "mastering",
        "description": "Mastering chain (dc_block -> EQ -> compress -> limit -> normalize)",
        "fn": "effects.master",
        "defaults": {"target_lufs": -14.0},
    },
    # --- Voice ---
    "vocal_chain": {
        "category": "voice",
        "description": "Vocal processing (de-esser -> EQ -> compress -> limit -> normalize)",
        "fn": "effects.vocal_chain",
        "defaults": {},
    },
    # --- Spatial ---
    "room": {
        "category": "spatial",
        "description": "Room reverb (FDN, short decay)",
        "fn": "effects.reverb",
        "defaults": {"preset": "room", "mix": 0.3, "decay": 0.6},
    },
    "hall": {
        "category": "spatial",
        "description": "Hall reverb (FDN, medium decay)",
        "fn": "effects.reverb",
        "defaults": {"preset": "hall", "mix": 0.3, "decay": 0.8},
    },
    "plate": {
        "category": "spatial",
        "description": "Plate reverb (FDN, bright character)",
        "fn": "effects.reverb",
        "defaults": {"preset": "plate", "mix": 0.25, "decay": 0.7},
    },
    "cathedral": {
        "category": "spatial",
        "description": "Cathedral reverb (FDN, long decay)",
        "fn": "effects.reverb",
        "defaults": {"preset": "cathedral", "mix": 0.4, "decay": 0.9},
    },
    "chamber": {
        "category": "spatial",
        "description": "Chamber reverb (FDN, moderate size)",
        "fn": "effects.reverb",
        "defaults": {"preset": "chamber", "mix": 0.3, "decay": 0.75},
    },
    # --- Dynamics ---
    "gentle_compress": {
        "category": "dynamics",
        "description": "Gentle compression (ratio 2:1, -20dB threshold)",
        "fn": "effects.compress",
        "defaults": {"ratio": 2.0, "threshold": -20.0, "attack": 0.01, "release": 0.1},
    },
    "heavy_compress": {
        "category": "dynamics",
        "description": "Heavy compression (ratio 8:1, -30dB threshold)",
        "fn": "effects.compress",
        "defaults": {
            "ratio": 8.0,
            "threshold": -30.0,
            "attack": 0.001,
            "release": 0.05,
        },
    },
    "brick_wall": {
        "category": "dynamics",
        "description": "Brick-wall limiter",
        "fn": "effects.limit",
        "defaults": {"pre_gain": 1.0},
    },
    # --- LoFi ---
    "telephone": {
        "category": "lofi",
        "description": "Telephone effect (bandpass 300-3400 Hz)",
        "chain": [
            ("effects", "highpass", {"cutoff_hz": 300.0}),
            ("effects", "lowpass", {"cutoff_hz": 3400.0}),
        ],
    },
    "lo_fi": {
        "category": "lofi",
        "description": "Lo-fi effect (bitcrush + sample rate reduction)",
        "chain": [
            ("effects", "bitcrush", {"bit_depth": 8}),
            ("effects", "sample_rate_reduce", {"freq": 0.3}),
        ],
    },
    "vinyl": {
        "category": "lofi",
        "description": "Vinyl warmth (low shelf boost + gentle saturation + highpass roll-off)",
        "chain": [
            ("effects", "low_shelf_db", {"cutoff_hz": 300.0, "db": 3.0}),
            ("effects", "saturate", {"drive": 0.2, "mode": "tape"}),
            ("effects", "lowpass", {"cutoff_hz": 14000.0}),
        ],
    },
    # --- Cleanup ---
    "dc_remove": {
        "category": "cleanup",
        "description": "Remove DC offset",
        "fn": "effects.dc_block",
        "defaults": {},
    },
    "de_noise": {
        "category": "cleanup",
        "description": "Highpass at 80 Hz + noise gate",
        "chain": [
            ("effects", "highpass", {"cutoff_hz": 80.0}),
            ("effects", "noise_gate", {"threshold_db": -40.0}),
        ],
    },
    "normalize": {
        "category": "cleanup",
        "description": "Peak normalize to 0 dBFS",
        "fn": "ops.normalize_peak",
        "defaults": {"target_db": 0.0},
    },
    "normalize_lufs": {
        "category": "cleanup",
        "description": "LUFS normalize to -14 LUFS",
        "fn": "analysis.normalize_lufs",
        "defaults": {"target_lufs": -14.0},
    },
    # --- Genre mastering ---
    "master_pop": {
        "category": "mastering",
        "description": "Pop mastering (bright top-end, moderate compression, -14 LUFS)",
        "chain": [
            ("effects", "dc_block", {}),
            ("effects", "highpass", {"cutoff_hz": 30.0}),
            ("effects", "high_shelf_db", {"cutoff_hz": 8000.0, "db": 2.0}),
            ("effects", "compress", {"ratio": 3.0, "threshold": -18.0}),
            ("effects", "limit", {}),
            ("analysis", "normalize_lufs", {"target_lufs": -14.0}),
        ],
    },
    "master_hiphop": {
        "category": "mastering",
        "description": "Hip-hop mastering (boosted lows, bright highs, heavy limiting, -14 LUFS)",
        "chain": [
            ("effects", "dc_block", {}),
            ("effects", "highpass", {"cutoff_hz": 25.0}),
            ("effects", "low_shelf_db", {"cutoff_hz": 100.0, "db": 3.0}),
            ("effects", "high_shelf_db", {"cutoff_hz": 10000.0, "db": 1.5}),
            ("effects", "compress", {"ratio": 4.0, "threshold": -16.0}),
            ("effects", "limit", {"pre_gain": 1.5}),
            ("analysis", "normalize_lufs", {"target_lufs": -14.0}),
        ],
    },
    "master_classical": {
        "category": "mastering",
        "description": "Classical mastering (gentle compression, wide dynamics, -18 LUFS)",
        "chain": [
            ("effects", "dc_block", {}),
            ("effects", "highpass", {"cutoff_hz": 20.0}),
            (
                "effects",
                "compress",
                {"ratio": 1.5, "threshold": -12.0, "attack": 0.05, "release": 0.3},
            ),
            ("analysis", "normalize_lufs", {"target_lufs": -18.0}),
        ],
    },
    "master_edm": {
        "category": "mastering",
        "description": "EDM mastering (sub boost, scooped mids, heavy compression, -11 LUFS)",
        "chain": [
            ("effects", "dc_block", {}),
            ("effects", "highpass", {"cutoff_hz": 30.0}),
            ("effects", "low_shelf_db", {"cutoff_hz": 80.0, "db": 2.0}),
            ("effects", "peak_db", {"center_hz": 3000.0, "db": -2.0}),
            (
                "effects",
                "compress",
                {"ratio": 6.0, "threshold": -20.0, "attack": 0.001, "release": 0.05},
            ),
            ("effects", "limit", {"pre_gain": 2.0}),
            ("analysis", "normalize_lufs", {"target_lufs": -11.0}),
        ],
    },
    "master_podcast": {
        "category": "mastering",
        "description": "Podcast mastering (voice clarity, reduced lows, -16 LUFS)",
        "chain": [
            ("effects", "dc_block", {}),
            ("effects", "highpass", {"cutoff_hz": 80.0}),
            ("effects", "low_shelf_db", {"cutoff_hz": 200.0, "db": -2.0}),
            ("effects", "peak_db", {"center_hz": 3000.0, "db": 2.0}),
            ("effects", "compress", {"ratio": 3.0, "threshold": -20.0}),
            ("effects", "limit", {}),
            ("analysis", "normalize_lufs", {"target_lufs": -16.0}),
        ],
    },
    # --- Creative ---
    "radio": {
        "category": "creative",
        "description": "AM radio effect (bandpass 500-5000 Hz, heavy compression)",
        "chain": [
            ("effects", "highpass", {"cutoff_hz": 500.0}),
            ("effects", "lowpass", {"cutoff_hz": 5000.0}),
            (
                "effects",
                "compress",
                {"ratio": 6.0, "threshold": -20.0, "attack": 0.001, "release": 0.05},
            ),
            ("effects", "limit", {"pre_gain": 1.5}),
        ],
    },
    "underwater": {
        "category": "creative",
        "description": "Underwater effect (heavy lowpass, chorus, wet reverb)",
        "chain": [
            ("effects", "lowpass", {"cutoff_hz": 600.0}),
            ("effects", "chorus", {"lfo_freq": 0.3, "lfo_depth": 0.6}),
            ("effects", "reverb", {"preset": "hall", "mix": 0.5, "decay": 0.9}),
        ],
    },
    "megaphone": {
        "category": "creative",
        "description": "Megaphone effect (bandpass, overdrive, heavy compression)",
        "chain": [
            ("effects", "highpass", {"cutoff_hz": 600.0}),
            ("effects", "lowpass", {"cutoff_hz": 4000.0}),
            ("effects", "overdrive", {"drive": 0.5}),
            (
                "effects",
                "compress",
                {"ratio": 8.0, "threshold": -15.0, "attack": 0.001, "release": 0.05},
            ),
        ],
    },
    "tape_warmth": {
        "category": "creative",
        "description": "Tape warmth (low shelf boost, tape saturation, gentle rolloff)",
        "chain": [
            ("effects", "low_shelf_db", {"cutoff_hz": 200.0, "db": 2.5}),
            ("effects", "saturate", {"drive": 0.3, "mode": "tape"}),
            ("effects", "lowpass", {"cutoff_hz": 12000.0}),
        ],
    },
    "shimmer": {
        "category": "creative",
        "description": "Shimmer reverb (octave-up pitch shift + plate reverb)",
        "chain": [
            ("effects", "pitch_shift", {"semitones": 12.0}),
            ("effects", "reverb", {"preset": "plate", "mix": 0.6, "decay": 0.85}),
        ],
    },
    "vaporwave": {
        "category": "creative",
        "description": "Vaporwave (pitch down, wet reverb, chorus)",
        "chain": [
            ("effects", "pitch_shift", {"semitones": -5.0}),
            ("effects", "reverb", {"preset": "hall", "mix": 0.4, "decay": 0.85}),
            ("effects", "chorus", {"lfo_freq": 0.5, "lfo_depth": 0.4}),
        ],
    },
    "walkie_talkie": {
        "category": "creative",
        "description": "Walkie-talkie (narrow bandpass, bitcrush, extreme compression)",
        "chain": [
            ("effects", "highpass", {"cutoff_hz": 800.0}),
            ("effects", "lowpass", {"cutoff_hz": 3000.0}),
            ("effects", "bitcrush", {"bit_depth": 12}),
            ("effects", "compress", {"ratio": 10.0, "threshold": -15.0}),
        ],
    },
    # --- Additional LoFi ---
    "8bit": {
        "category": "lofi",
        "description": "8-bit retro (4-bit crush + heavy sample rate reduction)",
        "chain": [
            ("effects", "bitcrush", {"bit_depth": 4}),
            ("effects", "sample_rate_reduce", {"freq": 0.15}),
        ],
    },
}


def _resolve_preset_fn(fn_str: str) -> Any:
    """Resolve a 'module.function' string to a callable."""
    module_name, func_name = fn_str.split(".", 1)
    module_map = {
        "effects": effects,
        "ops": ops,
        "spectral": spectral,
        "analysis": analysis,
        "synthesis": synthesis,
    }
    mod = module_map.get(module_name)
    if mod is None:
        raise KeyError(f"Unknown module in preset: {module_name!r}")
    fn = getattr(mod, func_name, None)
    if fn is None:
        raise KeyError(f"Unknown function: {module_name}.{func_name}")
    return fn


def apply_preset(name: str, buf: Any, overrides: dict[str, Any] | None = None) -> Any:
    """Apply a named preset to an AudioBuffer.

    Parameters
    ----------
    name : str
        Preset name (key in PRESETS).
    buf : AudioBuffer
        Input audio.
    overrides : dict or None
        Parameter overrides merged into preset defaults.

    Returns
    -------
    AudioBuffer
    """
    if name not in PRESETS:
        raise KeyError(f"Unknown preset: {name!r}")
    preset = PRESETS[name]
    overrides = overrides or {}

    if "chain" in preset:
        # Chain of (module_name, func_name, params) steps
        result = buf
        for module_name, func_name, params in preset["chain"]:
            fn = _resolve_preset_fn(f"{module_name}.{func_name}")
            merged = {**params, **overrides}
            result = fn(result, **merged)
        return result

    fn = _resolve_preset_fn(preset["fn"])
    params = {**preset.get("defaults", {}), **overrides}
    return fn(buf, **params)


def get_preset_categories() -> dict[str, list[str]]:
    """Return presets grouped by category."""
    cats: dict[str, list[str]] = {}
    for name, info in PRESETS.items():
        cat = info.get("category", "other")
        cats.setdefault(cat, []).append(name)
    return cats


# ---------------------------------------------------------------------------
# FX token parsing
# ---------------------------------------------------------------------------


def parse_fx_token(token: str) -> tuple[str, dict[str, str]]:
    """Parse a 'name:k=v,k=v' token into (name, raw_params).

    Returns raw string values; use coerce_params() to convert types.
    """
    if ":" in token:
        name, params_str = token.split(":", 1)
        params: dict[str, str] = {}
        for pair in params_str.split(","):
            if "=" not in pair:
                raise ValueError(f"Invalid parameter in fx token: {pair!r}")
            k, v = pair.split("=", 1)
            params[k.strip()] = v.strip()
        return name.strip(), params
    return token.strip(), {}


# ---------------------------------------------------------------------------
# Type coercion
# ---------------------------------------------------------------------------


def coerce_value(value: str, target_type: type | None) -> Any:
    """Coerce a string value to the target type.

    If target_type is None, tries float -> int -> str.
    """
    if target_type is bool or target_type is (bool | None):
        return value.lower() in ("true", "1", "yes")
    if target_type is int:
        return int(value)
    if target_type is float:
        return float(value)
    if target_type is str:
        return value
    # No target type: guess
    if target_type is None:
        try:
            f = float(value)
            if f == int(f) and "." not in value:
                return int(value)
            return f
        except ValueError:
            return value
    # For complex types (like enums), return as string
    return value


def coerce_params(fn: Any, raw_params: dict[str, str]) -> dict[str, Any]:
    """Coerce raw string params to the types expected by fn's signature.

    Skips 'buf', 'self', 'cls' parameters. Uses default value types
    to determine target type; falls back to guessing for params without defaults.
    """
    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError):
        # Can't inspect: return raw params, try float coercion
        return {k: coerce_value(v, None) for k, v in raw_params.items()}

    coerced: dict[str, Any] = {}
    for k, v in raw_params.items():
        param = sig.parameters.get(k)
        if param is not None and param.default is not inspect.Parameter.empty:
            default = param.default
            if default is None:
                # Optional param with None default: guess type
                coerced[k] = coerce_value(v, None)
            else:
                coerced[k] = coerce_value(v, type(default))
        else:
            coerced[k] = coerce_value(v, None)
    return coerced

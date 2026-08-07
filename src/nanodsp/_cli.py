"""Function registry, preset registry, fx token parser, and type coercion for CLI."""

from __future__ import annotations

import inspect
import json
import os
import types
from pathlib import Path
from typing import Any

from nanodsp import ops, spectral, analysis, synthesis, timestretch
from nanodsp.effects import filters, daisysp, dynamics, saturation, reverb, composed


# ---------------------------------------------------------------------------
# Function registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, tuple[Any, str]] = {}

# Kind of each registered name -- see _classify(). Only PROCESSOR entries are
# valid operands for `process -f` / `pipe -f` / `benchmark`.
_KINDS: dict[str, str] = {}

PROCESSOR = "processor"  # (buf, ...) -> AudioBuffer     -- chainable
ANALYZER = "analyzer"  # (buf, ...) -> measurement     -- `analyze`
SPECTRAL = "spectral"  # (spec, ...) -> ...            -- Spectrogram domain
GENERATOR = "generator"  # (frames, ...) -> AudioBuffer  -- `synth`
MULTI = "multi"  # needs a second buffer/operand

# Human-readable suffix shown by `nanodsp list` for non-chainable entries.
_KIND_NOTE: dict[str, str] = {
    ANALYZER: "returns a measurement, not audio",
    SPECTRAL: "operates on a Spectrogram",
    GENERATOR: "generates audio; use `nanodsp synth`",
    MULTI: "leading operand is not audio; Python API only",
}

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


def _classify(fn: Any) -> str | None:
    """Classify a callable for CLI use, or return None if it is not a DSP entry point.

    The registry is built by scanning module namespaces, which also picks up
    imported classes and ``typing`` constructs (``Callable``, ``Literal``,
    ``AudioBuffer``).  Admitting those made them appear in ``nanodsp list`` and
    accepted by ``-f``, where they produced either a confusing traceback or --
    worse -- a silently corrupt output file.  Only plain functions are admitted,
    and each is classified by its signature so that ``-f`` can reject
    non-chainable entries up front with a message naming the reason.

    Classification is by first parameter name and return annotation, which is
    reliable here because the package is fully annotated and follows a strict
    ``buf``/``spec`` naming convention for its leading parameter.
    """
    if not inspect.isfunction(fn):
        return None
    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError):
        return None
    params = list(sig.parameters.values())
    if not params:
        return None

    def _ann(obj) -> str:
        # `from __future__ import annotations` is in force package-wide, so
        # annotations arrive as strings; fall back to the object for safety.
        return obj if isinstance(obj, str) else getattr(obj, "__name__", str(obj))

    returns_audio = "AudioBuffer" in _ann(sig.return_annotation)
    first = params[0]
    takes_audio = "AudioBuffer" in _ann(first.annotation)

    # Classify on the leading parameter's *annotation* rather than its name.
    # Most processors call it `buf`, but not all -- `vocoder(modulator,
    # carrier)` and `crossfade(buf_a, buf_b)` lead with an AudioBuffer under a
    # different name, and are perfectly chainable now that `-f` can load a
    # second buffer from a file operand.
    if takes_audio:
        return PROCESSOR if returns_audio else ANALYZER
    if first.name == "spec":
        return SPECTRAL
    if first.name == "frames":
        return GENERATOR
    # Leading operand is not audio at all (iir_design, lfo, irfft).
    return MULTI


def _register(module: Any, module_name: str, include: set[str] | None = None) -> None:
    """Register public DSP functions from a module.

    Without *include*, only functions actually defined in *module* are taken, so
    that names re-exported via imports (e.g. ``stft`` imported into
    ``analysis``) are registered once, under their defining module.
    """
    explicit = include is not None
    names = include or {n for n in dir(module) if not n.startswith("_")}
    for name in sorted(names):
        fn = getattr(module, name, None)
        if fn is None:
            continue
        kind = _classify(fn)
        if kind is None:
            continue
        if not explicit and getattr(fn, "__module__", "") != module.__name__:
            continue
        _REGISTRY[name] = (fn, module_name)
        _KINDS[name] = kind
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


_EFFECTS_MODULES = [filters, daisysp, dynamics, saturation, reverb, composed]


def _build_registry() -> None:
    """Build the function registry from all modules."""
    if _REGISTRY:
        return
    for mod in _EFFECTS_MODULES:
        _register(mod, "effects")
    _register(ops, "ops")
    _register(spectral, "spectral")
    _register(timestretch, "spectral", include={"paulstretch", "signalsmith_stretch"})
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


def get_kinds() -> dict[str, str]:
    """Return the name -> kind map, building the registry on first call."""
    _build_registry()
    return _KINDS


def get_function(name: str) -> tuple[Any, str]:
    """Look up a function by name. Raises KeyError if not found."""
    reg = get_registry()
    if name not in reg:
        raise KeyError(f"Unknown function: {name!r}")
    return reg[name]


def get_processor(name: str) -> tuple[Any, str]:
    """Look up a chainable ``(buf, ...) -> AudioBuffer`` function by name.

    Raises
    ------
    KeyError
        If *name* is not registered at all.
    ValueError
        If *name* is registered but is not chainable -- a generator, analyzer,
        Spectrogram operator, or multi-operand function.  Chaining these was
        previously accepted and produced either a traceback or a silently
        corrupt output file.
    """
    fn, module_name = get_function(name)
    kind = get_kinds().get(name, PROCESSOR)
    if kind != PROCESSOR:
        raise ValueError(f"{name!r} is not a chainable effect ({_KIND_NOTE[kind]})")
    return fn, module_name


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


def _user_presets_path() -> Path:
    """Path to the user preset file.

    Uses ``$NANODSP_PRESETS`` if set, else ``~/.nanodsp/presets.json``.
    """
    env = os.environ.get("NANODSP_PRESETS")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".nanodsp" / "presets.json"


def load_user_presets() -> dict[str, dict[str, Any]]:
    """Load user-defined presets from JSON, or {} if no file is present.

    The JSON top level must be an object mapping preset names to preset
    definitions in the same shape as the built-in :data:`PRESETS` entries --
    either ``{"fn": "module.func", "defaults": {...}}`` or
    ``{"chain": [["module", "func", {...}], ...]}`` (plus optional
    ``description`` and ``category``).  Chain steps are lists rather than tuples
    because JSON has no tuple type; :func:`apply_preset` accepts both.

    Raises
    ------
    ValueError
        If the file is malformed JSON or its top level is not an object.
    """
    path = _user_presets_path()
    if not path.is_file():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"Failed to load user presets from {path}: {e}") from e
    if not isinstance(data, dict):
        raise ValueError(
            f"User presets file {path} must be a JSON object mapping names to presets"
        )
    return data


def get_presets() -> dict[str, dict[str, Any]]:
    """Return built-in presets merged with user presets.

    User presets override built-ins on a name collision.
    """
    merged: dict[str, dict[str, Any]] = dict(PRESETS)
    merged.update(load_user_presets())
    return merged


def _resolve_preset_fn(fn_str: str) -> Any:
    """Resolve a 'module.function' string to a callable."""
    module_name, func_name = fn_str.split(".", 1)
    if module_name == "effects":
        for mod in _EFFECTS_MODULES:
            fn = getattr(mod, func_name, None)
            if fn is not None:
                return fn
        raise KeyError(f"Unknown function: {fn_str}")
    module_map: dict[str, types.ModuleType] = {
        "ops": ops,
        "spectral": spectral,
        "analysis": analysis,
        "synthesis": synthesis,
    }
    if module_name not in module_map:
        raise KeyError(f"Unknown module in preset: {module_name!r}")
    mod = module_map[module_name]
    fn = getattr(mod, func_name, None)
    if fn is None:
        raise KeyError(f"Unknown function: {fn_str}")
    return fn


def _split_overrides(
    overrides: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Split overrides into unscoped and per-step-scoped groups.

    ``{"ratio": 6, "highpass.cutoff_hz": 40}`` becomes
    ``({"ratio": 6}, {"highpass": {"cutoff_hz": 40}})``.
    """
    plain: dict[str, Any] = {}
    scoped: dict[str, dict[str, Any]] = {}
    for key, value in overrides.items():
        step, sep, param = key.partition(".")
        if sep:
            scoped.setdefault(step, {})[param] = value
        else:
            plain[key] = value
    return plain, scoped


def _accepted_params(fn: Any, params: dict[str, Any]) -> dict[str, Any]:
    """Return the subset of *params* that *fn* actually accepts.

    A chain step only sees an unscoped override if its own signature declares
    that parameter.  Without this, any override applied to a chain preset was
    forwarded to every step and the first one that did not accept it raised
    ``TypeError`` -- which made overrides unusable on the 17 built-in presets
    that are chains rather than single functions.
    """
    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError):
        return dict(params)
    if any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values()):
        return dict(params)
    return {k: v for k, v in params.items() if k in sig.parameters}


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

        For a single-function preset every override is passed straight through.
        For a chain preset an unscoped key such as ``{"ratio": 6.0}`` is applied
        to each step whose signature accepts it, and is silently ignored by the
        rest.  Because several steps in a chain can share a parameter name --
        ``master_hiphop`` has a ``highpass`` and two shelving filters that all
        take ``cutoff_hz`` -- an unscoped override may hit more steps than
        intended.  Prefix the key with a step name to target one function:
        ``{"highpass.cutoff_hz": 40.0}``.

    Returns
    -------
    AudioBuffer

    Raises
    ------
    KeyError
        If *name* is not a known preset, or a scoped override names a step that
        is not in the chain.
    """
    presets = get_presets()
    if name not in presets:
        raise KeyError(f"Unknown preset: {name!r}")
    preset = presets[name]
    plain, scoped = _split_overrides(overrides or {})

    if "chain" in preset:
        # Chain of (module_name, func_name, params) steps (tuples or JSON lists)
        step_names = [str(step[1]) for step in preset["chain"]]
        unknown = sorted(set(scoped) - set(step_names))
        if unknown:
            raise KeyError(
                f"Preset {name!r} has no step(s) {unknown}; "
                f"available steps: {step_names}"
            )
        result = buf
        for module_name, func_name, params in preset["chain"]:
            fn = _resolve_preset_fn(f"{module_name}.{func_name}")
            merged = {
                **params,
                **_accepted_params(fn, plain),
                **scoped.get(func_name, {}),
            }
            result = fn(result, **merged)
        return result

    if "fn" in preset:
        fn = _resolve_preset_fn(preset["fn"])
        step_name = preset["fn"].split(".", 1)[-1]
        unknown = sorted(set(scoped) - {step_name})
        if unknown:
            raise KeyError(
                f"Preset {name!r} has no step(s) {unknown}; "
                f"the only step is {step_name!r}"
            )
        params = {
            **preset.get("defaults", {}),
            **plain,
            **scoped.get(step_name, {}),
        }
        return fn(buf, **params)

    raise ValueError(f"Preset {name!r} must define 'fn' or 'chain'")


def get_preset_categories() -> dict[str, list[str]]:
    """Return presets (built-in + user) grouped by category."""
    cats: dict[str, list[str]] = {}
    for name, info in get_presets().items():
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

    If target_type is None, tries bool -> int -> float -> str.
    """
    if target_type is bool:
        return value.lower() in ("true", "1", "yes", "on")
    if target_type is int:
        return int(value)
    if target_type is float:
        return float(value)
    if target_type is str:
        return value
    # No target type: guess. Only unambiguous boolean spellings are recognised
    # here -- "1"/"on"/"yes" stay as int/str, since a parameter with no default
    # is as likely to be a mode name as a flag.
    if target_type is None:
        lowered = value.lower()
        if lowered in ("true", "false"):
            return lowered == "true"
        # Narrow to int only for plain integer literals. Testing `f == int(f)`
        # instead accepted "1e3" and "inf", where int() then raised ValueError
        # (silently yielding the raw string) or OverflowError (uncaught).
        stripped = value.strip()
        if stripped.lstrip("+-").isdigit():
            return int(stripped)
        try:
            return float(value)
        except ValueError:
            return value
    # For complex types (like enums), return as string
    return value


def missing_required_params(
    fn: Any, supplied: dict[str, Any]
) -> tuple[list[str], list[str]]:
    """Return required parameters of *fn* that *supplied* does not cover.

    The leading ``buf`` operand comes from the file being processed, so it is
    always excluded.  Used to reject an ``-f`` token before it reaches the DSP
    layer, where the failure would otherwise be a bare ``TypeError``.

    Returns
    -------
    (missing, buffer_operands)
        *missing* is every unsatisfied required parameter; *buffer_operands* is
        the subset annotated ``AudioBuffer``, which the ``-f`` grammar has no
        way to express at all (``sidechain_compress``, ``vocoder``, ``convolve``
        and friends are Python-API-only until it grows a file-operand syntax).
    """
    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError):
        return [], []
    missing: list[str] = []
    buffers: list[str] = []
    for i, (pname, param) in enumerate(sig.parameters.items()):
        if i == 0 or pname in ("self", "cls"):
            continue
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        if param.default is inspect.Parameter.empty and pname not in supplied:
            missing.append(pname)
            ann = param.annotation
            ann_str = ann if isinstance(ann, str) else getattr(ann, "__name__", "")
            if "AudioBuffer" in ann_str:
                buffers.append(pname)
    return missing, buffers


#: Prefix marking a parameter value as a path to load, rather than a literal.
#: ``-f sidechain_compress:sidechain=@kick.wav`` reads kick.wav and passes the
#: resulting AudioBuffer. Without this the several effects taking a second
#: buffer -- sidechain compression, vocoding, convolution, EQ matching -- were
#: reachable only from the Python API.
FILE_OPERAND_PREFIX = "@"


def _load_operand(value: str) -> Any:
    """Load an ``@path`` operand into an AudioBuffer."""
    from pathlib import Path

    from nanodsp.io import read

    path = Path(value[len(FILE_OPERAND_PREFIX) :]).expanduser()
    if not path.is_file():
        raise ValueError(f"file operand not found: {path}")
    return read(path)


def coerce_params(fn: Any, raw_params: dict[str, str]) -> dict[str, Any]:
    """Coerce raw string params to the types expected by fn's signature.

    Skips 'buf', 'self', 'cls' parameters. Uses default value types
    to determine target type; falls back to guessing for params without defaults.

    A value beginning with :data:`FILE_OPERAND_PREFIX` is read as an audio file
    and passed as an ``AudioBuffer``, whatever the annotation says.
    """
    coerced: dict[str, Any] = {}
    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError):
        sig = None

    for k, v in raw_params.items():
        if v.startswith(FILE_OPERAND_PREFIX):
            coerced[k] = _load_operand(v)
            continue
        param = sig.parameters.get(k) if sig is not None else None
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

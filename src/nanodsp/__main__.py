"""nanodsp CLI -- process, analyze, synthesize, and convert audio files."""
# PYTHON_ARGCOMPLETE_OK

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import wave
from pathlib import Path
from typing import Any

import numpy as np

from nanodsp import __version__
from nanodsp.buffer import AudioBuffer


# Expected, user-facing error categories. Catching these (rather than a bare
# ``Exception``) reports a clean message and exits, while letting genuinely
# unexpected errors -- bugs such as AttributeError/ImportError/NameError --
# propagate with a full traceback instead of being masked.
_IO_ERRORS = (OSError, ValueError, EOFError, RuntimeError, wave.Error)
_DSP_ERRORS = (
    ValueError,
    TypeError,
    KeyError,
    IndexError,
    RuntimeError,
    ZeroDivisionError,
)


# ---------------------------------------------------------------------------
# Verbosity levels
# ---------------------------------------------------------------------------

QUIET = 0
NORMAL = 1
VERBOSE = 2


def _verbosity(args: argparse.Namespace) -> int:
    """Return verbosity level from parsed args."""
    if getattr(args, "quiet", False):
        return QUIET
    if getattr(args, "verbose", False):
        return VERBOSE
    return NORMAL


def _log(args: argparse.Namespace, msg: str, level: int = NORMAL) -> None:
    """Print *msg* if verbosity >= *level*."""
    if _verbosity(args) >= level:
        print(msg)


def _log_verbose(args: argparse.Namespace, msg: str) -> None:
    """Print only when --verbose."""
    _log(args, msg, level=VERBOSE)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_input(path: str, args: argparse.Namespace | None = None) -> AudioBuffer:
    """Read an audio file, exit on error."""
    from nanodsp.io import read

    if args:
        _log_verbose(args, f"  Reading {path}")
    try:
        buf = read(path)
    except _IO_ERRORS as e:
        print(f"Error reading {path}: {e}", file=sys.stderr)
        sys.exit(1)
    if args:
        _log_verbose(
            args,
            f"  Loaded: {buf.channels}ch, {buf.frames} frames, {buf.sample_rate:.0f} Hz",
        )
    return buf


def _write_output(
    path: str,
    buf: AudioBuffer,
    bit_depth: int = 16,
    args: argparse.Namespace | None = None,
) -> None:
    """Write an audio file, exit on error."""
    from nanodsp.io import write

    if args:
        _log_verbose(args, f"  Writing {path} ({bit_depth}-bit)")
    try:
        write(path, buf, bit_depth=bit_depth)
    except _IO_ERRORS as e:
        print(f"Error writing {path}: {e}", file=sys.stderr)
        sys.exit(1)


def _load_presets() -> dict:
    """Return built-in + user presets, exiting cleanly on a malformed user file."""
    from nanodsp._cli import get_presets

    try:
        return get_presets()
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Subcommand: info
# ---------------------------------------------------------------------------


def cmd_info(args: argparse.Namespace) -> None:
    """Print audio file metadata."""
    from nanodsp.analysis import loudness_lufs

    buf = _read_input(args.file, args)
    peak = float(np.max(np.abs(buf.data)))
    peak_db = 20.0 * np.log10(peak) if peak > 0 else float("-inf")
    lufs = loudness_lufs(buf)

    ext = Path(args.file).suffix.lower()
    info = {
        "path": str(args.file),
        "format": ext.lstrip(".").upper(),
        "duration": f"{buf.duration:.3f}s",
        "sample_rate": int(buf.sample_rate),
        "channels": buf.channels,
        "frames": buf.frames,
        "peak_db": f"{peak_db:.1f}",
        "loudness_lufs": f"{lufs:.1f}" if not np.isinf(lufs) else "-inf",
    }

    if args.json:
        print(json.dumps(info, indent=2))
    else:
        for k, v in info.items():
            print(f"  {k}: {v}")


# ---------------------------------------------------------------------------
# Subcommand: process
# ---------------------------------------------------------------------------


class _OrderedChainAction(argparse.Action):
    """Append ``(kind, value)`` to a single shared destination.

    ``-f`` and ``-p`` used to accumulate into two separate lists, and argparse
    does not record how the two interleaved on the command line.  The chain was
    therefore always built as "every effect, then every preset", silently
    reordering what the user typed -- and for a chain of DSP operations order is
    semantics, not presentation.  Routing both options through one destination
    preserves argv order exactly.
    """

    def __init__(self, option_strings, dest, kind: str = "fx", **kwargs):
        self._kind = kind
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        chain = getattr(namespace, self.dest, None)
        if chain is None:
            chain = []
            setattr(namespace, self.dest, chain)
        chain.append((self._kind, values))


def _add_chain_options(parser: argparse.ArgumentParser) -> argparse.Action:
    """Add the ``-f/--fx`` and ``-p/--preset`` pair, sharing one ordered dest."""
    fx_action = parser.add_argument(
        "-f",
        "--fx",
        dest="chain",
        action=_OrderedChainAction,
        kind="fx",
        metavar="NAME:K=V,...",
        help="Effect to apply (repeatable). Format: name:param=val,param=val",
    )
    _set_completer(fx_action, _function_completer)
    _set_completer(
        parser.add_argument(
            "-p",
            "--preset",
            dest="chain",
            action=_OrderedChainAction,
            kind="preset",
            metavar="NAME",
            help="Apply a named preset (repeatable)",
        ),
        _preset_name_completer,
    )
    return fx_action


def _build_chain(args: argparse.Namespace) -> list[dict]:
    """Build a list of chain steps from the ordered --fx/--preset list.

    Each step is a dict with keys:
      - "type": "fx" or "preset"
      - "name": function/preset name
      - "params": dict of coerced params (fx) or empty dict (preset)
      - "raw": original token string (fx only)

    Steps appear in the order the options were given on the command line.
    """
    from nanodsp._cli import (
        get_processor,
        parse_fx_token,
        coerce_params,
        missing_required_params,
    )

    chain = getattr(args, "chain", None) or []
    steps: list[dict] = []
    presets = _load_presets() if any(k == "preset" for k, _ in chain) else {}

    for kind, value in chain:
        if kind == "fx":
            name, raw_params = parse_fx_token(value)
            try:
                fn, mod = get_processor(name)
            except KeyError:
                print(f"Unknown function: {name!r}", file=sys.stderr)
                sys.exit(1)
            except ValueError as e:
                print(f"Cannot apply {name!r}: {e}", file=sys.stderr)
                sys.exit(1)
            try:
                params = coerce_params(fn, raw_params)
            except (ValueError, OverflowError) as e:
                print(f"Bad parameter for {name!r}: {e}", file=sys.stderr)
                sys.exit(1)
            missing, buffers = missing_required_params(fn, params)
            if buffers:
                first = buffers[0]
                print(
                    f"{name!r} needs a second audio buffer: "
                    f"{', '.join(buffers)}. Supply it with a file operand, "
                    f"e.g. {name}:{first}=@other.wav",
                    file=sys.stderr,
                )
                sys.exit(1)
            if missing:
                print(
                    f"{name!r} requires parameter(s) {', '.join(missing)}; "
                    f"pass them as {name}:{missing[0]}=...",
                    file=sys.stderr,
                )
                sys.exit(1)
            steps.append(
                {
                    "type": "fx",
                    "name": name,
                    "module": mod,
                    "fn": fn,
                    "params": params,
                    "raw": value,
                }
            )
        else:
            if value not in presets:
                print(f"Unknown preset: {value!r}", file=sys.stderr)
                sys.exit(1)
            steps.append(
                {
                    "type": "preset",
                    "name": value,
                    "description": presets[value].get("description", ""),
                }
            )

    return steps


def _format_chain(steps: list[dict]) -> str:
    """Format a chain of steps as a human-readable string."""
    parts = []
    for i, step in enumerate(steps, 1):
        if step["type"] == "fx":
            params_str = ", ".join(f"{k}={v}" for k, v in step["params"].items())
            if params_str:
                parts.append(f"  {i}. {step['name']}({params_str})")
            else:
                parts.append(f"  {i}. {step['name']}()")
        else:
            parts.append(f"  {i}. [preset] {step['name']} -- {step['description']}")
    return "\n".join(parts)


def _apply_chain(
    buf: AudioBuffer,
    steps: list[dict],
    args: argparse.Namespace,
) -> AudioBuffer:
    """Apply a chain of steps to a buffer."""
    from nanodsp._cli import apply_preset

    for step in steps:
        if step["type"] == "fx":
            _log_verbose(args, f"  Applying {step['name']}({step['params']})")
            try:
                buf = step["fn"](buf, **step["params"])
            except _DSP_ERRORS as e:
                print(f"Error applying {step['name']}: {e}", file=sys.stderr)
                sys.exit(1)
        else:
            _log_verbose(args, f"  Applying preset {step['name']!r}")
            try:
                buf = apply_preset(step["name"], buf)
            except _DSP_ERRORS as e:
                print(
                    f"Error applying preset {step['name']!r}: {e}",
                    file=sys.stderr,
                )
                sys.exit(1)
    return buf


def _resolve_output_path(
    input_path: str,
    output: str | None,
    output_dir: str | None,
) -> str:
    """Determine the output path for a given input file.

    If --output-dir is set, the output filename is derived from the input
    filename placed inside the output directory. If output is set (single
    file mode), use it directly.
    """
    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        return str(out_dir / Path(input_path).name)
    if output:
        return output
    # Should not happen -- parser validation catches this
    print("No output path specified", file=sys.stderr)
    sys.exit(1)


def _peak_db(buf: AudioBuffer) -> float:
    """Sample peak in dBFS, or -inf for digital silence."""
    peak = float(np.max(np.abs(buf.data))) if buf.frames else 0.0
    return 20.0 * np.log10(peak) if peak > 0 else float("-inf")


def _format_levels(buf: AudioBuffer, label: str) -> str:
    """One-line level summary. Includes true peak and loudness, which are not free."""
    from nanodsp.analysis import loudness_lufs, true_peak_dbtp

    lufs = loudness_lufs(buf)
    lufs_str = f"{lufs:6.1f} LUFS" if not np.isinf(lufs) else "  -inf LUFS"
    return (
        f"  {label:7} peak {_peak_db(buf):6.1f} dBFS   "
        f"true peak {true_peak_dbtp(buf):6.1f} dBTP   {lufs_str}"
    )


def _warn_if_clipped(
    path: str, buf: AudioBuffer, bit_depth: int, args: argparse.Namespace
) -> None:
    """Warn when the result will clip on write.

    Only checked for the integer formats: float output stores samples verbatim,
    so a peak above full scale survives intact and is not a loss.

    Only the sample peak is checked, not the true peak: this runs for every
    output file including in batch mode, and a full 4x-oversampled true-peak
    measurement is not worth paying for unconditionally. Integer WAV/FLAC output
    clips at +/-1.0, so this is the failure that actually costs the user audio.
    """
    from nanodsp.io import _FLOAT_WRITE_DEPTHS

    if _verbosity(args) == QUIET or not buf.frames:
        return
    if bit_depth in _FLOAT_WRITE_DEPTHS:
        return
    if float(np.max(np.abs(buf.data))) > 1.0:
        print(
            f"Warning: {path} clips (peak {_peak_db(buf):.1f} dBFS); "
            f"{bit_depth}-bit output is limited to 0 dBFS. Add a limiter, "
            "lower the gain, or write float with -b 32.",
            file=sys.stderr,
        )


def _process_file_streaming(
    input_path: str,
    output_path: str,
    steps: list[dict],
    bit_depth: int,
    args: argparse.Namespace,
) -> None:
    """Process one file a block at a time, in constant memory.

    Every step is rebuilt as a stateful processor that carries its state across
    blocks, so the result is identical to whole-file processing rather than
    restarting each effect at every block boundary. Peak memory is one block
    instead of the whole file.
    """
    from nanodsp.io import BlockWriter, read_blocks
    from nanodsp.stream import STREAMING_CAVEATS, build_streaming_chain

    blocks = read_blocks(input_path, block_size=args.block_size)
    try:
        first = next(blocks)
    except StopIteration:
        # Empty file: still produce a valid, empty output.
        BlockWriter(output_path, 48000.0, 1, bit_depth).close()
        return

    procs = build_streaming_chain(steps, first.channels, first.sample_rate)
    for name in dict.fromkeys(s["name"] for s in steps):
        if name in STREAMING_CAVEATS:
            _log(args, f"  note: --stream {name} {STREAMING_CAVEATS[name]}")

    _log_verbose(
        args,
        f"  Streaming {input_path}: {first.channels}ch, "
        f"{first.sample_rate:.0f} Hz, {args.block_size}-frame blocks",
    )
    clipped = False
    with BlockWriter(
        output_path, first.sample_rate, first.channels, bit_depth
    ) as writer:
        for block in itertools.chain([first], blocks):
            for proc in procs:
                block = proc.process(block)
            if not clipped and float(np.max(np.abs(block.data), initial=0.0)) > 1.0:
                clipped = True
            writer.write(block)
    if clipped:
        _warn_clipped_stream(output_path, bit_depth, args)


def _warn_clipped_stream(path: str, bit_depth: int, args: argparse.Namespace) -> None:
    """Clipping warning for the streaming path, where there is no final buffer."""
    from nanodsp.io import _FLOAT_WRITE_DEPTHS

    if _verbosity(args) == QUIET or bit_depth in _FLOAT_WRITE_DEPTHS:
        return
    print(
        f"Warning: {path} clips; {bit_depth}-bit output is limited to 0 dBFS. "
        "Add a limiter, lower the gain, or write float with -b 32.",
        file=sys.stderr,
    )


def _process_file(
    input_path: str,
    output_path: str,
    steps: list[dict],
    bit_depth: int,
    args: argparse.Namespace,
) -> None:
    """Read, process and write one file.

    Raises rather than exiting, so this is safe to call from a worker thread
    where ``sys.exit`` would only unwind that thread and be reported as a
    generic error.
    """
    from nanodsp.io import read, write

    _log_verbose(args, f"  Reading {input_path}")
    buf = read(input_path)
    _log_verbose(
        args,
        f"  Loaded: {buf.channels}ch, {buf.frames} frames, {buf.sample_rate:.0f} Hz",
    )
    if getattr(args, "stats", False) or _verbosity(args) >= VERBOSE:
        _log(args, _format_levels(buf, "input"), level=QUIET)

    for step in steps:
        if step["type"] == "fx":
            _log_verbose(args, f"  Applying {step['name']}({step['params']})")
            buf = step["fn"](buf, **step["params"])
        else:
            _log_verbose(args, f"  Applying preset {step['name']!r}")
            from nanodsp._cli import apply_preset

            buf = apply_preset(step["name"], buf)

    if getattr(args, "stats", False) or _verbosity(args) >= VERBOSE:
        _log(args, _format_levels(buf, "output"), level=QUIET)

    _warn_if_clipped(output_path, buf, bit_depth, args)
    _log_verbose(args, f"  Writing {output_path} ({bit_depth}-bit)")
    write(output_path, buf, bit_depth=bit_depth)


def cmd_process(args: argparse.Namespace) -> None:
    """Apply an effect chain to audio file(s)."""
    steps = _build_chain(args)

    # Dry run: show chain and exit
    if getattr(args, "dry_run", False):
        inputs = args.input
        if not steps:
            print("Chain: (empty -- no effects or presets specified)")
        else:
            print(f"Chain ({len(steps)} steps):")
            print(_format_chain(steps))
        print()
        n = len(inputs)
        label = "file" if n == 1 else "files"
        print(f"Input: {n} {label}")
        for p in inputs:
            print(f"  {p}")
        if args.output_dir:
            print(f"Output directory: {args.output_dir}")
        elif args.output:
            print(f"Output: {args.output}")
        print(f"Bit depth: {args.bit_depth or 16}")
        if (args.jobs or 1) != 1:
            print(f"Jobs: {_resolve_jobs(args.jobs)}")
        return

    bit_depth = args.bit_depth or 16
    inputs = args.input
    jobs = min(_resolve_jobs(args.jobs), len(inputs))

    targets = [
        (p, _resolve_output_path(p, args.output, args.output_dir)) for p in inputs
    ]

    runner = (
        _process_file_streaming if getattr(args, "stream", False) else _process_file
    )

    if jobs <= 1:
        failures = 0
        for input_path, output_path in targets:
            try:
                runner(input_path, output_path, steps, bit_depth, args)
            except _IO_ERRORS + _DSP_ERRORS as e:
                print(f"Error processing {input_path}: {e}", file=sys.stderr)
                failures += 1
                continue
            _log(args, f"Wrote {output_path}")
        if failures:
            sys.exit(1)
        return

    _run_parallel(targets, steps, bit_depth, jobs, args, runner)


def _resolve_jobs(jobs: int | None) -> int:
    """Resolve --jobs, where 0 means "one worker per CPU"."""
    if jobs is None or jobs < 0:
        return 1
    if jobs == 0:
        return os.cpu_count() or 1
    return jobs


def _run_parallel(
    targets: list[tuple[str, str]],
    steps: list[dict],
    bit_depth: int,
    jobs: int,
    args: argparse.Namespace,
    runner: Any,
) -> None:
    """Process files concurrently.

    Threads rather than processes: every C++ processing entry point releases the
    GIL, so the DSP work genuinely runs in parallel, and threads avoid pickling
    buffers across a process boundary. Each file is independent -- the chain
    steps hold only configuration, and every effect builds its own DSP objects
    per call -- so no state is shared between workers.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    _log_verbose(args, f"  Processing {len(targets)} files across {jobs} workers")
    failures = 0
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {
            pool.submit(runner, input_path, output_path, steps, bit_depth, args): (
                input_path,
                output_path,
            )
            for input_path, output_path in targets
        }
        for future in as_completed(futures):
            input_path, output_path = futures[future]
            try:
                future.result()
            except _IO_ERRORS + _DSP_ERRORS as e:
                print(f"Error processing {input_path}: {e}", file=sys.stderr)
                failures += 1
                continue
            _log(args, f"Wrote {output_path}")

    if failures:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Subcommand: analyze
# ---------------------------------------------------------------------------


_ANALYSIS_SUBCOMMANDS = {
    "loudness",
    "pitch",
    "onsets",
    "centroid",
    "bandwidth",
    "rolloff",
    "flux",
    "flatness",
    "chromagram",
    "info",
}


def cmd_analyze(args: argparse.Namespace) -> None:
    """Run an analysis function and print results."""

    buf = _read_input(args.input, args)
    subcmd = args.analysis_type

    if subcmd == "info":
        _analyze_info(buf, args)
        return

    result = _run_analysis(subcmd, buf, args)
    if result is None:
        return

    if args.json:
        if isinstance(result, np.ndarray):
            print(json.dumps(result.tolist()))
        elif isinstance(result, tuple):
            print(
                json.dumps(
                    [r.tolist() if isinstance(r, np.ndarray) else r for r in result]
                )
            )
        else:
            print(json.dumps(result))
    else:
        if isinstance(result, tuple):
            for i, r in enumerate(result):
                if isinstance(r, np.ndarray):
                    print(
                        f"  [{i}] shape={r.shape} min={r.min():.4f} max={r.max():.4f} mean={r.mean():.4f}"
                    )
                else:
                    print(f"  [{i}] {r}")
        elif isinstance(result, np.ndarray):
            print(f"  shape: {result.shape}")
            print(f"  min: {result.min():.4f}")
            print(f"  max: {result.max():.4f}")
            print(f"  mean: {result.mean():.4f}")
        else:
            print(f"  {result}")


def _run_analysis(subcmd: str, buf: AudioBuffer, args: argparse.Namespace):
    """Dispatch to the appropriate analysis function."""
    from nanodsp import analysis as _analysis

    if subcmd == "loudness":
        lufs = _analysis.loudness_lufs(buf)
        if not args.json:
            print(
                f"  loudness: {lufs:.1f} LUFS"
                if not np.isinf(lufs)
                else "  loudness: -inf LUFS"
            )
            return None
        return lufs

    if subcmd == "pitch":
        fmin = float(args.fmin) if args.fmin else 50.0
        fmax = float(args.fmax) if args.fmax else 2000.0
        freqs, confs = _analysis.pitch_detect(buf, fmin=fmin, fmax=fmax)
        return (freqs, confs)

    if subcmd == "onsets":
        method = args.method or "spectral_flux"
        onsets = _analysis.onset_detect(buf, method=method)
        return onsets

    if subcmd == "centroid":
        return _analysis.spectral_centroid(buf)

    if subcmd == "bandwidth":
        return _analysis.spectral_bandwidth(buf)

    if subcmd == "rolloff":
        return _analysis.spectral_rolloff(buf)

    if subcmd == "flux":
        return _analysis.spectral_flux(buf)

    if subcmd == "flatness":
        return _analysis.spectral_flatness_curve(buf)

    if subcmd == "chromagram":
        return _analysis.chromagram(buf)

    print(f"Unknown analysis type: {subcmd!r}", file=sys.stderr)
    sys.exit(1)


def _analyze_info(buf: AudioBuffer, args: argparse.Namespace) -> None:
    """Combined analysis summary."""
    from nanodsp import analysis as _analysis

    peak = float(np.max(np.abs(buf.data)))
    peak_db = 20.0 * np.log10(peak) if peak > 0 else float("-inf")
    lufs = _analysis.loudness_lufs(buf)

    info = {
        "duration": f"{buf.duration:.3f}s",
        "sample_rate": int(buf.sample_rate),
        "channels": buf.channels,
        "frames": buf.frames,
        "peak_db": round(peak_db, 1),
        "loudness_lufs": round(lufs, 1) if not np.isinf(lufs) else None,
    }

    # Spectral centroid mean
    try:
        cent = _analysis.spectral_centroid(buf)
        info["spectral_centroid_mean_hz"] = round(float(np.mean(cent)), 1)
    except _DSP_ERRORS:
        pass

    # Spectral flatness mean
    try:
        flat = _analysis.spectral_flatness_curve(buf)
        info["spectral_flatness_mean"] = round(float(np.mean(flat)), 4)
    except _DSP_ERRORS:
        pass

    if args.json:
        print(json.dumps(info, indent=2))
    else:
        for k, v in info.items():
            print(f"  {k}: {v}")


# ---------------------------------------------------------------------------
# Subcommand: synth
# ---------------------------------------------------------------------------


_SYNTH_SUBCOMMANDS = {"sine", "noise", "drum", "note", "sequence", "oscillator", "fm"}


def cmd_synth(args: argparse.Namespace) -> None:
    """Generate audio and write to file."""
    from nanodsp import synthesis as _synthesis

    sr = float(args.sample_rate or 48000)
    duration = float(args.duration or 1.0)
    frames = int(sr * duration)
    channels = int(args.channels or 1)
    subcmd = args.synth_type

    if subcmd == "sine":
        freq = float(args.freq or 440.0)
        buf = AudioBuffer.sine(freq, channels=channels, frames=frames, sample_rate=sr)

    elif subcmd == "noise":
        buf = AudioBuffer.noise(channels=channels, frames=frames, sample_rate=sr)

    elif subcmd == "drum":
        drum_type = args.type or "analog_bass_drum"
        freq = float(args.freq or 60.0)
        drum_fn = getattr(_synthesis, drum_type, None)
        if drum_fn is None:
            print(f"Unknown drum type: {drum_type!r}", file=sys.stderr)
            sys.exit(1)
        buf = drum_fn(frames, freq=freq, sample_rate=sr)

    elif subcmd == "oscillator":
        freq = float(args.freq or 440.0)
        waveform = args.waveform or "sine"
        buf = _synthesis.oscillator(
            frames,
            freq=freq,
            waveform=waveform,
            sample_rate=sr,
        )

    elif subcmd == "fm":
        freq = float(args.freq or 440.0)
        ratio = float(args.ratio or 2.0)
        index = float(args.index or 1.0)
        buf = _synthesis.fm2(
            frames,
            freq=freq,
            ratio=ratio,
            index=index,
            sample_rate=sr,
        )

    elif subcmd == "note":
        instrument = args.instrument or "clarinet"
        freq = float(args.freq or 440.0)
        velocity = float(args.velocity or 0.8)
        release = float(args.release or 0.1)
        buf = _synthesis.synth_note(
            instrument,
            freq=freq,
            duration=duration,
            velocity=velocity,
            release=release,
            sample_rate=sr,
        )

    elif subcmd == "sequence":
        instrument = args.instrument or "clarinet"
        if not args.notes:
            print("--notes is required for sequence synth", file=sys.stderr)
            sys.exit(1)
        try:
            raw = json.loads(args.notes)
        except json.JSONDecodeError as e:
            print(f"Invalid JSON for --notes: {e}", file=sys.stderr)
            sys.exit(1)
        notes = [(n["freq"], n["start"], n["dur"]) for n in raw]
        velocity = float(args.velocity or 0.8)
        release = float(args.release or 0.1)
        buf = _synthesis.synth_sequence(
            instrument,
            notes,
            sample_rate=sr,
            release=release,
            velocity=velocity,
        )

    else:
        print(f"Unknown synth type: {subcmd!r}", file=sys.stderr)
        sys.exit(1)

    bit_depth = args.bit_depth or 16
    _write_output(args.output, buf, bit_depth=bit_depth, args=args)
    _log(args, f"Wrote {args.output}")


# ---------------------------------------------------------------------------
# Subcommand: convert
# ---------------------------------------------------------------------------


def cmd_convert(args: argparse.Namespace) -> None:
    """Convert audio format, resample, change channels."""
    from nanodsp.analysis import resample

    buf = _read_input(args.input, args)

    # Resample if requested
    if args.sample_rate:
        target_sr = float(args.sample_rate)
        if target_sr != buf.sample_rate:
            _log_verbose(args, f"  Resampling {buf.sample_rate} -> {target_sr}")
            buf = resample(buf, target_sr)

    # Channel conversion
    if args.channels:
        target_ch = int(args.channels)
        if target_ch != buf.channels:
            _log_verbose(args, f"  Converting {buf.channels}ch -> {target_ch}ch")
            if target_ch == 1:
                buf = buf.to_mono()
            else:
                if buf.channels != 1:
                    buf = buf.to_mono()
                buf = buf.to_channels(target_ch)

    bit_depth = args.bit_depth or 16
    _write_output(args.output, buf, bit_depth=bit_depth, args=args)
    _log(args, f"Wrote {args.output}")


# ---------------------------------------------------------------------------
# Subcommand: preset
# ---------------------------------------------------------------------------


def cmd_preset(args: argparse.Namespace) -> None:
    """Manage and apply presets."""
    from nanodsp._cli import get_preset_categories, apply_preset

    presets = _load_presets()
    subcmd = args.preset_action

    if subcmd == "list":
        cats = get_preset_categories()
        filter_cat = (
            args.category if hasattr(args, "category") and args.category else None
        )
        if filter_cat:
            names = cats.get(filter_cat, [])
            if not names:
                print(f"No presets in category: {filter_cat!r}")
                return
            print(f"\n  {filter_cat}:")
            for name in sorted(names):
                desc = presets[name].get("description", "")
                print(f"    {name:20s} {desc}")
        else:
            for cat in sorted(cats):
                print(f"\n  {cat}:")
                for name in sorted(cats[cat]):
                    desc = presets[name].get("description", "")
                    print(f"    {name:20s} {desc}")
        print()

    elif subcmd == "info":
        name = args.name
        if name not in presets:
            print(f"Unknown preset: {name!r}", file=sys.stderr)
            sys.exit(1)
        preset = presets[name]
        print(f"\n  {name}")
        print(f"  Category: {preset.get('category', 'other')}")
        print(f"  Description: {preset.get('description', '')}")
        if "fn" in preset:
            print(f"  Function: {preset['fn']}")
        if "defaults" in preset and preset["defaults"]:
            print(f"  Defaults: {preset['defaults']}")
        if "chain" in preset:
            print("  Chain:")
            for mod, fn, params in preset["chain"]:
                print(f"    {mod}.{fn}({params})")
        print()

    elif subcmd == "apply":
        name = args.name
        if name not in presets:
            print(f"Unknown preset: {name!r}", file=sys.stderr)
            sys.exit(1)
        from nanodsp._cli import coerce_value

        buf = _read_input(args.input, args)
        # Collect overrides from extra KEY=VALUE args. Keys may be scoped to a
        # single chain step as STEP.KEY (see _cli.apply_preset).
        overrides = {}
        if args.overrides:
            for ov in args.overrides:
                if "=" not in ov:
                    print(
                        f"Invalid override (expected key=value): {ov!r}",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                k, v = ov.split("=", 1)
                # Shared coercion with the -f path, so that ints stay ints and
                # `flag=false` becomes False rather than the truthy string.
                try:
                    overrides[k] = coerce_value(v, None)
                except (ValueError, OverflowError) as e:
                    print(f"Bad override {ov!r}: {e}", file=sys.stderr)
                    sys.exit(1)
        try:
            buf = apply_preset(name, buf, overrides)
        except _DSP_ERRORS as e:
            print(f"Error applying preset {name!r}: {e}", file=sys.stderr)
            sys.exit(1)
        bit_depth = args.bit_depth or 16
        _write_output(args.output, buf, bit_depth=bit_depth, args=args)
        _log(args, f"Wrote {args.output}")

    else:
        print(f"Unknown preset action: {subcmd!r}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Subcommand: list
# ---------------------------------------------------------------------------


def cmd_pipe(args: argparse.Namespace) -> None:
    """Read WAV from stdin, apply effect chain, write WAV to stdout."""
    from nanodsp.io import read_wav_bytes, write_wav_bytes

    # Force quiet mode -- stdout is the audio stream
    args.quiet = True
    args.verbose = False

    try:
        raw_in = sys.stdin.buffer.read()
    except OSError as e:
        print(f"Error reading stdin: {e}", file=sys.stderr)
        sys.exit(1)

    if not raw_in:
        print("Error: no data on stdin", file=sys.stderr)
        sys.exit(1)

    try:
        buf = read_wav_bytes(raw_in)
    except _IO_ERRORS as e:
        print(f"Error parsing WAV from stdin: {e}", file=sys.stderr)
        sys.exit(1)

    steps = _build_chain(args)
    if steps:
        buf = _apply_chain(buf, steps, args)

    bit_depth = args.bit_depth or 16
    try:
        raw_out = write_wav_bytes(buf, bit_depth=bit_depth)
    except _IO_ERRORS as e:
        print(f"Error encoding WAV output: {e}", file=sys.stderr)
        sys.exit(1)

    sys.stdout.buffer.write(raw_out)


# ---------------------------------------------------------------------------
# Subcommand: benchmark
# ---------------------------------------------------------------------------


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Benchmark a DSP function by running it repeatedly on a test buffer."""
    import statistics
    import time

    from nanodsp._cli import get_processor, parse_fx_token, coerce_params

    token = args.function
    name, raw_params = parse_fx_token(token)
    try:
        fn, _mod = get_processor(name)
    except KeyError:
        print(f"Unknown function: {name!r}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Cannot benchmark {name!r}: {e}", file=sys.stderr)
        sys.exit(1)

    params = coerce_params(fn, raw_params)

    sr = float(args.sample_rate)
    duration = float(args.duration)
    channels = int(args.channels)
    frames = int(sr * duration)
    iterations = int(args.iterations)
    warmup = int(args.warmup)

    buf = AudioBuffer.sine(440.0, channels=channels, frames=frames, sample_rate=sr)

    # Warmup
    for _ in range(warmup):
        fn(buf, **params)

    # Timed runs
    times: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn(buf, **params)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    t_min = min(times)
    t_max = max(times)
    t_mean = statistics.mean(times)
    t_median = statistics.median(times)
    t_std = statistics.stdev(times) if len(times) > 1 else 0.0
    throughput = duration / t_mean if t_mean > 0 else float("inf")

    params_str = ", ".join(f"{k}={v}" for k, v in params.items())
    fn_label = f"{name}({params_str})" if params_str else f"{name}()"

    if getattr(args, "json", False):
        result = {
            "function": fn_label,
            "channels": channels,
            "frames": frames,
            "duration_s": duration,
            "sample_rate": sr,
            "iterations": iterations,
            "warmup": warmup,
            "time_min_ms": t_min * 1000,
            "time_max_ms": t_max * 1000,
            "time_mean_ms": t_mean * 1000,
            "time_median_ms": t_median * 1000,
            "time_std_ms": t_std * 1000,
            "throughput_x": round(throughput, 1),
        }
        print(json.dumps(result, indent=2))
    else:
        print(f"\n  Function: {fn_label}")
        print(
            f"  Buffer:   {channels}ch, {frames} frames ({duration:.3f}s @ {int(sr)} Hz)"
        )
        print(f"  Runs:     {iterations} iterations (+ {warmup} warmup)")
        print()
        print("  Time per call:")
        print(f"    min:    {t_min * 1000:.3f} ms")
        print(f"    max:    {t_max * 1000:.3f} ms")
        print(f"    mean:   {t_mean * 1000:.3f} ms")
        print(f"    median: {t_median * 1000:.3f} ms")
        print(f"    std:    {t_std * 1000:.3f} ms")
        print()
        print(f"  Throughput: {throughput:.0f}x realtime")
        print()


# ---------------------------------------------------------------------------
# Subcommand: list
# ---------------------------------------------------------------------------


def cmd_list(args: argparse.Namespace) -> None:
    """List available functions by category."""
    from nanodsp._cli import (
        get_categories,
        get_registry,
        get_kinds,
        format_signature,
        PROCESSOR,
        _KIND_NOTE,
    )

    cats = get_categories()
    reg = get_registry()
    kinds = get_kinds()
    filter_cat = args.category if hasattr(args, "category") and args.category else None

    def _print_category(cat: str, names: list[str]) -> None:
        chainable = sum(1 for n in names if kinds.get(n, PROCESSOR) == PROCESSOR)
        print(f"\n  {cat} ({len(names)} functions, {chainable} chainable):")
        for name in sorted(names):
            fn, _ = reg[name]
            sig = format_signature(fn)
            kind = kinds.get(name, PROCESSOR)
            # Non-chainable entries stay listed -- they are reachable from the
            # Python API and from `synth`/`analyze` -- but are marked so that
            # `-f` is not attempted on them.
            note = "" if kind == PROCESSOR else f"   [{_KIND_NOTE[kind]}]"
            print(f"    {name}{sig}{note}")

    if filter_cat:
        if filter_cat not in cats:
            print(f"Unknown category: {filter_cat!r}")
            print(f"Available: {', '.join(sorted(cats))}")
            sys.exit(1)
        _print_category(filter_cat, cats[filter_cat])
    else:
        for cat in sorted(cats):
            if cats[cat]:
                _print_category(cat, cats[cat])
    print()


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Tab completion (argcomplete)
#
# Completer callables are attached to argparse actions as a ``.completer``
# attribute. This is harmless when argcomplete is not installed (the attribute
# is simply ignored); when it is, ``main`` enables it. To activate shell
# completion, install argcomplete and run, e.g.:
#     eval "$(register-python-argcomplete nanodsp)"
# ---------------------------------------------------------------------------


def _preset_name_completer(prefix: str, **kwargs) -> list[str]:
    try:
        from nanodsp._cli import get_presets

        names = get_presets().keys()
    except (ValueError, ImportError):
        return []
    return [n for n in names if n.startswith(prefix)]


def _preset_category_completer(prefix: str, **kwargs) -> list[str]:
    try:
        from nanodsp._cli import get_preset_categories

        cats = get_preset_categories().keys()
    except (ValueError, ImportError):
        return []
    return [c for c in cats if c.startswith(prefix)]


def _function_completer(prefix: str, **kwargs) -> list[str]:
    from nanodsp._cli import get_registry

    return [n for n in get_registry() if n.startswith(prefix)]


def _category_completer(prefix: str, **kwargs) -> list[str]:
    from nanodsp._cli import get_categories

    return [c for c in get_categories() if c.startswith(prefix)]


def _set_completer(action: argparse.Action, completer) -> None:
    """Attach an argcomplete completer to an argparse action.

    argcomplete reads a ``.completer`` attribute off the action; argparse has no
    declared slot for it, hence the localized type-ignore.
    """
    action.completer = completer  # type: ignore[attr-defined]


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser."""
    parser = argparse.ArgumentParser(
        prog="nanodsp",
        description="nanodsp - audio DSP toolkit",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"nanodsp {__version__}",
    )

    # Global verbosity flags (mutually exclusive)
    verb_group = parser.add_mutually_exclusive_group()
    verb_group.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output (show details about each step)",
    )
    verb_group.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress all non-essential output",
    )

    sub = parser.add_subparsers(dest="command")

    # --- info ---
    p_info = sub.add_parser("info", help="Show audio file metadata")
    p_info.add_argument("file", help="Input audio file")
    p_info.add_argument("--json", action="store_true", help="Output as JSON")

    # --- process ---
    p_proc = sub.add_parser("process", help="Apply effect chain to audio")
    p_proc.add_argument(
        "input",
        nargs="+",
        help="Input audio file(s) (supports shell globs)",
    )
    p_proc.add_argument(
        "-o",
        "--output",
        help="Output audio file (single-file mode)",
    )
    p_proc.add_argument(
        "-O",
        "--output-dir",
        help="Output directory for batch mode (files keep original names)",
    )
    _add_chain_options(p_proc)
    p_proc.add_argument(
        "-b",
        "--bit-depth",
        type=int,
        choices=[16, 24, 32],
        help="Output bit depth: 16/24 = PCM, 32 = IEEE float (default: 16)",
    )
    p_proc.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Show the processing chain without reading or writing files",
    )
    p_proc.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        metavar="N",
        help="Process N files concurrently (0 = one per CPU). Batch mode only.",
    )
    p_proc.add_argument(
        "--stream",
        action="store_true",
        help="Process a block at a time in constant memory (large files). "
        "Requires every effect in the chain to have a streaming form.",
    )
    p_proc.add_argument(
        "--block-size",
        type=int,
        default=65536,
        metavar="N",
        help="Frames per block in --stream mode (default: 65536)",
    )
    p_proc.add_argument(
        "--stats",
        action="store_true",
        help="Report peak, true peak and loudness before and after the chain",
    )

    # --- analyze ---
    p_analyze = sub.add_parser("analyze", help="Analyze audio features")
    p_analyze.add_argument("input", help="Input audio file")
    p_analyze.add_argument(
        "analysis_type",
        choices=sorted(_ANALYSIS_SUBCOMMANDS),
        help="Analysis type",
    )
    p_analyze.add_argument("--json", action="store_true", help="Output as JSON")
    p_analyze.add_argument("--fmin", help="Minimum frequency for pitch detection")
    p_analyze.add_argument("--fmax", help="Maximum frequency for pitch detection")
    p_analyze.add_argument("--method", help="Method for onset detection")

    # --- synth ---
    p_synth = sub.add_parser("synth", help="Synthesize audio")
    p_synth.add_argument("output", help="Output audio file")
    p_synth.add_argument(
        "synth_type",
        choices=sorted(_SYNTH_SUBCOMMANDS),
        help="Synthesis type",
    )
    p_synth.add_argument("--freq", help="Frequency in Hz")
    p_synth.add_argument("--duration", help="Duration in seconds")
    p_synth.add_argument("--sample-rate", help="Sample rate (default: 48000)")
    p_synth.add_argument("--channels", help="Number of channels (default: 1)")
    p_synth.add_argument("--instrument", help="STK instrument name")
    p_synth.add_argument("--type", help="Drum type (e.g. analog_bass_drum)")
    p_synth.add_argument("--waveform", help="Oscillator waveform name")
    p_synth.add_argument("--ratio", help="FM ratio")
    p_synth.add_argument("--index", help="FM index")
    p_synth.add_argument("--velocity", help="Note velocity (0.0-1.0)")
    p_synth.add_argument("--release", help="Release time in seconds")
    p_synth.add_argument("--notes", help="JSON array for sequence synth")
    p_synth.add_argument(
        "-b",
        "--bit-depth",
        type=int,
        choices=[16, 24, 32],
        help="Output bit depth: 16/24 = PCM, 32 = IEEE float (default: 16)",
    )

    # --- convert ---
    p_conv = sub.add_parser("convert", help="Convert audio format/rate/channels")
    p_conv.add_argument("input", help="Input audio file")
    p_conv.add_argument("output", help="Output audio file")
    p_conv.add_argument("--sample-rate", type=float, help="Target sample rate")
    p_conv.add_argument("--channels", type=int, help="Target channel count")
    p_conv.add_argument(
        "-b",
        "--bit-depth",
        type=int,
        choices=[16, 24, 32],
        help="Output bit depth: 16/24 = PCM, 32 = IEEE float (default: 16)",
    )

    # --- preset ---
    p_preset = sub.add_parser("preset", help="Manage and apply presets")
    p_preset_sub = p_preset.add_subparsers(dest="preset_action")

    p_plist = p_preset_sub.add_parser("list", help="List presets")
    _set_completer(
        p_plist.add_argument("category", nargs="?", help="Filter by category"),
        _preset_category_completer,
    )

    p_pinfo = p_preset_sub.add_parser("info", help="Show preset details")
    _set_completer(
        p_pinfo.add_argument("name", help="Preset name"), _preset_name_completer
    )

    p_papply = p_preset_sub.add_parser("apply", help="Apply a preset")
    _set_completer(
        p_papply.add_argument("name", help="Preset name"), _preset_name_completer
    )
    p_papply.add_argument("input", help="Input audio file")
    p_papply.add_argument("output", help="Output audio file")
    p_papply.add_argument(
        "overrides",
        nargs="*",
        metavar="KEY=VALUE",
        help="Parameter overrides",
    )
    p_papply.add_argument(
        "-b",
        "--bit-depth",
        type=int,
        choices=[16, 24, 32],
        help="Output bit depth: 16/24 = PCM, 32 = IEEE float (default: 16)",
    )

    # --- pipe ---
    p_pipe = sub.add_parser(
        "pipe",
        help="Read WAV from stdin, apply effects, write WAV to stdout",
    )
    _add_chain_options(p_pipe)
    p_pipe.add_argument(
        "-b",
        "--bit-depth",
        type=int,
        choices=[16, 24, 32],
        help="Output bit depth: 16/24 = PCM, 32 = IEEE float (default: 16)",
    )

    # --- benchmark ---
    p_bench = sub.add_parser("benchmark", help="Benchmark a DSP function")
    p_bench.add_argument(
        "function",
        help="Function to benchmark (format: name:param=val,param=val)",
    )
    p_bench.add_argument(
        "-n",
        "--iterations",
        type=int,
        default=100,
        help="Number of timed iterations (default: 100)",
    )
    p_bench.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Warmup iterations (default: 5)",
    )
    p_bench.add_argument(
        "--duration",
        type=float,
        default=1.0,
        help="Test buffer duration in seconds (default: 1.0)",
    )
    p_bench.add_argument(
        "--sample-rate",
        type=float,
        default=48000.0,
        help="Test buffer sample rate (default: 48000)",
    )
    p_bench.add_argument(
        "--channels",
        type=int,
        default=1,
        help="Test buffer channels (default: 1)",
    )
    p_bench.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )

    # --- list ---
    p_list = sub.add_parser("list", help="List available functions")
    _set_completer(
        p_list.add_argument("category", nargs="?", help="Filter by category"),
        _category_completer,
    )

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = build_parser()

    # Enable shell tab completion when argcomplete is installed (optional).
    try:
        import argcomplete

        argcomplete.autocomplete(parser)
    except ImportError:
        pass

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Validate process subcommand output args
    if args.command == "process" and not getattr(args, "dry_run", False):
        n_inputs = len(args.input)
        has_output = bool(args.output)
        has_output_dir = bool(args.output_dir)

        if not has_output and not has_output_dir:
            print(
                "Error: process requires -o/--output or -O/--output-dir",
                file=sys.stderr,
            )
            sys.exit(1)
        if has_output and has_output_dir:
            print(
                "Error: use -o/--output or -O/--output-dir, not both",
                file=sys.stderr,
            )
            sys.exit(1)
        if n_inputs > 1 and has_output:
            print(
                "Error: use -O/--output-dir for multiple input files, not -o/--output",
                file=sys.stderr,
            )
            sys.exit(1)

    dispatch = {
        "info": cmd_info,
        "process": cmd_process,
        "analyze": cmd_analyze,
        "synth": cmd_synth,
        "convert": cmd_convert,
        "preset": cmd_preset,
        "pipe": cmd_pipe,
        "benchmark": cmd_benchmark,
        "list": cmd_list,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    handler(args)


if __name__ == "__main__":
    main()

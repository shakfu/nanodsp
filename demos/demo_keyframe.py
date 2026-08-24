#!/usr/bin/env python3
"""Demo: keyframe time-stretching via extrema sampling.

Renders time-stretches, pitch-shifts and sparsification sweeps using the
content-adaptive overlap-add stretcher. The signal is reduced to its local
extrema; their spacing tracks local bandwidth, so it doubles as a per-sample
estimate of information density and sizes each crossfade -- short at a
transient, long across a sustained note, with no separate transient detector,
no FFT and no correlation search.

The interesting knobs are not the stretch factor. They are:

- ``splice_keyframes`` -- a macro over splice duration that leaves the
  adaptation intact: raising it lengthens every splice proportionally.
- ``threshold`` (via ``keyframe_sparsify``) -- how much of the signal survives
  the sparse representation at all. Because high frequencies tend to carry
  lower amplitudes in practice, raising it behaves like an amplitude-dependent
  lowpass, which is a usable lo-fi effect well before it becomes destructive.

See nanodsp.timestretch.keyframe_stretch and .keyframe_sparsify.
"""

import argparse
import os
import time

import numpy as np

from nanodsp._core import keyframe as _kf
from nanodsp.buffer import AudioBuffer
from nanodsp.timestretch import (
    keyframe_sparsify,
    keyframe_stretch,
    paulstretch,
    signalsmith_stretch,
)


def peak_normalize(buf: AudioBuffer) -> AudioBuffer:
    """Scale so the loudest sample sits at 0 dBFS."""
    peak = np.max(np.abs(buf.data))
    if peak > 0:
        return buf.gain_db(-20.0 * np.log10(peak))
    return buf


def report_density(buf: AudioBuffer) -> None:
    """Print how sparse the representation is, and how much it varies.

    This is the measurement the whole method rests on: if extrema spacing did
    not vary with content, splice duration would not adapt and there would be
    no reason to prefer this over plain overlap-add.
    """
    idx, _ = _kf.analyze(buf.ensure_1d(0), 0.001)
    spacing = np.diff(idx)
    if spacing.size == 0:
        print("  (no extrema found)")
        return
    print(
        f"  keyframes {len(idx)} of {buf.frames} samples "
        f"(M/N = {len(idx) / buf.frames:.3f})"
    )
    print(
        f"  spacing: median {np.median(spacing):6.1f} samples, "
        f"10th pct {np.percentile(spacing, 10):6.1f}, "
        f"90th pct {np.percentile(spacing, 90):6.1f}"
    )
    print(
        "  -> splice duration tracks that spread, so it is roughly "
        f"{np.percentile(spacing, 90) / max(np.percentile(spacing, 10), 1e-9):.0f}x "
        "longer in the sparsest passages than the densest"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Demo: keyframe time-stretch via extrema sampling"
    )
    parser.add_argument("infile", help="Input .wav file")
    parser.add_argument(
        "-o", "--out-dir", default="build/demo-output", help="Output directory"
    )
    parser.add_argument(
        "-n", "--no-normalize", action="store_true", help="Skip peak normalization"
    )
    parser.add_argument(
        "--compare-seconds",
        type=float,
        default=2.0,
        help="Seconds of the source used for the three-way comparison",
    )
    args = parser.parse_args()

    normalize = (lambda b: b) if args.no_normalize else peak_normalize
    os.makedirs(args.out_dir, exist_ok=True)

    buf = AudioBuffer.from_file(args.infile)
    name = os.path.splitext(os.path.basename(args.infile))[0]
    in_secs = buf.frames / buf.sample_rate
    print(f"Source: {name} ({in_secs:.2f}s, {buf.channels}ch @ {buf.sample_rate:g} Hz)")
    report_density(buf)

    def render(out: AudioBuffer, label: str) -> None:
        out = normalize(out)
        path = os.path.join(args.out_dir, f"{name}_keyframe_{label}.wav")
        out.write(path)
        print(f"  {label:26s} {out.frames / out.sample_rate:6.2f}s -> {path}")

    # --- Time-stretch at several factors (pitch preserved) ---
    print("Time-stretch (pitch preserved):")
    render(keyframe_stretch(buf, stretch=0.5), "stretch-0.5x")
    render(keyframe_stretch(buf, stretch=1.5), "stretch-1.5x")
    render(keyframe_stretch(buf, stretch=2.0), "stretch-2x")
    render(keyframe_stretch(buf, stretch=4.0), "stretch-4x")

    # --- Pure pitch-shift (length unchanged) ---
    print("Pitch-shift (length unchanged):")
    render(keyframe_stretch(buf, semitones=12.0), "pitch-octave-up")
    render(keyframe_stretch(buf, semitones=-12.0), "pitch-octave-down")
    render(keyframe_stretch(buf, semitones=7.0), "pitch-fifth-up")

    # --- Both at once, decoupled ---
    print("Time and pitch together:")
    render(keyframe_stretch(buf, stretch=2.0, semitones=5.0), "stretch-2x-pitch-up5")
    render(keyframe_stretch(buf, stretch=0.7, semitones=7.0), "chipmunk-0.7x-up-fifth")

    # --- Splice threshold: the macro over splice duration ---
    # Same stretch factor throughout. Small values splice often and briefly,
    # which keeps transients crisp and can sound stuttery on sustained
    # material; large values splice rarely and over long spans, which is
    # smoother and blurs onsets. The adaptation to content is unaffected --
    # this scales it.
    print("Splice threshold (K, in keyframes) at a fixed 3x:")
    for k in (4, 16, 64, 256):
        render(
            keyframe_stretch(buf, stretch=3.0, splice_keyframes=k), f"3x-splice-k{k}"
        )

    # --- Splice duration cap ---
    # Section 3.7 of the paper: a long sparse passage before a transient
    # produces a very long splice, smearing everything leading up to it.
    print("Splice duration cap at a fixed 3x:")
    render(keyframe_stretch(buf, stretch=3.0, max_splice_ms=10.0), "3x-cap-10ms")
    render(keyframe_stretch(buf, stretch=3.0, max_splice_ms=0.0), "3x-cap-none")

    # --- Sparsification alone: the representation without any stretching ---
    print("Sparsify only (no time or pitch change) -- rising threshold:")
    for thr in (0.001, 0.01, 0.05, 0.2):
        out = keyframe_sparsify(buf, threshold=thr)
        idx, _ = _kf.analyze(buf.ensure_1d(0), thr)
        label = f"sparsify-{thr:g}"
        render(out, label)
        print(f"    {'':26s} kept {len(idx) / buf.frames:6.3f} of the samples")

    # --- Three-way comparison at the same factor, with timings ---
    # Cheapness is the point of the method, so it is worth measuring rather
    # than asserting. Note the gap to the phase-vocoder family is far smaller
    # here than on the embedded hardware the algorithm was designed for --
    # desktop FFTs are vectorised.
    n = min(buf.frames, int(args.compare_seconds * buf.sample_rate))
    src = buf.slice(0, n)
    factor = 4.0
    print(f"Compare @ {factor:g}x on first {n / buf.sample_rate:.2f}s:")
    for fn, label in (
        (lambda b: keyframe_stretch(b, stretch=factor), "compare-keyframe"),
        (lambda b: signalsmith_stretch(b, stretch=factor), "compare-signalsmith"),
        (lambda b: paulstretch(b, stretch=factor), "compare-paulstretch"),
    ):
        start = time.perf_counter()
        out = fn(src)
        elapsed = (time.perf_counter() - start) * 1000.0
        render(out, label)
        print(f"    {'':26s} {elapsed:7.1f} ms")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Demo: signalsmith-stretch time-stretching and pitch-shifting.

Renders high-quality, transient-aware time-stretches and independent
pitch-shifts using the MIT-licensed signalsmith-stretch library. Unlike
PaulStretch (smeared, ambient textures at extreme factors), this stays musical
at modest ratios and decouples duration from pitch.

See nanodsp.timestretch.signalsmith_stretch.
"""

import argparse
import os

import numpy as np

from nanodsp.buffer import AudioBuffer
from nanodsp.timestretch import paulstretch, signalsmith_stretch


def peak_normalize(buf: AudioBuffer) -> AudioBuffer:
    """Scale so the loudest sample sits at 0 dBFS."""
    peak = np.max(np.abs(buf.data))
    if peak > 0:
        return buf.gain_db(-20.0 * np.log10(peak))
    return buf


def main():
    parser = argparse.ArgumentParser(
        description="Demo: signalsmith-stretch time-stretch / pitch-shift"
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
        help="Seconds of the source used for the extreme/compare renders "
        "(keeps the large-factor outputs small)",
    )
    args = parser.parse_args()

    normalize = (lambda b: b) if args.no_normalize else peak_normalize
    os.makedirs(args.out_dir, exist_ok=True)

    buf = AudioBuffer.from_file(args.infile)
    name = os.path.splitext(os.path.basename(args.infile))[0]
    in_secs = buf.frames / buf.sample_rate
    print(f"Source: {name} ({in_secs:.2f}s, {buf.channels}ch @ {buf.sample_rate:g} Hz)")

    def render(out: AudioBuffer, label: str) -> None:
        out = normalize(out)
        path = os.path.join(args.out_dir, f"{name}_sigstretch_{label}.wav")
        out.write(path)
        print(f"  {label:24s} {out.frames / out.sample_rate:6.2f}s -> {path}")

    # --- Time-stretch at several factors (pitch preserved) ---
    render(signalsmith_stretch(buf, stretch=0.5), "stretch-0.5x")
    render(signalsmith_stretch(buf, stretch=1.5), "stretch-1.5x")
    render(signalsmith_stretch(buf, stretch=2.0), "stretch-2x")
    render(signalsmith_stretch(buf, stretch=4.0), "stretch-4x")

    # --- Pure pitch-shift (length unchanged) ---
    render(signalsmith_stretch(buf, stretch=1.0, semitones=12.0), "pitch-octave-up")
    render(signalsmith_stretch(buf, stretch=1.0, semitones=-12.0), "pitch-octave-down")
    render(signalsmith_stretch(buf, stretch=1.0, semitones=7.0), "pitch-fifth-up")

    # --- Tonality limit preserves high-frequency timbre on a big shift ---
    render(
        signalsmith_stretch(buf, stretch=1.0, semitones=7.0, tonality_hz=8000.0),
        "pitch-fifth-up-tonality",
    )

    # --- Stretch and pitch-shift together (decoupled) ---
    render(signalsmith_stretch(buf, stretch=2.0, semitones=5.0), "stretch-2x-pitch-up5")

    # --- Cheaper (lower-CPU) preset ---
    render(signalsmith_stretch(buf, stretch=2.0, cheaper=True), "stretch-2x-cheaper")

    # --- Fine detune (fractional semitones, length unchanged) ---
    render(signalsmith_stretch(buf, stretch=1.0, semitones=0.3), "detune-up-30cents")

    # --- "Monster": slower and pitched down an octave ---
    render(
        signalsmith_stretch(buf, stretch=1.5, semitones=-12.0),
        "monster-1.5x-down-octave",
    )

    # --- "Chipmunk": faster and pitched up, with tonality limit ---
    render(
        signalsmith_stretch(buf, stretch=0.7, semitones=7.0, tonality_hz=8000.0),
        "chipmunk-0.7x-up-fifth",
    )

    # --- Extreme slowdown: contrast signalsmith vs PaulStretch on the same
    # short source at the same factor. signalsmith stays recognizable and
    # transient-aware; PaulStretch smears it into an ambient wash. ---
    n = min(buf.frames, int(args.compare_seconds * buf.sample_rate))
    src = buf.slice(0, n)
    factor = 8.0
    print(
        f"Compare @ {factor:g}x on first {n / buf.sample_rate:.2f}s "
        "(signalsmith vs paulstretch):"
    )
    render(signalsmith_stretch(src, stretch=factor), "extreme-8x-signalsmith")
    render(paulstretch(src, stretch=factor), "extreme-8x-paulstretch")


if __name__ == "__main__":
    main()

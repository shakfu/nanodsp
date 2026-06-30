#!/usr/bin/env python3
"""Demo: PaulStretch extreme time-stretching.

Renders the smeared, ambient textures PaulStretch is known for: large
time-stretch factors, transient preservation, and spectral effects (pitch
shift, harmonics, spectral spread, and band filtering) applied during
resynthesis.

The algorithm is by Nasca Octavian Paul (public domain); see
nanodsp.timestretch.paulstretch.
"""

import argparse
import os

import numpy as np

from nanodsp.buffer import AudioBuffer
from nanodsp.timestretch import paulstretch


def peak_normalize(buf: AudioBuffer) -> AudioBuffer:
    """Scale so the loudest sample sits at 0 dBFS."""
    peak = np.max(np.abs(buf.data))
    if peak > 0:
        return buf.gain_db(-20.0 * np.log10(peak))
    return buf


def head(buf: AudioBuffer, seconds: float) -> AudioBuffer:
    """Return the first `seconds` of audio (PaulStretch is usually fed a short
    source to grow into a long texture)."""
    n = min(buf.frames, int(seconds * buf.sample_rate))
    return buf.slice(0, n)


def main():
    parser = argparse.ArgumentParser(description="Demo: PaulStretch time-stretching")
    parser.add_argument("infile", help="Input .wav file")
    parser.add_argument(
        "-o", "--out-dir", default="build/demo-output", help="Output directory"
    )
    parser.add_argument(
        "-n", "--no-normalize", action="store_true", help="Skip peak normalization"
    )
    parser.add_argument(
        "--source-seconds",
        type=float,
        default=3.0,
        help="Seconds of the input used as the stretch source (keeps outputs small)",
    )
    args = parser.parse_args()

    normalize = (lambda b: b) if args.no_normalize else peak_normalize
    os.makedirs(args.out_dir, exist_ok=True)

    buf = AudioBuffer.from_file(args.infile)
    src = head(buf, args.source_seconds)
    name = os.path.splitext(os.path.basename(args.infile))[0]
    in_secs = src.frames / src.sample_rate
    print(f"Source: {name} ({in_secs:.2f}s, {src.channels}ch @ {src.sample_rate:g} Hz)")

    def render(out: AudioBuffer, label: str) -> None:
        out = normalize(out)
        path = os.path.join(args.out_dir, f"{name}_paulstretch_{label}.wav")
        out.write(path)
        print(f"  {label:24s} {out.frames / out.sample_rate:6.1f}s -> {path}")

    # --- Core stretch at several factors (pitch preserved) ---
    for factor in (2.0, 4.0, 8.0):
        render(paulstretch(src, stretch=factor), f"stretch-{int(factor)}x")

    # --- Window size: small keeps detail, large is smoother/more diffuse ---
    render(paulstretch(src, stretch=8.0, window_size=1024), "stretch-8x-win1024")
    render(paulstretch(src, stretch=8.0, window_size=16384), "stretch-8x-win16384")

    # --- Transient preservation keeps attacks sharp inside the smear ---
    render(paulstretch(src, stretch=8.0, onset=0.6), "stretch-8x-onset")

    # --- Pitch / octave shift during resynthesis ---
    render(paulstretch(src, stretch=8.0, pitch_semitones=12.0), "stretch-8x-octave-up")
    render(paulstretch(src, stretch=8.0, pitch_semitones=-12.0), "stretch-8x-octave-down")

    # --- Added harmonics + spectral spread: thicker, more diffuse pad ---
    render(paulstretch(src, stretch=8.0, harmonics=3, spread=6.0), "stretch-8x-thick")

    # --- Spectral band filtering applied before resynthesis ---
    render(paulstretch(src, stretch=8.0, highpass_hz=500.0, lowpass_hz=6000.0), "stretch-8x-bandpass")

    # --- Extreme "drone": very long stretch + transient preservation + low-pass ---
    render(
        paulstretch(src, stretch=20.0, window_size=8192, onset=0.4, lowpass_hz=8000.0),
        "drone-20x",
    )


if __name__ == "__main__":
    main()

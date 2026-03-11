#!/usr/bin/env python3
"""Demo: dynamics processing on audio.

Applies compression, limiting, noise gating, parallel compression,
and multiband compression with various settings.
"""

import argparse
import os

import numpy as np

from nanodsp.buffer import AudioBuffer
from nanodsp.effects import (
    compress,
    limit,
    noise_gate,
    parallel_compress,
    multiband_compress,
)


def peak_normalize(buf: AudioBuffer) -> AudioBuffer:
    """Scale so the loudest sample sits at 0 dBFS."""
    peak = np.max(np.abs(buf.data))
    if peak > 0:
        return buf.gain_db(-20.0 * np.log10(peak))
    return buf


def main():
    parser = argparse.ArgumentParser(description="Demo: dynamics processing")
    parser.add_argument("infile", help="Input .wav file")
    parser.add_argument(
        "-o", "--out-dir", default="build/demo-output", help="Output directory"
    )
    parser.add_argument(
        "-n", "--no-normalize", action="store_true", help="Skip peak normalization"
    )
    args = parser.parse_args()

    normalize = (lambda b: b) if args.no_normalize else peak_normalize
    os.makedirs(args.out_dir, exist_ok=True)
    buf = AudioBuffer.from_file(args.infile)
    name = os.path.splitext(os.path.basename(args.infile))[0]

    demos = [
        ("compress-gentle", lambda b: compress(b, ratio=2.0, threshold=-18.0, attack=0.01, release=0.1)),
        ("compress-heavy", lambda b: compress(b, ratio=8.0, threshold=-24.0, attack=0.005, release=0.05)),
        ("compress-fast", lambda b: compress(b, ratio=4.0, threshold=-20.0, attack=0.001, release=0.02)),
        ("limit", lambda b: limit(b, pre_gain=2.0)),
        ("gate-tight", lambda b: noise_gate(b, threshold_db=-30.0, attack=0.001, release=0.02)),
        ("gate-loose", lambda b: noise_gate(b, threshold_db=-40.0, attack=0.005, release=0.1)),
        ("parallel-compress", lambda b: parallel_compress(b, mix=0.5, ratio=8.0, threshold_db=-30.0)),
        ("multiband-default", lambda b: multiband_compress(b)),
        ("multiband-aggressive", lambda b: multiband_compress(
            b,
            crossover_freqs=[150.0, 1500.0, 6000.0],
            ratios=[3.0, 4.0, 4.0, 3.0],
            thresholds=[-30.0, -24.0, -24.0, -22.0],
        )),
    ]

    for label, fn in demos:
        out = normalize(fn(buf))
        path = os.path.join(args.out_dir, f"{name}_dyn_{label}.wav")
        out.write(path)
        print(f"  {label} -> {path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Demo: distortion and saturation effects on audio.

Applies overdrive, wavefolding, bitcrushing, decimation, and
soft/hard/tape saturation at various intensities.
"""

import argparse
import os

import numpy as np

from nanodsp.buffer import AudioBuffer
from nanodsp.effects import (
    overdrive,
    wavefold,
    bitcrush,
    decimator,
    saturate,
    fold,
)


def peak_normalize(buf: AudioBuffer) -> AudioBuffer:
    """Scale so the loudest sample sits at 0 dBFS."""
    peak = np.max(np.abs(buf.data))
    if peak > 0:
        return buf.gain_db(-20.0 * np.log10(peak))
    return buf


def main():
    parser = argparse.ArgumentParser(description="Demo: distortion effects")
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
        ("overdrive-light", lambda b: overdrive(b, drive=0.3)),
        ("overdrive-heavy", lambda b: overdrive(b, drive=0.8)),
        ("wavefold-mild", lambda b: wavefold(b, gain=1.5)),
        ("wavefold-extreme", lambda b: wavefold(b, gain=4.0)),
        ("bitcrush-12bit", lambda b: bitcrush(b, bit_depth=12)),
        ("bitcrush-8bit", lambda b: bitcrush(b, bit_depth=8)),
        ("bitcrush-4bit", lambda b: bitcrush(b, bit_depth=4)),
        ("decimator-mild", lambda b: decimator(b, downsample_factor=0.3, bitcrush_factor=0.2)),
        ("decimator-harsh", lambda b: decimator(b, downsample_factor=0.7, bitcrush_factor=0.7)),
        ("saturate-soft", lambda b: saturate(b, drive=0.4, mode="soft")),
        ("saturate-hard", lambda b: saturate(b, drive=0.5, mode="hard")),
        ("saturate-tape", lambda b: saturate(b, drive=0.5, mode="tape")),
        ("fold-subtle", lambda b: fold(b, increment=0.8)),
        ("fold-intense", lambda b: fold(b, increment=0.3)),
    ]

    for label, fn in demos:
        out = normalize(fn(buf))
        path = os.path.join(args.out_dir, f"{name}_dist_{label}.wav")
        out.write(path)
        print(f"  {label} -> {path}")


if __name__ == "__main__":
    main()

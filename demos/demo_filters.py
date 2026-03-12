#!/usr/bin/env python3
"""Demo: biquad filter effects on audio.

Applies lowpass, highpass, bandpass, notch, peak, and shelving filters
at various settings to show how each shapes the frequency content.
"""

import argparse
import os

import numpy as np

from nanodsp.buffer import AudioBuffer
from nanodsp.effects.filters import (
    lowpass,
    highpass,
    bandpass,
    notch,
    peak_db,
    high_shelf_db,
    low_shelf_db,
)


def peak_normalize(buf: AudioBuffer) -> AudioBuffer:
    """Scale so the loudest sample sits at 0 dBFS."""
    peak = np.max(np.abs(buf.data))
    if peak > 0:
        return buf.gain_db(-20.0 * np.log10(peak))
    return buf


def main():
    parser = argparse.ArgumentParser(description="Demo: biquad filters")
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
        ("lowpass-1k", lambda b: lowpass(b, 1000.0)),
        ("lowpass-500", lambda b: lowpass(b, 500.0)),
        ("highpass-1k", lambda b: highpass(b, 1000.0)),
        ("highpass-3k", lambda b: highpass(b, 3000.0)),
        ("bandpass-1k", lambda b: bandpass(b, 1000.0, octaves=1.0)),
        ("bandpass-3k", lambda b: bandpass(b, 3000.0, octaves=0.5)),
        ("notch-1k", lambda b: notch(b, 1000.0, octaves=0.5)),
        ("peak-boost-1k", lambda b: peak_db(b, 1000.0, 12.0, octaves=1.0)),
        ("peak-cut-3k", lambda b: peak_db(b, 3000.0, -12.0, octaves=1.0)),
        ("high-shelf-boost", lambda b: high_shelf_db(b, 4000.0, 6.0)),
        ("high-shelf-cut", lambda b: high_shelf_db(b, 4000.0, -6.0)),
        ("low-shelf-boost", lambda b: low_shelf_db(b, 300.0, 6.0)),
        ("low-shelf-cut", lambda b: low_shelf_db(b, 300.0, -6.0)),
    ]

    for label, fn in demos:
        out = normalize(fn(buf))
        path = os.path.join(args.out_dir, f"{name}_filter_{label}.wav")
        out.write(path)
        print(f"  {label} -> {path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Demo: pitch shifting on audio.

Applies DaisySP time-domain pitch shifting and spectral (phase vocoder)
pitch shifting at various intervals.
"""

import argparse
import os

import numpy as np

from nanodsp.buffer import AudioBuffer
from nanodsp.effects import pitch_shift
from nanodsp.spectral import pitch_shift_spectral


def peak_normalize(buf: AudioBuffer) -> AudioBuffer:
    """Scale so the loudest sample sits at 0 dBFS."""
    peak = np.max(np.abs(buf.data))
    if peak > 0:
        return buf.gain_db(-20.0 * np.log10(peak))
    return buf


def main():
    parser = argparse.ArgumentParser(description="Demo: pitch shifting")
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
        # DaisySP time-domain pitch shifter
        ("td-down-octave", lambda b: pitch_shift(b, semitones=-12.0)),
        ("td-down-fifth", lambda b: pitch_shift(b, semitones=-7.0)),
        ("td-down-third", lambda b: pitch_shift(b, semitones=-4.0)),
        ("td-up-third", lambda b: pitch_shift(b, semitones=4.0)),
        ("td-up-fifth", lambda b: pitch_shift(b, semitones=7.0)),
        ("td-up-octave", lambda b: pitch_shift(b, semitones=12.0)),
        # Spectral (phase vocoder) pitch shifter
        ("spectral-down-octave", lambda b: pitch_shift_spectral(b, semitones=-12.0)),
        ("spectral-down-fifth", lambda b: pitch_shift_spectral(b, semitones=-7.0)),
        ("spectral-up-fifth", lambda b: pitch_shift_spectral(b, semitones=7.0)),
        ("spectral-up-octave", lambda b: pitch_shift_spectral(b, semitones=12.0)),
    ]

    for label, fn in demos:
        out = normalize(fn(buf))
        path = os.path.join(args.out_dir, f"{name}_pitch_{label}.wav")
        out.write(path)
        print(f"  {label} -> {path}")


if __name__ == "__main__":
    main()

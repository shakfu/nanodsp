#!/usr/bin/env python3
"""Demo: modulation effects on audio.

Applies chorus, flanger, phaser, and tremolo at various settings.
"""

import argparse
import os

import numpy as np

from nanodsp.buffer import AudioBuffer
from nanodsp.effects import chorus, flanger, phaser, tremolo


def peak_normalize(buf: AudioBuffer) -> AudioBuffer:
    """Scale so the loudest sample sits at 0 dBFS."""
    peak = np.max(np.abs(buf.data))
    if peak > 0:
        return buf.gain_db(-20.0 * np.log10(peak))
    return buf


def main():
    parser = argparse.ArgumentParser(description="Demo: modulation effects")
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
        ("chorus-subtle", lambda b: chorus(b, lfo_freq=0.2, lfo_depth=0.3, delay_ms=4.0, feedback=0.1)),
        ("chorus-deep", lambda b: chorus(b, lfo_freq=0.5, lfo_depth=0.8, delay_ms=8.0, feedback=0.4)),
        ("flanger-slow", lambda b: flanger(b, lfo_freq=0.1, lfo_depth=0.4, feedback=0.5, delay_ms=1.5)),
        ("flanger-fast", lambda b: flanger(b, lfo_freq=0.8, lfo_depth=0.7, feedback=0.6, delay_ms=2.0)),
        ("flanger-jet", lambda b: flanger(b, lfo_freq=0.15, lfo_depth=0.9, feedback=0.8, delay_ms=3.0)),
        ("phaser-gentle", lambda b: phaser(b, lfo_freq=0.2, lfo_depth=0.3, feedback=0.3, poles=4)),
        ("phaser-deep", lambda b: phaser(b, lfo_freq=0.5, lfo_depth=0.7, feedback=0.7, poles=4)),
        ("tremolo-slow", lambda b: tremolo(b, freq=3.0, depth=0.5)),
        ("tremolo-fast", lambda b: tremolo(b, freq=8.0, depth=0.8)),
        ("tremolo-square", lambda b: tremolo(b, freq=4.0, depth=0.9, waveform=2)),
    ]

    for label, fn in demos:
        out = normalize(fn(buf))
        path = os.path.join(args.out_dir, f"{name}_mod_{label}.wav")
        out.write(path)
        print(f"  {label} -> {path}")


if __name__ == "__main__":
    main()

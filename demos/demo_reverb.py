#!/usr/bin/env python3
"""Demo: reverb algorithms on audio.

Applies FDN reverb presets, DaisySP ReverbSc, and STK reverb algorithms
to show different reverberant spaces.
"""

import argparse
import os

import numpy as np

from nanodsp.buffer import AudioBuffer
from nanodsp.effects import reverb, reverb_sc, stk_reverb


def peak_normalize(buf: AudioBuffer) -> AudioBuffer:
    """Scale so the loudest sample sits at 0 dBFS."""
    peak = np.max(np.abs(buf.data))
    if peak > 0:
        return buf.gain_db(-20.0 * np.log10(peak))
    return buf


def main():
    parser = argparse.ArgumentParser(description="Demo: reverb algorithms")
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
        # FDN reverb presets
        ("fdn-room", lambda b: reverb(b, preset="room", mix=0.3, decay=0.6)),
        ("fdn-hall", lambda b: reverb(b, preset="hall", mix=0.35, decay=0.8)),
        ("fdn-plate", lambda b: reverb(b, preset="plate", mix=0.3, decay=0.7)),
        ("fdn-chamber", lambda b: reverb(b, preset="chamber", mix=0.3, decay=0.75)),
        ("fdn-cathedral", lambda b: reverb(b, preset="cathedral", mix=0.4, decay=0.85)),
        # DaisySP ReverbSc
        ("reverbsc-short", lambda b: reverb_sc(b, feedback=0.5, lp_freq=12000.0)),
        ("reverbsc-long", lambda b: reverb_sc(b, feedback=0.85, lp_freq=8000.0)),
        ("reverbsc-dark", lambda b: reverb_sc(b, feedback=0.75, lp_freq=3000.0)),
        # STK reverbs
        ("stk-freeverb", lambda b: stk_reverb(b, algorithm="freeverb", mix=0.3, room_size=0.7)),
        ("stk-jcrev", lambda b: stk_reverb(b, algorithm="jcrev", mix=0.3, t60=1.5)),
        ("stk-nrev", lambda b: stk_reverb(b, algorithm="nrev", mix=0.3, t60=2.0)),
        ("stk-prcrev", lambda b: stk_reverb(b, algorithm="prcrev", mix=0.3, t60=1.0)),
    ]

    for label, fn in demos:
        out = normalize(fn(buf))
        path = os.path.join(args.out_dir, f"{name}_reverb_{label}.wav")
        out.write(path)
        print(f"  {label} -> {path}")


if __name__ == "__main__":
    main()

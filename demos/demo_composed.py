#!/usr/bin/env python3
"""Demo: composed / chain effects on audio.

Applies autowah, sample-rate reduction, exciter, de-esser, vocal chain,
mastering chain, and STK chorus.
"""

import argparse
import os

import numpy as np

from nanodsp.buffer import AudioBuffer
from nanodsp.effects import (
    autowah,
    sample_rate_reduce,
    dc_block,
    exciter,
    de_esser,
    vocal_chain,
    master,
    stk_chorus,
)


def peak_normalize(buf: AudioBuffer) -> AudioBuffer:
    """Scale so the loudest sample sits at 0 dBFS."""
    peak = np.max(np.abs(buf.data))
    if peak > 0:
        return buf.gain_db(-20.0 * np.log10(peak))
    return buf


def main():
    parser = argparse.ArgumentParser(description="Demo: composed effects")
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
        ("autowah-subtle", lambda b: autowah(b, wah=0.3, dry_wet=0.7)),
        ("autowah-deep", lambda b: autowah(b, wah=0.8, dry_wet=1.0)),
        ("sr-reduce-mild", lambda b: sample_rate_reduce(b, freq=0.7)),
        ("sr-reduce-heavy", lambda b: sample_rate_reduce(b, freq=0.3)),
        ("dc-block", lambda b: dc_block(b)),
        ("exciter-subtle", lambda b: exciter(b, freq=3000.0, amount=0.2)),
        ("exciter-bright", lambda b: exciter(b, freq=2000.0, amount=0.5)),
        ("de-esser", lambda b: de_esser(b, freq=6000.0, threshold_db=-20.0)),
        ("stk-chorus-subtle", lambda b: stk_chorus(b, mod_depth=0.03, mod_freq=0.2, mix=0.3)),
        ("stk-chorus-deep", lambda b: stk_chorus(b, mod_depth=0.08, mod_freq=0.5, mix=0.6)),
        ("vocal-chain", lambda b: vocal_chain(b)),
        ("master-default", lambda b: master(b, target_lufs=-14.0)),
        ("master-loud", lambda b: master(b, target_lufs=-10.0)),
    ]

    for label, fn in demos:
        out = normalize(fn(buf))
        path = os.path.join(args.out_dir, f"{name}_composed_{label}.wav")
        out.write(path)
        print(f"  {label} -> {path}")


if __name__ == "__main__":
    main()

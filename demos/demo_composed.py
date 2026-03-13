#!/usr/bin/env python3
"""Demo: composed / chain effects on audio.

Applies autowah, sample-rate reduction, exciter, de-esser, vocal chain,
mastering chain, STK chorus, and derivative composed effects (shimmer reverb,
tape echo, lo-fi, telephone, gated reverb, auto-pan).
"""

import argparse
import os

import numpy as np

from nanodsp.buffer import AudioBuffer
from nanodsp.effects.composed import (
    auto_pan,
    de_esser,
    exciter,
    gated_reverb,
    lo_fi,
    master,
    shimmer_reverb,
    tape_echo,
    telephone,
    vocal_chain,
)
from nanodsp.effects.daisysp import autowah, sample_rate_reduce, dc_block
from nanodsp.effects.reverb import stk_chorus


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
        # --- Derivative composed effects ---
        ("shimmer-default", lambda b: shimmer_reverb(b)),
        ("shimmer-bright", lambda b: shimmer_reverb(b, shimmer=0.6, shift_semitones=12.0)),
        ("shimmer-fifth", lambda b: shimmer_reverb(b, shimmer=0.4, shift_semitones=7.0)),
        ("tape-echo-default", lambda b: tape_echo(b)),
        ("tape-echo-dark", lambda b: tape_echo(b, tone=1500.0, drive=0.5, feedback=0.6)),
        ("tape-echo-fast", lambda b: tape_echo(b, delay_ms=150.0, feedback=0.4)),
        ("lofi-default", lambda b: lo_fi(b)),
        ("lofi-crushed", lambda b: lo_fi(b, bit_depth=4, reduce=0.7, tone=2000.0)),
        ("telephone-default", lambda b: telephone(b)),
        ("telephone-radio", lambda b: telephone(b, low_cut=500.0, high_cut=5000.0, drive=0.2)),
        ("gated-reverb-default", lambda b: gated_reverb(b)),
        ("gated-reverb-tight", lambda b: gated_reverb(b, gate_threshold_db=-20.0, gate_hold_ms=30.0, gate_release=0.01)),
        ("autopan-default", lambda b: auto_pan(b)),
        ("autopan-slow", lambda b: auto_pan(b, rate=0.5, depth=0.8)),
        ("autopan-fast", lambda b: auto_pan(b, rate=6.0, depth=1.0)),
    ]

    for label, fn in demos:
        out = normalize(fn(buf))
        path = os.path.join(args.out_dir, f"{name}_composed_{label}.wav")
        out.write(path)
        print(f"  {label} -> {path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Demo: delay-based effects on audio.

Applies stereo delay, ping-pong delay, and STK echo at various settings.
"""

import argparse
import os

import numpy as np

from nanodsp.buffer import AudioBuffer
from nanodsp.effects.composed import stereo_delay
from nanodsp.effects.reverb import stk_echo


def peak_normalize(buf: AudioBuffer) -> AudioBuffer:
    """Scale so the loudest sample sits at 0 dBFS."""
    peak = np.max(np.abs(buf.data))
    if peak > 0:
        return buf.gain_db(-20.0 * np.log10(peak))
    return buf


def main():
    parser = argparse.ArgumentParser(description="Demo: delay effects")
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
        ("stereo-short", lambda b: stereo_delay(b, left_ms=150.0, right_ms=200.0, feedback=0.3, mix=0.4)),
        ("stereo-long", lambda b: stereo_delay(b, left_ms=375.0, right_ms=500.0, feedback=0.4, mix=0.4)),
        ("pingpong", lambda b: stereo_delay(b, left_ms=250.0, right_ms=375.0, feedback=0.4, mix=0.5, ping_pong=True)),
        ("pingpong-fast", lambda b: stereo_delay(b, left_ms=125.0, right_ms=187.0, feedback=0.5, mix=0.5, ping_pong=True)),
        ("slapback", lambda b: stereo_delay(b, left_ms=80.0, right_ms=80.0, feedback=0.0, mix=0.4)),
        ("echo-quarter", lambda b: stk_echo(b, delay_ms=250.0, mix=0.4)),
        ("echo-dotted-eighth", lambda b: stk_echo(b, delay_ms=187.5, mix=0.4)),
        ("echo-long", lambda b: stk_echo(b, delay_ms=500.0, mix=0.35)),
    ]

    for label, fn in demos:
        out = normalize(fn(buf))
        path = os.path.join(args.out_dir, f"{name}_delay_{label}.wav")
        out.write(path)
        print(f"  {label} -> {path}")


if __name__ == "__main__":
    main()

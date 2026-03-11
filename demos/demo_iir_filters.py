#!/usr/bin/env python3
"""Demo: multi-order IIR filter design.

Applies Butterworth, Chebyshev I/II, Elliptic, and Bessel filters at
various orders and configurations to show the difference in filter
characteristics.
"""

import argparse
import os

import numpy as np

from nanodsp.buffer import AudioBuffer
from nanodsp.effects import iir_filter


def peak_normalize(buf: AudioBuffer) -> AudioBuffer:
    """Scale so the loudest sample sits at 0 dBFS."""
    peak = np.max(np.abs(buf.data))
    if peak > 0:
        return buf.gain_db(-20.0 * np.log10(peak))
    return buf


def main():
    parser = argparse.ArgumentParser(description="Demo: IIR filter design")
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
        # --- Butterworth (maximally flat passband) ---
        ("butter-lp2-1k", dict(family="butter", filter_type="lp", order=2, freq=1000)),
        ("butter-lp4-1k", dict(family="butter", filter_type="lp", order=4, freq=1000)),
        ("butter-lp8-1k", dict(family="butter", filter_type="lp", order=8, freq=1000)),
        ("butter-hp4-500", dict(family="butter", filter_type="hp", order=4, freq=500)),
        ("butter-bp4-1k", dict(family="butter", filter_type="bp", order=4, freq=1000, width=500)),
        ("butter-bs4-1k", dict(family="butter", filter_type="bs", order=4, freq=1000, width=500)),
        # --- Chebyshev I (sharper rolloff, passband ripple) ---
        ("cheby1-lp4-1k-1db", dict(family="cheby1", filter_type="lp", order=4, freq=1000, ripple_db=1.0)),
        ("cheby1-lp4-1k-3db", dict(family="cheby1", filter_type="lp", order=4, freq=1000, ripple_db=3.0)),
        ("cheby1-hp4-500", dict(family="cheby1", filter_type="hp", order=4, freq=500, ripple_db=1.0)),
        ("cheby1-bp4-1k", dict(family="cheby1", filter_type="bp", order=4, freq=1000, width=500, ripple_db=1.0)),
        # --- Chebyshev II (flat passband, stopband ripple) ---
        ("cheby2-lp4-1k-40db", dict(family="cheby2", filter_type="lp", order=4, freq=1000, ripple_db=40.0)),
        ("cheby2-lp4-1k-60db", dict(family="cheby2", filter_type="lp", order=4, freq=1000, ripple_db=60.0)),
        ("cheby2-hp4-500", dict(family="cheby2", filter_type="hp", order=4, freq=500, ripple_db=40.0)),
        # --- Elliptic (sharpest transition) ---
        ("ellip-lp4-1k", dict(family="ellip", filter_type="lp", order=4, freq=1000, ripple_db=1.0)),
        ("ellip-hp4-500", dict(family="ellip", filter_type="hp", order=4, freq=500, ripple_db=1.0)),
        ("ellip-bp4-1k", dict(family="ellip", filter_type="bp", order=4, freq=1000, width=500, ripple_db=1.0)),
        # --- Bessel (linear phase, minimal ringing) ---
        ("bessel-lp4-1k", dict(family="bessel", filter_type="lp", order=4, freq=1000)),
        ("bessel-lp8-1k", dict(family="bessel", filter_type="lp", order=8, freq=1000)),
        ("bessel-hp4-500", dict(family="bessel", filter_type="hp", order=4, freq=500)),
        ("bessel-bp4-1k", dict(family="bessel", filter_type="bp", order=4, freq=1000, width=500)),
        # --- Order comparison at same cutoff ---
        ("butter-lp2-3k", dict(family="butter", filter_type="lp", order=2, freq=3000)),
        ("butter-lp6-3k", dict(family="butter", filter_type="lp", order=6, freq=3000)),
        ("butter-lp12-3k", dict(family="butter", filter_type="lp", order=12, freq=3000)),
    ]

    for label, kwargs in demos:
        out = normalize(iir_filter(buf, **kwargs))
        path = os.path.join(args.out_dir, f"{name}_iir-{label}.wav")
        out.write(path)
        print(f"  {label} -> {path}")


if __name__ == "__main__":
    main()

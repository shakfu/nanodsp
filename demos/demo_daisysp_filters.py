#!/usr/bin/env python3
"""Demo: DaisySP filter algorithms on audio.

Applies state-variable filters, ladder/Moog filters, tone filters,
modal resonator, and comb filter at various settings.
"""

import argparse
import os

import numpy as np

from nanodsp.buffer import AudioBuffer
from nanodsp.effects import (
    svf_lowpass,
    svf_highpass,
    svf_bandpass,
    svf_notch,
    svf_peak,
    ladder_filter,
    moog_ladder,
    tone_lowpass,
    tone_highpass,
    modal_bandpass,
    comb_filter,
)


def peak_normalize(buf: AudioBuffer) -> AudioBuffer:
    """Scale so the loudest sample sits at 0 dBFS."""
    peak = np.max(np.abs(buf.data))
    if peak > 0:
        return buf.gain_db(-20.0 * np.log10(peak))
    return buf


def main():
    parser = argparse.ArgumentParser(description="Demo: DaisySP filters")
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
        # State-variable filter
        ("svf-lp-1k", lambda b: svf_lowpass(b, freq_hz=1000.0, resonance=0.3)),
        ("svf-lp-resonant", lambda b: svf_lowpass(b, freq_hz=800.0, resonance=0.8)),
        ("svf-hp-500", lambda b: svf_highpass(b, freq_hz=500.0)),
        ("svf-bp-1k", lambda b: svf_bandpass(b, freq_hz=1000.0, resonance=0.5)),
        ("svf-notch-1k", lambda b: svf_notch(b, freq_hz=1000.0, resonance=0.5)),
        ("svf-peak-2k", lambda b: svf_peak(b, freq_hz=2000.0, resonance=0.7)),
        # Ladder filter
        ("ladder-lp24", lambda b: ladder_filter(b, freq_hz=1000.0, resonance=0.5, mode="lp24")),
        ("ladder-lp12", lambda b: ladder_filter(b, freq_hz=1500.0, resonance=0.3, mode="lp12")),
        ("ladder-bp24", lambda b: ladder_filter(b, freq_hz=1000.0, resonance=0.5, mode="bp24")),
        ("ladder-hp24", lambda b: ladder_filter(b, freq_hz=500.0, resonance=0.3, mode="hp24")),
        ("ladder-resonant", lambda b: ladder_filter(b, freq_hz=800.0, resonance=0.9, mode="lp24")),
        # Moog ladder
        ("moog-1k", lambda b: moog_ladder(b, freq_hz=1000.0, resonance=0.3)),
        ("moog-resonant", lambda b: moog_ladder(b, freq_hz=600.0, resonance=0.8)),
        # Tone (one-pole) filters
        ("tone-lp-2k", lambda b: tone_lowpass(b, freq_hz=2000.0)),
        ("tone-lp-500", lambda b: tone_lowpass(b, freq_hz=500.0)),
        ("tone-hp-500", lambda b: tone_highpass(b, freq_hz=500.0)),
        ("tone-hp-2k", lambda b: tone_highpass(b, freq_hz=2000.0)),
        # Modal bandpass
        ("modal-440", lambda b: modal_bandpass(b, freq_hz=440.0, q=200.0)),
        ("modal-1k", lambda b: modal_bandpass(b, freq_hz=1000.0, q=500.0)),
        # Comb filter
        ("comb-500", lambda b: comb_filter(b, freq_hz=500.0, rev_time=0.3)),
        ("comb-200", lambda b: comb_filter(b, freq_hz=200.0, rev_time=0.5)),
    ]

    for label, fn in demos:
        out = normalize(fn(buf))
        path = os.path.join(args.out_dir, f"{name}_dsyfilt_{label}.wav")
        out.write(path)
        print(f"  {label} -> {path}")


if __name__ == "__main__":
    main()

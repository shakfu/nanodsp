#!/usr/bin/env python3
"""Demo: spectral-domain processing on audio.

Applies time stretching, spectral gating, spectral emphasis (tilt EQ),
and spectral freeze via STFT/ISTFT round-trips.
"""

import argparse
import os

import numpy as np

from nanodsp.buffer import AudioBuffer
from nanodsp.spectral import (
    stft,
    istft,
    time_stretch,
    spectral_gate,
    spectral_emphasis,
    spectral_freeze,
    phase_lock,
)


def peak_normalize(buf: AudioBuffer) -> AudioBuffer:
    """Scale so the loudest sample sits at 0 dBFS."""
    peak = np.max(np.abs(buf.data))
    if peak > 0:
        return buf.gain_db(-20.0 * np.log10(peak))
    return buf


def main():
    parser = argparse.ArgumentParser(description="Demo: spectral processing")
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

    window_size = 2048

    # Time stretching
    for rate, label in [(0.5, "half-speed"), (0.75, "075x"), (1.5, "150x"), (2.0, "double-speed")]:
        spec = stft(buf, window_size=window_size)
        stretched = time_stretch(spec, rate)
        out = normalize(istft(stretched))
        path = os.path.join(args.out_dir, f"{name}_spectral_stretch-{label}.wav")
        out.write(path)
        print(f"  stretch {label} -> {path}")

    # Time stretching with phase locking (reduced phasiness)
    spec = stft(buf, window_size=window_size)
    stretched = time_stretch(spec, 0.5)
    locked = phase_lock(stretched)
    out = normalize(istft(locked))
    path = os.path.join(args.out_dir, f"{name}_spectral_stretch-half-phaselocked.wav")
    out.write(path)
    print(f"  stretch half-speed + phase-lock -> {path}")

    # Spectral gate
    for thresh, label in [(-30.0, "gentle"), (-20.0, "moderate"), (-10.0, "aggressive")]:
        spec = stft(buf, window_size=window_size)
        gated = spectral_gate(spec, threshold_db=thresh)
        out = normalize(istft(gated))
        path = os.path.join(args.out_dir, f"{name}_spectral_gate-{label}.wav")
        out.write(path)
        print(f"  gate {label} -> {path}")

    # Spectral emphasis (tilt EQ)
    for low, high, label in [(0.0, 6.0, "bright"), (0.0, -6.0, "dark"), (6.0, -6.0, "warm")]:
        spec = stft(buf, window_size=window_size)
        tilted = spectral_emphasis(spec, low_db=low, high_db=high)
        out = normalize(istft(tilted))
        path = os.path.join(args.out_dir, f"{name}_spectral_tilt-{label}.wav")
        out.write(path)
        print(f"  tilt {label} -> {path}")

    # Spectral freeze (repeats a single frame)
    spec = stft(buf, window_size=window_size)
    mid_frame = spec.num_frames // 2
    frozen = spectral_freeze(spec, frame_index=mid_frame)
    out = normalize(istft(frozen))
    path = os.path.join(args.out_dir, f"{name}_spectral_freeze-mid.wav")
    out.write(path)
    print(f"  freeze (mid-frame) -> {path}")


if __name__ == "__main__":
    main()

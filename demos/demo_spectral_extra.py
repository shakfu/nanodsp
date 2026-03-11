#!/usr/bin/env python3
"""Demo: additional spectral transforms on audio.

Applies spectral denoising, EQ matching, and spectral morphing.
"""

import argparse
import os

import numpy as np

from nanodsp.buffer import AudioBuffer
from nanodsp.effects import lowpass, highpass
from nanodsp.spectral import (
    stft,
    istft,
    spectral_denoise,
    spectral_morph,
    eq_match,
)


def peak_normalize(buf: AudioBuffer) -> AudioBuffer:
    """Scale so the loudest sample sits at 0 dBFS."""
    peak = np.max(np.abs(buf.data))
    if peak > 0:
        return buf.gain_db(-20.0 * np.log10(peak))
    return buf


def main():
    parser = argparse.ArgumentParser(description="Demo: spectral transforms (extra)")
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

    # Spectral denoise (uses first N frames as noise profile)
    for noise_frames, reduction, label in [
        (10, -15.0, "gentle"),
        (10, -30.0, "moderate"),
        (20, -40.0, "aggressive"),
    ]:
        spec = stft(buf, window_size=window_size)
        denoised = spectral_denoise(spec, noise_frames=noise_frames, reduction_db=reduction)
        out = normalize(istft(denoised))
        path = os.path.join(args.out_dir, f"{name}_specextra_denoise-{label}.wav")
        out.write(path)
        print(f"  denoise {label} -> {path}")

    # EQ match: match the input to a spectrally-modified target
    # Create targets by filtering, then match original toward that spectrum
    mono = buf.to_mono() if buf.channels > 1 else buf
    dark_target = lowpass(mono, 2000.0)
    matched_dark = eq_match(mono, dark_target, window_size=4096, smoothing=5)
    out = normalize(matched_dark)
    path = os.path.join(args.out_dir, f"{name}_specextra_eqmatch-dark.wav")
    out.write(path)
    print(f"  eq-match dark -> {path}")

    bright_target = highpass(mono, 500.0)
    matched_bright = eq_match(mono, bright_target, window_size=4096, smoothing=5)
    out = normalize(matched_bright)
    path = os.path.join(args.out_dir, f"{name}_specextra_eqmatch-bright.wav")
    out.write(path)
    print(f"  eq-match bright -> {path}")

    # Spectral morph: morph between original and a filtered version
    filtered = lowpass(mono, 1000.0)
    spec_a = stft(mono, window_size=window_size)
    spec_b = stft(filtered, window_size=window_size)
    for mix_val, label in [(0.25, "25pct"), (0.5, "50pct"), (0.75, "75pct")]:
        morphed = spectral_morph(spec_a, spec_b, mix=mix_val)
        out = normalize(istft(morphed))
        path = os.path.join(args.out_dir, f"{name}_specextra_morph-{label}.wav")
        out.write(path)
        print(f"  morph {label} -> {path}")


if __name__ == "__main__":
    main()

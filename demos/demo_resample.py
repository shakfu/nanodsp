#!/usr/bin/env python3
"""Demo: resampling algorithms on audio.

Resamples audio to different sample rates using madronalib-backed
and FFT-based methods.
"""

import argparse
import os

import numpy as np

from nanodsp.buffer import AudioBuffer
from nanodsp.analysis import resample, resample_fft


def peak_normalize(buf: AudioBuffer) -> AudioBuffer:
    """Scale so the loudest sample sits at 0 dBFS."""
    peak = np.max(np.abs(buf.data))
    if peak > 0:
        return buf.gain_db(-20.0 * np.log10(peak))
    return buf


def main():
    parser = argparse.ArgumentParser(description="Demo: resampling")
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
    orig_sr = buf.sample_rate

    # Madronalib-backed resample
    for target_sr, label in [
        (22050.0, "22k"),
        (44100.0, "44k"),
        (48000.0, "48k"),
        (96000.0, "96k"),
    ]:
        if target_sr == orig_sr:
            continue
        out = normalize(resample(buf, target_sr=target_sr))
        path = os.path.join(args.out_dir, f"{name}_resample_ml-{label}.wav")
        out.write(path)
        print(f"  resample (madronalib) -> {label} ({out.frames} frames) -> {path}")

    # FFT-based resample
    for target_sr, label in [
        (22050.0, "22k"),
        (44100.0, "44k"),
        (48000.0, "48k"),
        (96000.0, "96k"),
    ]:
        if target_sr == orig_sr:
            continue
        out = normalize(resample_fft(buf, target_sr=target_sr))
        path = os.path.join(args.out_dir, f"{name}_resample_fft-{label}.wav")
        out.write(path)
        print(f"  resample (FFT) -> {label} ({out.frames} frames) -> {path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Demo: audio analysis (prints measurements to stdout).

Runs loudness metering, spectral feature extraction, pitch detection,
and onset detection on the input file.
"""

import argparse
import os

import numpy as np

from nanodsp.buffer import AudioBuffer
from nanodsp.analysis import (
    loudness_lufs,
    spectral_centroid,
    spectral_bandwidth,
    spectral_rolloff,
    spectral_flux,
    spectral_flatness_curve,
    chromagram,
    pitch_detect,
    onset_detect,
)


def main():
    parser = argparse.ArgumentParser(description="Demo: audio analysis")
    parser.add_argument("infile", help="Input .wav file")
    args = parser.parse_args()

    buf = AudioBuffer.from_file(args.infile)
    sr = buf.sample_rate
    print(f"File: {args.infile}")
    print(f"  channels={buf.channels}  frames={buf.frames}  sample_rate={sr}  duration={buf.duration:.3f}s")
    print(f"  peak={np.max(np.abs(buf.data)):.4f} ({20*np.log10(np.max(np.abs(buf.data))+1e-20):.1f} dBFS)")

    # Loudness
    lufs = loudness_lufs(buf)
    print(f"\nLoudness: {lufs:.1f} LUFS")

    # Spectral centroid (mean over time)
    centroid = spectral_centroid(buf)
    if centroid.ndim > 1:
        centroid = np.mean(centroid, axis=0)
    print(f"\nSpectral centroid:  mean={np.mean(centroid):.0f} Hz  std={np.std(centroid):.0f} Hz")

    # Spectral bandwidth
    bw = spectral_bandwidth(buf)
    if bw.ndim > 1:
        bw = np.mean(bw, axis=0)
    print(f"Spectral bandwidth: mean={np.mean(bw):.0f} Hz  std={np.std(bw):.0f} Hz")

    # Spectral rolloff
    rolloff = spectral_rolloff(buf)
    if rolloff.ndim > 1:
        rolloff = np.mean(rolloff, axis=0)
    print(f"Spectral rolloff:   mean={np.mean(rolloff):.0f} Hz  std={np.std(rolloff):.0f} Hz")

    # Spectral flux
    flux = spectral_flux(buf)
    if flux.ndim > 1:
        flux = np.mean(flux, axis=0)
    print(f"Spectral flux:      mean={np.mean(flux):.2f}  max={np.max(flux):.2f}")

    # Spectral flatness
    flatness = spectral_flatness_curve(buf)
    if flatness.ndim > 1:
        flatness = np.mean(flatness, axis=0)
    print(f"Spectral flatness:  mean={np.mean(flatness):.4f}  (0=tonal, 1=noise)")

    # Chromagram (mean energy per pitch class)
    chroma = chromagram(buf)
    if chroma.ndim == 3:
        chroma = np.mean(chroma, axis=0)
    mean_chroma = np.mean(chroma, axis=1)
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    chroma_str = "  ".join(f"{n}={v:.2f}" for n, v in zip(note_names, mean_chroma))
    print(f"\nChromagram (mean): {chroma_str}")
    dominant = note_names[int(np.argmax(mean_chroma))]
    print(f"  dominant pitch class: {dominant}")

    # Pitch detection
    freqs, confs = pitch_detect(buf, fmin=50.0, fmax=2000.0)
    if freqs.ndim > 1:
        freqs = np.mean(freqs, axis=0)
        confs = np.mean(confs, axis=0)
    voiced = freqs[freqs > 0]
    if len(voiced) > 0:
        print(f"\nPitch (YIN): {len(voiced)}/{len(freqs)} voiced frames")
        print(f"  mean={np.mean(voiced):.1f} Hz  median={np.median(voiced):.1f} Hz  range=[{np.min(voiced):.1f}, {np.max(voiced):.1f}] Hz")
    else:
        print(f"\nPitch (YIN): no voiced frames detected")

    # Onset detection
    onsets = onset_detect(buf)
    onset_times = onsets / sr
    print(f"\nOnsets: {len(onsets)} detected")
    if len(onsets) > 0:
        shown = onset_times[:20]
        times_str = ", ".join(f"{t:.3f}s" for t in shown)
        suffix = f" ... ({len(onsets)} total)" if len(onsets) > 20 else ""
        print(f"  times: [{times_str}{suffix}]")


if __name__ == "__main__":
    main()

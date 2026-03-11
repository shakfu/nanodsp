#!/usr/bin/env python3
"""Demo: core DSP operations on audio.

Applies delay, vibrato (varying delay), convolution, envelope followers,
fades, panning, stereo widening, and crossfade.
"""

import argparse
import os

import numpy as np

from nanodsp.buffer import AudioBuffer
from nanodsp import ops


def peak_normalize(buf: AudioBuffer) -> AudioBuffer:
    """Scale so the loudest sample sits at 0 dBFS."""
    peak = np.max(np.abs(buf.data))
    if peak > 0:
        return buf.gain_db(-20.0 * np.log10(peak))
    return buf


def main():
    parser = argparse.ArgumentParser(description="Demo: core DSP operations")
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
    sr = buf.sample_rate

    # --- Delay ---
    for samples, interp, label in [
        (int(sr * 0.1), "linear", "100ms-linear"),
        (int(sr * 0.25), "linear", "250ms-linear"),
        (int(sr * 0.1), "cubic", "100ms-cubic"),
    ]:
        out = normalize(ops.delay(buf, delay_samples=samples, interpolation=interp))
        path = os.path.join(args.out_dir, f"{name}_ops_delay-{label}.wav")
        out.write(path)
        print(f"  delay {label} -> {path}")

    # --- Vibrato (time-varying delay) ---
    lfo_rate = 5.0 / sr  # 5 Hz vibrato
    t = np.arange(buf.frames, dtype=np.float32)
    delay_curve = 200.0 + 100.0 * np.sin(2.0 * np.pi * lfo_rate * t)
    out = normalize(ops.delay_varying(buf, delays=delay_curve, interpolation="cubic"))
    path = os.path.join(args.out_dir, f"{name}_ops_vibrato.wav")
    out.write(path)
    print(f"  vibrato -> {path}")

    # --- Convolution (with a synthetic impulse response) ---
    # Short reverb-like IR: decaying noise burst
    ir_len = int(sr * 0.3)
    ir_data = np.random.default_rng(42).normal(0, 1, ir_len).astype(np.float32)
    ir_data *= np.exp(-np.linspace(0, 8, ir_len)).astype(np.float32)
    ir_buf = AudioBuffer(ir_data.reshape(1, -1), sample_rate=sr)
    out = normalize(ops.convolve(buf, ir_buf, normalize=True))
    path = os.path.join(args.out_dir, f"{name}_ops_convolve-ir.wav")
    out.write(path)
    print(f"  convolve (synthetic IR) -> {path}")

    # --- Envelope followers ---
    for length, label in [(128, "short"), (1024, "long")]:
        out = normalize(ops.box_filter(buf, length=length))
        path = os.path.join(args.out_dir, f"{name}_ops_boxfilter-{label}.wav")
        out.write(path)
        print(f"  box-filter {label} -> {path}")

    out = normalize(ops.box_stack_filter(buf, size=64, layers=4))
    path = os.path.join(args.out_dir, f"{name}_ops_boxstack.wav")
    out.write(path)
    print(f"  box-stack-filter -> {path}")

    out = normalize(ops.peak_hold(buf, length=256))
    path = os.path.join(args.out_dir, f"{name}_ops_peakhold.wav")
    out.write(path)
    print(f"  peak-hold -> {path}")

    out = normalize(ops.peak_decay(buf, length=512))
    path = os.path.join(args.out_dir, f"{name}_ops_peakdecay.wav")
    out.write(path)
    print(f"  peak-decay -> {path}")

    # --- Fades ---
    for dur, curve, label in [
        (500.0, "linear", "in-linear"),
        (500.0, "ease_in", "in-easein"),
        (500.0, "smoothstep", "in-smooth"),
    ]:
        out = normalize(ops.fade_in(buf, duration_ms=dur, curve=curve))
        path = os.path.join(args.out_dir, f"{name}_ops_fade-{label}.wav")
        out.write(path)
        print(f"  fade {label} -> {path}")

    for dur, curve, label in [
        (500.0, "linear", "out-linear"),
        (500.0, "ease_out", "out-easeout"),
        (500.0, "smoothstep", "out-smooth"),
    ]:
        out = normalize(ops.fade_out(buf, duration_ms=dur, curve=curve))
        path = os.path.join(args.out_dir, f"{name}_ops_fade-{label}.wav")
        out.write(path)
        print(f"  fade {label} -> {path}")

    # --- Panning (mono input -> stereo output) ---
    mono = buf.to_mono() if buf.channels > 1 else buf
    for pos, label in [(-0.8, "left"), (0.0, "center"), (0.8, "right")]:
        out = normalize(ops.pan(mono, position=pos))
        path = os.path.join(args.out_dir, f"{name}_ops_pan-{label}.wav")
        out.write(path)
        print(f"  pan {label} -> {path}")

    # --- Stereo widening (needs stereo input) ---
    stereo = buf if buf.channels == 2 else buf.to_channels(2)
    for width, label in [(0.0, "mono"), (1.0, "normal"), (2.0, "wide"), (3.0, "extrawide")]:
        out = normalize(ops.stereo_widen(stereo, width=width))
        path = os.path.join(args.out_dir, f"{name}_ops_width-{label}.wav")
        out.write(path)
        print(f"  stereo-widen {label} -> {path}")

    # --- Crossfade (between original and filtered version) ---
    from nanodsp.effects import lowpass
    filtered = lowpass(buf, 800.0)
    for x, label in [(0.25, "25pct"), (0.5, "50pct"), (0.75, "75pct")]:
        out = normalize(ops.crossfade(buf, filtered, x=x))
        path = os.path.join(args.out_dir, f"{name}_ops_crossfade-{label}.wav")
        out.write(path)
        print(f"  crossfade {label} -> {path}")

    # --- Normalize peak ---
    out = ops.normalize_peak(buf, target_db=-6.0)
    path = os.path.join(args.out_dir, f"{name}_ops_normpeak-6db.wav")
    out.write(path)
    print(f"  normalize-peak -6dB -> {path}")

    # --- Trim silence ---
    out = normalize(ops.trim_silence(buf, threshold_db=-50.0, pad_frames=1024))
    path = os.path.join(args.out_dir, f"{name}_ops_trimsilence.wav")
    out.write(path)
    print(f"  trim-silence -> {path}")

    # --- Upsample 2x roundtrip ---
    out = normalize(ops.oversample_roundtrip(buf))
    path = os.path.join(args.out_dir, f"{name}_ops_oversample-roundtrip.wav")
    out.write(path)
    print(f"  oversample-roundtrip -> {path}")


if __name__ == "__main__":
    main()

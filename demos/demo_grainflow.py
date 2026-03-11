#!/usr/bin/env python3
"""Demo: granular synthesis with GrainflowLib.

Demonstrates grain clouds, time-stretching, pitch-shifting, granular recording,
and stereo panning using the low-level GrainflowLib bindings.
"""

import argparse
import os

import numpy as np

from nanodsp.buffer import AudioBuffer
from nanodsp._core import grainflow as gf


def peak_normalize(buf: AudioBuffer) -> AudioBuffer:
    """Scale so the loudest sample sits at 0 dBFS."""
    peak = np.max(np.abs(buf.data))
    if peak > 0:
        return buf.gain_db(-20.0 * np.log10(peak))
    return buf


def load_source_buffer(path: str) -> tuple[gf.GfBuffer, int, int]:
    """Load a wav file into a GfBuffer. Returns (gf_buf, frames, samplerate)."""
    buf = AudioBuffer.from_file(path)
    sr = int(buf.sample_rate)
    # Mix to mono for simplicity
    if buf.channels > 1:
        mono = buf.data.mean(axis=0, keepdims=True).astype(np.float32)
    else:
        mono = buf.data
    frames = mono.shape[1]
    gf_buf = gf.GfBuffer(frames, 1, sr)
    gf_buf.set_data(mono)
    return gf_buf, frames, sr


def granulate(
    source: gf.GfBuffer,
    sr: int,
    duration_s: float,
    num_grains: int,
    grain_rate_hz: float,
    rate: float = 1.0,
    transpose_st: float = 0.0,
    density: float = 1.0,
    random_delay_ms: float = 0.0,
    space: float = 0.0,
) -> np.ndarray:
    """Run granular processing and return mono output as float32 array."""
    gc = gf.GrainCollection(num_grains, sr)
    gc.set_buffer(source, gf.BUF_BUFFER, 0)

    gc.param_set(0, gf.PARAM_AMPLITUDE, gf.PTYPE_BASE, 1.0)
    gc.param_set(0, gf.PARAM_RATE, gf.PTYPE_BASE, rate)
    gc.param_set(0, gf.PARAM_DENSITY, gf.PTYPE_BASE, density)
    gc.param_set(0, gf.PARAM_SPACE, gf.PTYPE_BASE, space)
    if transpose_st != 0.0:
        gc.param_set(0, gf.PARAM_TRANSPOSE, gf.PTYPE_BASE, transpose_st)
    if random_delay_ms > 0.0:
        gc.param_set(0, gf.PARAM_DELAY, gf.PTYPE_RANDOM, random_delay_ms)

    total_frames = int(sr * duration_s)
    block_size = 256  # must be multiple of 64
    num_blocks = total_frames // block_size
    output = np.zeros(num_blocks * block_size, dtype=np.float32)

    phasor = gf.Phasor(grain_rate_hz, sr)

    for b in range(num_blocks):
        clock = phasor.perform(block_size).reshape(1, block_size)
        # Traversal: slow linear scan through buffer
        t_start = (b * block_size) / (num_blocks * block_size)
        t_end = ((b + 1) * block_size) / (num_blocks * block_size)
        traversal = np.linspace(t_start, t_end, block_size, dtype=np.float32).reshape(
            1, block_size
        )
        fm = np.zeros((1, block_size), dtype=np.float32)
        am = np.zeros((1, block_size), dtype=np.float32)

        result = gc.process(clock, traversal, fm, am, sr)
        grain_output = result[0]  # [num_grains, block_size]

        # Sum all grains
        mixed = grain_output.sum(axis=0)
        output[b * block_size : (b + 1) * block_size] = mixed

    return output


def granulate_stereo_panned(
    source: gf.GfBuffer,
    sr: int,
    duration_s: float,
    num_grains: int,
    grain_rate_hz: float,
    pan_spread: float = 0.5,
) -> np.ndarray:
    """Granulate with stereo panning. Returns [2, frames] float32."""
    gc = gf.GrainCollection(num_grains, sr)
    gc.set_buffer(source, gf.BUF_BUFFER, 0)

    gc.param_set(0, gf.PARAM_AMPLITUDE, gf.PTYPE_BASE, 1.0)
    gc.param_set(0, gf.PARAM_RATE, gf.PTYPE_BASE, 1.0)

    panner = gf.Panner(num_grains, 2, gf.PAN_STEREO)
    panner.set_pan_position(0.5)
    panner.set_pan_spread(pan_spread)

    total_frames = int(sr * duration_s)
    block_size = 256
    num_blocks = total_frames // block_size
    output = np.zeros((2, num_blocks * block_size), dtype=np.float32)

    phasor = gf.Phasor(grain_rate_hz, sr)

    for b in range(num_blocks):
        clock = phasor.perform(block_size).reshape(1, block_size)
        t_start = (b * block_size) / (num_blocks * block_size)
        t_end = ((b + 1) * block_size) / (num_blocks * block_size)
        traversal = np.linspace(t_start, t_end, block_size, dtype=np.float32).reshape(
            1, block_size
        )
        fm = np.zeros((1, block_size), dtype=np.float32)
        am = np.zeros((1, block_size), dtype=np.float32)

        result = gc.process(clock, traversal, fm, am, sr)
        grain_output = result[0]  # [num_grains, block_size]
        grain_state = result[1]  # [num_grains, block_size]

        panned = panner.process(grain_output, grain_state, 2)  # [2, block_size]
        output[:, b * block_size : (b + 1) * block_size] = panned

    return output


def demo_recording(sr: int, duration_s: float) -> np.ndarray:
    """Record a generated signal into a GrainflowLib buffer, then read it back."""
    rec = gf.Recorder(sr)
    buf_frames = int(sr * duration_s)
    rec.set_target(buf_frames, 1, sr)
    rec.state = True
    rec.overdub = 0.0

    block_size = 256
    num_blocks = buf_frames // block_size

    # Feed a sine wave into the recorder
    t = np.arange(buf_frames, dtype=np.float32) / sr
    sine = (np.sin(2 * np.pi * 440.0 * t) * 0.5).astype(np.float32)

    for b in range(num_blocks):
        chunk = sine[b * block_size : (b + 1) * block_size].reshape(1, block_size)
        rec.process(chunk)

    return rec.get_buffer_data()  # [1, buf_frames]


def main():
    parser = argparse.ArgumentParser(description="Demo: granular synthesis")
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
    name = os.path.splitext(os.path.basename(args.infile))[0]

    source, source_frames, sr = load_source_buffer(args.infile)
    duration = 3.0

    # --- 1. Basic grain cloud ---
    out = granulate(source, sr, duration, num_grains=8, grain_rate_hz=15.0)
    buf = normalize(AudioBuffer(out.reshape(1, -1), sample_rate=float(sr)))
    path = os.path.join(args.out_dir, f"grainflow_{name}_cloud-basic.wav")
    buf.write(path)
    print(f"  grain cloud (basic) -> {path}")

    # --- 2. Dense cloud (many grains, high rate) ---
    out = granulate(source, sr, duration, num_grains=16, grain_rate_hz=30.0, space=0.2)
    buf = normalize(AudioBuffer(out.reshape(1, -1), sample_rate=float(sr)))
    path = os.path.join(args.out_dir, f"grainflow_{name}_cloud-dense.wav")
    buf.write(path)
    print(f"  grain cloud (dense) -> {path}")

    # --- 3. Pitch-shifted (+7 semitones) ---
    out = granulate(
        source, sr, duration, num_grains=8, grain_rate_hz=15.0, transpose_st=7.0
    )
    buf = normalize(AudioBuffer(out.reshape(1, -1), sample_rate=float(sr)))
    path = os.path.join(args.out_dir, f"grainflow_{name}_pitch-up7.wav")
    buf.write(path)
    print(f"  pitch shift +7st -> {path}")

    # --- 4. Pitch-shifted (-12 semitones / octave down) ---
    out = granulate(
        source, sr, duration, num_grains=8, grain_rate_hz=15.0, transpose_st=-12.0
    )
    buf = normalize(AudioBuffer(out.reshape(1, -1), sample_rate=float(sr)))
    path = os.path.join(args.out_dir, f"grainflow_{name}_pitch-down12.wav")
    buf.write(path)
    print(f"  pitch shift -12st -> {path}")

    # --- 5. Sparse / stochastic (low density) ---
    out = granulate(
        source,
        sr,
        duration,
        num_grains=8,
        grain_rate_hz=10.0,
        density=0.4,
        random_delay_ms=50.0,
    )
    buf = normalize(AudioBuffer(out.reshape(1, -1), sample_rate=float(sr)))
    path = os.path.join(args.out_dir, f"grainflow_{name}_sparse.wav")
    buf.write(path)
    print(f"  sparse stochastic -> {path}")

    # --- 6. Stereo panned grain cloud ---
    out = granulate_stereo_panned(
        source, sr, duration, num_grains=8, grain_rate_hz=15.0, pan_spread=0.8
    )
    buf = normalize(AudioBuffer(out, sample_rate=float(sr)))
    path = os.path.join(args.out_dir, f"grainflow_{name}_stereo-panned.wav")
    buf.write(path)
    print(f"  stereo panned -> {path}")

    # --- 7. Recording demo ---
    rec_data = demo_recording(sr, 1.0)
    buf = normalize(AudioBuffer(rec_data, sample_rate=float(sr)))
    path = os.path.join(args.out_dir, "grainflow_recorder.wav")
    buf.write(path)
    print(f"  recorder -> {path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Demo: sound synthesis (no input file required).

Generates oscillators, FM, noise, drums, physical modeling sounds,
and STK instrument notes.
"""

import argparse
import os

import numpy as np

from nanodsp.buffer import AudioBuffer
from nanodsp import synthesis


def peak_normalize(buf: AudioBuffer) -> AudioBuffer:
    """Scale so the loudest sample sits at 0 dBFS."""
    peak = np.max(np.abs(buf.data))
    if peak > 0:
        return buf.gain_db(-20.0 * np.log10(peak))
    return buf


def main():
    parser = argparse.ArgumentParser(description="Demo: sound synthesis")
    parser.add_argument(
        "-o", "--out-dir", default="build/demo-output", help="Output directory"
    )
    parser.add_argument(
        "-n", "--no-normalize", action="store_true", help="Skip peak normalization"
    )
    args = parser.parse_args()

    normalize = (lambda b: b) if args.no_normalize else peak_normalize
    os.makedirs(args.out_dir, exist_ok=True)

    sr = 48000.0
    frames_1s = int(sr)
    frames_half = int(sr * 0.5)

    # --- Oscillators ---
    for wf in ["sine", "tri", "saw", "square", "polyblep_saw", "polyblep_square"]:
        out = normalize(synthesis.oscillator(frames_1s, freq=440.0, waveform=wf, sample_rate=sr))
        path = os.path.join(args.out_dir, f"synth_osc-{wf}.wav")
        out.write(path)
        print(f"  oscillator {wf} -> {path}")

    # Band-limited oscillator
    for wf in ["saw", "square", "tri"]:
        out = normalize(synthesis.bl_oscillator(frames_1s, freq=440.0, waveform=wf, sample_rate=sr))
        path = os.path.join(args.out_dir, f"synth_blosc-{wf}.wav")
        out.write(path)
        print(f"  bl_oscillator {wf} -> {path}")

    # --- FM synthesis ---
    for ratio, index, label in [(2.0, 1.0, "2x-mild"), (3.0, 4.0, "3x-bright"), (1.414, 2.0, "inharmonic")]:
        out = normalize(synthesis.fm2(frames_1s, freq=220.0, ratio=ratio, index=index, sample_rate=sr))
        path = os.path.join(args.out_dir, f"synth_fm-{label}.wav")
        out.write(path)
        print(f"  fm2 {label} -> {path}")

    # Formant oscillator
    for ff, label in [(800.0, "800hz"), (1500.0, "1500hz"), (3000.0, "3000hz")]:
        out = normalize(synthesis.formant_oscillator(frames_1s, carrier_freq=220.0, formant_freq=ff, sample_rate=sr))
        path = os.path.join(args.out_dir, f"synth_formant-{label}.wav")
        out.write(path)
        print(f"  formant {label} -> {path}")

    # --- Noise ---
    out = normalize(synthesis.white_noise(frames_1s, sample_rate=sr))
    path = os.path.join(args.out_dir, "synth_noise-white.wav")
    out.write(path)
    print(f"  white noise -> {path}")

    for freq, label in [(500.0, "500hz"), (2000.0, "2khz")]:
        out = normalize(synthesis.clocked_noise(frames_1s, freq=freq, sample_rate=sr))
        path = os.path.join(args.out_dir, f"synth_noise-clocked-{label}.wav")
        out.write(path)
        print(f"  clocked noise {label} -> {path}")

    out = normalize(synthesis.dust(frames_1s, density=50.0, sample_rate=sr))
    path = os.path.join(args.out_dir, "synth_noise-dust.wav")
    out.write(path)
    print(f"  dust -> {path}")

    # --- Drums ---
    drums = [
        ("kick-analog", lambda: synthesis.analog_bass_drum(frames_half, freq=55.0, sample_rate=sr)),
        ("kick-synth", lambda: synthesis.synthetic_bass_drum(frames_half, freq=50.0, sample_rate=sr)),
        ("snare-analog", lambda: synthesis.analog_snare_drum(frames_half, freq=180.0, sample_rate=sr)),
        ("snare-synth", lambda: synthesis.synthetic_snare_drum(frames_half, freq=200.0, sample_rate=sr)),
        ("hihat", lambda: synthesis.hihat(frames_half, freq=3000.0, sample_rate=sr)),
    ]
    for label, fn in drums:
        out = normalize(fn())
        path = os.path.join(args.out_dir, f"synth_drum-{label}.wav")
        out.write(path)
        print(f"  drum {label} -> {path}")

    # --- Physical modeling ---
    for freq, label in [(220.0, "A3"), (440.0, "A4"), (880.0, "A5")]:
        out = normalize(synthesis.pluck(frames_1s, freq=freq, sample_rate=sr))
        path = os.path.join(args.out_dir, f"synth_pluck-{label}.wav")
        out.write(path)
        print(f"  pluck {label} -> {path}")

    out = normalize(synthesis.modal_voice(frames_1s, freq=440.0, sample_rate=sr))
    path = os.path.join(args.out_dir, "synth_modalvoice.wav")
    out.write(path)
    print(f"  modal voice -> {path}")

    out = normalize(synthesis.string_voice(frames_1s, freq=330.0, sample_rate=sr))
    path = os.path.join(args.out_dir, "synth_stringvoice.wav")
    out.write(path)
    print(f"  string voice -> {path}")

    out = normalize(synthesis.drip(frames_1s, sample_rate=sr))
    path = os.path.join(args.out_dir, "synth_drip.wav")
    out.write(path)
    print(f"  drip -> {path}")

    # Karplus-Strong (uses noise as excitation)
    excitation = synthesis.white_noise(frames_1s, amp=0.3, sample_rate=sr)
    out = normalize(synthesis.karplus_strong(excitation, freq_hz=220.0, brightness=0.7))
    path = os.path.join(args.out_dir, "synth_karplus.wav")
    out.write(path)
    print(f"  karplus-strong -> {path}")

    # --- STK instruments ---
    instruments = [
        "clarinet", "flute", "brass", "bowed", "plucked", "sitar",
        "stifkarp", "saxofony", "recorder", "blowbotl", "blowhole", "whistle",
    ]
    for inst in instruments:
        out = normalize(synthesis.synth_note(inst, freq=440.0, duration=0.8, release=0.2, sample_rate=sr))
        path = os.path.join(args.out_dir, f"synth_stk-{inst}.wav")
        out.write(path)
        print(f"  stk {inst} -> {path}")

    # --- Sequence ---
    notes = [
        (262.0, 0.0, 0.4),    # C4
        (294.0, 0.5, 0.4),    # D4
        (330.0, 1.0, 0.4),    # E4
        (349.0, 1.5, 0.4),    # F4
        (392.0, 2.0, 0.4),    # G4
        (440.0, 2.5, 0.4),    # A4
        (494.0, 3.0, 0.4),    # B4
        (523.0, 3.5, 0.6),    # C5
    ]
    out = normalize(synthesis.synth_sequence("flute", notes=notes, sample_rate=sr))
    path = os.path.join(args.out_dir, "synth_sequence-flute-scale.wav")
    out.write(path)
    print(f"  sequence (flute scale) -> {path}")


if __name__ == "__main__":
    main()

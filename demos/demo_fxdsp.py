#!/usr/bin/env python3
"""Demo: FX DSP algorithms -- waveshaping, reverbs, formant, PSOLA, minBLEP, ping-pong delay, freq shift, ring mod.

Applies waveshaping, reverb, formant, ping-pong delay, frequency shifting,
and ring modulation effects to an input file.
Also generates minBLEP oscillator waveforms (no input needed for those).
"""

import argparse
import os

import numpy as np

from nanodsp.buffer import AudioBuffer
from nanodsp.effects.composed import (
    formant_filter,
    freq_shift,
    ping_pong_delay,
    psola_pitch_shift,
    ring_mod,
)
from nanodsp.effects.reverb import schroeder_reverb, moorer_reverb
from nanodsp.effects.saturation import aa_hard_clip, aa_soft_clip, aa_wavefold
from nanodsp.synthesis import minblep


def peak_normalize(buf: AudioBuffer) -> AudioBuffer:
    """Scale so the loudest sample sits at 0 dBFS."""
    peak = np.max(np.abs(buf.data))
    if peak > 0:
        return buf.gain_db(-20.0 * np.log10(peak))
    return buf


def main():
    parser = argparse.ArgumentParser(description="Demo: FX DSP algorithms")
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

    # --- Antialiased waveshaping ---
    fx_demos = [
        ("aa-hardclip-mild", lambda b: aa_hard_clip(b, drive=1.0)),
        ("aa-hardclip-heavy", lambda b: aa_hard_clip(b, drive=3.0)),
        ("aa-softclip-mild", lambda b: aa_soft_clip(b, drive=1.5)),
        ("aa-softclip-heavy", lambda b: aa_soft_clip(b, drive=4.0)),
        ("aa-wavefold-mild", lambda b: aa_wavefold(b, drive=1.5)),
        ("aa-wavefold-heavy", lambda b: aa_wavefold(b, drive=3.0)),
        # --- Reverbs ---
        ("schroeder-default", lambda b: schroeder_reverb(b)),
        ("schroeder-long", lambda b: schroeder_reverb(b, feedback=0.85, diffusion=0.7)),
        ("schroeder-mod", lambda b: schroeder_reverb(b, feedback=0.7, mod_depth=0.3)),
        ("moorer-default", lambda b: moorer_reverb(b)),
        ("moorer-long", lambda b: moorer_reverb(b, feedback=0.85, diffusion=0.8)),
        ("moorer-mod", lambda b: moorer_reverb(b, feedback=0.7, mod_depth=0.3)),
        # --- Formant filter ---
        ("formant-a", lambda b: formant_filter(b, vowel="a")),
        ("formant-e", lambda b: formant_filter(b, vowel="e")),
        ("formant-i", lambda b: formant_filter(b, vowel="i")),
        ("formant-o", lambda b: formant_filter(b, vowel="o")),
        ("formant-u", lambda b: formant_filter(b, vowel="u")),
        # --- PSOLA pitch shift ---
        ("psola-up-2", lambda b: psola_pitch_shift(b, semitones=2.0)),
        ("psola-up-5", lambda b: psola_pitch_shift(b, semitones=5.0)),
        ("psola-up-12", lambda b: psola_pitch_shift(b, semitones=12.0)),
        ("psola-down-3", lambda b: psola_pitch_shift(b, semitones=-3.0)),
        ("psola-down-7", lambda b: psola_pitch_shift(b, semitones=-7.0)),
        ("psola-down-12", lambda b: psola_pitch_shift(b, semitones=-12.0)),
        # --- Ping-pong delay ---
        ("pingpong-default", lambda b: ping_pong_delay(b, delay_ms=375.0, feedback=0.5, mix=0.5)),
        ("pingpong-short", lambda b: ping_pong_delay(b, delay_ms=125.0, feedback=0.3, mix=0.4)),
        ("pingpong-long", lambda b: ping_pong_delay(b, delay_ms=750.0, feedback=0.6, mix=0.5)),
        # --- Frequency shifter ---
        ("freqshift-up50", lambda b: freq_shift(b, shift_hz=50.0)),
        ("freqshift-up200", lambda b: freq_shift(b, shift_hz=200.0)),
        ("freqshift-down100", lambda b: freq_shift(b, shift_hz=-100.0)),
        # --- Ring modulator ---
        ("ringmod-300hz", lambda b: ring_mod(b, carrier_freq=300.0)),
        ("ringmod-100hz", lambda b: ring_mod(b, carrier_freq=100.0)),
        ("ringmod-lfo", lambda b: ring_mod(b, carrier_freq=300.0, lfo_freq=3.0, lfo_width=30.0)),
        ("ringmod-subtle", lambda b: ring_mod(b, carrier_freq=200.0, mix=0.4)),
    ]

    for label, fn in fx_demos:
        out = normalize(fn(buf))
        path = os.path.join(args.out_dir, f"{name}_fxdsp-{label}.wav")
        out.write(path)
        print(f"  {label} -> {path}")

    # --- MinBLEP oscillator waveforms (generated, no input needed) ---
    frames_1s = int(sr)
    for wf in ["saw", "rsaw", "square", "triangle"]:
        out = normalize(minblep(frames_1s, freq=220.0, waveform=wf, sample_rate=sr))
        path = os.path.join(args.out_dir, f"synth_minblep-{wf}.wav")
        out.write(path)
        print(f"  minblep {wf} -> {path}")

    # MinBLEP square with narrow pulse width
    out = normalize(
        minblep(frames_1s, freq=220.0, waveform="square", pulse_width=0.25, sample_rate=sr)
    )
    path = os.path.join(args.out_dir, "synth_minblep-square-pw25.wav")
    out.write(path)
    print(f"  minblep square pw=0.25 -> {path}")


if __name__ == "__main__":
    main()

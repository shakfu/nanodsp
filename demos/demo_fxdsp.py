#!/usr/bin/env python3
"""Demo: FX DSP algorithms -- antialiased waveshaping, reverbs, formant, PSOLA, minBLEP.

Applies waveshaping, reverb, and formant effects to an input file.
Also generates minBLEP oscillator waveforms (no input needed for those).
"""

import argparse
import os

import numpy as np

from nanodsp.buffer import AudioBuffer
from nanodsp.effects import (
    aa_hard_clip,
    aa_soft_clip,
    aa_wavefold,
    schroeder_reverb,
    moorer_reverb,
    formant_filter,
    psola_pitch_shift,
)
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

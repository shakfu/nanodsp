# Time-stretching (PaulStretch)

PaulStretch is an *extreme* time-stretching technique by Nasca Octavian Paul (public domain). Where the phase vocoder in [Spectral](spectral.md) (`spectral.time_stretch`) is built to change duration by modest ratios while keeping a signal recognizable, PaulStretch is built for very large factors -- 8x, 20x, 50x -- where the result is intentionally smeared into ambient, pad-like textures rather than a faithful slow-down.

It works by stepping through the input in overlapping windows, taking an FFT of each, **keeping the magnitude spectrum but replacing every bin's phase with a random value**, and overlap-adding the inverse transforms. Randomizing the phase removes the temporal structure within each window, so stretching far past the original length produces a smooth wash instead of the metallic/"stuttering" artifacts a phase vocoder gives at extreme ratios. Pitch is preserved unless you ask for a shift.

This is an original implementation built on the signalsmith FFT; it does not use the GPLv3 [paulxstretch](https://github.com/essej/paulxstretch) application sources -- only the public-domain algorithm is reproduced.

## Core stretch

```python
from nanodsp import AudioBuffer
from nanodsp.timestretch import paulstretch

buf = AudioBuffer.from_file("input.wav")

# 8x longer, pitch preserved
out = paulstretch(buf, stretch=8.0)
```

The output length is approximately `frames * stretch`. PaulStretch is usually fed a short source that it grows into a long texture, so beware that large factors produce large files. All channels share the same output length, and stereo material is decorrelated (each channel uses a different phase seed) for a wider image. Output is reproducible for a given `seed`.

## Window size

The window size sets the trade-off between time and frequency detail.

```python
# Smaller window keeps more detail (more movement)
detailed = paulstretch(buf, stretch=8.0, window_size=1024)

# Larger window is smoother and more diffuse (more "frozen")
smooth = paulstretch(buf, stretch=8.0, window_size=16384)
```

Typical values are 2048--16384. Smaller windows track fast changes; larger windows blur them into a steadier drone.

## Transient preservation

Pure phase randomization softens attacks. The `onset` parameter detects transients and keeps the *original* phase on those frames, so percussive hits stay defined inside the smear.

```python
# 0 disables (full smear); (0, 1] preserves onsets, higher = more sensitive
out = paulstretch(buf, stretch=8.0, onset=0.6)
```

## Spectral effects

These reshape the magnitude spectrum before resynthesis.

```python
# Pitch / octave shift (formants move with the pitch)
up = paulstretch(buf, stretch=8.0, pitch_semitones=12.0)
down = paulstretch(buf, stretch=8.0, pitch_semitones=-12.0)

# Added harmonics (integer-multiple copies, geometric decay) + spectral
# spread (blur across neighbouring bins): thicker, more diffuse pad
thick = paulstretch(buf, stretch=8.0, harmonics=3, spread=6.0)

# Spectral band filtering -- zero bins outside the band before resynthesis
band = paulstretch(buf, stretch=8.0, highpass_hz=500.0, lowpass_hz=6000.0)
```

## A long drone

Combine the parameters for a sustained ambient texture:

```python
drone = paulstretch(
    buf,
    stretch=20.0,
    window_size=8192,
    onset=0.4,
    lowpass_hz=8000.0,
)
```

## CLI

The same effect is available as the `paulstretch` filter in the [CLI](../cli.md):

```bash
nanodsp process input.wav -o out.wav -f paulstretch:stretch=8
nanodsp process input.wav -o out.wav -f paulstretch:stretch=20,pitch_semitones=12,onset=0.5
```

## When to use which stretcher

| | `spectral.time_stretch` (phase vocoder) | `timestretch.paulstretch` |
|---|---|---|
| Best for | Modest ratios, keeping the signal recognizable | Extreme ratios, ambient/textural results |
| Character | Faithful slow-down/speed-up | Smeared, diffuse, pad-like |
| Typical range | ~0.5x--2x | ~4x--50x+ |

# Time-stretching and pitch-shifting

Two complementary backends:

- **PaulStretch** -- extreme time-stretching via phase-randomized spectral resynthesis. The algorithm is by Nasca Octavian Paul (public domain); an original implementation built on the signalsmith FFT, not the GPLv3 [paulxstretch](https://github.com/essej/paulxstretch) sources.
- **Signalsmith stretch** -- the MIT-licensed [signalsmith-stretch](https://github.com/Signalsmith-Audio/signalsmith-stretch) library: a transient-aware, phase-vocoder-derived stretcher that stays musical at modest ratios and decouples time-stretch from pitch-shift.

## PaulStretch usage

### Core stretch

```python
from nanodsp import AudioBuffer
from nanodsp.timestretch import paulstretch

buf = AudioBuffer.from_file("input.wav")

# 8x longer, pitch preserved
out = paulstretch(buf, stretch=8.0)
out.write("stretched.wav")
```

### Window size

```python
# Smaller window keeps more detail; larger is smoother / more diffuse
detailed = paulstretch(buf, stretch=8.0, window_size=1024)
smooth = paulstretch(buf, stretch=8.0, window_size=16384)
```

### Transient preservation

```python
# Keep attacks sharp inside the smear (0 disables; (0, 1] enables)
out = paulstretch(buf, stretch=8.0, onset=0.6)
```

### Spectral effects

```python
# Pitch / octave shift applied during resynthesis
up = paulstretch(buf, stretch=8.0, pitch_semitones=12.0)    # up one octave
down = paulstretch(buf, stretch=8.0, pitch_semitones=-12.0)  # down one octave

# Added harmonics + spectral spread: thicker, more diffuse pad
thick = paulstretch(buf, stretch=8.0, harmonics=3, spread=6.0)

# Spectral band filtering (zeroes bins outside the band before resynthesis)
band = paulstretch(buf, stretch=8.0, highpass_hz=500.0, lowpass_hz=6000.0)
```

### Long drone

```python
# Very long stretch + transient preservation + low-pass
drone = paulstretch(buf, stretch=20.0, window_size=8192, onset=0.4, lowpass_hz=8000.0)
```

Output length is approximately `frames * stretch`; all channels share the same length, and stereo material is decorrelated (per-channel seeds) for a wider image. Output is reproducible for a given `seed`. The effect is also available as the CLI filter `paulstretch:stretch=...`.

## Signalsmith stretch usage

### Time-stretch

```python
from nanodsp.timestretch import signalsmith_stretch

# 2x longer, pitch preserved
out = signalsmith_stretch(buf, stretch=2.0)

# Speed up (shorter), pitch preserved
out = signalsmith_stretch(buf, stretch=0.75)
```

### Pitch-shift (independent of stretch)

```python
# Pure pitch-shift, length unchanged
up = signalsmith_stretch(buf, stretch=1.0, semitones=12.0)    # up one octave
down = signalsmith_stretch(buf, stretch=1.0, semitones=-7.0)   # down a perfect fifth

# Stretch and pitch-shift together; they are decoupled
both = signalsmith_stretch(buf, stretch=1.5, semitones=5.0)
```

### Tonality limit

```python
# Above tonality_hz the shift rolls back toward the original, preserving
# high-frequency timbre/"air" on large shifts (a common choice for voice)
out = signalsmith_stretch(buf, stretch=1.0, semitones=7.0, tonality_hz=8000.0)
```

### Cheaper preset

```python
# Lower-CPU preset (slightly lower quality)
out = signalsmith_stretch(buf, stretch=2.0, cheaper=True)
```

Output length is approximately `frames * stretch`; time-stretch and pitch-shift are independent, all channels are processed coherently in a single pass, and output is reproducible for a given `seed`. Also available as the CLI filter `signalsmith_stretch:stretch=...`.

## API reference

::: nanodsp.timestretch
    options:
      show_if_no_docstring: false

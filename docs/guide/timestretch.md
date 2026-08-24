# Time-stretching and pitch-shifting

The `timestretch` module has two backends for changing duration and pitch: **PaulStretch** for extreme, textural stretches, and **Signalsmith stretch** for clean, musical stretching and independent pitch-shifting. See [When to use which stretcher](#when-to-use-which-stretcher) at the end for picking between these and the [Spectral](spectral.md) phase vocoder.

## PaulStretch

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

## Constant-Q spread

`spread` blurs across a fixed number of FFT bins. Because bin spacing is linear in frequency, a given radius covers many more octaves down low than up high, so bass partials smear into mush while the top end barely moves.

`spread_octaves` blurs on a log-frequency axis instead, so every partial is smeared across the same fraction of its own frequency. The result is musically even across the range, and the width is independent of `window_size` and sample rate.

```python
# Each partial smeared by ~0.3 octaves, top to bottom
even = paulstretch(buf, stretch=8.0, spread_octaves=0.3)
```

Typical values are 0.05--0.5. Prefer this over `spread` unless the low-end bias is the effect you are after.

## Tonal vs. noise

Most material is a mix of steady partials and a noise floor underneath them. Comparing the spectrum against a smoothed copy of itself separates the two: anything standing above its own local envelope is tonal, the rest is noise. `tonal_vs_noise` then blends the output toward one part or the other.

```python
# Keep the pitches, drop the hiss -- a cleaner, more harmonic drone
pitched = paulstretch(buf, stretch=8.0, tonal_vs_noise=1.0)

# Keep the hiss, drop the pitches -- a breathy, unpitched wash
airy = paulstretch(buf, stretch=8.0, tonal_vs_noise=-1.0)

# Partial application
subtle = paulstretch(buf, stretch=8.0, tonal_vs_noise=0.4)
```

The parameter runs from -1 to +1 with 0 leaving the spectrum untouched, and the effect increases monotonically across the whole range. `tonal_noise_octaves` sets the width of the envelope used for the comparison: narrow settings (0.1) are a high bar and keep only sharp partials, wider settings (0.5) let more of the spectrum count as tonal.

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

## Signalsmith stretch

`signalsmith_stretch` wraps the MIT-licensed [signalsmith-stretch](https://github.com/Signalsmith-Audio/signalsmith-stretch) library (Geraint Luff / Signalsmith Audio), a transient-aware, phase-vocoder-derived algorithm. Where PaulStretch *intentionally* smears the signal, this aims to keep it recognizable and musical -- a clean slow-down or speed-up -- and it treats **pitch and duration as independent controls**. All channels are processed together so a stereo image stays coherent.

```python
from nanodsp.timestretch import signalsmith_stretch

# Time-stretch, pitch preserved
slower = signalsmith_stretch(buf, stretch=2.0)    # 2x longer
faster = signalsmith_stretch(buf, stretch=0.75)   # shorter
```

### Pitch-shifting

`semitones` shifts pitch independently of `stretch`, so `stretch=1.0` gives a pure pitch-shift with the duration unchanged.

```python
up = signalsmith_stretch(buf, stretch=1.0, semitones=12.0)   # up one octave
down = signalsmith_stretch(buf, stretch=1.0, semitones=-7.0)  # down a perfect fifth

# Decoupled: change both at once
both = signalsmith_stretch(buf, stretch=1.5, semitones=5.0)
```

### Tonality limit

Large pitch shifts move the whole spectrum, which can sound thin (up) or dark (down). The `tonality_hz` limit rolls the shift back toward the original above that frequency, keeping the high-end timbre and "air" more natural -- around 8000 Hz is a common choice for voice.

```python
out = signalsmith_stretch(buf, stretch=1.0, semitones=7.0, tonality_hz=8000.0)
```

### Cheaper preset and seed

```python
# Lower-CPU preset (slightly lower quality)
out = signalsmith_stretch(buf, stretch=2.0, cheaper=True)

# Past ~2x the algorithm randomizes phase; `seed` makes that reproducible
out = signalsmith_stretch(buf, stretch=3.0, seed=42)
```

## CLI

Both effects are available as filters in the [CLI](../cli.md):

```bash
nanodsp process input.wav -o out.wav -f paulstretch:stretch=8
nanodsp process input.wav -o out.wav -f paulstretch:stretch=20,pitch_semitones=12,onset=0.5

nanodsp process input.wav -o out.wav -f signalsmith_stretch:stretch=2
nanodsp process input.wav -o out.wav -f signalsmith_stretch:stretch=1,semitones=-5,tonality_hz=8000
```

## When to use which stretcher

| | `spectral.time_stretch` (phase vocoder) | `timestretch.signalsmith_stretch` | `timestretch.paulstretch` | `timestretch.keyframe_stretch` |
|---|---|---|---|---|
| Best for | Modest ratios, keeping the signal recognizable | Clean stretch + independent pitch-shift | Extreme ratios, ambient/textural results | Percussive and dense material, cheaply |
| Character | Faithful slow-down/speed-up | Musical, transient-aware | Smeared, diffuse, pad-like | Organic, transient-preserving, some spectral haze |
| Pitch control | Separate pitch-shift | Built-in, decoupled from stretch | Spectral shift (formants move) | Built-in, decoupled from stretch |
| Typical range | ~0.5x--2x | ~0.5x--4x | ~4x--50x+ | ~0.5x--4x |
| Cost | FFT per frame | FFT per frame | FFT per frame | No FFT; ~15 multiplies per sample |

Reach for `signalsmith_stretch` when the result should sound like the input.
Reach for `keyframe_stretch` when the material is percussive or busy, when you
want its particular character, or when you need sample-by-sample output with no
block latency -- it is the only stretcher here that has no frame to fill before
it can emit anything. Its concession is spectral: tonally nuanced material
(solo voice, glockenspiel) picks up an audible haze.

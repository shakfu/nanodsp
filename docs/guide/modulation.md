# Modulation Effects

Effects that vary a parameter (usually delay time or amplitude) with an LFO.

```python
from nanodsp.buffer import AudioBuffer
from nanodsp.effects.daisysp import (
    chorus, flanger, phaser, tremolo, autowah,
    pitch_shift,
)

buf = AudioBuffer.from_file("input.wav")
```

## Chorus

Mixes the dry signal with a delayed copy whose delay time is modulated by an LFO. The varying delay creates pitch vibrato; mixing with dry creates the characteristic thickening. Mono input produces stereo output.

```python
# Standard chorus
wide = chorus(buf, lfo_freq=0.3, lfo_depth=0.5, delay_ms=5.0, feedback=0.2)

# Fast, shallow chorus (subtle doubling)
double = chorus(buf, lfo_freq=1.0, lfo_depth=0.2, delay_ms=3.0)

# Deep, slow chorus (ensemble-like)
ensemble = chorus(buf, lfo_freq=0.1, lfo_depth=0.8, delay_ms=10.0, feedback=0.4)
```

## Flanger

Similar to chorus but with shorter delays (< 5 ms) and feedback. The comb-filtering effect sweeps through the spectrum, creating a jet-engine swoosh.

```python
# Classic flanging
flanged = flanger(buf, lfo_freq=0.2, lfo_depth=0.5, feedback=0.5, delay_ms=1.0)

# Metallic resonant flanger
metallic = flanger(buf, lfo_freq=0.1, lfo_depth=0.8, feedback=0.8, delay_ms=2.0)
```

## Phaser

Cascaded allpass filters whose center frequencies are swept by an LFO. Unlike flanging (evenly-spaced notches), phasing creates irregularly-spaced notches that sound more organic.

```python
# Standard phaser
phased = phaser(buf, lfo_freq=0.3, lfo_depth=0.5, feedback=0.5)

# Deep phaser with more stages
deep = phaser(buf, lfo_freq=0.2, lfo_depth=0.7, feedback=0.6, poles=8)

# Subtle, slow phaser
gentle = phaser(buf, lfo_freq=0.05, lfo_depth=0.3, feedback=0.3, poles=4)
```

## Tremolo

Amplitude modulation by an LFO. Multiplies the signal by a low-frequency waveform.

```python
# Standard tremolo
trem = tremolo(buf, freq=5.0, depth=0.5)

# Fast, deep tremolo (helicopter effect)
fast = tremolo(buf, freq=12.0, depth=0.9)
```

## Autowah

An envelope-controlled bandpass filter. The input amplitude modulates the filter frequency, creating a wah-wah that responds to playing dynamics.

```python
wah = autowah(buf, wah=0.7, dry_wet=0.8, level=0.5)
```

## Pitch shifting (DaisySP)

Time-domain pitch shifting with a "fun" parameter for spectral smearing.

```python
# Up one octave
up = pitch_shift(buf, semitones=12.0)

# Down a fifth
down = pitch_shift(buf, semitones=-7.0)

# With spectral smearing
weird = pitch_shift(buf, semitones=5.0, fun=0.5)
```

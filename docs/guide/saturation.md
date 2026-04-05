# Saturation and Distortion

Waveshaping algorithms that add harmonics by applying nonlinear transfer functions to the signal.

## Basic saturation

Three modes with different transfer curves:

- **Soft (tanh)** -- smooth, symmetrical clipping. Adds primarily odd harmonics. Sounds warm.
- **Hard (clip)** -- abrupt clipping at +/-1.0. Harsh, buzzy harmonics.
- **Tape** -- asymmetric soft clip (`x - x^3/3`). Adds both even and odd harmonics, emulating tape saturation.

```python
from nanodsp.buffer import AudioBuffer
from nanodsp.effects.saturation import saturate, aa_hard_clip, aa_soft_clip, aa_wavefold

buf = AudioBuffer.from_file("input.wav")

# Warm tape saturation
warm = saturate(buf, drive=0.5, mode="tape")

# Aggressive hard clipping
harsh = saturate(buf, drive=0.8, mode="hard")

# Subtle soft saturation
gentle = saturate(buf, drive=0.2, mode="soft")
```

## Antialiased waveshaping

Naive waveshaping creates aliasing -- frequencies above Nyquist fold back as inharmonic distortion. These functions use antiderivative antialiasing (ADAA) to suppress it.

```python
# Antialiased hard clipper (1st-order ADAA)
clipped = aa_hard_clip(buf, drive=2.0)

# Antialiased soft clipper (sin-based, 1st-order ADAA)
soft = aa_soft_clip(buf, drive=1.5)

# Antialiased wavefolder (Buchla 259 style, 2nd-order ADAA)
folded = aa_wavefold(buf, drive=3.0)
```

## DaisySP distortion

```python
from nanodsp.effects.daisysp import (
    overdrive, wavefold, bitcrush, decimator, fold, sample_rate_reduce,
)

# Smooth overdrive
driven = overdrive(buf, drive=0.6)

# Wavefolding with DC offset
folded = wavefold(buf, gain=2.0, offset=0.1)

# Bitcrushing -- quantize to 8 bits
crushed = bitcrush(buf, bit_depth=8)

# Decimator -- downsampling + bitcrushing combined
lofi = decimator(buf, downsample_factor=0.5, bitcrush_factor=0.5)

# Sample rate reduction
reduced = sample_rate_reduce(buf, freq=0.3)

# Fold distortion
fold_dist = fold(buf, increment=1.5)
```

## Comparing clean vs antialiased

```python
import numpy as np
from nanodsp.buffer import AudioBuffer
from nanodsp.effects.saturation import saturate, aa_hard_clip

# High-frequency sine -- aliasing is most audible here
buf = AudioBuffer.sine(8000.0, frames=48000, sample_rate=48000.0)

# Naive hard clip -- will alias
naive = saturate(buf, drive=0.9, mode="hard")

# Antialiased hard clip -- clean
clean = aa_hard_clip(buf, drive=3.0)
```

# Saturation

Saturation, distortion, and antialiased waveshaping.

## Usage examples

### Basic saturation

```python
from nanodsp.effects import saturation
from nanodsp.buffer import AudioBuffer

buf = AudioBuffer.from_file("input.wav")

# Soft saturation (tanh) -- warm, symmetrical
warm = saturation.saturate(buf, drive=0.5, mode="soft")

# Hard clipping -- harsh, buzzy
hard = saturation.saturate(buf, drive=0.7, mode="hard")

# Tape saturation -- asymmetric, adds even harmonics
tape = saturation.saturate(buf, drive=0.4, mode="tape")
```

### Antialiased waveshaping

These use antiderivative antialiasing (ADAA) to suppress aliasing artifacts that occur with naive waveshaping:

```python
# Antialiased hard clipper (clean even at high drive)
clipped = saturation.aa_hard_clip(buf, drive=3.0)

# Antialiased soft clipper (sin-based saturation)
soft = saturation.aa_soft_clip(buf, drive=2.0)

# Antialiased wavefolder (Buchla 259 style, 2nd-order ADAA)
folded = saturation.aa_wavefold(buf, drive=4.0)
```

### Comparison: naive vs. antialiased

```python
# Naive hard clip introduces aliasing at high drive
naive = saturation.saturate(buf, drive=0.9, mode="hard")

# Antialiased version is much cleaner
clean = saturation.aa_hard_clip(buf, drive=10.0)
```

## API reference

::: nanodsp.effects.saturation
    options:
      show_if_no_docstring: false

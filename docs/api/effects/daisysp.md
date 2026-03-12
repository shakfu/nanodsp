# DaisySP Effects

Autowah, chorus, decimator, flanger, overdrive, phaser, pitch shift, sample-rate reduction, tremolo, wavefold, bitcrush, fold, reverb, and DC blocking.

## Usage examples

### Modulation effects

```python
from nanodsp.effects import daisysp
from nanodsp.buffer import AudioBuffer

buf = AudioBuffer.from_file("input.wav")

# Chorus (mono input -> stereo output)
chorused = daisysp.chorus(buf, lfo_freq=0.3, lfo_depth=0.5, delay_ms=5.0)

# Flanger
flanged = daisysp.flanger(buf, lfo_freq=0.2, lfo_depth=0.5, feedback=0.5)

# Phaser (6-stage allpass cascade)
phased = daisysp.phaser(buf, lfo_freq=0.3, lfo_depth=0.5, feedback=0.5, poles=6)

# Tremolo
tremolo = daisysp.tremolo(buf, freq=5.0, depth=0.8)

# Autowah (envelope-controlled filter)
wah = daisysp.autowah(buf, wah=0.7, dry_wet=1.0)
```

### Distortion and lo-fi

```python
# Overdrive
driven = daisysp.overdrive(buf, drive=0.7)

# Wavefolder
folded = daisysp.wavefold(buf, gain=2.0, offset=0.0)

# Bitcrusher (reduce bit depth + sample rate)
crushed = daisysp.bitcrush(buf, bit_depth=8, crush_rate=11025.0)

# Decimator (combined downsampling + bit reduction)
decimated = daisysp.decimator(
    buf, downsample_factor=0.5, bitcrush_factor=0.5, bits_to_crush=8
)

# Fold distortion
fold = daisysp.fold(buf, increment=2.0)

# Sample-rate reduction
reduced = daisysp.sample_rate_reduce(buf, freq=0.3)
```

### Pitch shifting

```python
# Shift up one octave (+12 semitones)
up = daisysp.pitch_shift(buf, semitones=12.0)

# Shift down a perfect fifth (-7 semitones)
down = daisysp.pitch_shift(buf, semitones=-7.0)

# With spectral smearing ("fun" parameter)
weird = daisysp.pitch_shift(buf, semitones=5.0, fun=0.5)
```

### Reverb and utility

```python
# ReverbSc -- stereo reverb (mono or stereo input, always stereo output)
reverbed = daisysp.reverb_sc(buf, feedback=0.8, lp_freq=10000.0)

# Remove DC offset
clean = daisysp.dc_block(buf)
```

## API reference

::: nanodsp.effects.daisysp
    options:
      show_if_no_docstring: false

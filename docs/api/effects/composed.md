# Composed Effects

Higher-level effects built by combining multiple primitives: exciter, de-esser, parallel compression, stereo delay, multiband compression, formant filtering, PSOLA pitch shifting, mastering, vocal processing chains, shimmer reverb, tape echo, lo-fi, telephone, gated reverb, and auto-pan.

## Usage examples

### Exciter

```python
from nanodsp.effects import composed
from nanodsp.buffer import AudioBuffer

buf = AudioBuffer.from_file("input.wav")

# Add brightness and presence above 3 kHz
bright = composed.exciter(buf, freq=3000.0, amount=0.3)

# Subtle air boost above 8 kHz
airy = composed.exciter(buf, freq=8000.0, amount=0.15)
```

### De-esser

```python
# Tame sibilance around 6 kHz
deessed = composed.de_esser(buf, freq=6000.0, threshold_db=-20.0)

# Aggressive de-essing with higher ratio
strong = composed.de_esser(buf, freq=5000.0, threshold_db=-25.0, ratio=8.0)
```

### Parallel compression

```python
# "New York compression" -- heavy compression mixed with dry
punchy = composed.parallel_compress(buf, mix=0.5, ratio=8.0, threshold_db=-30.0)
```

### Stereo delay

```python
# Stereo delay with different left/right times
delayed = composed.stereo_delay(
    buf, left_ms=250.0, right_ms=375.0, feedback=0.3, mix=0.4
)

# Ping-pong delay
pp = composed.stereo_delay(
    buf, left_ms=250.0, right_ms=250.0, feedback=0.4, mix=0.5, ping_pong=True
)
```

### Multiband compression

```python
# 4-band compression with default crossovers
multi = composed.multiband_compress(buf)

# Custom crossover frequencies and per-band settings
multi = composed.multiband_compress(
    buf,
    crossover_freqs=[200.0, 2000.0, 8000.0],
    ratios=[2.0, 4.0, 3.0, 2.0],
    thresholds=[-24.0, -20.0, -18.0, -16.0],
)
```

### Formant filter

```python
# Apply vowel formant (a, e, i, o, u)
vowel_a = composed.formant_filter(buf, vowel="a")
vowel_e = composed.formant_filter(buf, vowel="e")
vowel_o = composed.formant_filter(buf, vowel="o")
```

### PSOLA pitch shifting

```python
# Pitch shift up 5 semitones (time-domain, best for monophonic)
shifted = composed.psola_pitch_shift(buf, semitones=5.0)

# Down one octave
down = composed.psola_pitch_shift(buf, semitones=-12.0)
```

### Ping-pong delay

```python
# Stereo ping-pong delay with crossed feedback
pp = composed.ping_pong_delay(buf, delay_ms=375.0, feedback=0.5, mix=0.5)

# Shorter delay, less feedback
pp = composed.ping_pong_delay(buf, delay_ms=125.0, feedback=0.3, mix=0.4)
```

### Frequency shifter

```python
# Shift all frequencies up by 100 Hz (inharmonic -- not pitch shifting)
shifted = composed.freq_shift(buf, shift_hz=100.0)

# Shift down by 50 Hz
shifted = composed.freq_shift(buf, shift_hz=-50.0)
```

### Ring modulator

```python
# Classic ring mod at 300 Hz carrier
ring = composed.ring_mod(buf, carrier_freq=300.0)

# With LFO modulating the carrier frequency
wobble = composed.ring_mod(buf, carrier_freq=300.0, lfo_freq=5.0, lfo_width=20.0)

# 50/50 dry/wet mix
subtle = composed.ring_mod(buf, carrier_freq=200.0, mix=0.5)
```

### Shimmer reverb

```python
# Ethereal octave-up shimmer reverb
shimmer = composed.shimmer_reverb(buf, mix=0.4, shimmer=0.3)

# Brighter shimmer with more pitch blend
bright = composed.shimmer_reverb(buf, shimmer=0.6, shift_semitones=12.0, preset="cathedral")

# Fifth-up shimmer for a different harmonic flavor
fifth = composed.shimmer_reverb(buf, shimmer=0.4, shift_semitones=7.0)
```

### Tape echo

```python
# Classic tape delay with darkening repeats
echo = composed.tape_echo(buf, delay_ms=300.0, feedback=0.5, tone=3000.0)

# Dark, saturated echoes
warm = composed.tape_echo(buf, delay_ms=400.0, feedback=0.6, tone=1500.0, drive=0.5)

# Fast slapback-style
slap = composed.tape_echo(buf, delay_ms=100.0, feedback=0.3, repeats=3)
```

### Lo-fi

```python
# Default lo-fi degradation
lofi = composed.lo_fi(buf)

# Heavy bitcrushing with bandwidth reduction
crushed = composed.lo_fi(buf, bit_depth=4, reduce=0.7, tone=2000.0)
```

### Telephone

```python
# Standard telephone bandpass (300-3400 Hz) with saturation
phone = composed.telephone(buf)

# AM radio simulation (wider bandwidth, less distortion)
radio = composed.telephone(buf, low_cut=500.0, high_cut=5000.0, drive=0.2)
```

### Gated reverb

```python
# Classic 80s gated reverb
gated = composed.gated_reverb(buf, preset="plate", mix=0.5)

# Tight, punchy gate
tight = composed.gated_reverb(
    buf, gate_threshold_db=-20.0, gate_hold_ms=30.0, gate_release=0.01
)
```

### Auto-pan

```python
# Moderate stereo auto-panning
panned = composed.auto_pan(buf, rate=2.0, depth=1.0)

# Slow, subtle movement
gentle = composed.auto_pan(buf, rate=0.5, depth=0.5)

# Fast tremolo-like panning
fast = composed.auto_pan(buf, rate=6.0, depth=1.0)
```

### Mastering chain

```python
# Full mastering chain: DC block -> EQ -> compress -> limit -> loudness normalize
mastered = composed.master(buf, target_lufs=-14.0)

# With custom EQ settings
mastered = composed.master(buf, target_lufs=-14.0, eq={
    "low_shelf_hz": 80.0,
    "low_shelf_db": 1.0,
    "high_shelf_hz": 12000.0,
    "high_shelf_db": 0.5,
})

# Skip compression stage
mastered = composed.master(buf, compress_on=False)
```

### Vocal processing chain

```python
# Full vocal chain: de-ess -> EQ -> compress -> limit
vocals = composed.vocal_chain(buf, de_ess_freq=6000.0)

# With loudness target
vocals = composed.vocal_chain(buf, target_lufs=-16.0)

# Custom settings
vocals = composed.vocal_chain(
    buf,
    de_ess=True,
    de_ess_freq=5500.0,
    compress_on=True,
    limit_on=True,
    eq={"low_shelf_hz": 100.0, "low_shelf_db": -2.0},
)
```

## API reference

::: nanodsp.effects.composed
    options:
      show_if_no_docstring: false

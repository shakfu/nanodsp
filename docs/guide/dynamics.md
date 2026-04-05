# Dynamics

Dynamics processors control the amplitude envelope of a signal.

## Compressor

Reduces dynamic range by attenuating signals above a threshold. The `ratio` controls how much (4:1 means 4 dB above threshold produces 1 dB of output above threshold).

```python
from nanodsp.buffer import AudioBuffer
from nanodsp.effects.dynamics import compress, limit, noise_gate, agc
from nanodsp.effects.composed import parallel_compress, multiband_compress

buf = AudioBuffer.from_file("input.wav")

# Standard compression
compressed = compress(buf, ratio=4.0, threshold=-18.0, attack=0.01, release=0.1)

# Heavy compression with auto-makeup gain
heavy = compress(buf, ratio=12.0, threshold=-30.0, auto_makeup=True)

# Gentle bus compression
glue = compress(buf, ratio=2.0, threshold=-12.0, attack=0.03, release=0.2)
```

## Limiter

A compressor with effectively infinite ratio -- prevents the signal from exceeding a ceiling. Use as a final safety stage.

```python
# Basic limiting
limited = limit(buf)

# Pre-gain into limiter (pushes signal harder)
loud = limit(buf, pre_gain=2.0)
```

## Noise gate

Attenuates signals *below* a threshold. Silences quiet passages (mic bleed, background noise). The hold parameter prevents chattering on transients.

```python
# Gate out noise below -40 dBFS
gated = noise_gate(buf, threshold_db=-40.0, attack=0.001, release=0.05)

# Tight gate for drums
tight = noise_gate(buf, threshold_db=-30.0, hold_ms=20.0, release=0.02)
```

## Automatic Gain Control

A slow-acting compressor that maintains consistent average level over time. Good for speech, podcasts, or signals with varying loudness.

```python
# Normalize to a target RMS level
leveled = agc(buf, target_level=0.5, attack=0.01, release=0.05)

# Conservative AGC with limited max boost
safe = agc(buf, target_level=0.3, max_gain_db=20.0)
```

## Parallel compression

Also called "New York compression." Mixes a heavily compressed copy with the dry signal. Preserves transients while bringing up quiet details.

```python
# 50/50 dry/compressed blend
punchy = parallel_compress(buf, mix=0.5, ratio=8.0, threshold_db=-30.0)

# Subtle parallel compression for vocals
subtle = parallel_compress(buf, mix=0.3, ratio=6.0, threshold_db=-24.0)
```

## Multiband compression

Splits the signal into frequency bands, compresses each independently, then sums. Essential for mastering where bass and treble need different treatment.

```python
# Default 3-band compression
multi = multiband_compress(buf)

# Custom crossovers and per-band settings
multi = multiband_compress(
    buf,
    crossover_freqs=[200.0, 2000.0, 8000.0],
    ratios=[2.0, 4.0, 3.0, 2.0],
    thresholds=[-24.0, -20.0, -18.0, -16.0],
)
```

## Chaining dynamics

```python
from nanodsp.effects.filters import highpass

# Vocal dynamics chain
result = (
    buf
    .pipe(highpass, cutoff_hz=80.0)           # remove rumble
    .pipe(compress, ratio=3.0, threshold=-20.0)  # control dynamics
    .pipe(limit)                               # safety limiter
)
```

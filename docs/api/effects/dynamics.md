# Dynamics

Compression, limiting, noise gate, and automatic gain control.

## Usage examples

### Compression

```python
from nanodsp.effects import dynamics
from nanodsp.buffer import AudioBuffer

buf = AudioBuffer.from_file("input.wav")

# Gentle compression (4:1, -20 dB threshold)
compressed = dynamics.compress(buf, ratio=4.0, threshold=-20.0)

# Aggressive compression with fast attack
squashed = dynamics.compress(
    buf, ratio=8.0, threshold=-30.0, attack=0.001, release=0.05
)

# With makeup gain
loud = dynamics.compress(
    buf, ratio=4.0, threshold=-20.0, makeup=6.0
)

# Auto makeup gain
auto = dynamics.compress(
    buf, ratio=4.0, threshold=-20.0, auto_makeup=True
)
```

### Limiting

```python
# Brick-wall limiter
limited = dynamics.limit(buf, pre_gain=1.0)

# Boost into limiter for loudness
hot = dynamics.limit(buf, pre_gain=2.0)
```

### Noise gate

```python
# Gate signal below -40 dB
gated = dynamics.noise_gate(buf, threshold_db=-40.0)

# Tight gate for drums
tight = dynamics.noise_gate(
    buf, threshold_db=-30.0, attack=0.001, release=0.05, hold_ms=10.0
)
```

### Automatic gain control

```python
# Normalize speech to consistent level
level = dynamics.agc(buf, target_level=1.0, max_gain_db=60.0)

# Faster response for dynamic material
fast = dynamics.agc(buf, attack=0.005, release=0.005, average_len=50)
```

## API reference

::: nanodsp.effects.dynamics
    options:
      show_if_no_docstring: false

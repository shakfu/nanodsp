# Getting Started

## Requirements

- Python >= 3.10
- numpy
- C++17 compiler (for building from source)
- CMake >= 3.15 (for building from source)

## Installation

### From PyPI

```bash
pip install nanodsp
```

### From source

```bash
git clone https://github.com/shakfu/nanodsp.git
cd nanodsp
uv sync            # install dependencies + build extension
uv run pytest      # run tests
uv build           # build wheel
```

Use `make help` for additional targets (build, test, lint, format, typecheck, qa, coverage, etc.).

## Basic usage

### Loading audio

```python
from nanodsp.buffer import AudioBuffer

# From file
buf = AudioBuffer.from_file("input.wav")

# Generate test signals
buf = AudioBuffer.sine(440.0, frames=44100, sample_rate=44100)
buf = AudioBuffer.noise(channels=2, frames=44100)
buf = AudioBuffer.impulse(frames=1024)
buf = AudioBuffer.zeros(channels=1, frames=4096)
```

### Processing audio

```python
from nanodsp.effects import filters, dynamics

# Direct function calls
filtered = filters.lowpass(buf, freq=1000.0)
compressed = dynamics.compress(filtered, threshold=-18.0, ratio=4.0)

# Pipeline style
result = (
    buf
    .pipe(filters.highpass, freq=80.0)
    .pipe(filters.lowpass, freq=12000.0)
    .pipe(dynamics.compress, threshold=-18.0)
)
```

### Writing output

```python
result.write("output.wav")
result.write("output.flac", bit_depth=24)
```

### Channel operations

```python
# Mono/stereo conversion
mono = buf.to_mono("mean")
stereo = mono.to_channels(2)

# Channel access
left = buf.channel(0)   # 1D numpy view
channels = buf.split()  # list of mono AudioBuffers

# Stack channels
merged = AudioBuffer.concat_channels(left_buf, right_buf)
```

### Arithmetic

```python
quiet = buf * 0.5
boosted = buf.gain_db(6.0)
mixed = buf_a + buf_b
```

## Architecture

All DSP functions accept and return `AudioBuffer` objects. The processing pipeline is:

1. **AudioBuffer** wraps a 2D float32 numpy array `[channels, frames]` with metadata
2. **Python functions** validate parameters and delegate to C++ bindings
3. **C++ layer** processes audio with GIL released for thread safety
4. **Result** is returned as a new AudioBuffer (functions are non-mutating)

```
AudioBuffer --> Python API --> C++ bindings --> AudioBuffer
                  |                |
           param validation   GIL released
           freq normalization  float32 processing
```

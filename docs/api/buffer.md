# AudioBuffer

The central data type. A 2D `float32` numpy array with shape `[channels, frames]` plus metadata (`sample_rate`, `channel_layout`, `label`).

## Usage examples

### Construction

```python
import numpy as np
from nanodsp.buffer import AudioBuffer

# From a numpy array
arr = np.zeros((2, 44100), dtype=np.float32)
buf = AudioBuffer(arr, sample_rate=44100)

# From file
buf = AudioBuffer.from_file("input.wav")

# Factory methods
buf = AudioBuffer.sine(440.0, channels=1, frames=44100, sample_rate=44100)
buf = AudioBuffer.noise(channels=2, frames=44100, seed=42)
buf = AudioBuffer.impulse(channels=1, frames=1024)
buf = AudioBuffer.zeros(channels=1, frames=4096, sample_rate=48000)
buf = AudioBuffer.ones(channels=2, frames=1024)
```

### Inspecting properties

```python
buf = AudioBuffer.sine(440.0, frames=48000, sample_rate=48000)

buf.channels      # 1
buf.frames        # 48000
buf.sample_rate   # 48000.0
buf.duration      # 1.0 (seconds)
buf.channel_layout  # 'mono'
buf.data.shape    # (1, 48000)
```

### Channel operations

```python
stereo = AudioBuffer.noise(channels=2, frames=44100)

# Downmix
mono = stereo.to_mono("mean")       # average channels
mono = stereo.to_mono("left")       # take left channel

# Upmix
stereo = mono.to_channels(2)        # duplicate mono to stereo

# Split into individual channels
left, right = stereo.split()

# Stack channels
merged = AudioBuffer.concat_channels(left, right)
```

### Arithmetic

```python
buf = AudioBuffer.sine(440.0, frames=4096)

quiet = buf * 0.5                    # scale amplitude
boosted = buf.gain_db(6.0)           # +6 dB
inverted = -buf                      # phase invert
mixed = buf_a + buf_b                # sum two buffers
diff = buf_a - buf_b                 # difference
```

### Pipeline processing

```python
from nanodsp.effects import filters, dynamics

result = (
    AudioBuffer.from_file("input.wav")
    .pipe(filters.highpass, cutoff_hz=80.0)
    .pipe(filters.lowpass, cutoff_hz=12000.0)
    .pipe(dynamics.compress, threshold=-18.0, ratio=4.0)
)
result.write("output.wav")
```

### Slicing

```python
buf = AudioBuffer.sine(440.0, frames=48000, sample_rate=48000)

# Time slice (view, no copy)
first_half = buf.slice(0, 24000)

# Channel indexing
ch0 = buf[0]              # 1D numpy array (channel 0)
samples = buf[0, 100:200] # numpy slice of channel 0, frames 100-199
```

## API reference

::: nanodsp.buffer.AudioBuffer
    options:
      members:
        - __init__
        - data
        - sample_rate
        - channels
        - frames
        - duration
        - channel_layout
        - label
        - dtype
        - channel
        - mono
        - __getitem__
        - __len__
        - zeros
        - ones
        - impulse
        - sine
        - noise
        - from_numpy
        - from_file
        - write
        - to_mono
        - to_channels
        - split
        - concat_channels
        - slice
        - gain_db
        - pipe
        - copy
        - ensure_1d
        - ensure_2d

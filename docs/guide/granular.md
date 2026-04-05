# Granular Synthesis

Granular synthesis decomposes audio into tiny fragments ("grains") and reassembles them with independent control over timing, pitch, density, and spatialization. nanodsp wraps GrainflowLib for block-based granular processing.

## Basic granular cloud

```python
import numpy as np
from nanodsp._core import grainflow as gf
from nanodsp.buffer import AudioBuffer

# Load source audio into a GfBuffer
source = AudioBuffer.from_file("input.wav")
gf_buf = gf.GfBuffer(source.data, source.sample_rate)

# Create a granulator with 32 grains
grains = gf.GrainCollection(n_grains=32, sample_rate=source.sample_rate)
grains.set_buffer(gf_buf, gf.BUF_MAIN)

# Configure grain parameters
grains.set_param(gf.PARAM_WINDOW, gf.PTYPE_VALUE, 0.5)       # grain envelope shape
grains.set_param(gf.PARAM_RATE, gf.PTYPE_VALUE, 1.0)         # playback rate
grains.set_param(gf.PARAM_DELAY, gf.PTYPE_VALUE, 0.0)        # position in source

# Set up a clock to trigger grains
phasor = gf.Phasor()
phasor.set_frequency(20.0, source.sample_rate)    # 20 grains/sec

# Process in blocks
block_size = 512
n_blocks = source.frames // block_size
output = np.zeros((1, n_blocks * block_size), dtype=np.float32)

for i in range(n_blocks):
    clock = phasor.process(block_size)
    grains.set_auto_overlap(True)
    block = grains.process(clock, block_size)
    output[0, i * block_size:(i + 1) * block_size] = block[0]

result = AudioBuffer(output, sample_rate=source.sample_rate)
```

## Stereo panning

The `Panner` distributes grains across the stereo field using equal-power quarter-sine interpolation.

```python
# Create a stereo panner
panner = gf.Panner(mode=gf.PAN_BIPOLAR)   # full L-R spread

# Process mono grains into stereo
mono_block = grains.process(clock, block_size)
stereo_block = panner.process(mono_block)
# stereo_block shape: [2, block_size]
```

Pan modes:

- `PAN_BIPOLAR` -- grains pan across full stereo field (-1 to +1)
- `PAN_UNIPOLAR` -- grains pan from center to one side (0 to +1)
- `PAN_STEREO` -- preserves stereo source positioning

## Live recording

The `Recorder` enables live recording into granular buffers with overdub and freeze.

```python
recorder = gf.Recorder(sample_rate=48000.0)
recorder.set_buffer(gf_buf, gf.BUF_MAIN)

# Record a block of live audio
live_input = np.random.randn(1, block_size).astype(np.float32)
recorder.process(live_input, block_size)
```

## Parameter control

GrainflowLib parameters are set via string names or enum constants. Each parameter supports multiple types:

```python
# Direct value
grains.set_param(gf.PARAM_RATE, gf.PTYPE_VALUE, 1.0)

# Random offset (each grain gets value +/- random amount)
grains.set_param(gf.PARAM_RATE, gf.PTYPE_RANDOM, 0.1)

# Via string names
grains.set_param_str("rateOffset", gf.PTYPE_VALUE, 0.5)
grains.set_param_str("delayRandom", gf.PTYPE_RANDOM, 0.2)
```

Key parameters: `PARAM_RATE` (playback speed), `PARAM_DELAY` (source position), `PARAM_WINDOW` (envelope shape), `PARAM_GLISSON` (pitch glide), `PARAM_DIRECTION` (forward/reverse).

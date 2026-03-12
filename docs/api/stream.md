# Streaming

Block-based processing, ring buffers, and processor chains for streaming audio.

## Usage examples

### Ring buffer

```python
from nanodsp.stream import RingBuffer
from nanodsp.buffer import AudioBuffer
import numpy as np

# Create a stereo ring buffer with 8192 frames of capacity
rb = RingBuffer(channels=2, capacity=8192, sample_rate=48000)

# Write audio data
chunk = AudioBuffer.noise(channels=2, frames=512)
written = rb.write(chunk)

# Check available data
print(f"Available to read: {rb.available_read}")
print(f"Available to write: {rb.available_write}")

# Read audio data (consumes from buffer)
out = rb.read(256)

# Peek without consuming
peeked = rb.peek(256)

# Clear the buffer
rb.clear()
```

### Block processor (subclass)

```python
from nanodsp.stream import BlockProcessor
from nanodsp.effects import filters

class LowpassProcessor(BlockProcessor):
    def __init__(self, cutoff_hz, sample_rate=48000):
        super().__init__(block_size=512, channels=1, sample_rate=sample_rate)
        self.cutoff_hz = cutoff_hz

    def process_block(self, block):
        return filters.lowpass(block, cutoff_hz=self.cutoff_hz)

proc = LowpassProcessor(cutoff_hz=1000.0)
buf = AudioBuffer.from_file("input.wav")
filtered = proc.process(buf)
```

### Callback processor

```python
from nanodsp.stream import CallbackProcessor

# Quick inline processor with a lambda
gain_proc = CallbackProcessor(
    callback=lambda block: block * 0.5,
    block_size=512,
)
quiet = gain_proc.process(buf)
```

### Processor chain

```python
from nanodsp.stream import ProcessorChain, CallbackProcessor

# Chain multiple processors
chain = ProcessorChain(
    CallbackProcessor(lambda b: b * 2.0, block_size=512),       # boost
    CallbackProcessor(lambda b: b.pipe(filters.lowpass, cutoff_hz=2000), block_size=512),
    CallbackProcessor(lambda b: b * 0.5, block_size=512),       # attenuate
)
result = chain.process(buf)
chain.reset()
```

### Overlap-add block processing

```python
from nanodsp.stream import process_blocks

# Apply a function to overlapping blocks with automatic reconstruction
def spectral_fn(block):
    # Any per-block processing
    return block * 0.8

out = process_blocks(buf, fn=spectral_fn, block_size=2048, hop_size=512)
```

## API reference

::: nanodsp.stream
    options:
      show_if_no_docstring: false

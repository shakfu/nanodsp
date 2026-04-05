# Streaming

Block-based processing, ring buffers, and processor chains for streaming audio workflows.

## Ring buffer

A fixed-capacity circular buffer for producer/consumer audio workflows. Write audio in, read it out FIFO.

```python
from nanodsp.stream import RingBuffer
from nanodsp.buffer import AudioBuffer

# Create a stereo ring buffer with 8192 frames
rb = RingBuffer(channels=2, capacity=8192, sample_rate=48000)

# Write audio
chunk = AudioBuffer.noise(channels=2, frames=512)
written = rb.write(chunk)

# Check capacity
print(f"Available to read: {rb.available_read}")
print(f"Available to write: {rb.available_write}")

# Read (consumes from buffer)
out = rb.read(256)

# Peek without consuming
peeked = rb.peek(256)

# Clear
rb.clear()
```

!!! warning
    RingBuffer is **not thread-safe**. Use external synchronization if accessing from multiple threads.

## Block processor

Subclass `BlockProcessor` for stateful block-based processing, or use `CallbackProcessor` with a function.

```python
from nanodsp.stream import BlockProcessor, CallbackProcessor
from nanodsp.effects import filters

# Subclass approach
class LowpassProcessor(BlockProcessor):
    def __init__(self, cutoff_hz, sample_rate=48000):
        super().__init__(block_size=512, channels=1, sample_rate=sample_rate)
        self.cutoff_hz = cutoff_hz

    def process_block(self, block):
        return filters.lowpass(block, cutoff_hz=self.cutoff_hz)

proc = LowpassProcessor(cutoff_hz=1000.0)
buf = AudioBuffer.from_file("input.wav")
filtered = proc.process(buf)

# Callback approach
gain_proc = CallbackProcessor(
    callback=lambda block: block * 0.5,
    block_size=512,
)
quiet = gain_proc.process(buf)
```

## Processor chain

Chains multiple processors in series. Each block passes through all processors before the next block.

```python
from nanodsp.stream import ProcessorChain, CallbackProcessor

chain = ProcessorChain(
    CallbackProcessor(lambda b: b * 2.0, block_size=512),
    CallbackProcessor(
        lambda b: b.pipe(filters.lowpass, cutoff_hz=2000),
        block_size=512,
    ),
    CallbackProcessor(lambda b: b * 0.5, block_size=512),
)
result = chain.process(buf)
chain.reset()
```

## Overlap-add processing

`process_blocks` slices input into overlapping blocks, applies a function, and reconstructs with overlap-add.

```python
from nanodsp.stream import process_blocks

def apply_gate(block):
    """Zero out quiet frames."""
    import numpy as np
    peak = float(np.max(np.abs(block.data)))
    return block if peak > 0.01 else block * 0.0

out = process_blocks(buf, fn=apply_gate, block_size=2048, hop_size=512)
```

# I/O

Read and write WAV (8/16/24/32-bit PCM) and FLAC (16/24-bit) files. WAV uses the Python stdlib `wave` module; FLAC uses the CHOC codec.

## Usage examples

### Read and write files

```python
from nanodsp import io

# Auto-detect format by extension
buf = io.read("input.wav")
buf = io.read("input.flac")

io.write("output.wav", buf)                  # 16-bit default
io.write("output.wav", buf, bit_depth=24)    # 24-bit WAV
io.write("output.flac", buf, bit_depth=24)   # 24-bit FLAC
```

### Format-specific functions

```python
buf = io.read_wav("file.wav")
io.write_wav("out.wav", buf, bit_depth=24)

buf = io.read_flac("file.flac")
io.write_flac("out.flac", buf, bit_depth=16)
```

### Byte-level I/O (for pipes and streaming)

```python
# Parse WAV from raw bytes (e.g., from stdin)
import sys
raw = sys.stdin.buffer.read()
buf = io.read_wav_bytes(raw)

# Serialize to WAV bytes (e.g., for stdout)
out_bytes = io.write_wav_bytes(buf, bit_depth=16)
sys.stdout.buffer.write(out_bytes)
```

## API reference

::: nanodsp.io
    options:
      members:
        - read
        - write
        - read_wav
        - write_wav
        - read_wav_bytes
        - write_wav_bytes
        - read_flac
        - write_flac

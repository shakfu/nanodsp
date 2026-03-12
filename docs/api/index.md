# API Reference

nanodsp is organized into modules that accept and return [`AudioBuffer`](buffer.md) objects.

| Module | Description |
|--------|-------------|
| [`nanodsp.buffer`](buffer.md) | `AudioBuffer` -- the central data type |
| [`nanodsp.io`](io.md) | Audio file I/O (WAV, FLAC) |
| [`nanodsp.ops`](ops.md) | Core DSP operations (delay, FFT, convolution, mixing, panning) |
| [`nanodsp.effects`](effects/index.md) | Filters, effects, dynamics, reverb, mastering |
| [`nanodsp.spectral`](spectral.md) | STFT, spectral transforms, EQ matching |
| [`nanodsp.synthesis`](synthesis.md) | Oscillators, noise, drums, physical modeling |
| [`nanodsp.analysis`](analysis.md) | Loudness, spectral features, pitch, onsets, resampling |
| [`nanodsp.stream`](stream.md) | Ring buffers, block processors, streaming |

# nanodsp

High-performance Python DSP toolkit built on C++ libraries via [nanobind](https://github.com/wjakob/nanobind). All processing uses float32 in a planar `[channels, frames]` layout with block-based APIs that accept and return `AudioBuffer` objects.

## Features

- **79+ DSP functions** -- filters, effects, dynamics, reverb, synthesis, analysis, spectral processing
- **11 C++ backends** -- signalsmith, DaisySP, STK, madronalib, HISSTools, CHOC, GrainflowLib, fxdsp, DspFilters, vafilters, PolyBLEP
- **Zero-copy where possible** -- numpy arrays pass directly to C++ with GIL release for true multi-threaded parallelism
- **CLI included** -- process, analyze, synthesize, convert, and benchmark audio from the command line
- **30 built-in presets** -- mastering, voice, spatial, dynamics, creative, lo-fi, cleanup

## Quick start

```python
from nanodsp.buffer import AudioBuffer
from nanodsp.effects import filters, dynamics
from nanodsp import analysis

buf = (
    AudioBuffer.from_file("input.wav")
    .pipe(filters.highpass, freq=80.0)
    .pipe(dynamics.compress, threshold=-18.0, ratio=4.0)
    .pipe(analysis.normalize_lufs, target_lufs=-14.0)
)
buf.write("output.wav")
```

## Install

```bash
pip install nanodsp
```

Or build from source:

```bash
git clone https://github.com/shakfu/nanodsp.git
cd nanodsp
uv sync
uv run pytest
```

## Backends

| Library | License | What it provides |
|---------|---------|------------------|
| [signalsmith-dsp](https://signalsmith-audio.co.uk/code/dsp/) | MIT | Filters, FFT, delay, envelopes, spectral processing, rates, mix |
| [DaisySP](https://github.com/electro-smith/DaisySP) | MIT | Oscillators, effects, dynamics, drums, physical modeling, noise |
| [STK](https://github.com/thestk/stk) | MIT | Physical modeling instruments, generators, filters, delays, effects |
| [madronalib](https://github.com/madronalabs/madronalib) | MIT | FDN reverbs, resampling, generators, projections, windows |
| [HISSTools Library](https://github.com/AlexHarker/HISSTools_Library) | BSD-3 | Convolution, spectral processing, statistical analysis, windows |
| [CHOC](https://github.com/Tracktion/choc) | ISC | FLAC codec (read/write) |
| [GrainflowLib](https://github.com/composingcap/GrainflowLib) | MIT | Granular synthesis |
| [fxdsp](https://github.com/hamiltonkibbe/FXDsp) | MIT | Antialiased waveshaping, classic reverbs, formant filter, PSOLA, ping-pong delay, frequency shifter, ring modulator |
| [DspFilters](https://github.com/vinniefalco/DSPFilters) | MIT | Multi-order IIR filter design |
| [vafilters](https://github.com/music-dsp-collection/va-filters) | MIT | Virtual analog filters (Moog, Diode, Korg35, Oberheim) |
| [PolyBLEP et al.](https://github.com/martinfinke/PolyBLEP) | MIT | Band-limited oscillators |

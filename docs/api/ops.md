# Operations

Low-level DSP building blocks: delay, envelopes, FFT, convolution, sample rates, mixing, panning, normalization, cross-correlation, Hilbert transform, median filter, LMS adaptive filter.

## Usage examples

### Delay

```python
from nanodsp import ops
from nanodsp.buffer import AudioBuffer

buf = AudioBuffer.from_file("input.wav")

# Fixed delay (100 samples)
delayed = ops.delay(buf, delay_samples=100)

# Cubic interpolation for fractional delays
delayed = ops.delay(buf, delay_samples=50.5, interpolation="cubic")

# Time-varying delay (vibrato effect)
import numpy as np
t = np.arange(buf.frames, dtype=np.float32) / buf.sample_rate
delay_curve = 20 + 10 * np.sin(2 * np.pi * 5 * t)  # 5 Hz modulation
vibrato = ops.delay_varying(buf, delays=delay_curve)
```

### Envelopes

```python
# Smooth an envelope with a box filter
smoothed = ops.box_filter(buf, length=64)

# Multi-stage smoothing (cascaded box filters approximate Gaussian)
smooth = ops.box_stack_filter(buf, size=32, layers=4)

# Peak tracking
peaks = ops.peak_hold(buf, length=128)
decayed = ops.peak_decay(buf, length=256)
```

### FFT

```python
buf = AudioBuffer.sine(440.0, frames=1024, sample_rate=44100)

# Forward FFT (returns list of complex arrays, one per channel)
spectra = ops.rfft(buf)

# Inverse FFT back to time domain
reconstructed = ops.irfft(spectra, size=1024, sample_rate=44100)
```

### Convolution

```python
signal = AudioBuffer.from_file("dry.wav")
ir = AudioBuffer.from_file("impulse_response.wav")

# Convolve (applies reverb IR to signal)
wet = ops.convolve(signal, ir, normalize=True, trim=True)
```

### Mixing and panning

```python
a = AudioBuffer.sine(440.0, frames=44100)
b = AudioBuffer.sine(880.0, frames=44100)

# Crossfade between two buffers
blended = ops.crossfade(a, b, x=0.5)   # 50/50 mix

# Mix multiple buffers with gains
mixed = ops.mix_buffers(a, b, gains=[1.0, 0.5])

# Equal-power panning (mono -> stereo)
panned = ops.pan(a, position=0.3)   # slightly right

# Stereo widening
wide = ops.stereo_widen(stereo_buf, width=1.5)
```

### Mid-side processing

```python
stereo = AudioBuffer.noise(channels=2, frames=44100)

# Encode to mid-side
ms = ops.mid_side_encode(stereo)
# Process mid and side independently...
# Decode back to left-right
lr = ops.mid_side_decode(ms)
```

### Normalization and fades

```python
buf = AudioBuffer.from_file("input.wav")

# Peak normalize to -1 dBFS
normalized = ops.normalize_peak(buf, target_db=-1.0)

# Trim leading/trailing silence
trimmed = ops.trim_silence(buf, threshold_db=-60.0, pad_frames=100)

# Apply fades
faded = ops.fade_in(buf, duration_ms=10.0)
faded = ops.fade_out(buf, duration_ms=50.0, curve="exp")
```

### LFO

```python
# Generate a 2 Hz LFO signal
lfo_signal = ops.lfo(frames=44100, low=0.0, high=1.0, rate=2.0, sample_rate=44100)
```

### Cross-correlation and Hilbert transform

```python
# Cross-correlation between two signals
corr = ops.xcorr(buf_a, buf_b)

# Autocorrelation
auto = ops.xcorr(buf_a)

# Analytic signal envelope via Hilbert transform
env = ops.hilbert(buf)
env = ops.envelope(buf)  # alias
```

### Adaptive filtering

```python
# LMS adaptive noise cancellation
output, error = ops.lms_filter(
    buf, ref, filter_len=32, step_size=0.01, normalized=True
)
```

## API reference

::: nanodsp.ops
    options:
      show_if_no_docstring: false

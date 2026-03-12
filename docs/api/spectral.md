# Spectral

Short-time Fourier transform, spectral transforms, and EQ matching.

## Usage examples

### STFT round-trip

```python
from nanodsp import spectral
from nanodsp.buffer import AudioBuffer

buf = AudioBuffer.from_file("input.wav")

# Analyze
spec = spectral.stft(buf, window_size=2048, hop_size=512)

# Inspect
mag = spectral.magnitude(spec)    # magnitude array
ph = spectral.phase(spec)         # phase array
print(f"Frames: {spec.num_frames}, Bins: {spec.bins}")

# Reconstruct
reconstructed = spectral.istft(spec)
```

### Window types

```python
# Available: "hann", "hamming", "blackman", "bartlett", "rectangular"
spec = spectral.stft(buf, window_size=2048, window="blackman")
out = spectral.istft(spec, window="blackman")
```

### Spectral processing

```python
spec = spectral.stft(buf, window_size=2048, hop_size=512)

# Gate: silence bins below threshold
cleaned = spectral.spectral_gate(spec, threshold_db=-40.0)

# Tilt EQ: boost highs, cut lows (or vice versa)
tilted = spectral.spectral_emphasis(spec, low_db=-3.0, high_db=3.0)

# Convert between polar and complex
mag = spectral.magnitude(spec)
ph = spectral.phase(spec)
spec2 = spectral.from_polar(mag, ph, spec)

# Apply a binary mask
import numpy as np
mask = np.ones_like(mag)
mask[:, :, :10] = 0.0   # zero first 10 bins
masked = spectral.apply_mask(spec, mask)
```

### Time stretching and pitch shifting

```python
buf = AudioBuffer.from_file("input.wav")

# Slow down to half speed (double duration)
spec = spectral.stft(buf, window_size=2048, hop_size=512)
stretched = spectral.time_stretch(spec, rate=0.5)
slow = spectral.istft(stretched)

# Pitch shift up 5 semitones (preserves duration)
shifted = spectral.pitch_shift_spectral(buf, semitones=5.0)
```

### Spectral effects

```python
spec = spectral.stft(buf, window_size=2048, hop_size=512)

# Freeze a single frame into a sustained texture
frozen = spectral.spectral_freeze(spec, frame_index=10, num_frames=200)

# Morph between two sounds
spec_a = spectral.stft(buf_a, window_size=2048, hop_size=512)
spec_b = spectral.stft(buf_b, window_size=2048, hop_size=512)
morphed = spectral.spectral_morph(spec_a, spec_b, mix=0.5)

# Phase locking (identity phase-lock for cleaner stretching)
locked = spectral.phase_lock(spec)
```

### Noise reduction

```python
# Assumes first 10 frames are noise-only
spec = spectral.stft(buf, window_size=2048, hop_size=512)
denoised = spectral.spectral_denoise(spec, noise_frames=10, reduction_db=-20.0)
clean = spectral.istft(denoised)
```

### EQ matching

```python
# Make source sound like target in tonal balance
matched = spectral.eq_match(source_buf, target_buf, window_size=4096)
```

### Frequency / bin conversion

```python
spec = spectral.stft(buf, window_size=2048)

freq = spectral.bin_freq(spec, bin_index=10)     # bin -> Hz
b = spectral.freq_to_bin(spec, freq_hz=1000.0)   # Hz -> bin
```

## API reference

::: nanodsp.spectral
    options:
      show_if_no_docstring: false

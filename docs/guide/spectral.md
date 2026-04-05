# Spectral Processing

Algorithms that operate in the frequency domain via the Short-Time Fourier Transform (STFT).

## STFT round-trip

The STFT decomposes a signal into overlapping windowed frames, applies an FFT to each, and produces a complex-valued spectrogram. The ISTFT reverses this with overlap-add.

```python
from nanodsp import spectral
from nanodsp.buffer import AudioBuffer

buf = AudioBuffer.from_file("input.wav")

# Analyze
spec = spectral.stft(buf, window_size=2048, hop_size=512)

# Inspect
print(f"Channels: {spec.channels}")
print(f"STFT frames: {spec.num_frames}")
print(f"Frequency bins: {spec.bins}")

mag = spectral.magnitude(spec)     # magnitude array
ph = spectral.phase(spec)          # phase array

# Reconstruct
reconstructed = spectral.istft(spec)
```

## Window types

```python
# Available: "hann", "hamming", "blackman", "bartlett", "rectangular"
spec = spectral.stft(buf, window_size=2048, window="blackman")
out = spectral.istft(spec, window="blackman")
```

## Time stretching

Changes duration without changing pitch. Uses the phase vocoder technique (Flanagan & Golden, 1966; Laroche & Dolson, 1999).

```python
spec = spectral.stft(buf, window_size=2048, hop_size=512)

# Half speed (double duration)
slow = spectral.istft(spectral.time_stretch(spec, rate=0.5))

# Double speed (half duration)
fast = spectral.istft(spectral.time_stretch(spec, rate=2.0))

# Phase-locked stretching (cleaner for tonal material)
locked = spectral.phase_lock(spectral.time_stretch(spec, rate=0.75))
slow_clean = spectral.istft(locked)
```

## Spectral pitch shifting

Changes pitch without changing duration by combining time stretching with resampling.

```python
# Up 5 semitones
shifted = spectral.pitch_shift_spectral(buf, semitones=5.0)

# Down one octave
low = spectral.pitch_shift_spectral(buf, semitones=-12.0)
```

## Spectral processing

```python
spec = spectral.stft(buf, window_size=2048, hop_size=512)

# Gate: silence bins below threshold
cleaned = spectral.spectral_gate(spec, threshold_db=-40.0, noise_floor_db=-80.0)

# Tilt EQ: boost highs, cut lows
tilted = spectral.spectral_emphasis(spec, low_db=-3.0, high_db=3.0)

# Apply a custom mask
import numpy as np
mask = np.ones(spec.bins, dtype=np.float32)
mask[:10] = 0.0     # zero first 10 bins (remove low frequencies)
masked = spectral.apply_mask(spec, mask)

# Reconstruct any of these
result = spectral.istft(cleaned)
```

## Spectral freeze

Repeats a single STFT frame indefinitely, creating a sustained "frozen" texture from an instant of the input.

```python
spec = spectral.stft(buf, window_size=2048, hop_size=512)
frozen = spectral.spectral_freeze(spec, frame_index=10, num_frames=200)
sustained = spectral.istft(frozen)
```

## Spectral morphing

Interpolates the magnitude spectra of two spectrograms while preserving the phase of the first. Creates smooth timbral transitions.

```python
buf_a = AudioBuffer.from_file("guitar.wav")
buf_b = AudioBuffer.from_file("voice.wav")

spec_a = spectral.stft(buf_a, window_size=2048, hop_size=512)
spec_b = spectral.stft(buf_b, window_size=2048, hop_size=512)

# 50/50 blend of timbres
morphed = spectral.spectral_morph(spec_a, spec_b, mix=0.5)
hybrid = spectral.istft(morphed)
```

## Noise reduction

Estimates a noise floor from the first N frames, then attenuates bins at or below it. Works best when the signal starts with noise-only content.

```python
spec = spectral.stft(buf, window_size=2048, hop_size=512)

# Use first 10 frames as noise profile
denoised = spectral.spectral_denoise(spec, noise_frames=10, reduction_db=-20.0)
clean = spectral.istft(denoised)

# More aggressive with smoothing to reduce musical noise artifacts
strong = spectral.spectral_denoise(spec, noise_frames=15, reduction_db=-40.0, smoothing=3)
```

## EQ matching

Analyzes the spectral envelope of a target, computes the ratio to the source, and applies it as a filter. Makes one recording's tonal balance match another.

```python
source = AudioBuffer.from_file("my_mix.wav")
reference = AudioBuffer.from_file("pro_mix.wav")

# Match tonal balance
matched = spectral.eq_match(source, reference, window_size=4096, smoothing=8)
```

## Frequency / bin conversion

```python
spec = spectral.stft(buf, window_size=2048)

freq = spectral.bin_freq(spec, bin_index=10)      # bin -> Hz
b = spectral.freq_to_bin(spec, freq_hz=1000.0)    # Hz -> bin
```

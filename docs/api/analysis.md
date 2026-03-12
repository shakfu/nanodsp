# Analysis

Loudness metering, spectral features, pitch detection, onset detection, and resampling.

## Usage examples

### Loudness metering

```python
from nanodsp import analysis
from nanodsp.buffer import AudioBuffer

buf = AudioBuffer.from_file("input.wav")

# Measure integrated loudness (ITU-R BS.1770-4)
lufs = analysis.loudness_lufs(buf)
print(f"Loudness: {lufs:.1f} LUFS")

# Normalize to -14 LUFS (streaming target)
normalized = analysis.normalize_lufs(buf, target_lufs=-14.0)
```

### Spectral features

```python
# Brightness tracking
centroid = analysis.spectral_centroid(buf, window_size=2048)

# Spectral spread
bandwidth = analysis.spectral_bandwidth(buf)

# High-frequency rolloff (85th percentile)
rolloff = analysis.spectral_rolloff(buf, percentile=0.85)

# Onset-correlated spectral change
flux = analysis.spectral_flux(buf, rectify=True)

# Noisiness measure (0 = tonal, 1 = noise-like)
flatness = analysis.spectral_flatness_curve(buf)

# Pitch class distribution (12 bins)
chroma = analysis.chromagram(buf, n_chroma=12, tuning_hz=440.0)
```

### Pitch detection

```python
# YIN algorithm for monophonic f0 estimation
f0, confidence = analysis.pitch_detect(
    buf, method="yin", fmin=80.0, fmax=800.0, threshold=0.2
)
# f0: array of frequency estimates per frame
# confidence: array of confidence values (higher = more reliable)
```

### Onset detection

```python
# Detect note onsets
onsets = analysis.onset_detect(buf, method="spectral_flux", threshold=0.5)
# onsets: array of frame indices

# With backtracking (move to nearest energy minimum)
onsets = analysis.onset_detect(buf, backtrack=True)
```

### Resampling

```python
# Polyphase resampling (madronalib backend)
buf_48k = analysis.resample(buf, target_sr=48000.0)

# FFT-based resampling
buf_22k = analysis.resample_fft(buf, target_sr=22050.0)
```

### Delay estimation (GCC-PHAT)

```python
# Estimate time delay between two microphone signals
delay_sec, correlation = analysis.gcc_phat(mic1, mic2)
print(f"Estimated delay: {delay_sec * 1000:.2f} ms")
```

## API reference

::: nanodsp.analysis
    options:
      show_if_no_docstring: false

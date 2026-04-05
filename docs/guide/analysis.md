# Analysis

Loudness metering, spectral features, pitch detection, onset detection, and resampling.

## Loudness metering (ITU-R BS.1770-4)

Measures integrated loudness per the broadcast standard. Applies K-weighting, computes per-block power in 400 ms windows, then applies absolute (-70 LUFS) and relative (-10 dB) gating.

```python
from nanodsp import analysis
from nanodsp.buffer import AudioBuffer

buf = AudioBuffer.from_file("input.wav")

# Measure integrated loudness
lufs = analysis.loudness_lufs(buf)
print(f"Loudness: {lufs:.1f} LUFS")

# Normalize to streaming target (-14 LUFS)
normalized = analysis.normalize_lufs(buf, target_lufs=-14.0)

# Normalize to broadcast target (-23 LUFS)
broadcast = analysis.normalize_lufs(buf, target_lufs=-23.0)
```

## Spectral features

Frame-by-frame measurements computed from the STFT magnitude:

```python
# Brightness tracking (weighted mean frequency)
centroid = analysis.spectral_centroid(buf, window_size=2048)

# Spectral spread around the centroid
bandwidth = analysis.spectral_bandwidth(buf)

# Frequency below which 85% of energy lies
rolloff = analysis.spectral_rolloff(buf, percentile=0.85)

# Frame-to-frame spectral change (useful for onset detection)
flux = analysis.spectral_flux(buf, rectify=True)

# Noisiness measure (0 = tonal, 1 = noise-like)
flatness = analysis.spectral_flatness_curve(buf)

# Pitch class distribution (12-bin chromagram)
chroma = analysis.chromagram(buf, n_chroma=12, tuning_hz=440.0)
```

## Pitch detection (YIN)

The YIN algorithm (de Cheveigne & Kawahara, 2002) estimates fundamental frequency using the cumulative mean normalized difference function with parabolic interpolation. Good for monophonic signals; degrades with polyphonic content.

```python
# Detect pitch in speech range
f0, confidence = analysis.pitch_detect(
    buf, method="yin", fmin=80.0, fmax=800.0, threshold=0.2
)
# f0: Hz per frame (0.0 where unvoiced)
# confidence: 0.0--1.0 per frame

# Detect pitch in instrument range
f0, confidence = analysis.pitch_detect(buf, fmin=50.0, fmax=2000.0)

# Stricter voicing detection (lower threshold)
f0, confidence = analysis.pitch_detect(buf, threshold=0.1)
```

## Onset detection

Detects note onsets using spectral flux with adaptive peak-picking.

```python
# Detect onsets
onsets = analysis.onset_detect(buf, method="spectral_flux")
print(f"Found {len(onsets)} onsets at samples: {onsets}")

# With backtracking (move to nearest energy minimum)
onsets = analysis.onset_detect(buf, backtrack=True)

# Custom threshold
onsets = analysis.onset_detect(buf, threshold=0.5)
```

## Resampling

```python
# Polyphase resampling (madronalib) -- high quality for standard rate conversions
buf_48k = analysis.resample(buf, target_sr=48000.0)
buf_22k = analysis.resample(buf, target_sr=22050.0)

# FFT-based resampling -- mathematically exact for bandlimited signals
buf_96k = analysis.resample_fft(buf, target_sr=96000.0)
```

## Delay estimation (GCC-PHAT)

Generalized Cross-Correlation with Phase Transform (Knapp & Carter, 1976). Estimates time delay between two signals using phase-weighted cross-correlation. Robust to reverberation and noise.

```python
# Estimate delay between two microphone signals
delay_sec, correlation = analysis.gcc_phat(mic1, mic2)
print(f"Estimated delay: {delay_sec * 1000:.2f} ms")

# With explicit sample rate override
delay_sec, corr = analysis.gcc_phat(buf, ref, sample_rate=44100.0)
```

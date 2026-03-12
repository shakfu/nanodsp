# nanodsp

High-performance Python DSP toolkit built on C++ libraries via [nanobind](https://github.com/wjakob/nanobind). All processing uses float32 in a planar `[channels, frames]` layout with block-based APIs that accept and return `AudioBuffer` objects.

## Backends

| Library | License | What it provides |
|---------|---------|------------------|
| [signalsmith-dsp](https://signalsmith-audio.co.uk/code/dsp/) | MIT | Filters, FFT, delay, envelopes, spectral processing, rates, mix |
| [DaisySP](https://github.com/electro-smith/DaisySP) | MIT | Oscillators, effects, dynamics, drums, physical modeling, noise |
| [STK](https://github.com/thestk/stk) | MIT | Physical modeling instruments, generators, filters, delays, effects |
| [madronalib](https://github.com/madronalabs/madronalib) | MIT | FDN reverbs, resampling, generators, projections, windows |
| [HISSTools Library](https://github.com/AlexHarker/HISSTools_Library) | BSD-3 | Convolution, spectral processing, statistical analysis, windows |
| [CHOC](https://github.com/Tracktion/choc) | ISC | FLAC codec (read/write) |
| [GrainflowLib](https://github.com/composingcap/GrainflowLib) | MIT | Granular synthesis (grain collections, panning, recording, phasor) |
| [fxdsp](https://github.com/hamiltonkibbe/FXDsp) | MIT | Antialiased waveshaping, Schroeder/Moorer reverbs, formant filter, PSOLA pitch shift, MinBLEP oscillators |
| [DspFilters](https://github.com/vinniefalco/DSPFilters) | MIT | Multi-order IIR filter design (Butterworth, Chebyshev I/II, Elliptic, Bessel) |
| [vafilters](https://github.com/music-dsp-collection/va-filters) | MIT | Virtual analog filters (Moog, Diode, Korg35, Oberheim) via Faust |
| [PolyBLEP et al.](https://github.com/martinfinke/PolyBLEP) | MIT | Band-limited oscillators (PolyBLEP, BLIT, DPW, MinBLEP, wavetable) |

## Requirements

- Python >= 3.10
- numpy
- C++17 compiler
- CMake >= 3.15

## Install

```bash
pip install nanodsp
```

Or if you prefer to build from source (requires `uv` and `cmake`):

```bash
git clone https://github.com/shakfu/nanodsp.git
cd nanodsp
uv sync            # install dependencies + build extension
uv run pytest      # run tests
uv build           # build wheel
```

Use `make help` for additional targets (build, test, lint, format, typecheck, qa, coverage, etc.).

## CLI

nanodsp ships with a command-line interface accessible via `nanodsp` or `python -m nanodsp`.

```bash
# File info
nanodsp info drums.wav
nanodsp info drums.wav --json

# Process with effect chain (single file)
nanodsp process input.wav -o output.wav \
  -f highpass:cutoff_hz=80 \
  -f compress:ratio=4,threshold=-18 \
  -f normalize_peak:target_db=-1

# Apply a preset
nanodsp process vocals.wav -o out.wav -p vocal_chain
nanodsp process input.wav -o out.wav -f lowpass:cutoff_hz=12000 -p master

# Batch mode -- process multiple files to a directory
nanodsp process *.wav -O out/ -f lowpass:cutoff_hz=2000

# Dry run -- show chain without reading/writing files
nanodsp process input.wav -n -f highpass:cutoff_hz=80 -f compress:ratio=4

# Verbose / quiet
nanodsp -v process input.wav -o out.wav -f lowpass:cutoff_hz=1000
nanodsp -q process input.wav -o out.wav -p master

# Analyze
nanodsp analyze input.wav loudness
nanodsp analyze input.wav pitch --fmin=80 --fmax=800
nanodsp analyze input.wav onsets --json
nanodsp analyze input.wav info

# Synthesize
nanodsp synth tone.wav sine --freq=440 --duration=2.0
nanodsp synth kick.wav drum --type=analog_bass_drum --freq=60
nanodsp synth melody.wav note --instrument=clarinet --freq=440 --duration=1.0
nanodsp synth seq.wav sequence --instrument=flute \
  --notes='[{"freq":440,"start":0,"dur":0.5},{"freq":554,"start":0.5,"dur":0.5}]'

# Convert
nanodsp convert input.wav output.flac
nanodsp convert input.wav output.wav --sample-rate=44100 --channels=1 -b 24

# Presets
nanodsp preset list
nanodsp preset list spatial
nanodsp preset info master
nanodsp preset apply master input.wav output.wav target_lufs=-16

# Pipe (stdin/stdout streaming)
cat input.wav | nanodsp pipe -f lowpass:cutoff_hz=1000 > output.wav
cat input.wav | nanodsp pipe -p telephone > output.wav
nanodsp pipe -f highpass:cutoff_hz=80 < in.wav | nanodsp pipe -f compress:ratio=4 > out.wav

# Benchmark
nanodsp benchmark lowpass:cutoff_hz=1000
nanodsp benchmark compress:ratio=4,threshold=-20 -n 100 --duration=2.0
nanodsp benchmark reverb:preset=hall --channels=2 --json

# List available functions
nanodsp list
nanodsp list filters
nanodsp list effects
```

### Presets

30 built-in presets across 8 categories:

| Category | Presets |
|----------|---------|
| mastering | `master`, `master_pop`, `master_hiphop`, `master_classical`, `master_edm`, `master_podcast` |
| voice | `vocal_chain` |
| spatial | `room`, `hall`, `plate`, `cathedral`, `chamber` |
| dynamics | `gentle_compress`, `heavy_compress`, `brick_wall` |
| creative | `radio`, `underwater`, `megaphone`, `tape_warmth`, `shimmer`, `vaporwave`, `walkie_talkie` |
| lofi | `telephone`, `lo_fi`, `vinyl`, `8bit` |
| cleanup | `dc_remove`, `de_noise`, `normalize`, `normalize_lufs` |

## Quick start

```python
from nanodsp.buffer import AudioBuffer
from nanodsp import effects, analysis

# Read, process, write
buf = (
    AudioBuffer.from_file("input.wav")
    .pipe(effects.highpass, freq=80.0)
    .pipe(effects.compress, threshold=-18.0, ratio=4.0)
    .pipe(analysis.normalize_lufs, target_lufs=-14.0)
)
buf.write("output.wav")
```

## Modules

### `nanodsp.buffer` -- AudioBuffer

The central data type. A 2D float32 array with shape `[channels, frames]` plus metadata (sample_rate, channel_layout, label).

```python
from nanodsp.buffer import AudioBuffer

# Construction
buf = AudioBuffer(np.zeros((2, 44100), dtype=np.float32), sample_rate=44100)
buf = AudioBuffer.from_file("input.wav")       # read WAV/FLAC
buf = AudioBuffer.sine(440.0, channels=1, frames=44100, sample_rate=44100)
buf = AudioBuffer.noise(channels=2, frames=44100, sample_rate=44100)
buf = AudioBuffer.zeros(channels=1, frames=1024, sample_rate=44100)
buf = AudioBuffer.impulse(channels=1, frames=1024, sample_rate=44100)

# Properties
buf.channels        # number of channels
buf.frames          # number of frames
buf.sample_rate     # sample rate in Hz
buf.duration        # duration in seconds
buf.mono            # 1D view (mono buffers only)
buf.channel(0)      # 1D view of channel 0

# Channel operations
buf.to_mono("mean")                    # downmix
buf.to_channels(2)                     # upmix mono to stereo
AudioBuffer.concat_channels(a, b)      # stack channels

# Arithmetic
buf + other          # add
buf * 0.5            # scale
buf.gain_db(-6.0)    # apply dB gain

# I/O
buf.write("output.wav")                # write WAV/FLAC (detected by extension)
buf.write("output.flac", bit_depth=24)

# Pipeline
buf.pipe(effects.lowpass, freq=1000.0)
```

### `nanodsp.io` -- Audio file I/O

Read and write WAV (8/16/24/32-bit PCM) and FLAC (16/24-bit) files. Zero external dependencies for WAV (uses stdlib `wave`); FLAC uses the CHOC codec.

```python
from nanodsp import io

buf = io.read("file.wav")          # auto-detect by extension
buf = io.read("file.flac")
io.write("out.wav", buf)           # 16-bit default
io.write("out.flac", buf, bit_depth=24)

# Format-specific
buf = io.read_wav("file.wav")
io.write_wav("out.wav", buf, bit_depth=24)
buf = io.read_flac("file.flac")
io.write_flac("out.flac", buf, bit_depth=16)

# Byte-level (for stdin/stdout/pipe workflows)
buf = io.read_wav_bytes(raw_bytes)           # parse WAV from bytes
raw = io.write_wav_bytes(buf, bit_depth=16)  # serialize to WAV bytes
```

### `nanodsp.ops` -- Core DSP operations

Low-level building blocks: delay, envelopes, FFT, convolution, sample rates, mixing, panning, normalization, cross-correlation, Hilbert transform, median filter, LMS adaptive filter.

```python
from nanodsp import ops

# Delay
ops.delay(buf, delay_samples=100)
ops.delay_varying(buf, delays=delay_curve)

# Envelopes
ops.box_filter(buf, length=64)
ops.box_stack_filter(buf, size=32, layers=4)
ops.peak_hold(buf, length=128)
ops.peak_decay(buf, length=256)

# FFT
spectra = ops.rfft(buf)                        # forward real FFT
buf = ops.irfft(spectra, size=1024, sample_rate=44100)  # inverse

# Convolution
ops.convolve(buf, ir, normalize=True)

# Sample rates
ops.upsample_2x(buf)
ops.oversample_roundtrip(buf)

# Mixing
ops.hadamard(buf)              # Hadamard matrix mixing
ops.householder(buf)           # Householder reflection
ops.crossfade(buf_a, buf_b, x=0.5)
ops.mix_buffers(a, b, c, gains=[1.0, 0.5, 0.8])

# LFO
lfo_signal = ops.lfo(frames=44100, low=0.0, high=1.0, rate=2.0)

# Normalization
ops.normalize_peak(buf, target_db=-1.0)
ops.trim_silence(buf, threshold_db=-60.0)

# Fades
ops.fade_in(buf, duration_ms=10.0)
ops.fade_out(buf, duration_ms=50.0, curve="exp")

# Panning and stereo
ops.pan(buf, position=0.3)     # equal-power pan
ops.mid_side_encode(buf)
ops.mid_side_decode(buf)
ops.stereo_widen(buf, width=1.5)

# Cross-correlation
corr = ops.xcorr(buf_a, buf_b)        # cross-correlation
auto = ops.xcorr(buf)                  # autocorrelation

# Hilbert / envelope
env = ops.hilbert(buf)                 # analytic signal envelope
env = ops.envelope(buf)                # alias for hilbert

# Median filter
ops.median_filter(buf, kernel_size=5)

# LMS adaptive filter
output, error = ops.lms_filter(buf, ref, filter_len=32, step_size=0.01)
```

### `nanodsp.effects` -- Filters, effects, dynamics, mastering

70 functions covering signalsmith biquad filters, DaisySP effects/filters/dynamics, composed effects, reverbs, mastering chains, STK effects, automatic gain control, antialiased waveshaping, classic reverbs, formant filtering, PSOLA pitch shifting, and multi-order IIR filters.

#### Biquad filters (signalsmith)

All filter functions take `freq` in Hz, auto-converted to normalized frequency.

```python
from nanodsp import effects

effects.lowpass(buf, freq=1000.0, q=0.707)
effects.highpass(buf, freq=80.0)
effects.bandpass(buf, freq=1000.0, q=2.0)
effects.notch(buf, freq=50.0)
effects.peak(buf, freq=1000.0, gain=2.0, q=1.0)
effects.peak_db(buf, freq=1000.0, db=6.0)
effects.high_shelf(buf, freq=8000.0, gain=1.5)
effects.high_shelf_db(buf, freq=8000.0, db=3.0)
effects.low_shelf(buf, freq=200.0, gain=0.8)
effects.low_shelf_db(buf, freq=200.0, db=-2.0)
effects.allpass(buf, freq=1000.0)
```

#### DaisySP effects

```python
effects.autowah(buf, wah=0.5)
effects.chorus(buf, rate=1.0, depth=0.5)
effects.decimator(buf, downsample_factor=0.5, bitcrush_factor=0.5)
effects.flanger(buf, rate=0.2, depth=0.5, feedback=0.5)
effects.overdrive(buf, drive=0.7)
effects.phaser(buf, rate=0.3, depth=0.5, feedback=0.5)
effects.pitch_shift(buf, semitones=12.0)
effects.sample_rate_reduce(buf, factor=0.5)
effects.tremolo(buf, rate=5.0, depth=0.8)
effects.wavefold(buf, gain=2.0)
effects.bitcrush(buf, bits=8)
effects.fold(buf, gain=2.0)
effects.reverb_sc(buf, feedback=0.8, lpfreq=10000.0)
effects.dc_block(buf)
```

#### DaisySP filters

```python
effects.svf_lowpass(buf, freq=1000.0, res=0.5)
effects.svf_highpass(buf, freq=200.0, res=0.5)
effects.svf_bandpass(buf, freq=1000.0, res=0.7)
effects.svf_notch(buf, freq=1000.0, res=0.5)
effects.svf_peak(buf, freq=1000.0, res=0.8)
effects.ladder_filter(buf, freq=800.0, res=0.6, mode="lp24")
effects.moog_ladder(buf, freq=1000.0, res=0.7)
effects.tone_lowpass(buf, freq=2000.0)
effects.tone_highpass(buf, freq=100.0)
effects.modal_bandpass(buf, freq=440.0, q=50.0)
effects.comb_filter(buf, freq=500.0, revtime=0.5)
```

#### Dynamics

```python
effects.compress(buf, threshold=-20.0, ratio=4.0, attack=0.01, release=0.1)
effects.limit(buf, threshold=-1.0)
```

#### Composed effects

```python
effects.saturate(buf, drive=0.7, mode="soft")    # soft, hard, or tape
effects.exciter(buf, freq=3000.0, drive=0.4)
effects.de_esser(buf, freq=6000.0, threshold=-20.0)
effects.parallel_compress(buf, mix=0.5, threshold=-24.0, ratio=8.0)
effects.noise_gate(buf, threshold_db=-40.0)
effects.stereo_delay(buf, delay_l=0.25, delay_r=0.375, feedback=0.4, ping_pong=True)
effects.multiband_compress(buf, crossovers=[200.0, 2000.0, 8000.0])
```

#### Reverb

```python
effects.reverb(buf, preset="hall", decay=2.0, mix=0.3)
# Presets: room, hall, plate, chamber, cathedral
```

#### Mastering and vocal chains

```python
effects.master(buf, target_lufs=-14.0)
effects.vocal_chain(buf, de_ess_freq=6000.0, comp_threshold=-18.0)
```

#### STK effects

```python
effects.stk_reverb(buf, algorithm="freeverb", decay=1.5, mix=0.3)
effects.stk_chorus(buf, mod_depth=0.02, mod_freq=1.0, mix=0.5)
effects.stk_echo(buf, delay=0.25, max_delay=1.0, mix=0.5)
```

#### Automatic gain control

```python
effects.agc(buf, target_level=1.0, max_gain_db=60.0, average_len=100, attack=0.01, release=0.01)
```

#### Antialiased waveshaping (fxdsp)

```python
effects.aa_hard_clip(buf, drive=2.0)       # 1st-order antiderivative hard clip
effects.aa_soft_clip(buf, drive=2.0)       # 1st-order antiderivative soft clip
effects.aa_wavefold(buf, drive=2.0)        # 2nd-order Buchla-style wavefolder
```

#### Classic reverbs (fxdsp)

```python
effects.schroeder_reverb(buf, feedback=0.7, diffusion=0.5, mod_depth=0.0)
effects.moorer_reverb(buf, feedback=0.7, diffusion=0.7, mod_depth=0.1)
```

#### Formant filter and PSOLA pitch shifting (fxdsp)

```python
effects.formant_filter(buf, vowel="a")           # 'a','e','i','o','u'
effects.psola_pitch_shift(buf, semitones=5.0)     # pitch-synchronous overlap-add
```

#### Multi-order IIR filters (DspFilters)

```python
effects.iir_filter(buf, family="butterworth", filter_type="lowpass", order=4, freq=1000.0)
effects.iir_filter(buf, family="chebyshev1", filter_type="highpass", order=6, freq=200.0, ripple_db=1.0)
effects.iir_filter(buf, family="elliptic", filter_type="bandpass", order=4, freq=1000.0, width=500.0)
effects.iir_filter(buf, family="bessel", filter_type="lowpass", order=8, freq=5000.0)

# Return SOS coefficients without applying
sos = effects.iir_design("butterworth", "lowpass", order=4, sample_rate=44100, freq=1000.0)
```

### `nanodsp.spectral` -- STFT and spectral processing

Short-time Fourier transform, spectral utilities, and spectral transforms.

```python
from nanodsp import spectral

# STFT
spec = spectral.stft(buf, window_size=2048, hop_size=512)
buf = spectral.istft(spec)

# Spectral utilities
mag = spectral.magnitude(spec)
ph = spectral.phase(spec)
spec = spectral.from_polar(mag, ph)
spec = spectral.apply_mask(spec, mask)
spec = spectral.spectral_gate(spec, threshold_db=-40.0)
spec = spectral.spectral_emphasis(spec, low_db=-3.0, high_db=3.0)
freq = spectral.bin_freq(bin_index=10, fft_size=2048, sample_rate=44100)
b = spectral.freq_to_bin(freq=1000.0, fft_size=2048, sample_rate=44100)

# Spectral transforms
stretched = spectral.time_stretch(buf, rate=0.5)        # half speed
locked = spectral.phase_lock(spec)                       # identity phase-locking
frozen = spectral.spectral_freeze(buf, frame_index=10)   # frozen texture
morphed = spectral.spectral_morph(spec_a, spec_b, x=0.5)
shifted = spectral.pitch_shift_spectral(buf, semitones=5.0)
denoised = spectral.spectral_denoise(buf, noise_frames=10)

# EQ matching
matched = spectral.eq_match(source_buf, target_buf, strength=1.0)
```

### `nanodsp.synthesis` -- Oscillators, noise, drums, physical modeling

Sound generators using DaisySP and STK backends.

```python
from nanodsp import synthesis

# Oscillators
synthesis.oscillator(freq=440.0, waveform="saw", frames=44100)
synthesis.fm2(freq=440.0, ratio=2.0, index=1.0, frames=44100)
synthesis.formant_oscillator(freq=440.0, formant_freq=800.0, frames=44100)
synthesis.bl_oscillator(freq=440.0, waveform="saw", frames=44100)

# Noise
synthesis.white_noise(frames=44100, amp=0.5)
synthesis.clocked_noise(freq=1000.0, frames=44100)
synthesis.dust(density=100.0, frames=44100)

# Drums
synthesis.analog_bass_drum(freq=60.0, frames=44100)
synthesis.analog_snare_drum(freq=200.0, frames=44100)
synthesis.hihat(freq=3000.0, frames=44100)
synthesis.synthetic_bass_drum(freq=60.0, frames=44100)
synthesis.synthetic_snare_drum(freq=200.0, frames=44100)

# Physical modeling
synthesis.karplus_strong(buf, freq=440.0, brightness=0.5, damping=0.5)
synthesis.modal_voice(freq=440.0, frames=44100)
synthesis.string_voice(freq=440.0, frames=44100)
synthesis.pluck(freq=440.0, frames=44100)
synthesis.drip(freq=1000.0, frames=44100)

# STK synthesis
synthesis.synth_note("clarinet", freq=440.0, amplitude=0.8, duration=1.0)
synthesis.synth_sequence("flute", notes=[
    {"freq": 440.0, "amp": 0.8, "start": 0.0, "dur": 0.5},
    {"freq": 554.37, "amp": 0.7, "start": 0.5, "dur": 0.5},
])

# MinBLEP oscillators (fxdsp)
synthesis.minblep(frames=44100, freq=440.0, waveform="saw")     # saw, rsaw, square, triangle
synthesis.minblep(frames=44100, freq=440.0, waveform="square", pulse_width=0.3)
```

Available STK instruments: `clarinet`, `flute`, `brass`, `bowed`, `plucked`, `sitar`, `stifkarp`, `saxofony`, `recorder`, `blowbotl`, `blowhole`, `whistle`.

### `nanodsp.analysis` -- Loudness, spectral features, pitch, onsets, resampling

```python
from nanodsp import analysis

# Loudness (ITU-R BS.1770-4)
lufs = analysis.loudness_lufs(buf)
buf = analysis.normalize_lufs(buf, target_lufs=-14.0)

# Spectral features
centroid = analysis.spectral_centroid(buf)
bandwidth = analysis.spectral_bandwidth(buf)
rolloff = analysis.spectral_rolloff(buf, percentile=0.85)
flux = analysis.spectral_flux(buf, rectify=True)
flatness = analysis.spectral_flatness_curve(buf)
chroma = analysis.chromagram(buf, n_chroma=12, tuning_hz=440.0)

# Pitch detection (YIN algorithm)
f0, confidence = analysis.pitch_detect(buf, method="yin", fmin=50.0, fmax=2000.0)

# Onset detection
onsets = analysis.onset_detect(buf, method="spectral_flux", threshold=0.5)

# Resampling
buf_48k = analysis.resample(buf, target_sr=48000.0)       # madronalib backend
buf_22k = analysis.resample_fft(buf, target_sr=22050.0)   # FFT-based

# GCC-PHAT delay estimation
delay_sec, corr = analysis.gcc_phat(buf, ref)
```

### `nanodsp.stream` -- Real-time streaming infrastructure

Block-based processing, ring buffers, and processor chains for streaming audio.

```python
from nanodsp.stream import (
    RingBuffer, BlockProcessor, CallbackProcessor, ProcessorChain, process_blocks
)

# Ring buffer
rb = RingBuffer(channels=2, capacity=8192)
rb.write(frame_data)
chunk = rb.read(512)

# Block processor
class MyProcessor(BlockProcessor):
    def process_block(self, block):
        return block * 0.5

proc = MyProcessor(block_size=512)
out = proc.process(buf)

# Callback processor
proc = CallbackProcessor(block_size=512, fn=lambda b: b * 0.5)

# Chain processors
chain = ProcessorChain([proc1, proc2, proc3])
out = chain.process(buf)

# Process with overlap-add
out = process_blocks(buf, block_size=2048, hop_size=512, fn=my_spectral_fn)
```

### `nanodsp._core.grainflow` -- Granular synthesis (low-level)

Direct access to GrainflowLib's granular synthesis engine.

```python
from nanodsp._core import grainflow as gf

# Create a buffer and fill with audio data
buf = gf.GfBuffer(4096, 1, 48000)
buf.set_data(audio_array)  # [channels, frames] float32

# Create a grain collection
gc = gf.GrainCollection(num_grains=8, samplerate=48000)
gc.set_buffer(buf, gf.BUF_BUFFER, 0)  # 0 = set for all grains

# Set parameters (enum-based or string reflection)
gc.param_set(0, gf.PARAM_RATE, gf.PTYPE_BASE, 1.0)
gc.param_set_str(0, "delayRandom", 10.0)

# Generate a clock and process
phasor = gf.Phasor(rate=10.0, samplerate=48000)
clock = phasor.perform(256).reshape(1, 256)
traversal = np.linspace(0, 0.5, 256, dtype=np.float32).reshape(1, 256)
fm = np.zeros((1, 256), dtype=np.float32)
am = np.zeros((1, 256), dtype=np.float32)

# Returns 8-element tuple: (output, state, progress, playhead, amp, envelope, buf_ch, stream_ch)
result = gc.process(clock, traversal, fm, am, 48000)
grain_output = result[0]  # [num_grains, block_size]

# Pan grains to stereo
panner = gf.Panner(in_channels=8, out_channels=2, pan_mode=gf.PAN_STEREO)
panner.set_pan_spread(0.5)
stereo = panner.process(grain_output, result[1], out_channels=2)  # [2, block_size]
```

### `nanodsp._core` -- C++ bindings (low-level)

Direct access to the C++ extension module with 17 submodules. All high-level Python modules build on these.

| Submodule | Backend | Contents |
|-----------|---------|----------|
| `filters` | signalsmith | `Biquad` with 16 filter designs, `BiquadDesign` enum |
| `fft` | signalsmith | `FFT` (complex), `RealFFT` (real) |
| `delay` | signalsmith | `Delay` (linear), `DelayCubic` (cubic interpolation) |
| `envelopes` | signalsmith | `CubicLfo`, `BoxFilter`, `BoxStackFilter`, `PeakHold`, `PeakDecayLinear` |
| `spectral` | signalsmith | `STFT` (multi-channel analysis/synthesis) |
| `rates` | signalsmith | `Oversampler2x` |
| `mix` | signalsmith | `Hadamard`, `Householder`, `cheap_energy_crossfade` |
| `daisysp` | DaisySP | 9 submodules, ~60 classes (oscillators, filters, effects, dynamics, drums, noise, physical modeling, control, utility) |
| `stk` | STK | 5 submodules, 39 classes (instruments, generators, filters, delays, effects) |
| `madronalib` | madronalib | FDN reverbs, resampling, generators, 18 projection functions, 6 window functions |
| `hisstools` | HISSTools | Convolution (mono/multi), spectral processing, 24 statistics functions, 28 window functions, partial tracking |
| `choc` | CHOC | FLAC read/write |
| `grainflow` | GrainflowLib | `GfBuffer`, `GrainCollection`, `Panner`, `Recorder`, `Phasor`, 37 enum constants |
| `vafilters` | vafilters (Faust) | 6 virtual analog filters (Moog ladder, Diode ladder, Korg35 LP/HP, Oberheim multi-mode) |
| `bloscillators` | PolyBLEP et al. | 5 band-limited oscillator algorithms (PolyBLEP, BLIT, DPW, MinBLEP, wavetable) |
| `fxdsp` | fxdsp | Antialiased clippers/wavefolder, Schroeder/Moorer reverbs, formant filter, PSOLA, MinBLEP oscillator |
| `iirdesign` | DspFilters | Multi-order IIR filter design (Butterworth, Chebyshev I/II, Elliptic, Bessel, orders 1-16) |

```python
from nanodsp._core import filters, fft, delay, daisysp, stk, madronalib, hisstools, grainflow, vafilters, bloscillators, fxdsp, iirdesign

# Example: direct biquad usage
bq = filters.Biquad()
bq.lowpass(0.1, 0.707)
out = bq.process(input_array)

# Example: direct FFT
f = fft.RealFFT(1024)
spectrum = f.fft(signal)

# Example: DaisySP oscillator
osc = daisysp.oscillators.Oscillator()
osc.init(44100.0)
osc.set_freq(440.0)
samples = osc.process(1024)
```

Full type stubs are provided in `_core.pyi` for IDE autocompletion and type checking.

## Architecture

```text
nanodsp/
  __init__.py          # package root (__version__ only)
  __main__.py          # CLI entry point (argparse, subcommand handlers)
  _cli.py              # function/preset registries, fx parser, type coercion
  _core.cpython-*.so   # compiled C++ extension (nanobind)
  _core.pyi            # type stubs for C++ extension
  _helpers.py          # shared private utilities
  buffer.py            # AudioBuffer class
  io.py                # audio file I/O (WAV + FLAC)
  ops.py               # delay, envelopes, FFT, convolution, rates, mix, pan, xcorr, hilbert, median, LMS
  effects.py           # filters, effects, dynamics, reverb, mastering, AGC
  spectral.py          # STFT, spectral transforms, eq_match
  synthesis.py         # oscillators, noise, drums, physical modeling
  analysis.py          # loudness, spectral features, pitch, onsets, resample, gcc_phat
  stream.py            # ring buffer, block processors, overlap-add
```

## Performance Guidance

### Computational cost tiers

Cost estimates assume a typical stereo buffer at 44.1 kHz. Actual times vary with buffer length, sample rate, and hardware.

| Tier | Typical latency | Functions |
|------|----------------|-----------|
| Cheap (< 1 ms) | Near-instant | `lowpass`, `highpass`, `bandpass`, `notch`, `peak`, `allpass`, shelving filters, `gain_db`, `normalize_peak`, `box_filter`, `peak_hold`, `peak_decay`, `delay`, `pan`, `mix_buffers`, `crossfade`, `hadamard`, `householder`, `fade_in`, `fade_out`, `trim_silence`, `dc_block` |
| Moderate (1--10 ms) | Noticeable in tight loops | `chorus`, `flanger`, `phaser`, `tremolo`, `autowah`, `compress`, `limit`, `noise_gate`, `saturate`, `overdrive`, `exciter`, `de_esser`, `parallel_compress`, `svf_*`, `ladder_filter`, `moog_ladder`, `iir_filter`, `agc`, `aa_hard_clip`, `aa_soft_clip`, `aa_wavefold`, `formant_filter` |
| Expensive (> 10 ms) | Dominates processing time | `stft`/`istft`, `convolve`, `reverb` (FDN, `reverb_sc`), `schroeder_reverb`, `moorer_reverb`, `time_stretch`, `pitch_shift_spectral`, `eq_match`, `spectral_denoise`, `spectral_freeze`, `psola_pitch_shift`, `resample`, `resample_fft`, `multiband_compress`, `lms_filter`, `master`, `vocal_chain` |

### Block size recommendations

- **Offline processing**: Pass the full file as a single `AudioBuffer`. This minimizes per-block overhead and is the simplest approach.
- **Streaming / real-time**: Use `BlockProcessor` or `process_blocks` with 256--1024 samples per block. This range balances throughput against latency.
- **Throughput vs. latency**: Larger blocks amortize fixed overhead (function calls, GIL acquire/release) but increase latency proportionally. At 44.1 kHz, a 512-sample block is ~11.6 ms of latency.
- **Stateful effects**: Effects with internal state (IIR filters, compressors, FDN reverb, delays) must be initialized once and reused across blocks. `BlockProcessor` and `ProcessorChain` handle this automatically.

### GIL release

All C++ processing functions release the Python GIL during computation. This means you can process multiple `AudioBuffer` objects in parallel using `threading` or `concurrent.futures.ThreadPoolExecutor` and achieve true multi-core parallelism -- no need for `multiprocessing`.

### Benchmarking

The CLI provides a built-in benchmark command for measuring function throughput:

```bash
nanodsp benchmark lowpass:cutoff_hz=1000
nanodsp benchmark compress:ratio=4,threshold=-20 -n 100 --duration=2.0
nanodsp benchmark reverb:preset=hall --channels=2 --json
```

This reports iterations per second, mean time per call, and buffer throughput in seconds-of-audio per wall-second.

## Demos

18 demo scripts in `demos/` showcase the full API surface. Run them all at once:

```bash
make demos                              # uses demos/s01.wav
make demos DEMO_INPUT=my_audio.wav      # use a custom input file
```

Or run individual demos:

```bash
uv run python demos/demo_filters.py demos/s01.wav
uv run python demos/demo_reverb.py demos/s01.wav -o /tmp/reverb-output
uv run python demos/demo_distortion.py demos/s01.wav --no-normalize
uv run python demos/demo_synthesis.py                # no input file needed
uv run python demos/demo_analysis.py demos/s01.wav   # prints to stdout
```

| Script | Variants | What it demonstrates |
|--------|----------|----------------------|
| `demo_filters.py` | 13 | Lowpass, highpass, bandpass, notch, peak EQ, high/low shelf |
| `demo_modulation.py` | 10 | Chorus, flanger, phaser, tremolo |
| `demo_distortion.py` | 14 | Overdrive, wavefold, bitcrush, decimator, saturation, fold |
| `demo_reverb.py` | 12 | FDN presets, ReverbSc, STK freeverb/jcrev/nrev/prcrev |
| `demo_dynamics.py` | 9 | Compression, limiting, noise gate, parallel/multiband compression |
| `demo_delay.py` | 8 | Stereo delay, ping-pong, slapback, STK echo |
| `demo_pitch.py` | 10 | Time-domain and spectral pitch shifting |
| `demo_spectral.py` | 12 | Time stretch, phase lock, spectral gate, tilt EQ, freeze |
| `demo_daisysp_filters.py` | 21 | SVF, ladder, moog, tone, modal, comb filters |
| `demo_composed.py` | 13 | Autowah, SR reduce, DC block, exciter, de-esser, vocal chain, mastering, STK chorus |
| `demo_spectral_extra.py` | 8 | Spectral denoise, EQ match, spectral morph |
| `demo_ops.py` | 29 | Delay, vibrato, convolution, envelopes, fades, panning, stereo widening, crossfade |
| `demo_resample.py` | 6 | Madronalib and FFT resampling at 22k/48k/96k |
| `demo_synthesis.py` | 44 | Oscillators, FM, noise, drums, physical modeling, STK instruments (no input file) |
| `demo_analysis.py` | -- | Loudness, spectral features, pitch, onsets, chromagram (stdout only) |
| `demo_grainflow.py` | 7 | Granular clouds (basic, dense), pitch shift, sparse stochastic, stereo panning, recorder |
| `demo_fxdsp.py` | 28 | Antialiased waveshaping, Schroeder/Moorer reverbs, formant filter, PSOLA pitch shift, MinBLEP oscillators |
| `demo_iir_filters.py` | 23 | Butterworth, Chebyshev I/II, Elliptic, Bessel filters at various orders |

File-processing scripts share the same interface:

```
usage: demo_*.py [-h] [-o OUT_DIR] [-n] infile

positional arguments:
  infile                Input .wav file

options:
  -o, --out-dir DIR     Output directory (default: build/demo-output)
  -n, --no-normalize    Skip peak normalization (may clip on PCM output)
```

`demo_synthesis.py` generates sounds from scratch (no input file; takes `-o` and `-n` only). `demo_analysis.py` prints measurements to stdout (no audio output). `demo_grainflow.py` processes an input file through granular synthesis.

## Development

```bash
make build    # rebuild extension after C++ changes
make test     # run 1431 tests
make demos    # run all 18 demo scripts
make qa       # test + lint + typecheck + format
make coverage # tests with coverage report
```

## License

MIT

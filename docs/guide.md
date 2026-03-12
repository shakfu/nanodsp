# DSP Algorithm Guide

An overview of the signal processing algorithms available in nanodsp, organized by category. Each section explains the underlying technique, when to use it, and which functions implement it.

## Filters

Filters selectively pass or reject frequencies in a signal. nanodsp provides four filter families at different abstraction levels.

### Biquad filters (signalsmith)

The biquad is a second-order IIR filter -- the workhorse of audio DSP. Two poles and two zeros give enough flexibility for lowpass, highpass, bandpass, notch, peak EQ, and shelving responses, all with minimal computation (5 multiplies + 4 adds per sample).

nanodsp wraps signalsmith's biquad implementation, which offers four filter design methods:

| Design | Technique | Best for |
|--------|-----------|----------|
| `"bilinear"` | Bilinear transform of analog prototype | General-purpose LP/HP, matches analog response |
| `"cookbook"` | Robert Bristow-Johnson's Audio EQ Cookbook | Classic EQ curves, wide compatibility |
| `"one_sided"` | Optimized for one-sided frequency response | Bandpass, notch, peak, allpass |
| `"vicanek"` | Martin Vicanek's matched-response design | Accurate magnitude at high frequencies |

**Functions:** `lowpass`, `highpass`, `bandpass`, `notch`, `peak`, `peak_db`, `high_shelf`, `high_shelf_db`, `low_shelf`, `low_shelf_db`, `allpass`

### DaisySP filters

DaisySP provides several filter topologies beyond the basic biquad:

- **SVF (State Variable Filter)** -- a 2nd-order filter that produces lowpass, highpass, bandpass, notch, and peak outputs simultaneously from a single topology. The `resonance` parameter controls the Q, and `drive` adds soft saturation at the input. Five dedicated functions (`svf_lowpass`, `svf_highpass`, etc.) select a single output mode.

- **Ladder filter** -- a 4-pole (24 dB/oct) lowpass inspired by the Moog ladder topology. Supports multiple modes: `"lp24"`, `"lp12"`, `"bp24"`, `"bp12"`, `"hp24"`, `"hp12"`. The `drive` parameter is an input multiplier (1.0 = unity, higher values add saturation).

- **Moog ladder** -- a simpler Moog ladder variant with just cutoff and resonance.

- **Tone filters** -- simple 1-pole lowpass and highpass filters. Cheap and useful for gentle tilt EQ or DC blocking.

- **Modal bandpass** -- a high-Q resonant bandpass for simulating resonant bodies (drums, strings, rooms).

- **Comb filter** -- creates a series of evenly-spaced notches or peaks, useful for flanger-like effects and Karplus-Strong synthesis.

**Functions:** `svf_lowpass`, `svf_highpass`, `svf_bandpass`, `svf_notch`, `svf_peak`, `ladder_filter`, `moog_ladder`, `tone_lowpass`, `tone_highpass`, `modal_bandpass`, `comb_filter`

### Virtual analog filters (Faust)

These are zero-delay feedback (ZDF) filter models derived from analog circuit schematics using the Faust DSP language. They accurately model the nonlinear behavior of classic analog synthesizer filters:

| Filter | Model | Character |
|--------|-------|-----------|
| `va_moog_ladder` | 24 dB/oct Moog transistor ladder | Warm, fat, classic subtractive synth |
| `va_moog_half_ladder` | 12 dB/oct Moog half ladder | Gentler rolloff, less resonance |
| `va_diode_ladder` | 24 dB/oct diode ladder (TB-303 style) | Acidic, with internal soft clipping |
| `va_korg35_lpf` | Korg MS-20 lowpass | Aggressive, screamy at high resonance |
| `va_korg35_hpf` | Korg MS-20 highpass | Same character, highpass mode |
| `va_oberheim` | Oberheim SEM multi-mode SVF | Clean, precise, 4 simultaneous outputs |

### Multi-order IIR filters (DspFilters)

For sharper transitions than a biquad can provide, `iir_filter` offers classical IIR designs at orders 1--16, implemented as cascaded second-order sections (SOS):

| Family | Character |
|--------|-----------|
| Butterworth | Maximally flat passband, no ripple |
| Chebyshev Type I | Passband ripple, sharper transition than Butterworth |
| Chebyshev Type II | Flat passband, stopband ripple |
| Elliptic | Sharpest transition for a given order, ripple in both bands |
| Bessel | Maximally flat group delay (best phase linearity) |

Each family supports lowpass, highpass, bandpass, and bandstop modes.

**Functions:** `iir_filter`, `iir_design`

---

## Dynamics

Dynamics processors control the amplitude envelope of a signal.

### Compressor

Reduces the dynamic range by attenuating signals above a threshold. The `ratio` controls how much attenuation (4:1 means 4 dB of input above threshold produces 1 dB of output above threshold). Attack and release times control how quickly the compressor responds.

**Function:** `compress`

### Limiter

A compressor with an effectively infinite ratio -- prevents the signal from exceeding a ceiling. Useful as a final safety stage.

**Function:** `limit`

### Noise gate

The inverse of a compressor: attenuates signals *below* a threshold. Silences quiet passages (mic bleed, background noise, amp hiss). The hold parameter prevents the gate from chattering on transients.

**Function:** `noise_gate`

### AGC (Automatic Gain Control)

A slow-acting compressor that maintains a consistent average level over time. Uses a moving-average power estimator with asymmetric attack/release smoothing. Good for normalizing speech or podcast audio where the speaker's distance from the mic varies.

**Function:** `agc`

### Parallel compression

Also called "New York compression." Mixes a heavily compressed copy with the dry signal. This preserves transients while bringing up quiet details -- a common mixing technique for drums and vocals.

**Function:** `parallel_compress`

### Multiband compression

Splits the signal into frequency bands (using cascaded biquad crossovers), compresses each band independently, then sums. This allows different compression settings for bass, mids, and highs -- essential for mastering.

**Function:** `multiband_compress`

---

## Saturation and Distortion

Waveshaping algorithms that add harmonics by applying nonlinear transfer functions.

### Basic saturation

Three modes with different transfer curves:

- **Soft (tanh)** -- smooth, symmetrical clipping. Adds primarily odd harmonics. Sounds warm.
- **Hard (clip)** -- abrupt clipping at +/-1.0. Harsh, buzzy harmonics.
- **Tape** -- asymmetric soft clip (`x - x^3/3`). Adds both even and odd harmonics, emulating tape saturation.

**Function:** `saturate`

### Antialiased waveshaping (fxdsp)

Naive waveshaping creates aliasing -- frequencies above Nyquist that fold back as inharmonic distortion. These functions use antiderivative antialiasing (ADAA) to suppress aliasing:

- **`aa_hard_clip`** -- 1st-order ADAA hard clipper
- **`aa_soft_clip`** -- 1st-order ADAA sin-based soft clipper
- **`aa_wavefold`** -- 2nd-order ADAA wavefolder (Buchla 259 style)

ADAA computes the antiderivative of the waveshaping function and differences adjacent samples, effectively integrating the transfer function over each sample interval. The result is dramatically cleaner than naive clipping, especially at high drive settings.

### DaisySP distortion

- **`overdrive`** -- soft overdrive with drive amount 0-1
- **`wavefold`** -- gain + offset into a folding function
- **`bitcrush`** -- quantize to N bits with sample-and-hold
- **`decimator`** -- combined downsampling + bitcrushing
- **`fold`** -- fold distortion with increment control
- **`sample_rate_reduce`** -- sample-rate reduction

---

## Reverb

Reverb simulates the acoustic reflections of a physical space.

### FDN (Feedback Delay Network)

The primary reverb algorithm. Uses a matrix of delay lines with feedback through a mixing matrix (Hadamard) and per-line damping filters. The madronalib backend provides optimized 64-sample DSPVector processing.

Five presets configure the delay line lengths and damping to model different spaces:

| Preset | Character |
|--------|-----------|
| `room` | Small room, short decay |
| `hall` | Concert hall, medium decay |
| `plate` | Plate reverb, bright and dense |
| `chamber` | Small chamber, intimate |
| `cathedral` | Large cathedral, long decay |

**Function:** `reverb`

### Schroeder reverb (fxdsp)

The classic reverb topology: 4 parallel feedback comb filters summed into 2 series allpass filters. Simple but effective. Optional LFO modulation on comb delay lengths reduces metallic ringing.

**Function:** `schroeder_reverb`

### Moorer reverb (fxdsp)

Extends Schroeder's design with an 18-tap early reflections delay network before the comb filters. This separates early reflections (which convey room size) from the late diffuse tail.

**Function:** `moorer_reverb`

### ReverbSc (DaisySP)

Sean Costello's reverb algorithm -- a high-quality stereo reverb with lowpass damping. Accepts mono or stereo input, always produces stereo output.

**Function:** `reverb_sc`

### STK reverbs

Four reverb algorithms from the Synthesis ToolKit:

- **freeverb** -- Jezar's Freeverb (8 parallel combs + 4 series allpasses)
- **jcrev** -- John Chowning's reverb
- **nrev** -- CCRMA NRev (6 combs + allpass chain)
- **prcrev** -- Perry Cook's simple reverb

**Function:** `stk_reverb`

---

## Modulation Effects

Effects that vary a parameter (usually delay time) with an LFO.

### Chorus

Mixes the dry signal with a delayed copy whose delay time is modulated by an LFO. The varying delay creates pitch vibrato; mixing with dry creates the characteristic thickening effect. Mono input produces stereo output via `process_stereo`.

**Function:** `chorus`

### Flanger

Similar to chorus but with shorter delay times (< 5ms) and feedback. The comb-filtering effect sweeps through the spectrum as the LFO modulates the delay, creating a jet-engine swoosh.

**Function:** `flanger`

### Phaser

Cascaded allpass filters whose center frequencies are swept by an LFO. Unlike flanging (which creates evenly-spaced notches), phasing creates irregularly-spaced notches that sound more organic. The `poles` parameter controls how many allpass stages are used (more = deeper effect).

**Function:** `phaser`

### Tremolo

Amplitude modulation by an LFO. Simple but effective -- multiplies the signal by a low-frequency waveform.

**Function:** `tremolo`

### Autowah

An envelope-controlled bandpass filter. The input amplitude modulates the filter frequency, creating a wah-wah effect that responds to playing dynamics.

**Function:** `autowah`

---

## Spectral Processing

Algorithms that operate in the frequency domain via the Short-Time Fourier Transform (STFT).

### STFT / ISTFT

The STFT decomposes a signal into overlapping windowed frames, applies an FFT to each, and produces a complex-valued spectrogram. The ISTFT reverses this with overlap-add reconstruction. The `window` parameter selects the analysis/synthesis window (`"hann"`, `"hamming"`, `"blackman"`, `"bartlett"`, `"rectangular"`).

**Functions:** `stft`, `istft`

### Time stretching

Changes duration without changing pitch. Implemented by interpolating STFT magnitudes between frames and accumulating phase. The `rate` parameter controls speed (0.5 = half speed / double length, 2.0 = double speed / half length).

**Function:** `time_stretch`

### Spectral pitch shifting

Changes pitch without changing duration. Combines time stretching (to shift the spectrum) with resampling (to restore the original duration).

**Function:** `pitch_shift_spectral`

### Spectral denoising

Estimates a noise floor from the first N frames of the spectrogram, then attenuates frequency bins below the noise floor. A simple but effective noise reduction technique -- works best when the signal starts with a few frames of noise-only content.

**Function:** `spectral_denoise`

### Spectral freeze

Repeats a single STFT frame indefinitely, creating a sustained "frozen" texture from an instant of the input signal.

**Function:** `spectral_freeze`

### Spectral morphing

Interpolates the magnitude spectra of two spectrograms while preserving the phase of the first. Creates smooth timbral transitions between two sounds.

**Function:** `spectral_morph`

### EQ matching

Analyzes the spectral envelope of a target signal, computes the ratio to the source's envelope, and applies it as a filter. Makes one recording sound like another in terms of tonal balance.

**Function:** `eq_match`

---

## Synthesis

### DaisySP oscillators

- **`oscillator`** -- basic waveforms (sine, triangle, saw, square, polyblep variants) with anti-aliased options
- **`fm2`** -- two-operator FM synthesis (carrier + modulator)
- **`formant_oscillator`** -- carrier with formant frequency control
- **`bl_oscillator`** -- band-limited oscillator (DaisySP)

### Band-limited oscillators

Aliasing-free oscillator algorithms for clean digital synthesis:

- **PolyBLEP** -- polynomial band-limited step. Corrects discontinuities at waveform transitions by adding a polynomial residual. Supports 14 waveforms. Very efficient.
- **BLIT** -- band-limited impulse train. Generates a mathematically perfect bandlimited impulse train, then integrates to produce saw/square. Configurable number of harmonics.
- **DPW** -- differentiated parabolic wave. Generates a parabola, then differentiates to produce a sawtooth. Simple and effective, low aliasing.
- **MinBLEP** -- minimum-phase band-limited step. Precomputes a 2048-element correction table at 64x oversampling. Most accurate antialiasing of the four methods, but higher memory use.

**Functions:** `polyblep`, `blit_saw`, `blit_square`, `dpw_saw`, `dpw_pulse`, `minblep`

### Noise generators

- **`white_noise`** -- uniform random samples
- **`clocked_noise`** -- sample-and-hold noise at a given frequency
- **`dust`** -- sparse random impulses (Poisson process)

### Drum synthesis (DaisySP)

Physically-inspired analog drum models:

- **`analog_bass_drum`** -- resonant body with FM self-modulation
- **`analog_snare_drum`** -- tonal body + noise burst
- **`hihat`** -- metallic noise with bandpass
- **`synthetic_bass_drum`** -- synthesis-focused variant with dirtiness/FM controls
- **`synthetic_snare_drum`** -- synthesis-focused variant with FM

### Physical modeling

- **`karplus_strong`** -- plucked string model (excitation into a filtered delay line)
- **`modal_voice`** -- resonant body excited by an impulse
- **`string_voice`** -- bowed/plucked string model
- **`pluck`** -- simple Karplus-Strong pluck
- **`drip`** -- water drop physical model

### STK instruments

Physical modeling instruments from the Synthesis ToolKit. Available instruments: `clarinet`, `flute`, `brass`, `bowed`, `plucked`, `sitar`, `stifkarp`, `saxofony`, `recorder`, `blowbotl`, `blowhole`, `whistle`.

**Functions:** `synth_note`, `synth_sequence`

---

## Analysis

### Loudness metering

`loudness_lufs` measures integrated loudness per ITU-R BS.1770-4 (the broadcast standard). Returns a value in LUFS (Loudness Units Full Scale). `normalize_lufs` adjusts gain to hit a target loudness.

### Spectral features

Frame-by-frame measurements computed from the STFT magnitude:

| Feature | What it measures | Typical use |
|---------|-----------------|-------------|
| Spectral centroid | "Center of mass" of the spectrum | Brightness tracking |
| Spectral bandwidth | Spread around the centroid | Timbre analysis |
| Spectral rolloff | Frequency below which N% of energy lies | High-frequency content |
| Spectral flux | Frame-to-frame spectral change | Onset detection, transient analysis |
| Spectral flatness | Ratio of geometric to arithmetic mean | Noisiness vs. tonality |
| Chromagram | Energy in each of 12 pitch classes | Chord/key detection |

### Pitch detection (YIN)

The YIN algorithm estimates fundamental frequency by computing the cumulative mean normalized difference function (autocorrelation-based). Returns f0 estimates and confidence values per frame. Good for monophonic signals; degrades with polyphonic content.

**Function:** `pitch_detect`

### Onset detection

Detects note onsets using spectral flux (frame-to-frame magnitude change) with peak-picking. Returns frame indices of detected onsets. The `backtrack` option moves onsets to the nearest preceding energy minimum.

**Function:** `onset_detect`

### Resampling

Two methods:

- **`resample`** -- madronalib's polyphase resampler. High quality, efficient for standard rate conversions (44.1k to 48k, etc.).
- **`resample_fft`** -- FFT-based resampling. Zero-pads or truncates the spectrum, then inverse-transforms. Mathematically exact for bandlimited signals but more expensive for large rate changes.

### GCC-PHAT

Generalized Cross-Correlation with Phase Transform. Estimates the time delay between two signals by cross-correlating in the frequency domain with phase-only weighting. Used for microphone array processing and time-of-arrival estimation.

**Function:** `gcc_phat`

---

## Streaming

### RingBuffer

A fixed-capacity circular buffer for producer/consumer audio workflows. Write audio in, read it out FIFO. Useful for bridging real-time audio callbacks with offline processing.

### BlockProcessor / CallbackProcessor

Base class for stateful block-based processing. Subclass `BlockProcessor` and override `process_block`, or use `CallbackProcessor` with a lambda. Handles block slicing, tail handling, and state persistence automatically.

### ProcessorChain

Chains multiple `BlockProcessor` instances in series. Each block passes through all processors before the next block is processed.

### process_blocks

Utility function for overlap-add processing. Slices the input into overlapping blocks, applies a function to each, and reconstructs with overlap-add. Useful for frame-by-frame spectral processing.

**Function:** `process_blocks`

---

## Composed Effects

Higher-level effects built by combining multiple primitives.

- **Exciter** -- highpass filter + saturation + mix with dry. Adds brightness and presence.
- **De-esser** -- sidechain compressor triggered by a bandpass around sibilant frequencies (typically 4--8 kHz).
- **Stereo delay** -- independent left/right delays with optional ping-pong routing.
- **Formant filter** -- 3 cascaded bandpass biquads tuned to vowel formant frequencies (A/E/I/O/U) with interpolation.
- **PSOLA pitch shift** -- pitch-synchronous overlap-add. Detects pitch via autocorrelation, then time-aligns and overlaps grains to shift pitch without changing duration. Best for monophonic voiced signals (speech, solo instruments).
- **Master** -- mastering chain: DC block, EQ, compression, limiting, loudness normalization.
- **Vocal chain** -- voice processing chain: de-essing, EQ, compression, limiting, optional loudness normalization.

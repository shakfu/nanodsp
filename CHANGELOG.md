# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.6]

### Added

- **Integration tests** -- 10 new composed effect chain tests (`TestEffectChains`) verifying multi-stage pipelines produce finite, correctly-shaped output: exciter->compress->limit, vocal_chain, master chain, lowpass->saturate->reverb, stereo_delay->compress->limit, de_esser->EQ->compress, multiband_compress->limit, shimmer_reverb->normalize, lo_fi->reverb, noise_gate->reverb->limit
- **Edge-case tests** -- 7 new tests for boundary conditions: `sample_rate=0` and negative sample_rate rejection, `pitch_detect(fmin >= fmax)` returns unvoiced, extreme `time_stretch` rates (0.05 and 10.0), near-silence LUFS metering, near-silence AGC stability
- **Algorithm citations** -- added References sections with paper citations to 4 algorithm implementations:
  - `loudness_lufs`: ITU-R BS.1770-4 (2015)
  - `pitch_detect`: de Cheveigne & Kawahara, "YIN," JASA 2002
  - `gcc_phat`: Knapp & Carter, IEEE 1976
  - `time_stretch`: Flanagan & Golden (1966), Laroche & Dolson (1999)

### Changed

- **Sample rate validation** -- `AudioBuffer.__init__` now raises `ValueError` if `sample_rate <= 0`, preventing downstream `ZeroDivisionError` in `.duration` and other computations
- **`Literal` type unions for string mode parameters** -- 14 function signatures now use `Literal[...]` instead of bare `str` for IDE autocompletion and static checking:
  - `buffer.py`: `to_mono(method=)`
  - `ops.py`: `delay(interpolation=)`, `delay_varying(interpolation=)`, `fade_in(curve=)`, `fade_out(curve=)`
  - `effects/saturation.py`: `saturate(mode=)`
  - `effects/reverb.py`: `reverb(preset=)`, `stk_reverb(algorithm=)`
  - `effects/filters.py`: `ladder_filter(mode=)`, `va_oberheim(mode=)`
  - `effects/composed.py`: `shimmer_reverb(preset=)`, `gated_reverb(preset=)`
  - `spectral.py`: `stft(window=)`, `istft(window=)` (via `WindowType` alias)
- **Type annotations in `buffer.py`** -- full parameter and return type annotations for all operator overloads (`__add__`, `__sub__`, `__mul__`, `__truediv__`, `__neg__`, `__radd__`, `__rsub__`, `__rmul__`), `__getitem__` (with `@overload`), `__array__`, and `pipe()`. Added `ArrayLike` type for `__init__` data parameter.
- **Parameter range documentation** -- added valid ranges and typical values to docstrings across 10 modules (~60 parameters): dynamics (ratio, threshold, attack, release, makeup, etc.), reverb (mix, decay, damping, feedback, t60, etc.), saturation (drive), daisysp effects (lfo_freq, lfo_depth, feedback, depth, etc.), ops (delay_samples, crossfade x, lfo rate, normalize target_db, trim threshold, lms step_size, stereo_widen width), analysis (target_lufs, percentile, fmin/fmax, threshold), spectral (window_size, hop_size, threshold_db, noise_floor_db, reduction_db, smoothing), filters (cutoff_hz, q for all VA filters), synthesis (freq, amp, pw, ratio, index, phase_shift), composed (exciter amount, de_esser freq/threshold/ratio/bandwidth, parallel_compress mix/ratio/threshold/attack/release)
- **Named epsilon constants in `analysis.py`** -- replaced 3 bare `eps = 1e-20` / `eps = 1e-10` values with module-level `_LOG_EPS` and `_DIV_EPS` constants with a comment explaining why they differ
- **Inline comment for EQ match gain ceiling** -- `spectral.py` `eq_match()` now documents the `100.0` clipping ceiling as "+40 dB cap to prevent EQ runaway on near-silent bins"
- **Mono-summing deduplication in `reverb.py`** -- extracted `_to_mono()` helper, replacing 3 identical mono-downmix patterns in `reverb()`, `stk_reverb()`, and `stk_chorus()`
- **STK version clarification** -- `VERSIONS.md` now identifies the vendored STK as a `~5.0.0-dev` snapshot, explains the version string inconsistency across source files, and notes `configure.ac` (5.0.0) as authoritative
- **fxdsp VERSIONS.md entry** -- updated stale "unlicensed" description to reflect the MIT LICENSE file present in `thirdparty/fxdsp/`

## [0.1.5]

### Added

- **Ping-pong delay** (`nanodsp._core.fxdsp.PingPongDelay`, `nanodsp.effects.composed.ping_pong_delay`) -- stereo ping-pong delay with crossed feedback and linear interpolation. Based on FX/KHPingPongDelay.hpp, rewritten as clean header.
  - Configurable delay time, feedback (-0.99 to 0.99), and dry/wet mix
  - Stereo processing: `[2, N]` input/output with crossed feedback paths
  - Accepts mono (duplicated to stereo) or stereo input

- **Frequency shifter** (`nanodsp._core.fxdsp.FreqShifter`, `nanodsp.effects.composed.freq_shift`) -- Bode-style frequency shifter using allpass Hilbert transform approximation. Based on FX/BodeShifter.hpp, rewritten from scratch (original had bugs).
  - 4-stage allpass pair (Wardle coefficients) for wideband 90-degree phase split
  - Quadrature oscillator for single-sideband modulation
  - Positive or negative shift in Hz; does not preserve harmonic relationships

- **Ring modulator** (`nanodsp._core.fxdsp.RingMod`, `nanodsp.effects.composed.ring_mod`) -- ring modulation with carrier sine oscillator and optional LFO frequency modulation. Based on FX/AudioEffectRingMod.hpp, rewritten as clean header.
  - Configurable carrier frequency, dry/wet mix
  - Optional LFO with rate and depth controls for carrier FM
  - Produces sum and difference tones (e.g., 440 Hz input * 300 Hz carrier = 140 Hz + 740 Hz)

- 40 new tests for ping-pong delay, frequency shifter, and ring modulator (C++ bindings + Python API)

- **6 derivative composed effects** that combine existing primitives:
  - `shimmer_reverb` -- FDN reverb + PSOLA pitch shift blended as a shimmer layer (ambient/post-rock)
  - `tape_echo` -- multi-tap delay with progressive lowpass filtering and tape saturation per repeat
  - `lo_fi` -- bitcrush + sample-rate reduction + tape saturation + lowpass chain
  - `telephone` -- tight bandpass (300-3400 Hz) + hard saturation (codec/radio simulation)
  - `gated_reverb` -- FDN reverb + noise gate for truncated punchy tails (80s production)
  - `auto_pan` -- sine LFO-driven equal-power stereo panning
- 34 new tests for the 6 derivative composed effects
- 15 new demo variants in `demo_composed.py` for the derivative effects

### Changed

- **FDN reverb demo parameters** -- widened mix, decay, and damping spread across presets to make each sound distinctly different (room=dry/damped, plate=bright, cathedral=wet/long/dark)

### Fixed

- **165 pytest-review assertion warnings** -- added explicit `assert` keyword statements to tests that relied solely on `np.testing.assert_*` (not detected by the plugin) or had no assertions at all. Replaced `try/assert False/except` anti-patterns with idiomatic `pytest.raises`. Removed trivial `assert True` statements. All 1522 tests pass with 0 review issues.

## [0.1.4]

### Changed

- **BREAKING: `effects` module split into subpackage** -- the monolithic `effects.py` (68KB) is now `effects/` with 6 public submodules:
  - `nanodsp.effects.filters` -- signalsmith biquads, DaisySP SVF/ladder/moog/tone/modal/comb, virtual analog (Faust), IIR (DspFilters)
  - `nanodsp.effects.daisysp` -- autowah, chorus, decimator, flanger, overdrive, phaser, pitch_shift, sample_rate_reduce, tremolo, wavefold, bitcrush, fold, reverb_sc, dc_block
  - `nanodsp.effects.dynamics` -- compress, limit, noise_gate, agc
  - `nanodsp.effects.saturation` -- saturate, aa_hard_clip, aa_soft_clip, aa_wavefold
  - `nanodsp.effects.reverb` -- FDN reverb (with presets), schroeder_reverb, moorer_reverb, stk_reverb, stk_chorus, stk_echo
  - `nanodsp.effects.composed` -- exciter, de_esser, parallel_compress, stereo_delay, ping_pong_delay, freq_shift, ring_mod, multiband_compress, formant_filter, psola_pitch_shift, master, vocal_chain, shimmer_reverb, tape_echo, lo_fi, telephone, gated_reverb, auto_pan
- **BREAKING: `effects/__init__.py` no longer re-exports** -- import from specific submodules (e.g. `from nanodsp.effects.filters import lowpass` instead of `from nanodsp.effects import lowpass`)
- **BREAKING: Biquad filter `design` parameter now accepts strings** -- `lowpass(buf, 1000, design="bilinear")` instead of `design=filters.BiquadDesign.bilinear`. Raw enum/int values still accepted for backward compatibility. Valid strings: `"bilinear"`, `"cookbook"`, `"one_sided"`, `"vicanek"`
- All effects submodules use relative imports internally
- Updated all tests, demos, and CLI to use new submodule import paths
- **`io.py` deduplication** -- extracted shared WAV sample decode/encode logic into `_decode_wav_frames` and `_encode_wav_frames`, eliminating ~60 lines of duplication between file and bytes I/O variants
- **Frequency validation for DaisySP/VA filters** -- SVF, ladder, moog, tone, modal, comb, and all VA filter functions now validate frequency against Nyquist at function entry, matching the behavior of signalsmith biquad wrappers
- **Improved error messages** -- frequency validation errors now include `sample_rate`; WAV I/O errors include the file path; channel index errors include the valid range; `concat_channels` sample rate errors include both rates
- **C++ uint8_t bounds checks** -- 6 DaisySP binding locations now validate parameter ranges before casting to `uint8_t`, raising `IndexError` with descriptive messages instead of silently truncating:
  - `Oscillator.set_waveform`: 0-7
  - `BlOsc.set_waveform`: 0-3
  - `Decimator.set_bits_to_crush`: 1-32
  - `CrossFade.set_curve`: 0-3
  - `AdEnv.set_time` segment: 0-2
  - `Adsr.set_time` segment: 0-3
- **Thirdparty version documentation** -- added `thirdparty/VERSIONS.md` documenting version, license, and upstream URL for all 10 C++ dependencies
- **NumPy-style docstrings** -- added comprehensive docstrings to ~50 public functions across `ops.py`, `synthesis.py`, `analysis.py`, `stream.py`, and `spectral.py`
- **CLI bool coercion** -- added `"on"` to truthy values in CLI parameter parsing (joins `"true"`, `"yes"`, `"1"`)
- **Configurable STFT window type** -- `stft` and `istft` now accept a `window` parameter (`"hann"`, `"hamming"`, `"blackman"`, `"bartlett"`, `"rectangular"`)
- **Test parametrization** -- consolidated 13 biquad filter type tests into a single parametrized test in `test_filters.py`; consolidated oscillator, noise, drum, physical modeling, and instrument tests in `test_synthesis.py`
- **Performance guidance** -- added Performance Guidance section to README.md covering buffer sizing, channel layouts, C++ vs Python paths, and GIL release behavior

### Fixed

- **`ladder_filter` silent output** -- default `drive` was `0.0`, which zeroed the input signal before filtering (DaisySP's `drive` is an input multiplier). Changed default to `1.0` (unity gain)
- **RingBuffer docstring** -- replaced misleading "lock-free-style" label with explicit thread safety warning (not safe for concurrent access without external synchronization)
- **BlockProcessor docstring** -- added note about stateful DSP objects needing instantiation in `__init__` to avoid state loss between blocks

## [0.1.3]

### Added

- **Virtual Analog filter bindings** (`nanodsp._core.vafilters`, `nanodsp.effects`) -- 6 Faust-derived analog-modeled filters
  - `MoogLadder` -- 24 dB/oct Moog ladder lowpass with resonance
  - `MoogHalfLadder` -- 12 dB/oct Moog half-ladder lowpass
  - `DiodeLadder` -- 24 dB/oct diode ladder lowpass with internal soft clipping
  - `Korg35LPF` -- 24 dB/oct Korg-35 lowpass
  - `Korg35HPF` -- 24 dB/oct Korg-35 highpass
  - `OberheimSVF` -- multi-mode state-variable filter with 4 simultaneous outputs (LPF, HPF, BPF, BSF)
  - Python wrappers: `va_moog_ladder`, `va_moog_half_ladder`, `va_diode_ladder`, `va_korg35_lpf`, `va_korg35_hpf`, `va_oberheim`
  - 44 tests (`tests/test_vafilters.py`)

- **Band-limited oscillator bindings** (`nanodsp._core.bloscillators`, `nanodsp.synthesis`) -- 5 anti-aliased oscillator algorithms
  - `PolyBLEP` -- polynomial band-limited step oscillator with 14 waveforms (sine, cosine, triangle, square, rectangle, sawtooth, ramp, modified triangle/square, half/full-wave rectified sine, triangular pulse, trapezoid fixed/variable)
  - `BlitSaw` -- BLIT (band-limited impulse train) sawtooth with configurable harmonics
  - `BlitSquare` -- BLIT square wave with DC blocker
  - `DPWSaw` -- DPW (differentiated parabolic wave) sawtooth
  - `DPWPulse` -- DPW pulse with variable duty cycle
  - Python wrappers: `polyblep`, `blit_saw`, `blit_square`, `dpw_saw`, `dpw_pulse`
  - 83 tests (`tests/test_bloscillators.py`)

- **FX DSP algorithms** (`nanodsp._core.fxdsp`, `nanodsp.effects`, `nanodsp.synthesis`) -- 9 algorithms from cleaned/rewritten third-party sources
  - `HardClipper` -- first-order antiderivative antialiased hard clipping
  - `SoftClipper` -- first-order antiderivative antialiased soft clipping (sin saturation)
  - `Wavefolder` -- second-order antiderivative antialiased wavefolding
  - `SchroederReverb` -- classic 4 parallel feedback combs + 2 series allpasses with optional LFO modulation
  - `MoorerReverb` -- Schroeder extension with 18-tap early reflections delay network
  - `MinBLEP` -- minimum band-limited step oscillator (saw, reverse saw, square, triangle) with precomputed 2048-element table at 64x oversampling
  - `PsolaShifter` -- PSOLA pitch shifting with autocorrelation pitch detection and grain-based resynthesis
  - `FormantFilter` -- 3 cascaded bandpass biquads tuned to vowel formant frequencies (A/E/I/O/U) with blending
  - `PingPongDelay` -- stereo ping-pong delay with crossed feedback and linear interpolation
  - `FreqShifter` -- Bode-style frequency shifter using allpass Hilbert transform
  - `RingMod` -- ring modulator with carrier oscillator and optional LFO FM
  - Python wrappers: `aa_hard_clip`, `aa_soft_clip`, `aa_wavefold`, `schroeder_reverb`, `moorer_reverb`, `formant_filter`, `psola_pitch_shift`, `minblep`, `ping_pong_delay`, `freq_shift`, `ring_mod`
  - 105 tests (`tests/test_fxdsp.py`)

- **Multi-order IIR filter design** (`nanodsp._core.iirdesign`, `nanodsp.effects`) -- 5 classical filter families via DspFilters (Vinnie Falco, MIT)
  - Butterworth (maximally flat passband)
  - Chebyshev Type I (passband ripple, sharper rolloff)
  - Chebyshev Type II (stopband ripple, flat passband)
  - Elliptic (sharpest transition, ripple in both bands)
  - Bessel (linear phase, minimal ringing)
  - Each family supports lowpass, highpass, bandpass, bandstop configurations
  - Orders 1-16, returning SOS (second-order sections) coefficients
  - `IIRFilter` class for stateful processing with `setup()`/`process()`/`reset()`/`sos()`
  - `iir_design()` returns SOS coefficient array `[n_sections, 6]`
  - `iir_filter()` applies multi-order IIR filter to AudioBuffer
  - 41 tests (`tests/test_iirdesign.py`)

- **Pure NumPy DSP algorithms** -- 7 new functions for API completeness without scipy dependency
  - `ops.xcorr(buf_a, buf_b=None)` -- FFT-based cross-correlation (or autocorrelation in single-arg form)
  - `ops.hilbert(buf)` -- amplitude envelope via analytic signal (FFT method)
  - `ops.envelope(buf)` -- alias for `hilbert`
  - `ops.median_filter(buf, kernel_size=3)` -- per-channel median filtering via stride tricks
  - `ops.lms_filter(buf, ref, filter_len=32, step_size=0.01, normalized=True)` -- NLMS adaptive filter returning `(output, error)`
  - `effects.agc(buf, target_level, max_gain_db, average_len, attack, release)` -- automatic gain control with asymmetric attack/release
  - `analysis.gcc_phat(buf, ref, sample_rate)` -- GCC-PHAT time-delay estimation returning `(delay_seconds, correlation)`

- **GrainflowLib bindings** (`nanodsp._core.grainflow`) -- granular synthesis engine (header-only, MIT license)
  - `GfBuffer` -- buffer wrapper bridging numpy `[channels, frames]` arrays to GrainflowLib's internal AudioFile storage
  - `GrainCollection` -- core multi-grain granulator with block-based processing, parameter control (enum and string reflection), buffer assignment, stream management, and auto-overlap
  - `Panner` -- stereo grain panning with three modes (bipolar, unipolar, stereo) using equal-power quarter-sine interpolation
  - `Recorder` -- live recording into buffers with overdub, freeze, sync, and multi-band filter support
  - `Phasor` -- clock generator for grain triggering (continuous-phase ramp [0, 1))
  - 37 enum constants: `PARAM_*` (23 parameter names), `PTYPE_*` (5 parameter types), `STREAM_*` (4 stream modes), `BUF_*` (6 buffer types), `BUFMODE_*` (3 buffer modes), `PAN_*` (3 pan modes)
  - String-based parameter reflection (e.g. `"delayRandom"`, `"rateOffset"`, `"channelMode"`)
  - All processing methods release the GIL for thread safety
  - 49 tests (`tests/test_grainflow.py`)
  - Patched two GrainflowLib upstream bugs for `SigType=float`: `gf_utils::mod` template deduction failure, `stream` method vs member access

- **Demo scripts** (`demos/`) -- 16 runnable demo scripts showcasing the full API surface
  - `demo_filters.py` -- 13 biquad filter variants (lowpass, highpass, bandpass, notch, peak, shelving)
  - `demo_modulation.py` -- 10 modulation effects (chorus, flanger, phaser, tremolo)
  - `demo_distortion.py` -- 14 distortion/saturation effects (overdrive, wavefold, bitcrush, decimator, saturate, fold)
  - `demo_reverb.py` -- 12 reverb algorithms (FDN presets, ReverbSc, STK freeverb/jcrev/nrev/prcrev)
  - `demo_dynamics.py` -- 9 dynamics processors (compression, limiting, gating, parallel/multiband compression)
  - `demo_delay.py` -- 8 delay effects (stereo delay, ping-pong, slapback, echo)
  - `demo_pitch.py` -- 10 pitch shifters (time-domain and spectral at various intervals)
  - `demo_spectral.py` -- 12 spectral transforms (time stretch, phase lock, spectral gate, tilt EQ, freeze)
  - `demo_daisysp_filters.py` -- 21 DaisySP filter variants (SVF, ladder, moog, tone, modal, comb)
  - `demo_composed.py` -- 13 composed effects (autowah, sample rate reduce, DC block, exciter, de-esser, vocal chain, mastering, STK chorus)
  - `demo_spectral_extra.py` -- 8 additional spectral transforms (denoise, EQ match, spectral morph)
  - `demo_ops.py` -- 29 core DSP operations (delay, vibrato, convolution, envelopes, fades, panning, stereo widening, crossfade, normalization, trim, oversample)
  - `demo_resample.py` -- 6 resampling variants (madronalib and FFT methods at 22k/48k/96k)
  - `demo_synthesis.py` -- 44 synthesis sounds (oscillators, FM, formant, noise, drums, physical modeling, STK instruments, sequence) -- no input file required
  - `demo_analysis.py` -- audio analysis printout (loudness, spectral features, pitch detection, onset detection, chromagram) -- no audio output
  - `demo_grainflow.py` -- 7 granular synthesis variants (basic cloud, dense cloud, pitch shift up/down, sparse stochastic, stereo panned, recorder)
  - `demo_fxdsp.py` -- 38 FX DSP outputs: antialiased waveshaping (6), Schroeder/Moorer reverbs (6), formant vowels (5), PSOLA pitch shifts (6), ping-pong delay (3), frequency shifter (3), ring modulator (4), minBLEP waveforms (5)
  - `demo_iir_filters.py` -- 23 IIR filter outputs: Butterworth (6), Chebyshev I (4), Chebyshev II (3), Elliptic (3), Bessel (4), order comparison (3)
  - All file-processing scripts accept positional `infile`, optional `-o`/`--out-dir` (default `build/demo-output/`), and `-n`/`--no-normalize` to skip peak normalization
  - Peak normalization (0 dBFS) applied by default to prevent clipping on PCM output
- `make demos` target -- runs all 18 demo scripts in sequence (`DEMO_INPUT=demos/s01.wav` by default)

### Fixed

- **Moorer reverb early reflections routing** -- early reflections now bypass comb filters and mix directly to output (classic Moorer design)
- **Moorer reverb delay read direction** -- fixed EarlyReflections reading forward (unwritten buffer) instead of backward (past samples)
- **Schroeder reverb bugs** (from original source) -- all 4 combs incorrectly used same filter instance; allpass path used uninitialized variable
- **DPW oscillator startup transient** -- seeded differentiator state in `reset()` to eliminate first-sample amplitude spike (~25x) caused by uninitialized `last_value_`
- **Faust VA filter NaN** -- seeded parameter smoothing registers in MoogLadder, MoogHalfLadder, and DiodeLadder to prevent `log10(~0)` producing -inf/NaN on first samples

## [0.1.2]

### Changed

- **GIL release in C++ bindings** -- all ~160 processing functions across 6 binding files now release the Python GIL during computation via `nb::gil_scoped_release`, enabling true multi-threaded parallelism
  - `_core_signalsmith.cpp` -- Biquad, FFT, RealFFT, Delay, LFO, envelope, STFT, Oversampler processing
  - `_core_daisysp.cpp` -- 73 functions: oscillators, filters, effects, dynamics, control, noise, drums, physical modeling, utility
  - `_core_stk.cpp` -- generators, filters (via macro), reverbs (via macro), instruments (via macro), effects, Guitar, Twang
  - `_core_madronalib.cpp` -- `ml_process`/`ml_process_stereo`/`ml_process2` templates (propagates to FDN reverbs, delay, resampling, generators), projections, amp/dB conversions
  - `_core_hisstools.cpp` -- MonoConvolve, Convolver, SpectralProcessor (convolve/correlate/change_phase), KernelSmoother
  - `_core_choc.cpp` -- FLAC read/write file I/O

### Fixed

- **Cross-platform build** (Linux, macOS, Windows)
  - Linux: `CMAKE_POSITION_INDEPENDENT_CODE` for static libs linked into shared `.so`
  - Linux: Suppressed GCC `-Wmaybe-uninitialized` false positives from HISSTools `Statistics.hpp`
  - Linux: Dropped aarch64 wheels (HISSTools NEON code requires Apple Clang-specific implicit type conversions)
  - macOS: Set `MACOSX_DEPLOYMENT_TARGET=10.15` for `std::filesystem::path` and nanobind aligned deallocation
  - macOS: Architecture detection via compiler built-in defines (`__aarch64__`) instead of `CMAKE_SYSTEM_PROCESSOR` (correct under cross-compilation)
  - macOS: `cmake/hisstools_arch_compat.h` -- bridges `__aarch64__` (GCC/Linux) to `__arm64__` (Apple/HISSTools)
  - Windows: `NOMINMAX` and `_USE_MATH_DEFINES` for MSVC across all targets
  - Windows: `cmake/msvc_compat.h` -- `__attribute__` no-op and `<cmath>` includes for DaisySP
  - Python < 3.12: Guarded `AudioBuffer.__buffer__` (PEP 688) behind version check

- **CI/CD** (`.github/workflows/`)
  - `build-publish.yml` -- cibuildwheel v3.3.1 wheel builds for Linux x86_64, macOS arm64+x86_64, Windows AMD64; TestPyPI + PyPI publish via trusted publishing
  - `ci.yml` -- QA (ruff lint/format, mypy typecheck) + native build/test matrix (ubuntu/macOS/Windows, Python 3.10+3.14)
  - Cross-compile macOS x86_64 wheels from ARM64 runner (macos-latest); tests skipped for x86_64

## [0.1.1]

### Added

- **CLI** (`nanodsp.__main__`, `nanodsp._cli`)
  - `nanodsp info <file>` -- audio file metadata (path, format, duration, sample_rate, channels, frames, peak_db, loudness_lufs), `--json` output
  - `nanodsp process <inputs...> -o OUT|-O DIR` -- chainable effect pipeline with `--fx`/`-f` (repeatable) and `--preset`/`-p` (repeatable)
  - Batch mode: `nanodsp process *.wav -O out/` processes multiple files to an output directory
  - Dry-run: `nanodsp process in.wav -n -f lowpass:cutoff_hz=1000` shows the chain without reading or writing files
  - Global `-v`/`--verbose` flag for detailed step-by-step output, `-q`/`--quiet` to suppress non-essential output (mutually exclusive)
  - `nanodsp analyze <file> <type>` -- 10 analysis subcommands (loudness, pitch, onsets, centroid, bandwidth, rolloff, flux, flatness, chromagram, info), `--json` output
  - `nanodsp synth <out> <type>` -- 7 synth types (sine, noise, drum, oscillator, fm, note, sequence)
  - `nanodsp convert <in> <out>` -- format conversion (WAV/FLAC), resampling (`--sample-rate`), channel conversion (`--channels`), bit depth (`-b`)
  - `nanodsp pipe` -- read WAV from stdin, apply `-f`/`-p` effect chain, write WAV to stdout; supports Unix pipe chaining
  - `nanodsp benchmark <function>` -- profile a DSP function with configurable iterations, warmup, buffer size; reports min/max/mean/median/std timing and realtime throughput multiplier, `--json` output
  - `nanodsp preset list|info|apply` -- 30 presets across 8 categories (mastering, voice, spatial, dynamics, lofi, cleanup, creative)
  - `nanodsp list [category]` -- browse all registered functions with signatures across 7 categories (filters, effects, dynamics, spectral, analysis, synthesis, ops)
  - 13 new presets: genre mastering (`master_pop`, `master_hiphop`, `master_classical`, `master_edm`, `master_podcast`), creative effects (`radio`, `underwater`, `megaphone`, `tape_warmth`, `shimmer`, `vaporwave`, `walkie_talkie`), lofi (`8bit`)
  - Function registry with auto-discovery from all modules, `inspect.signature`-based parameter display
  - Preset registry with single-function and chain-based presets, parameter overrides
  - FX token parser (`name:k=v,k=v`) with type coercion from signature defaults
  - `[project.scripts]` entry point: `nanodsp` command

- **Audio I/O** (`nanodsp.io`)
  - `read_wav_bytes(data)` -- parse WAV from raw bytes (for stdin/pipe workflows)
  - `write_wav_bytes(buf, bit_depth)` -- serialize AudioBuffer to WAV bytes (for stdout/pipe workflows)

- **CHOC FLAC codec** -- read/write FLAC files (16/24-bit) via header-only CHOC library
  - `nanodsp._core.choc` C++ bindings for FLAC read/write
  - `io.read_flac()`, `io.write_flac()` Python wrappers
  - `io.read()`, `io.write()` auto-detect WAV vs FLAC by extension
  - Fixed CHOC upstream bug in 24-bit float-to-int scale factor

- **Streaming infrastructure** (`nanodsp.stream`)
  - `RingBuffer` -- multi-channel ring buffer with independent read/write positions
  - `BlockProcessor` -- base class for block-based audio processors
  - `CallbackProcessor` -- wrap a callable as a block processor
  - `ProcessorChain` -- chain multiple processors in series
  - `process_blocks()` -- process a buffer through a function in blocks with optional overlap-add

- **DaisySP effects** (via `nanodsp.effects`)
  - Effects: `autowah`, `chorus`, `decimator`, `flanger`, `overdrive`, `phaser`, `pitch_shift`, `sample_rate_reduce`, `tremolo`, `wavefold`, `bitcrush`, `fold`, `reverb_sc`, `dc_block`
  - Filters: `svf_lowpass`, `svf_highpass`, `svf_bandpass`, `svf_notch`, `svf_peak`, `ladder_filter`, `moog_ladder`, `tone_lowpass`, `tone_highpass`, `modal_bandpass`, `comb_filter`
  - Dynamics: `compress`, `limit`

- **DaisySP synthesis** (via `nanodsp.synthesis`)
  - Oscillators: `oscillator`, `fm2`, `formant_oscillator`, `bl_oscillator`
  - Noise: `white_noise`, `clocked_noise`, `dust`
  - Drums: `analog_bass_drum`, `analog_snare_drum`, `hihat`, `synthetic_bass_drum`, `synthetic_snare_drum`
  - Physical modeling: `karplus_strong`, `modal_voice`, `string_voice`, `pluck`, `drip`

- **STK bindings** (`nanodsp._core.stk`) -- 5 submodules, 39 classes
  - Instruments: `Clarinet`, `Flute`, `Brass`, `Bowed`, `Plucked`, `Sitar`, `StifKarp`, `Saxofony`, `Recorder`, `BlowBotl`, `BlowHole`, `Whistle`, `Guitar`, `Twang`
  - Generators: `SineWave`, `Noise`, `Blit`, `BlitSaw`, `BlitSquare`, `ADSR`, `Asymp`, `Envelope`, `Modulate`
  - Filters: `BiQuad`, `OnePole`, `OneZero`, `TwoPole`, `TwoZero`, `PoleZero`, `FormSwep`
  - Delays: `Delay`, `DelayA`, `DelayL`, `TapDelay`
  - Effects: `FreeVerb`, `JCRev`, `NRev`, `PRCRev`, `Echo`, `Chorus`, `PitShift`, `LentPitShift`
  - High-level wrappers: `stk_reverb`, `stk_chorus`, `stk_echo`, `synth_note`, `synth_sequence`

- **Madronalib bindings** (`nanodsp._core.madronalib`) -- 7 submodules
  - FDN reverbs: `FDN4`, `FDN8`, `FDN16` with configurable delays, cutoffs, and feedback
  - Delays: `PitchbendableDelay`
  - Resampling: `Downsampler`, `Upsampler`
  - Generators: `OneShotGen`, `LinearGlide`, `SampleAccurateLinearGlide`, `TempoLock`
  - Projections: 18 easing functions (`smoothstep`, `bell`, `ease_in`, `ease_out`, etc.)
  - Windows: `hamming`, `blackman`, `flat_top`, `triangle`, `raised_cosine`, `rectangle`
  - Utilities: `amp_to_db`, `db_to_amp` (scalar and array overloads)

- **HISSTools bindings** (`nanodsp._core.hisstools`) -- 4 submodules
  - Convolution: `MonoConvolve`, `Convolver` (multi-channel) with selectable latency modes
  - Spectral processing: `SpectralProcessor` (convolve, correlate, phase change), `KernelSmoother`
  - Analysis: 24 statistics functions (`stat_mean`, `stat_rms`, `stat_centroid`, `stat_kurtosis`, etc.), `PartialTracker`
  - Windows: 28 window functions (Hann, Blackman-Harris variants, Nuttall variants, flat-top variants, Kaiser, Tukey, etc.)

- **Spectral processing** (`nanodsp.spectral`)
  - STFT/ISTFT with Hann window and COLA overlap-add reconstruction
  - Spectral utilities: `magnitude`, `phase`, `from_polar`, `apply_mask`, `spectral_gate`, `spectral_emphasis`, `bin_freq`, `freq_to_bin`
  - Spectral transforms: `time_stretch`, `phase_lock`, `spectral_freeze`, `spectral_morph`, `pitch_shift_spectral`, `spectral_denoise`
  - `eq_match` -- match spectral envelope between two buffers

- **Analysis** (`nanodsp.analysis`)
  - Loudness: `loudness_lufs` (ITU-R BS.1770-4), `normalize_lufs`
  - Spectral features: `spectral_centroid`, `spectral_bandwidth`, `spectral_rolloff`, `spectral_flux`, `spectral_flatness_curve`, `chromagram`
  - Pitch detection: `pitch_detect` (YIN algorithm)
  - Onset detection: `onset_detect` (spectral flux with peak picking)
  - Resampling: `resample` (madronalib backend), `resample_fft` (FFT-based)
  - Delay estimation: `gcc_phat` (GCC-PHAT)

- **Composed effects** (`nanodsp.effects`)
  - `saturate` (soft/hard/tape modes), `exciter`, `de_esser`, `parallel_compress`
  - `noise_gate`, `stereo_delay` (with ping-pong mode), `multiband_compress`
  - `reverb` with FDN backend and presets (room, hall, plate, chamber, cathedral)
  - `master` -- mastering chain (dc_block, EQ, compress, limit, normalize_lufs)
  - `vocal_chain` -- vocal processing chain (de-esser, EQ, compress, limit, normalize)
  - `agc` -- automatic gain control with asymmetric attack/release

- **Core DSP operations** (`nanodsp.ops`)
  - Delay: `delay`, `delay_varying`
  - Envelopes: `box_filter`, `box_stack_filter`, `peak_hold`, `peak_decay`
  - FFT: `rfft`, `irfft`
  - `convolve` -- FFT-based overlap-add convolution
  - Rate conversion: `upsample_2x`, `oversample_roundtrip`
  - Mixing: `hadamard`, `householder`, `crossfade`, `mix_buffers`
  - `lfo` -- cubic LFO with rate/depth variation
  - Normalization: `normalize_peak`, `trim_silence`, `fade_in`, `fade_out`
  - Stereo: `pan`, `mid_side_encode`, `mid_side_decode`, `stereo_widen`
  - Correlation: `xcorr` (FFT-based cross-/auto-correlation)
  - Analytic signal: `hilbert`, `envelope`
  - Filtering: `median_filter`, `lms_filter`

- **Biquad filter wrappers** (`nanodsp.effects`)
  - `lowpass`, `highpass`, `bandpass`, `notch`, `peak`, `peak_db`
  - `high_shelf`, `high_shelf_db`, `low_shelf`, `low_shelf_db`, `allpass`
  - `biquad_process` -- process through a pre-configured Biquad
  - All accept frequency in Hz with automatic normalization

- **AudioBuffer I/O methods**
  - `AudioBuffer.from_file(path)` -- classmethod to read WAV/FLAC by extension
  - `buf.write(path, bit_depth=16)` -- instance method to write WAV/FLAC by extension
- `nanodsp._core.pyi` -- complete type stubs for all 12 C++ submodules
- `Spectrogram` data class for STFT output (`[channels, frames, bins]` complex64)

### Changed

- **Module split** -- monolithic `dsp.py` replaced by focused modules:
  - `_helpers.py` -- shared private utilities
  - `ops.py` -- delay, envelopes, FFT, convolution, rates, mix, pan, normalization
  - `effects.py` -- filters, effects, dynamics, reverb, mastering chains
  - `spectral.py` -- STFT, spectral utilities, spectral transforms, eq_match
  - `synthesis.py` -- oscillators, noise, drums, physical modeling, STK synth
  - `analysis.py` -- loudness, spectral features, pitch/onset detection, resampling
- `__init__.py` stripped to `__version__` only -- no re-exports; use explicit imports
- `io.py` now supports both WAV and FLAC formats
- Test suite reorganized into per-module test files (1114 tests)
- Removed `disable_error_code = ["import-untyped"]` from mypy config (stubs fix this)

## [0.1.0]

### Added

- Initial project structure with scikit-build-core + CMake + uv
- Core C++ bindings via nanobind (`nanodsp._core`):
  - `filters` -- `Biquad` with 16 filter designs, `BiquadDesign` enum
  - `fft` -- `FFT` (complex-to-complex), `RealFFT` (real-to-complex)
  - `delay` -- `Delay` (linear interpolation), `DelayCubic` (cubic interpolation)
  - `envelopes` -- `CubicLfo`, `BoxFilter`, `BoxStackFilter`, `PeakHold`, `PeakDecayLinear`
  - `spectral` -- `STFT` (multi-channel analysis/synthesis)
  - `rates` -- `Oversampler2x`
  - `mix` -- `Hadamard`, `Householder`, `cheap_energy_crossfade`
- `AudioBuffer` class (pure Python, 2D `[channels, frames]` float32 with metadata)
  - Factory methods: `zeros`, `ones`, `impulse`, `sine`, `noise`, `from_numpy`
  - Channel operations: `to_mono`, `to_channels`, `split`, `concat_channels`
  - Arithmetic operators: `+`, `-`, `*`, `/`, `gain_db`
  - Pipeline: `pipe()` for chaining DSP functions
- `io.read_wav()`, `io.write_wav()` -- WAV file I/O (8/16/24/32-bit PCM, stdlib `wave`)
- Test suite with pytest (203 tests)

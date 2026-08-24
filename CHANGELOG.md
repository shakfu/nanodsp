# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0]

Correctness work dominates this release, most of it from a second full project
review. Two defects fired on the default arguments of public functions:
`daisysp.pitch_shift` returned an all-NaN buffer, and `daisysp.bitcrush`
returned audio at roughly 126 times full scale with an inverted sign. Both were
covered by passing tests that asserted only on shape, which is the gap
`tests/test_default_output.py` now closes across the registry.

One entry changes existing behaviour and is worth checking before upgrading:
**`bitcrush` and `lo_fi` output moves by about 42 dB and by the sign**, so any
downstream compensation for the old level has to come off. Everything else
either fixes output that was already wrong or adds new API.

The rest is a streaming path (`nanodsp.stream`, `io.read_blocks`,
`process --stream`), a third time-stretcher built on a published algorithm,
honest licence metadata now that the wheel is known to contain LGPL-2.1 code,
and several guards against the failure modes that let the above ship: numeric
fingerprints that actually run on CI, magnitude checks across every registry
entry, and a drift check on the hand-maintained type stub.

### Added

- **Streaming for any per-channel effect** (`nanodsp.stream`) -- `StatefulFilter` grew from five biquad constructors into a general `StatefulProcessor` (the old name remains as an alias) with eighteen ready-made constructors spanning dynamics, filters, modulation and the per-channel reverbs. Each holds one DSP object per channel across calls, so feeding a signal in blocks gives bit-identical output to processing it whole -- verified for all eighteen at two block sizes. Effects that cannot work this way now say so rather than being quietly absent: channel-linked dynamics need every channel at once, and the mono-to-stereo effects (the FDN `reverb`, `chorus`, the STK reverbs) have no per-channel form.

- **Chunked out-of-core I/O** (`io.read_blocks`, `io.BlockWriter`) -- read and write WAV a block at a time, so file size is bounded by disk rather than RAM. `read()` decodes whole-file, which caps usable input at a fraction of memory: an hour of stereo 96 kHz is ~2.7 GB as float32.

- **`nanodsp process --stream`** -- processes a file in constant memory by rebuilding the chain from stateful processors. Measured on a 60 s stereo 96 kHz file: peak RSS 379 MB whole-file versus 42 MB streamed, with byte-identical output. `--block-size` tunes the block. A chain containing anything without a streaming form is rejected up front, naming every offender; `compress` and `limit` report that they run unlinked rather than silently differing.

- **File operands in the CLI** (`-f name:param=@path.wav`) -- effects taking a second buffer were previously reachable only from Python. `sidechain_compress`, `vocoder`, `convolve`, `eq_match`, `crossfade` and the new convolution reverb now work from the command line: `-f sidechain_compress:sidechain=@kick.wav,ratio=8`. Omitting the operand suggests the syntax rather than saying the function is unusable.

- **Convolution reverb** (`effects.reverb.convolution_reverb`) -- IR-based reverb with wet/dry mix, pre-delay, optional tail extension and IR normalisation (on by default, since recorded IRs vary in level by orders of magnitude and `mix` would otherwise mean something different for every file). Pre-delay is applied to the IR rather than the wet signal, so the tail is not clipped by the delay amount. Unlike the algorithmic `reverb`, it preserves the input channel count.

- **178 verified docstring examples** across fifteen modules, and `tests/test_doctests.py` to run them. The package previously had none, so every documented example was prose that nothing executed -- writing these immediately found three wrong parameter names and one wrong return value in examples I had just written. Examples avoid printing bare numpy scalars, whose repr differs across numpy versions.

- **`tests/test_default_output.py` -- magnitude checks across the registry.** The suite asserted on shapes, dtypes and error types and almost never on magnitude, which is why both defects below shipped under passing tests: `test_pitch_shift_shape` received 24000 NaNs, checked `result.frames`, and passed. The new sweep calls every registry entry that needs nothing beyond its leading argument -- 86 chainable effects and 22 generators -- and asserts that output is finite, is not wildly out of range, and that silence in gives silence out. Bounds are set from measurement: the loudest processor is `agc` at 3.4 from a 0.5-peak input, so the limit is 10.

  Verified against the defects it exists for: reverting the `bitcrush` fix fails six of its cases. Reverting the `pitch_shift` fix fails none of them, for the reason given under Fixed -- which is why that one is pinned numerically instead.

- **`tests/GOLDEN.json` holds fingerprints per platform, and CI runs them.** The fixture stamped one platform and skipped everywhere else, so the 66-case numeric regression corpus -- the only guard on the vendored DSP's actual output, and the thing `TODO.md` relies on to test the unexplained `daisysp.chorus` hypothesis -- never ran on CI, which is Linux, while the committed block was macOS. Worse, `--update` replaced rather than merged, so regenerating on a second machine silently dropped the first machine's numbers and nobody found out, because a missing platform skips rather than fails. The schema is now `{"platforms": {key: cases}}`, `--update` merges and reports what it kept, a legacy fixture fails with migration instructions rather than skipping, and a platform with no block skips with a message naming the ones that do exist.

- **`THIRD_PARTY_LICENSES.md`, and it ships inside the wheel.** Every backend compiled into `_core` is MIT, BSD-3-Clause, ISC or LGPL-2.1, and all of those require their copyright notice to accompany a binary redistribution. The wheel carried only nanodsp's own LICENSE, so fifteen notices were missing from the artefact almost everyone installs. The file is generated by `tests/test_licenses.py --update` rather than hand-maintained, because a stale attribution file is worse than none, and a test fails when it drifts from `thirdparty/`. Writing the coverage check against licence *files* rather than top-level directories immediately caught two nested dependencies that a directory-level check would have missed: `AudioFile` inside GrainflowLib and `sse2neon` inside madronalib, both compiled in through their headers. `clap` and `utf` are listed separately as vendored-but-not-compiled -- nanodsp includes madronalib through `mldsp.h`, which never reaches `source/app`.

- **`tests/test_core_stubs.py` -- a drift guard for `_core.pyi`.** The 2600-line hand-maintained stub had nothing checking it, and a stub that has drifted is worse than no stub: mypy reports it as fact, so a binding renamed in C++ keeps type-checking against the old name. `mypy.stubtest` cannot help, because the stub models nanobind submodules as `class` blocks and stubtest stops at 19 "is not a type" errors without inspecting their contents -- which is where the whole stub lives. This compares names directly, two levels deep. It found zero drift on the existing stub, which is a credit to whoever maintained it by hand; it now stays that way by construction.

  The first version of this guard had a hole, and adding `keyframe` walked straight into it: the checks enumerated submodules from the stub, so they compared the contents of everything the stub declared but could not see a submodule the extension exposed and the stub omitted entirely. A whole new binding passed silently. `test_every_runtime_submodule_is_declared` closes that direction, and fails with the block to add.

- **Keyframe time stretching** (`timestretch.keyframe_stretch`, `timestretch.keyframe_sparsify`) -- a third stretcher, occupying a different point in the design space from the two already here. The signal is reduced to a sparse set of its local extrema; because their spacing tracks local bandwidth, it doubles as a per-sample estimate of information density, and each overlap-add crossfade is sized in extrema rather than samples. Splices therefore shorten at a transient and lengthen across a sustained note with no separate transient detector, no FFT and no correlation search. Measured here on a 60 Hz tone against broadband noise, mean extrema spacing differs by a factor of ~200, which is the adaptation the method rests on; `tests/test_keyframe.py` asserts it. `keyframe_sparsify` exposes the representation on its own -- the input as the stretcher sees it, with no time or pitch change -- which is both an audit tool and a usable lo-fi effect as the threshold rises. It is the only stretcher here that emits output sample by sample with no block latency. The registry goes from 169 entries to 171, and from 114 chainable effects to 116; both are reachable from the CLI as `keyframe_stretch:stretch=...` and `keyframe_sparsify:threshold=...`. `demos/demo_keyframe.py`, wired into `make demos`, renders the parameter sweeps that matter -- the splice-threshold macro, the duration cap, and a threshold sweep over the sparse representation on its own -- alongside a timed three-way comparison against the other two stretchers. `demos/play.sh` was added at the same time: `make demos` now writes 344 files, which is past the point of auditioning them by hand, so it filters them by name and plays the matches through sox. Extra keywords narrow rather than widen, and matches sort naturally so a parameter sweep plays as k4, k16, k64, k256. Its concession is spectral haze on tonally nuanced material; `signalsmith_stretch` remains the one to reach for when the result should sound like the input.

  An original implementation of the algorithm in M. Nielsen, "Keyframe Time Stretching via Extrema Sampling", DAFx26, whose paper is CC BY 4.0. The author's own firmware is AGPL-3.0 and was deliberately not consulted -- vendoring or linking it would have replaced this project's MIT licensing with AGPL-3.0. Provenance and the licensing reasoning are recorded in `thirdparty/VERSIONS.md` beside the equivalent note for PaulStretch.

  Validated against the figures the paper publishes rather than against a reference implementation, since by construction there is none to diff: odd-harmonic THD on a 1 kHz sine reproduces the paper's -38.1 dB exactly, and peak reconstruction error reproduces its Figure 6 at 0.0216 against an analytic bound of 0.0196. Two under-specified points in the paper's Algorithm 1 needed decisions, both at the derivative's sign when it is exactly zero -- which is routine, not exotic, because any tone whose period divides the sample rate puts every extremum on the sample grid where the central difference cancels. Taken literally the pseudocode plants a spurious keyframe at the start of every signal; guarding that naively then drops every on-grid extremum instead, halving the keyframe count on a 1 kHz sine at 48 kHz. Comparing against the last *nonzero* derivative handles both, and the subsample refinement then lands exactly on the sample. Both decisions are commented at the code and covered by regression tests.

### Changed

- **Effect classification keys on the leading parameter's annotation, not its name** -- previously a chainable effect had to call its first parameter `buf`, which excluded `vocoder(modulator, carrier)` and `crossfade(buf_a, buf_b)`. Both are chainable now that `-f` can load a second buffer from a file. The registry gained `convolution_reverb` and reports 114 chainable effects of 169 entries.

- **Licence metadata follows PEP 639 and names what is actually in the wheel.** `license = { text = "MIT" }` became the SPDX expression `MIT AND LGPL-2.1-only AND BSD-3-Clause AND ISC`, with `license-files` carrying both `LICENSE` and `THIRD_PARTY_LICENSES.md`; the now-superseded MIT classifier is gone. The wheel's metadata says `License-Expression` instead of a bare `MIT` that understated it. The README and docs backend tables listed DaisySP as "MIT" where `CMakeLists.txt` has always said "MIT + LGPL-2.1"; both now say so, and the README's License section spells out that DaisySP-LGPL supplies `Compressor`, `ReverbSc`, `MoogLadder`, `BlOsc`, `Bitcrush`, `Fold`, `Pluck`, `Tone` and `Comb` -- all reachable from the public API, so it is not an optional component. `twine check` passes on both wheel and sdist.

- **CI type-checks the package rather than one file.** The workflow ran `mypy src/nanodsp/__init__.py tests/`, which is seven lines, while the `Makefile` target ran `mypy src/nanodsp tests/`. Full-tree mypy already passed, so this costs nothing and closes a gap that would only have mattered once it did.

### Fixed

- **`effects.daisysp.pitch_shift` returned an all-NaN buffer at its default arguments.** DaisySP's `PitchShifter::Init` assigned only part of the object, leaving `transpose_`, the crossfade history and the random-modulation state indeterminate. `Init` ends by calling `SetDelSize`, which computes the first transposition from that indeterminate `transpose_`; and at transposition 0 -- the wrapper's default -- the modulation frequency solves to zero and freezes both phasors, so the two branches in `Process()` that are the only writers of `slewed_mod_` and `mod_coeff_` never run. The first read of that state turned every output sample into NaN. Patched in `thirdparty/`; see `thirdparty/VERSIONS.md`.

  The symptom was not reliably reproducible, which is characteristic of the fault rather than incidental to it: whether the indeterminate values happened to be usable floats depended on what the allocator last left in that memory. It reproduced in three consecutive fresh interpreters and did not reproduce inside the test suite, where other DSP had run first. `daisysp.pitch_shift` is now pinned in `tests/GOLDEN.json` at transposition 0 and 7, because a numeric fingerprint is the only guard that does not depend on allocator history.

- **Behaviour change: `effects.daisysp.bitcrush` amplified by about 2^(bit_depth-1) and inverted the signal.** DaisySP's decode step read `out *= (65536.0f / bits) - 32768;`, which parses as one multiply by `((65536/bits) - 32768)` rather than the multiply-then-subtract that undoes the encode half above it. At the default 8 bits a 0.5-peak input came back at 126.5, and silence came back at a 66.5 DC offset; from the CLI the result was a file that was 99.998% full-scale. `effects.composed.lo_fi` calls `bitcrush` and was affected the same way.

  Output of `bitcrush` and `lo_fi` now differs from previous releases by roughly 42 dB at the default bit depth, and by the sign. Anyone who compensated for the old level downstream will need to remove that compensation.

- **A shared `static Fold` in the same file made `Bitcrush` instances interfere.** It was a file-scope object used by every `Bitcrush` in the process, so two instances shared fold state and concurrent `Process()` calls -- routine under `nanodsp process -j`, since the bindings release the GIL -- raced on it. It is a member now.

- **Heap buffer overflow in the generator bindings when asked for zero samples.** `util_trigger_generate_mono` wrote the first sample before testing the count, so `frames=0` wrote one element past a zero-size allocation; confirmed under AddressSanitizer. The same pattern was open-coded a second time in the `Pluck` binding. Reachable from the documented public API as `synthesis.analog_bass_drum(0)` and seven other generators. Both sites are guarded, and a negative count now raises `ValueError` naming the argument instead of `MemoryError: std::bad_array_new_length`.

- **Documentation claims that were not true.** `docs/api/io.md` said WAV goes through the stdlib `wave` module; it goes through a RIFF chunk walker in `io.py`, which is why float and `WAVE_FORMAT_EXTENSIBLE` files work at all -- `wave` reads neither. The README described `--stream` as producing "identical output" without qualification; channel-linked dynamics run unlinked when streamed, measured at 0.038 peak absolute on stereo `compress`, and the CLI note saying so is suppressed by `-q`. `_run_parallel`'s docstring claimed "no state is shared between workers", which skips the two pieces of process-global state inside vendored C++ -- the C `rand()` stream that `set_random_seed` reseeds process-wide, and the function-local `static` seed in DaisySP's `PitchShifter` that `Process()` mutates with the GIL released, making a `-j` chain containing `pitch_shift` neither reproducible nor race-free. The io docs also now state that the writers take the destination first, `io.write(path, buf)`, unlike `soundfile.write` and `scipy.io.wavfile.write`.

- **A preset naming a non-callable attribute failed late and obscurely.** `_resolve_preset_fn` allowlists the module but resolved any public attribute within it, so `ops.np` was a valid lookup returning the numpy module and a useless preset step that only failed later inside `inspect`. It now rejects a non-callable at resolution, with the name.

## [0.2.0]

This release is dominated by correctness work following a full project review.
Several entries change existing behaviour; those are marked **Behaviour change**
and are listed first under Changed -- hence the minor bump rather than a patch
release. Four are worth checking before upgrading: `AudioBuffer` no longer
aliases the array it is constructed from, `compress` and `limit` are now
stereo-linked by default, the CLI effect chain runs in the order given rather
than effects-then-presets, and `-f` now rejects anything that is not a chainable
effect.

### Added

- **32/64-bit IEEE float WAV, read and write** (`nanodsp.io`) -- WAV is now parsed from RIFF chunks directly instead of through the stdlib `wave` module, which accepts only `WAVE_FORMAT_PCM` and rejects everything else outright. That excluded two very common cases: 32-bit float WAV (what ffmpeg, Audacity and most DAWs write from a float pipeline) and `WAVE_FORMAT_EXTENSIBLE`, which many writers emit for anything above two channels or 16 bits. `wave` also cannot *write* float at all. `bit_depth` now accepts 32 and 64 for IEEE float alongside 16 and 24 for PCM; float output is written verbatim and is not clipped, so material above full scale survives a gain stage intact. Unknown codecs now name themselves in the error (`Unsupported WAV encoding: IMA ADPCM`) and RF64/BW64 files are rejected with a clear message rather than a parse failure. No new dependency: the parser is about 120 lines of Python.

- **Parallel batch processing** (`nanodsp process -j/--jobs N`, `0` = one worker per CPU) -- every C++ processing entry point releases the GIL, so batch work scales across cores using threads, with no multiprocessing and no pickling of buffers. Measured 2.67x end-to-end on 8 cores for 8 x 20 s stereo files through a reverb/compress/saturate chain (closer to 3.9x on the DSP alone, once fixed interpreter startup is subtracted). Output is verified bit-identical to serial. A failure in one file now reports and continues, exiting non-zero at the end, instead of aborting the batch.

- **Level reporting and clipping detection in the CLI** -- `nanodsp process --stats` reports peak, true peak and integrated loudness before and after the chain. Independently, a warning is now printed whenever integer output would exceed full scale, which is the most common way a chain silently costs the user audio. The warning is suppressed for float output, where the peak survives intact, and under `-q`. Only the sample peak is checked, not the true peak: this runs for every file in a batch, and a 4x-oversampled true-peak measurement is not worth paying for unconditionally.

- **Step-scoped preset overrides** -- an override key may now name a single chain step, as in `preset apply master_hiphop in.wav out.wav highpass.cutoff_hz=40`. Several steps in one chain can share a parameter name (`master_hiphop` has a highpass and two shelving filters that all take `cutoff_hz`), so an unscoped override can reach more steps than intended; the scoped form removes the ambiguity. Naming a step that is not in the chain is an error rather than a silent no-op.

- **Reproducible synthesis** -- a `seed` parameter on the 11 synthesis functions that draw from the C library `rand()` (`synth_note`, `synth_sequence`, `pluck`, `drip`, `string_voice`, `hihat`, `analog_snare_drum`, `synthetic_snare_drum`, `synthetic_bass_drum`, `clocked_noise`, `dust`), plus `nanodsp._core.stk.set_random_seed()` underneath. These are now pure functions of their arguments, like the rest of the library; pass a different seed for a different variation. See the STK seeding fix below for why they were not.

- **`copy` parameter on `AudioBuffer`** -- `AudioBuffer(data, copy=False)` hands over an array the caller no longer references, avoiding one full-buffer copy. Used internally by the per-channel effects driver, so the safer default costs nothing on the effects path.

- **`make asan`** -- rebuilds the extension with AddressSanitizer and runs the suite under it. Memory errors in the vendored C++ are invisible from Python and invisible to the test suite: an out-of-bounds write corrupts the allocator's metadata and the process traps later, inside an unrelated allocation, so the reported location is meaningless. This target is the only reliable way to find them, and is worth running after any vendored-library upgrade. (macOS strips `DYLD_INSERT_LIBRARIES` across `uv run`, so the target invokes the venv interpreter directly.)

- **Four contract-style test modules** -- `tests/test_golden.py` pins the numeric output of 65 cases covering every backend, so a vendored library that starts producing different numbers is caught even when it still satisfies every property the suite asserts; `tests/test_channel_contract.py` pins channel-count, length and sample-rate invariants as explicit allow-lists; `tests/test_stk_determinism.py` pins reproducibility of the `rand()`-drawing voices, including that a voice's first render in a process matches every later one; `tests/test_docs.py` checks documented counts against reality. The suite went from 1699 to 2342 tests at ~94.5% coverage.

  The golden fixtures are stamped with the platform that produced them and skip elsewhere, because the corpus is reference-platform-scoped rather than universal: glibc's `rand()` is not macOS's, so a seeded render reproduces only within a platform, and the FFT-heavy paths accumulate enough floating-point difference across libm and compiler versions to exceed any useful tolerance (measured on CI: `spectral.pitch_shift`'s peak moved 0.18% between macOS and Linux, while every case matched between two different macOS machines). Regenerate locally to pin your own platform.

### Changed

- **Behaviour change: `compress` and `limit` are now channel-linked by default** (`link=True`) -- one gain curve is computed from the per-frame maximum across channels and applied to all of them, preserving inter-channel ratios. Previously each channel ran its own detector, so a transient in one channel ducked only that channel and the stereo image shifted -- wrong for the stereo programme material the mastering presets target. `link=False` restores the previous behaviour for per-channel utility work. Mono output is bit-identical in both modes. (`noise_gate` was already linked and is unchanged.)

- **Behaviour change: `AudioBuffer` no longer aliases the array it is given** -- the constructor previously adopted a contiguous float32 ndarray as-is while copying anything that needed conversion, so whether writes through the buffer reached the caller's array depended on that array's dtype and memory layout -- not a distinction callers can reasonably track. It now copies by default; pass `copy=False` for the hand-over case. `slice()` had the matching problem (a frame slice of a mono buffer is contiguous and was adopted, the same slice of a stereo buffer was not and got copied) and now never shares storage; use `buf.data[:, start:end]` for a genuine view. Measured cost on the effects path: none (2.60 ms vs 2.61 ms for a 10 s stereo lowpass).

- **Behaviour change: the CLI effect chain now runs in the order given** -- `-f` and `-p` accumulated into two separate lists, and argparse does not record how they interleaved, so the chain was always built as "every effect, then every preset", silently reordering what the user typed. For a chain of DSP operations order is semantics: filter-then-saturate and saturate-then-filter are different effects.

- **Behaviour change: `-f` accepts only chainable effects** -- the CLI function registry collected everything in a module that was callable and not underscore-prefixed, which admitted imported typing constructs and classes (`Callable`, `Literal`, `AudioBuffer`, `Spectrogram`) as "DSP functions". Registration now admits only functions defined in the module being scanned and classifies each by signature. `-f` accepts the 110 chainable `(buf, ...) -> AudioBuffer` processors and rejects the rest by name and reason; generators, analyzers and Spectrogram-domain operators remain listed by `nanodsp list`, annotated with why they are not chainable. The registry went from 174 entries to 168.

- **WAV quantisation now rounds to nearest, with matched encode/decode scales** -- the encoder truncated toward zero (`astype`) and scaled by `2**(n-1) - 1` while the decoder divided by `2**(n-1)`. Rounding alone was not enough: the scale mismatch leaves a residual gain error of up to a full LSB near full scale, which swamps the rounding it was meant to protect. With both fixed, worst-case 16-bit round-trip error drops from 1.889 LSB to exactly 0.500 (the theoretical optimum), mean error from -0.008 LSB to under 0.001, and every exactly representable level round-trips bit-exactly. Integer dtypes are also now spelled little-endian explicitly, since WAV is little-endian regardless of host byte order.

- **`-b/--bit-depth` accepts 32** across `process`, `synth`, `convert`, `preset apply` and `pipe`, selecting IEEE float output.

- **Channel-count behaviour documented** -- `reverb` returns 2 channels whatever it is given, mono-sums the input before the FDN, and folds more than two channels to a stereo pair; none of that was documented, and a chain that assumes channel count is invariant changes shape there. Writing the check found the behaviour was not unique to `reverb`: eleven processors widen mono to stereo and four fold above two channels down. All are now documented and pinned.

- **CLI override coercion unified** -- `preset apply` used a weaker float-or-string coercion than `-f`, so `flag=false` became the truthy string `"false"`. Both paths now share one implementation.

### Fixed

- **Two heap buffer overflows in vendored STK** -- an intermittent crash (SIGTRAP, shell exit 133) during an ordinary test run turned out to be the allocator trapping in its own free list while servicing an unrelated `new`. That is the signature of memory corruption: the crash surfaces wherever the next allocation lands, so the reported location is meaningless and moves between runs. AddressSanitizer located two upstream bugs: `PitShift::PitShift()` looped `i <= window_.size()` and wrote one `StkFloat` past a 5000-element buffer (fires on construction alone), and `LentPitShift::tick()` read `dpt[delay_]` after the pitch-tracking loop had incremented `delay_` one past the end of the array (input-dependent, so it fired only for some signals). Both are patched in the vendored source and recorded in `thirdparty/VERSIONS.md`; the suite is now ASan-clean. Neither is detectable from Python, and the suite passed 2326 tests while the corruption was happening.

- **`drip` fed two of its three resonator bands uninitialised memory** -- vendored DaisySP's `Drip::Process` assigns `outputs10_ = inputs1_` and `outputs20_ = inputs2_`, the *members*, where it means the locals `inputs1` / `inputs2` computed two lines above; the members are never assigned anywhere in the class. Two of the three resonator bands were therefore driven by whatever was in memory while the values actually computed for them were discarded. **This changes what `drip` sounds like** -- it is now the algorithm as written, rather than one band plus two reading stale heap. Band 0 immediately above uses the local, which is the intended form.

- **`synthetic_bass_drum`'s first render differed from every later one** -- `SyntheticBassDrum::Init` zeroes the body-envelope state but not its transient counterpart, and `transient_env_lp_` is only ever written through `fonepole()`, which reads the previous value first. The first `Process()` after construction therefore read uninitialised memory. Freshly mapped pages read as zero, so a cold render looked perfectly deterministic across runs while every later render in the same process reused its predecessor's freed block and converged on a different value -- which is why an ordinary test suite could not see it. Both this and the `drip` defect were caught by CI running the test suite in a different order than a local run.

- **Uninitialised read in the DaisySP modulation effects** -- `ChorusEngine::Init`, `Flanger::Init` and `PhaserEngine::Init` each call `SetLfoFreq`, which reads `lfo_freq_` to decide the LFO direction before anything has assigned it. An indeterminate negative value latches a reversed LFO for the object's entire lifetime, and every later `SetLfoFreq` preserves that sign, so no wrapper can correct it after construction. Patched in the vendored source (one line each) and recorded in `thirdparty/VERSIONS.md`; chorus, flanger and phaser are pinned in the golden corpus.

- **STK seeded its noise from the wall clock** -- `Noise::setSeed` called `srand(time(NULL))` for the default seed, which is every `Noise` inside every STK voice. STK renders therefore differed on every run landing in a different wall-clock second, with no way to pin them from Python. Worse, `srand()` is process-global and DaisySP draws from the same `rand()` stream, so merely constructing an STK instrument silently randomised unrelated DaisySP generators (`pluck`, `drip`, `dust`, `clocked_noise`, `string_voice`, `hihat`, the snare and bass drums). The vendored `setSeed` no longer touches `rand()` for the default seed, and the affected functions seed it explicitly on entry.

- **Preset overrides crashed on chain presets** -- an override was merged into every step of a chain, so the first step that did not accept it raised `TypeError`. This made overrides unusable on 17 of the 30 built-in presets, despite being documented. Overrides are now filtered per step by signature.

- **`-f` on a non-effect silently corrupted the output or crashed** -- `-f stft` wrote a garbage file and exited 0 (a `ComplexWarning` was the only hint), and `-f loudness_lufs` produced a raw `AttributeError` traceback. Both are now rejected before any processing, naming the reason. `-f` on a processor needing a second buffer (`sidechain_compress`, `eq_match`, `convolve`) now says so instead of failing deep in the DSP layer.

- **Parameter coercion mishandled two common inputs** -- for parameters with no default, `1e3` silently became the string `"1e3"` (`int("1e3")` raises, and the resulting `ValueError` was swallowed) and `inf` raised an uncaught `OverflowError`. Integer narrowing is now guarded by a digit test.

- **`numpy` copy protocol** -- `AudioBuffer.__array__` ignored the `copy` argument numpy passes, so `np.array(buf)` could return shared storage despite the caller asking for a copy. It is now honoured, and an impossible `copy=False` conversion raises as numpy's contract requires.

- **Stale and incorrect documentation counts** -- the README claimed 1522 tests and 18 demo scripts (actual: 1699 and 20 at the time), and `docs/index.md` claimed 12 C++ backends while its table listed 11, having lost the signalsmith-stretch row. All corrected, and `tests/test_docs.py` now fails when a documented count stops matching reality.

## [0.1.9]

### Added

- **Constant-Q spectral spread for PaulStretch** (`spread_octaves` on `nanodsp.timestretch.paulstretch`, `PaulStretch.spread_octaves`) -- spectral smearing on a log-frequency axis, so every partial is spread across a fixed fraction of its own frequency instead of a fixed number of FFT bins. The existing `spread` blurs a fixed bin radius, and because bin spacing is linear in frequency it smears bass partials into mush while barely touching the top end; measured on a 4096-point window, a 32-bin radius covers 1.11 octaves at 220 Hz but only 0.08 at 3520 Hz. The new control holds that width constant to within 7% across the same range, and is independent of `window_size` and sample rate. Implemented by warping the magnitude spectrum onto a log axis, smoothing with a forward/backward one-pole cascade, and warping back; `spread` is retained for the cases where its low-end bias is the wanted effect.

- **Tonal/noise separation for PaulStretch** (`tonal_vs_noise` and `tonal_noise_octaves` on `nanodsp.timestretch.paulstretch`, plus the matching `PaulStretch` properties) -- splits the spectrum against a smoothed copy of itself, treating whatever stands above its own local envelope as tonal and the rest as the noise floor, then blends the output toward one component. `+1` keeps only the pitched peaks for a cleaner, more harmonic drone; `-1` keeps only the floor for a breathy, unpitched wash; `0` leaves the spectrum untouched. `tonal_noise_octaves` sets the width of the envelope used for the comparison, and so how sharp a partial must be to count as tonal. The effect increases monotonically across the whole `[-1, 1]` range.

Both modules reproduce techniques described by the GPLv3 [paulxstretch](https://github.com/essej/paulxstretch) application, but are written from scratch against the vendored signalsmith RealFFT -- no upstream source is copied or linked, and the project stays MIT. Both also deviate deliberately from the original parameter mappings, which measurement showed to be poorly conditioned: upstream's spread dial produces no smoothing at all below roughly 0.8 (and the dead zone widens as the FFT shrinks), and its tonal/noise control collapses to silence at both extremes, because scaling the detection threshold eventually destroys the peaks along with the noise. See the notes in `thirdparty/VERSIONS.md`.

### Changed

- **PaulStretch demo extended** (`demos/demo_paulstretch.py`) -- 11 to 15 examples, adding narrow and wide constant-Q spreads and the two tonal/noise extremes.

## [0.1.8]

### Added

- **PaulStretch extreme time-stretching** (`nanodsp.timestretch.paulstretch`, CLI filter `paulstretch`) -- the PaulStretch algorithm (by Nasca Octavian Paul, public domain) for extreme time-stretching via phase-randomized spectral resynthesis, producing the smeared, pad-like textures it is known for at large stretch factors where a phase-vocoder breaks down. Implemented as a new C++ backend (`paulstretch.PaulStretch`) on the signalsmith RealFFT; it is an original implementation and does not vendor the GPLv3 paulxstretch application sources. Supports onset/transient preservation and spectral effects (pitch/octave shift, added harmonics, spectral spread, and spectral band-pass filtering). Phase randomization is seeded for reproducible output, and stereo channels are decorrelated for a wider image.

- **PaulStretch demo** (`demos/demo_paulstretch.py`, wired into `make demos`) -- renders 11 examples covering stretch factors, window size, transient preservation, octave shift, harmonics/spread, spectral band-pass, and a long drone. A `--source-seconds` flag trims the source so large stretch factors stay reasonably sized.

- **Signalsmith time-stretching / pitch-shifting** (`nanodsp.timestretch.signalsmith_stretch`, CLI filter `signalsmith_stretch`) -- the MIT-licensed [signalsmith-stretch](https://github.com/Signalsmith-Audio/signalsmith-stretch) library (Geraint Luff / Signalsmith Audio), a transient-aware, phase-vocoder-derived stretcher that stays musical at modest ratios and decouples time-stretch from pitch-shift. Implemented as a new C++ backend (`signalsmith_stretch.SignalsmithStretch`) driving the library's offline whole-buffer pattern (seek/process/flush with pre-roll fold-back) for an exact-length, latency-trimmed result; all channels are processed together so the stereo image stays coherent. Exposes independent pitch-shift in semitones, a tonality limit for preserving high-frequency timbre on large shifts, a lower-CPU "cheaper" preset, and a seed for reproducible output. Vendored at release 1.1.1 on top of the already-vendored signalsmith-dsp -- no new FFT library is needed (newer upstream releases depend on the separate signalsmith-linear library, which is not vendored).

- **Signalsmith stretch demo** (`demos/demo_signalsmith_stretch.py`, wired into `make demos`) -- renders 15 examples covering time-stretch factors, pure pitch-shifts (octave up/down, fifth, fine detune), a tonality-limited shift, combined stretch-plus-pitch ("monster"/"chipmunk"), the cheaper preset, and an extreme-factor comparison of signalsmith vs PaulStretch on the same source.

### Fixed

- **Clean rebuilds on newer toolchains** -- vendored DaisySP sources use unqualified `size_t`, which recent libc++/SDK versions (e.g. current Xcode) no longer leak into the global namespace, breaking a full `make build`. A force-included compat shim (`cmake/daisysp_compat.h`) now restores it for the DaisySP targets without editing the vendored sources, mirroring the existing `msvc_compat.h` / `hisstools_arch_compat.h` shims.

## [0.1.7]

### Added

- **User-defined CLI presets** -- the CLI now loads custom presets from `~/.nanodsp/presets.json` (or the path in `$NANODSP_PRESETS`) and merges them with the built-ins; a user preset overrides a built-in of the same name. They work in `preset list`, `preset info`, `preset apply`, and `process -p`. Each entry uses the same JSON shape as a built-in (a single `fn` + `defaults`, or a `chain` of `[module, function, params]` steps). Malformed preset files produce a clean error.

- **CLI tab completion** -- optional shell completion via `argcomplete` (`eval "$(register-python-argcomplete nanodsp)"`), covering subcommands, options, preset names, preset categories, and function names. `argcomplete` is an optional dependency; the CLI works normally without it.

- **Stateful streaming filters** (`nanodsp.stream.StatefulFilter` plus `stateful_lowpass`, `stateful_highpass`, `stateful_bandpass`, `stateful_notch`, and `stateful_moog_ladder`) -- filters that retain per-channel state across `process()` calls by holding one persistent C++ DSP object per channel. Feeding a signal in arbitrary blocks produces bit-exactly the same result as processing it whole, so filters can now be streamed in real time or over long files without discontinuities at block boundaries (which the stateless `nanodsp.effects.filters` functions cannot do, since they rebuild their filter every call). `StatefulFilter` subclasses `BlockProcessor`, composes inside `ProcessorChain`, supports `reset()`, and accepts a custom factory to wrap any stateful per-channel DSP object.

- **Top-level `AudioBuffer` export** -- `AudioBuffer` is now re-exported from the package root, so `from nanodsp import AudioBuffer` (and `nanodsp.AudioBuffer`) work directly. Importing the package stays cheap: `buffer.py` depends only on numpy, so `import nanodsp` does not load the compiled `_core` extension. All other functionality continues to be imported from its specific submodule (e.g. `from nanodsp.effects.filters import lowpass`).

### Changed

- **DaisySP binding deduplication** -- extracted three templated helpers in `_core_common.h` (`util_process_mono`, `util_generate_mono`, `util_trigger_generate_mono`) that factor out the repeated per-sample binding pattern (output allocation, GIL release, processing loop, ownership transfer to numpy). Applied them across `_core_daisysp.cpp` at 57 call sites, reducing the file from 1560 to 1266 lines (~19%) and centralizing the GIL-release/allocation contract. Behavior is unchanged; `Pluck` retains a bespoke binding because its `Process(float&)` signature is incompatible with the shared helper.

- **Centralized mono/multi-channel return-shape policy** -- added a `_squeeze_mono` helper and applied it across the six `nanodsp.analysis` spectral features and `pitch_detect`, replacing the copy-pasted squeeze-if-mono idiom. Behavior is unchanged: a mono input drops the leading channel axis, multi-channel input keeps it.

- **Factored synthesis/ops boilerplate** -- introduced `_synth_triggered` (drives the five DaisySP drums plus the modal and string voices), `_apply_envelope` (backs `box_filter`, `box_stack_filter`, `peak_hold`, `peak_decay`), and `_process_mix_frames` (backs `hadamard` and `householder`), removing the repeated init/configure/trigger/render and per-frame mixing loops. No behavior change.

- **Numerical-correctness test tier** -- added `tests/test_numerical.py` (38 tests) that verify signal behavior rather than just shape/dtype: a `parametrize`-driven filter harness measuring passband/stopband gain via single-bin DFT across the biquad, state-variable, ladder, tone, and IIR designs; oscillator-fundamental and YIN pitch-accuracy checks; and alias-suppression checks comparing band-limited oscillators (PolyBLEP, BLIT, DPW) against a naive sawtooth.
- **Coverage gate** -- coverage reporting now fails below 90% (`[tool.coverage.report] fail_under` in `pyproject.toml`), enforced by `make coverage` and the CI `--cov` run. Current coverage is ~94%.

- **Narrowed CLI exception handling** -- replaced the ten broad `except Exception` blocks in the CLI with targeted `_IO_ERRORS` (file/stream I/O) and `_DSP_ERRORS` (effect/preset application) tuples, so expected failures still exit cleanly with a message while genuine bugs surface with a full traceback instead of being masked.

### Fixed

- **README API reference and quick start** -- corrected the documented examples to match the current API after the `effects` package split and parameter renames. Imports now use the real submodule paths (e.g. `nanodsp.effects.filters`, `nanodsp.effects.dynamics`); parameter names were fixed (e.g. `cutoff_hz`/`center_hz`, `resonance`, `lfo_freq`, `bit_depth`, `lp_freq`); spectral examples reflect that most transforms operate on a `Spectrogram`; and the `synth_sequence`, `CallbackProcessor`, and `ProcessorChain` call shapes were corrected. The reference catalog is now grouped by submodule, and all documented snippets were verified to execute.

## [0.1.6]

### Added

- **Sidechain compressor** (`nanodsp._core.fxdsp.SidechainCompressor`, `nanodsp.effects.dynamics.sidechain_compress`) -- compressor driven by an external sidechain signal for ducking effects. C++ implementation with one-pole attack/release smoothing in dB domain. GIL released during processing.
  - Configurable ratio, threshold, attack, release
  - Accepts mono sidechain envelope applied across all channels of the input

- **Transient shaper** (`nanodsp._core.fxdsp.TransientShaper`, `nanodsp.effects.dynamics.transient_shape`) -- independent attack and sustain envelope control using dual envelope followers. C++ implementation with configurable fast/slow attack/release times. GIL released during processing.
  - `attack_gain` scales transient component; `sustain_gain` scales sustained component
  - Fast envelope tracks transients (~5 ms); slow envelope tracks body (~50 ms)

- **Lookahead limiter** (`nanodsp._core.fxdsp.LookaheadLimiter`, `nanodsp.effects.dynamics.lookahead_limit`) -- brick-wall limiter with lookahead buffer for transparent peak control. C++ implementation with forward-looking minimum gain curve and smooth release. GIL released during processing.
  - Audio delayed by `lookahead_ms` so gain reduction starts before peaks arrive
  - Output guaranteed not to exceed `threshold_db`

- **True peak metering** (`nanodsp.analysis.true_peak_dbtp`) -- ITU-R BS.1770-4 true-peak measurement via 4x oversampling (two passes of `upsample_2x`). Detects inter-sample peaks that exceed the sample peak.

- **Channel vocoder** (`nanodsp.effects.composed.vocoder`) -- classic channel vocoder with logarithmically-spaced bandpass filterbank, per-band envelope extraction (rectify + lowpass), and carrier modulation. Configurable band count, frequency range, and envelope smoothing.
  - Modulator provides spectral envelope; carrier provides timbre
  - Delegates to C++ bandpass/lowpass filters per band

- C++ header `thirdparty/fxdsp/dynamics.h` with 3 header-only classes (`SidechainCompressor`, `TransientShaper`, `LookaheadLimiter`)
- nanobind bindings for all 3 new C++ classes in `_core_fxdsp.cpp` with GIL release
- Type stubs for the 3 new classes in `_core.pyi`
- 25 new tests: sidechain compress (6), transient shaper (4), lookahead limiter (4), true peak (4), vocoder (7)

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

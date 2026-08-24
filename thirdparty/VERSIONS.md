# Third-Party Library Versions

All libraries are vendored directly into the `thirdparty/` directory (no git submodules).

| Library | Version / Ref | License | Upstream URL |
|---------|---------------|---------|--------------|
| choc | unversioned (2025 snapshot) | ISC | https://github.com/Tracktion/choc |
| DaisySP | 0.0.1 (CMakeLists.txt) | MIT | https://github.com/electro-smith/DaisySP |
| DspFilters | unversioned | MIT | https://github.com/vinniefalco/DSPFilters |
| fxdsp | unversioned (custom/cleaned) | MIT | local / rewritten from various sources |
| GrainflowLib | unversioned | MIT | https://github.com/composingcap/GrainflowLib |
| HISSTools_Library | unversioned | BSD 3-Clause | https://github.com/AlexHarker/HISSTools_Library |
| madronalib | 0.1.0 (CMakeLists.txt) | MIT | https://github.com/madronalabs/madronalib |
| signalsmith | 1.7.0 (header macro) | MIT | https://signalsmith-audio.co.uk/code/dsp/ |
| signalsmith-stretch | 1.1.1 (release tag) | MIT | https://github.com/Signalsmith-Audio/signalsmith-stretch |
| stk (STK) | ~5.0.0-dev (see notes) | MIT | https://github.com/thestk/stk |
| vafilters | unversioned (Faust-generated) | MIT-style STK-4.3 | local / derived from Faust DSP |

## Local patches

Changes made to vendored sources. Each is marked in place with a
`nanodsp local patch` comment, so `grep -rn "nanodsp local patch" thirdparty/`
lists them all. **Re-apply these after any upgrade of the library concerned.**

| File | Change | Why |
|------|--------|-----|
| `DaisySP/Source/Effects/chorus.cpp` | `lfo_freq_ = 0.f;` added in `ChorusEngine::Init` | `Init` calls `SetLfoFreq`, which reads `lfo_freq_` to decide LFO direction before anything has assigned it. An indeterminate negative value latches a reversed LFO for the object's lifetime, and every later `SetLfoFreq` preserves that sign, so no wrapper can correct it after construction. |
| `DaisySP/Source/Effects/flanger.cpp` | same, in `Flanger::Init` | same defect |
| `DaisySP/Source/Effects/phaser.cpp` | same, in `PhaserEngine::Init` | same defect |
| `DaisySP/Source/Drums/synthbassdrum.cpp` | `transient_env_` / `transient_env_lp_` zeroed in `Init` | `Init` zeroes the body-envelope state but not its transient counterpart. `transient_env_lp_` is only ever written through `fonepole()`, which reads the previous value first, so the first `Process()` after construction read uninitialised memory. |
| `DaisySP/Source/PhysicalModeling/drip.cpp` | `outputs10_ = inputs1` / `outputs20_ = inputs2` (was `inputs1_` / `inputs2_`) | A stray trailing underscore made two of the three resonator bands read the *members* `inputs1_`/`inputs2_`, which are never assigned anywhere, and discard the locals actually computed for them. Both an uninitialised read and a wrong result; band 0 immediately above uses the local, which is the intended form. |
| `DaisySP/Source/Effects/pitchshifter.h` | `transpose_`, `pitch_shift_`, `prev_phs_a_`, `prev_phs_b_`, `mod_a_amt_`, `mod_b_amt_`, `slewed_mod_[2]` and `mod_coeff_[2]` zeroed in `Init` | `Init` assigned only part of the object. It ends by calling `SetDelSize`, which calls `SetTransposition(transpose_)`, so the first transposition was computed from indeterminate memory. Worse, at transposition 0 -- the wrapper's default -- the modulation frequency solves to zero and freezes both phasors, so the two `prev_phs_ > fade` branches in `Process()`, the only writers of `slewed_mod_` and `mod_coeff_`, never ran. The first `slewed_mod_[i] += mod_coeff_[i] * (mod_a_amt_ - slewed_mod_[i])` then read indeterminate memory and NaN propagated through `SetDelay()` into every output sample: `effects.daisysp.pitch_shift(buf)` returned an all-NaN buffer. |
| `DaisySP/DaisySP-LGPL/Source/Effects/bitcrush.cpp` | `out *= (65536.0f / bits) - 32768;` split into `out *= (65536.0f / bits); out -= 32768;` | The encode half of `Process` is `out = in*65536; out += 32768; out *= bits/65536; out = floor(out)`, so the decode half has to undo it in the same two steps. As written it parses as a single multiply by `((65536/bits) - 32768)`, applying a gain of roughly `2^(bit_depth-1)` with an inverted sign. At the default 8 bits a 0.5-peak input came back at 126.5, and silence came back at a 66.5 DC offset. |
| `DaisySP/DaisySP-LGPL/Source/Effects/bitcrush.{cpp,h}` | file-scope `static Fold fold;` moved to a `Fold fold_;` member | One `Fold` was shared by every `Bitcrush` in the process, so two instances interfered with each other and concurrent `Process()` calls -- which the CLI's `-j` batch mode makes routine, since the bindings release the GIL -- raced on its state. |
| `stk/src/Noise.cpp` | `setSeed(0)` no longer calls `srand(time(NULL))` | Upstream reseeds from the wall clock for the default seed, which is every `Noise` inside every STK voice. That made STK renders irreproducible across wall-clock seconds and -- because `srand()` is process-global and DaisySP draws from the same `rand()` stream -- silently randomised unrelated DaisySP generators as a side effect of constructing an STK instrument. Callers wanting variation seed explicitly via `nanodsp._core.stk.set_random_seed()`, which the synthesis wrappers expose as a `seed` parameter. |
| `stk/src/PitShift.cpp` | window init loop `i <= size()` -> `i < size()` | **Heap buffer overflow.** The constructor wrote one `StkFloat` past the end of its 5000-element window buffer, corrupting the allocator's metadata. The process then trapped inside `malloc` at some later, unrelated allocation, so the crash appeared to come from whatever code happened to allocate next. |
| `stk/include/LentPitShift.h` | clamp `delay_` to `tMax_` before the final-period test | **Heap buffer overflow (read).** When the pitch-tracking loop completes without finding a minimum under the threshold, `delay_` ends at `tMax_+1`, one past the end of the `dt`/`cumDt`/`dpt` arrays; the test immediately after then read out of bounds. Input-dependent, so it fired only for some signals. |

Both overflows were found with AddressSanitizer (`make asan`) after an
intermittent SIGTRAP inside `malloc` during an ordinary test run. They are
upstream STK bugs, not binding bugs, and neither is detectable from Python --
the corruption is silent until an unrelated allocation trips over it.

Note that ASan does **not** cover the uninitialised-read family above
(`pitchshifter.h`, `chorus.cpp`, `synthbassdrum.cpp`, `drip.cpp`). Reading
indeterminate memory is not a memory-safety violation ASan instruments for;
MemorySanitizer (`-fsanitize=memory`) is the tool for it, and it requires an
instrumented standard library to be usable in practice. Until that exists here,
these bugs are caught by their symptoms -- which are themselves unreliable,
because whether they reproduce depends on what the allocator last left in the
reused memory. `effects.daisysp.pitch_shift` returned all-NaN in a fresh
interpreter but finite audio inside the test suite, purely because of what ran
first. The durable guard is a numeric fingerprint, not an assertion on
behaviour.

The DaisySP patches are one line each and could not be done through the
force-include shim in `cmake/daisysp_compat.h`, which can only add
declarations, not alter a function body.

The chorus/flanger/phaser fix has a longer write-up in
`docs/devs/daisysp-chorus-anomaly.md`: it records what the original symptom was,
what was ruled out, and -- importantly -- that the fix has *not* been shown to
be the cause of that symptom.

Regression coverage: `tests/test_stk_determinism.py` pins the seeding
behaviour, and `tests/GOLDEN.json` pins the numeric output of chorus, flanger,
phaser, bitcrush and pitch_shift (at transposition 0 and 7).
`tests/test_default_output.py` sweeps every registry entry callable with
default parameters and asserts finite, in-range output and silence-in
silence-out; that sweep catches the bitcrush defect on six counts but, for the
reason given above, cannot be relied on for the uninitialised-read family.

## Notes

- **choc**: Copyright 2025 Tracktion Corporation. No version tags or macros found in the vendored snapshot.
- **DaisySP**: CMakeLists.txt declares version 0.0.1, which appears to be a placeholder. Copyright 2020 Electrosmith, Corp.
- **DspFilters**: "A Collection of Useful C++ Classes for Digital Signal Processing" by Vinnie Falco (2009). No version numbering scheme.
- **fxdsp**: Headers reference original sources (e.g., `FX/Waveshaping.hpp`, `Reverbs/SchroederReverb.hpp`) but have been rewritten and cleaned for nanodsp. Licensed under MIT (see `fxdsp/LICENSE`).
- **GrainflowLib**: Copyright 2024 Christopher Poovey. Header-only granulation library. No version tags.
- **HISSTools_Library**: Copyright 2019 Alex Harker. No version numbering found.
- **madronalib**: Copyright 2025 Madrona Labs LLC. Version 0.1.0 per CMakeLists.txt variables.
- **signalsmith**: Copyright 2021 Geraint Luff / Signalsmith Audio Ltd. Version definitively 1.7.0 via `SIGNALSMITH_DSP_VERSION_STRING`.
- **signalsmith-stretch**: Copyright 2022 Geraint Luff / Signalsmith Audio Ltd. Vendored at release tag 1.1.1 (`version[3] = {1, 1, 1}` in the header), the last release that builds on the signalsmith DSP library; newer releases depend on the separate `signalsmith-linear` FFT library, which is not vendored here. The upstream `signalsmith-stretch.h` is unmodified -- its `dsp/...` includes resolve through forwarding shims in `signalsmith-stretch/dsp/` to the vendored signalsmith-dsp (1.7.0 >= the required 1.6.0).
- **stk**: Copyright 1995-2023 Perry R. Cook and Gary P. Scavone. Development snapshot from the 5.0.0 branch. Version strings are inconsistent across source files (`configure.ac` = 5.0.0, `STK.podspec` = 4.6.2, `CMakeLists.txt` = 4.6.1) because the vendored copy was taken between tagged releases. The `configure.ac` value (5.0.0) is authoritative as it is the autotools primary version source.
- **vafilters**: Faust-generated VA filter implementations by Eric Tarr / Christopher Arndt, cleaned for nanodsp. Includes PolyBLEP oscillator based on Kleimola et al. (SMC 2010).

## Not vendored

- **PaulStretch** (`src/nanodsp/_core_paulstretch.cpp`): the extreme time-stretch backend is an *original* implementation of the PaulStretch algorithm by Nasca Octavian Paul, which the author placed in the public domain. It is built on the vendored signalsmith RealFFT and does **not** include any source from the GPLv3 [paulxstretch](https://github.com/essej/paulxstretch) application -- that code is incompatible with this project's MIT license. Only the public-domain algorithm is reproduced.

  The same applies to the spectral modules added on top of it. The constant-Q spread and tonal/noise separation reproduce techniques *described* by the PaulXStretch sources, but the implementations here are written from scratch against the vendored signalsmith RealFFT, and both deviate deliberately where the original parameterization is poorly conditioned:

    - **Spread width** is specified in octaves and the one-pole coefficient is solved in closed form from the target kernel variance. PaulXStretch raises the coefficient to a power proportional to the bin count, which only behaves in the regime where that coefficient is already near 1 -- below roughly 0.8 on its 0--1 dial it produces no smoothing at all, and the dead zone widens as the FFT gets smaller.
    - **Tonal/noise** splits the spectrum once against a fixed threshold and blends toward one component. PaulXStretch scales the threshold with the parameter instead, which is degenerate at both extremes: the envelope is a smoothed copy of the same spectrum, so once the threshold passes a peak's own ratio to its neighbourhood the peaks are destroyed along with the noise and the output collapses to silence.

  If the GPLv3 sources are ever checked out locally for reference, keep them **outside** this repository. Vendoring them -- or linking any part of them into `_core` -- would make the extension a derivative work and end its MIT licensing. Note also that the core algorithm files (`Stretch.{h,cpp}`, `ProcessedStretch.{h,cpp}`) are GPL**v2**-only, and that they depend on JUCE, which is itself GPLv3-or-commercial.

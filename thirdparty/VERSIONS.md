# Third-Party Library Versions

All libraries are vendored directly into the `thirdparty/` directory (no git submodules).

| Library | Version / Ref | License | Upstream URL |
|---------|---------------|---------|--------------|
| choc | unversioned (2025 snapshot) | ISC | https://github.com/Tracktion/choc |
| DaisySP | 0.0.1 (CMakeLists.txt) | MIT | https://github.com/electro-smith/DaisySP |
| DspFilters | unversioned | MIT | https://github.com/vinniefalco/DSPFilters |
| fxdsp | unversioned (custom/cleaned) | unlicensed (no license file) | local / unknown origin |
| GrainflowLib | unversioned | MIT | https://github.com/composingcap/GrainflowLib |
| HISSTools_Library | unversioned | BSD 3-Clause | https://github.com/AlexHarker/HISSTools_Library |
| madronalib | 0.1.0 (CMakeLists.txt) | MIT | https://github.com/madronalabs/madronalib |
| signalsmith | 1.7.0 (header macro) | MIT | https://signalsmith-audio.co.uk/code/dsp/ |
| stk (STK) | 5.0.0 (configure.ac) / 4.6.2 (podspec) | MIT | https://github.com/thestk/stk |
| vafilters | unversioned (Faust-generated) | MIT-style STK-4.3 | local / derived from Faust DSP |

## Notes

- **choc**: Copyright 2025 Tracktion Corporation. No version tags or macros found in the vendored snapshot.
- **DaisySP**: CMakeLists.txt declares version 0.0.1, which appears to be a placeholder. Copyright 2020 Electrosmith, Corp.
- **DspFilters**: "A Collection of Useful C++ Classes for Digital Signal Processing" by Vinnie Falco (2009). No version numbering scheme.
- **fxdsp**: Headers reference original sources (e.g., `FX/Waveshaping.hpp`, `Reverbs/SchroederReverb.hpp`) but have been rewritten and cleaned for nanodsp. No license file is present.
- **GrainflowLib**: Copyright 2024 Christopher Poovey. Header-only granulation library. No version tags.
- **HISSTools_Library**: Copyright 2019 Alex Harker. No version numbering found.
- **madronalib**: Copyright 2025 Madrona Labs LLC. Version 0.1.0 per CMakeLists.txt variables.
- **signalsmith**: Copyright 2021 Geraint Luff / Signalsmith Audio Ltd. Version definitively 1.7.0 via `SIGNALSMITH_DSP_VERSION_STRING`.
- **stk**: Copyright 1995-2023 Perry R. Cook and Gary P. Scavone. Version varies by source file: `configure.ac` says 5.0.0, `STK.podspec` says 4.6.2, `CMakeLists.txt` says 4.6.1. Likely a development snapshot.
- **vafilters**: Faust-generated VA filter implementations by Eric Tarr / Christopher Arndt, cleaned for nanodsp. Includes PolyBLEP oscillator based on Kleimola et al. (SMC 2010).

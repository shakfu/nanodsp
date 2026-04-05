# TODO

- [ ] Tab completion (via argcomplete or shell scripts)

## High impact

- [x] Vocoder (channel vocoder) -- `effects.composed.vocoder()`
- [x] Sidechain compression -- `effects.dynamics.sidechain_compress()`
- [x] Transient shaper -- `effects.dynamics.transient_shape()`
- [x] True peak metering -- `analysis.true_peak_dbtp()`
- [x] Lookahead limiter -- `effects.dynamics.lookahead_limit()`

## Medium impact

- [ ] Convolution reverb -- high-level wrapper around `ops.convolve()` with mix, pre-delay, tail handling
- [ ] Linear-phase FIR EQ -- FIR-based EQ for mastering (preserves phase)
- [ ] MFCCs -- mel-frequency cepstral coefficients for speech/timbre analysis
- [ ] Wavetable oscillator -- user-supplied single-cycle waveform with interpolation
- [ ] Beat/tempo detection -- tempo estimation and beat tracking from onset function

## Lower impact

- [ ] Stereo correlation meter -- phase correlation between L/R channels
- [ ] Pitch correction (auto-tune) -- combine YIN pitch detection + PSOLA correction
- [ ] Additive synthesis -- harmonic series with per-partial control
- [ ] De-reverb -- spectral dereverberation
- [ ] Filtered feedback delay -- delay with LP/HP in the feedback path

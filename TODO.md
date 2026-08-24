# TODO

- [x] Tab completion (via argcomplete or shell scripts)

## High impact

- [ ] **MP3 / OGG / Opus decoding** -- the last unimplemented part of the I/O work. WAV now covers PCM 8/16/24/32-bit and IEEE float 32/64-bit (read and write, including `WAVE_FORMAT_EXTENSIBLE`), and FLAC 16/24-bit via CHOC, all with no external dependency. Lossy formats are the remaining gap, and the one most likely to stop someone using the library at all: a user whose material is MP3 has to convert it first, at which point they already have ffmpeg installed and the argument for nanodsp weakens.

  This fits the existing vendoring model. [`dr_libs`](https://github.com/mackron/dr_libs) (`dr_mp3`, public domain / MIT-0, single header) or [`miniaudio`](https://github.com/mackron/miniaudio) (same author, MIT-0 or public domain) drops into `thirdparty/` beside CHOC with no new Python dependency and no license friction. Both decode to interleaved float, which the existing `_decode_samples` path already handles.

  Scope notes for whoever picks this up:
  - Decode only. Encoding MP3 needs a separate encoder (LAME) with a different license; there is no good reason to write lossy formats from a DSP toolkit.

  - Route through `io.read()`'s extension table, so `nanodsp process song.mp3` works with no other change.

  - `read_blocks()` should get an MP3 path too, or explicitly refuse: the current implementation seeks by byte offset within a `data` chunk, which has no equivalent in a frame-based format.

  - Add the vendored library to `thirdparty/VERSIONS.md` and run `make asan` afterwards -- both STK heap overflows were found that way.

- [x] Vocoder (channel vocoder) -- `effects.composed.vocoder()`

- [x] Sidechain compression -- `effects.dynamics.sidechain_compress()`

- [x] Transient shaper -- `effects.dynamics.transient_shape()`

- [x] True peak metering -- `analysis.true_peak_dbtp()`

- [x] Lookahead limiter -- `effects.dynamics.lookahead_limit()`

## Medium impact

- [x] Convolution reverb -- `effects.reverb.convolution_reverb()`, with mix, pre-delay, tail handling and IR normalisation. Reachable from the CLI via the file-operand syntax: `-f convolution_reverb:ir=@church.wav,mix=0.4`

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

## Known issues

- [ ] **`daisysp.chorus` anomaly is unexplained.** An uninitialised read of `lfo_freq_` in the DaisySP modulation family was found and patched, but was never demonstrated to be the cause of the observed divergence that started the investigation. `chorus`, `flanger` and `phaser` are pinned in `tests/GOLDEN.json`; if one of those fingerprints ever moves, the hypothesis was wrong and the real cause is still open. That check is only now actually running: the fixture used to be stamped for a single platform and skipped everywhere else, including CI, so nothing had been verifying it. Full write-up, including what was ruled out and how to investigate a recurrence: [`docs/devs/daisysp-chorus-anomaly.md`](docs/devs/daisysp-chorus-anomaly.md).

- [ ] **Vendored patches must be re-applied on upgrade.** Eleven local fixes live in `thirdparty/` (two STK heap overflows, STK wall-clock seeding, six DaisySP uninitialised-state bugs, the DaisySP bitcrush gain/sign defect, and the shared `static Fold` in the same file). They are all marked in place -- `grep -rn "nanodsp local patch" thirdparty/` -- and tabulated in `thirdparty/VERSIONS.md`. Run `make asan` after any vendored upgrade.

  Note that `make asan` does not cover the uninitialised-read family; see the note under "Local patches" in `thirdparty/VERSIONS.md`. The golden fingerprints are what actually pin those.

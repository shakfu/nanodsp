# TODO

## Critical

## High

### High impact

- [ ] **MP3 / OGG / Opus decoding** -- the last unimplemented part of the I/O work. WAV now covers PCM 8/16/24/32-bit and IEEE float 32/64-bit (read and write, including `WAVE_FORMAT_EXTENSIBLE`), and FLAC 16/24-bit via CHOC, all with no external dependency. Lossy formats are the remaining gap, and the one most likely to stop someone using the library at all: a user whose material is MP3 has to convert it first, at which point they already have ffmpeg installed and the argument for nanodsp weakens.

  This fits the existing vendoring model. [`dr_libs`](https://github.com/mackron/dr_libs) (`dr_mp3`, public domain / MIT-0, single header) or [`miniaudio`](https://github.com/mackron/miniaudio) (same author, MIT-0 or public domain) drops into `thirdparty/` beside CHOC with no new Python dependency and no license friction. Both decode to interleaved float, which the existing `_decode_samples` path already handles.

  Scope notes for whoever picks this up:
  - Decode only. Encoding MP3 needs a separate encoder (LAME) with a different license; there is no good reason to write lossy formats from a DSP toolkit.

  - Route through `io.read()`'s extension table, so `nanodsp process song.mp3` works with no other change.

  - `read_blocks()` should get an MP3 path too, or explicitly refuse: the current implementation seeks by byte offset within a `data` chunk, which has no equivalent in a frame-based format.

  - Add the vendored library to `thirdparty/VERSIONS.md` and run `make asan` afterwards -- both STK heap overflows were found that way.

### Known issues

- [ ] **`tests/GOLDEN.json` darwin-arm64 block is stale for 17 cases.** The 0.3.x review fixes changed the output of `fm2`, `multiband_compress`, `master`, `daisysp.pitch_shift`, loudness, `onset_detect` and every STFT-based case. Only the Linux block was regenerated, so macOS CI fails on those cases until `python tests/test_golden.py --update` runs on a Mac.

## Medium

### Medium impact

- [ ] Linear-phase FIR EQ -- FIR-based EQ for mastering (preserves phase)

- [ ] MFCCs -- mel-frequency cepstral coefficients for speech/timbre analysis

- [ ] Wavetable oscillator -- user-supplied single-cycle waveform with interpolation

- [ ] Beat/tempo detection -- tempo estimation and beat tracking from onset function

### Known issues

- [ ] **`daisysp.chorus` anomaly is unexplained.** An uninitialised read of `lfo_freq_` in the DaisySP modulation family was found and patched, but was never demonstrated to be the cause of the observed divergence that started the investigation. `chorus`, `flanger` and `phaser` are pinned in `tests/GOLDEN.json`; if one of those fingerprints ever moves, the hypothesis was wrong and the real cause is still open. That check is only now actually running: the fixture used to be stamped for a single platform and skipped everywhere else, including CI, so nothing had been verifying it. Full write-up, including what was ruled out and how to investigate a recurrence: [`docs/devs/daisysp-chorus-anomaly.md`](docs/devs/daisysp-chorus-anomaly.md).

- [ ] **`istft` tapers the first ~6% of a window.** Where under 0.1% of full overlap remains (65 samples at window 1024), output is attenuated. The symmetric window also zeroes sample 0. Fix: centred frames with a periodic window. That moves every frame time, so the analysis functions and their golden cases change with it.

- [ ] **`eq_match` correction is noisy and biased on short input.** The per-bin correction is unsmoothed by default, and padded frames bias the source average: a 1000-frame input gets an RMS gain of 5.35 against a matched-noise reference (1.06 at 48000 frames). A smoothing default changes every caller's output.

- [ ] **`RealFFT` packs the Nyquist bin into `imag(bin 0)`.** `magnitude`, `phase`, `time_stretch` and `spectral_centroid` treat it as 0 Hz energy. Fixing it changes the `Spectrogram` data layout.

- [ ] **Release workflow decisions.**
  - `build-publish.yml` triggers on `v*` tags, but this repo tags `0.3.0`, so automatic publishing has never fired. Making the trigger match turns on automatic PyPI publishing.
  - Only `gh-action-pypi-publish` is SHA-pinned. `actions/checkout`, `upload-artifact`, `download-artifact` and `pypa/cibuildwheel` are still on tag refs.

## Low

### Lower impact

- [ ] Stereo correlation meter -- phase correlation between L/R channels

- [ ] Pitch correction (auto-tune) -- combine YIN pitch detection + PSOLA correction

- [ ] Additive synthesis -- harmonic series with per-partial control

- [ ] De-reverb -- spectral dereverberation

- [ ] Filtered feedback delay -- delay with LP/HP in the feedback path

### Known issues

- [ ] **Vendored patches must be re-applied on upgrade.** The local fixes in `thirdparty/` include two STK heap overflows, the STK `LentPitShift` delete mismatch, STK wall-clock seeding, six DaisySP uninitialised-state bugs, the DaisySP bitcrush gain/sign defect and shared `static Fold`, the DaisySP `ReverbSc` overflow, the `Fm2` tuning and `PitchShifter` ratio defects, and the CHOC FLAC end-of-stream hang. They are all marked in place -- `grep -rn "nanodsp local patch" thirdparty/` -- and tabulated in `thirdparty/VERSIONS.md`. Run `make asan` after any vendored upgrade.

  Note that `make asan` does not cover the uninitialised-read family; see the note under "Local patches" in `thirdparty/VERSIONS.md`. The golden fingerprints are what actually pin those.

- [ ] **CLI edge cases.**
  - `process --stream` on an empty input writes 48 kHz mono, ignoring the input format.
  - `--dry-run` reads `@file` operands despite its help text.
  - `process` overwrites an existing output without prompting.
  - The `-f` grammar cannot express paths containing commas, a literal leading `@`, or tuple parameters (`vocoder:freq_range`). With duplicate keys the last wins silently.
  - `convert` to more than two channels downmixes stereo to mono first, discarding the image.

- [ ] **WAV I/O conformance.**
  - Odd-sized data chunks get no RIFF pad byte, and the RIFF size omits it.
  - A truncated fmt chunk raises `struct.error` rather than `ValueError`.
  - `read_blocks` rejects data-before-fmt files that `read` accepts.
  - NaN written to PCM becomes 0 with a `RuntimeWarning`.

- [ ] **`AudioBuffer` accepts invalid metadata.** NaN/inf `sample_rate` and 0-channel arrays pass construction; `write_wav_bytes` then emits a channels=0 header that `read` rejects.

- [ ] **`stream.py` edge cases.** Overlap-add with the symmetric Hann window does not reconstruct the first samples, and `BlockProcessor.process`/`process_blocks` raise on a 0-frame buffer.

- [ ] **`paulstretch` output is never shorter than `window_size`.** 0 or 1 input frames, or `stretch=1e-9`, give 4096 frames; the docstring says about `frames * stretch`.

- [ ] **Binding input handling.**
  - `Biquad.process_inplace` silently does nothing on float64 or strided input (needs `noconvert()`).
  - `ArrayF` rejects read-only arrays (needs `ndarray<const float>`).
  - A negative `n` raises `MemoryError` in STK `.process(n)`, `CubicLfo` and DaisySP `Line`.
  - The `iirdesign` `Family`/`Type` enums are not accepted by the functions that take `int`.

- [ ] **Build warnings are suppressed.** `-w` on the vendored targets hides `-Wparentheses` and `-Wmaybe-uninitialized`, the classes behind the bitcrush and uninitialised-state bugs; a CI build with them enabled would catch recurrences. `__LITTLE_ENDIAN__` is hard-coded for STK.

- [ ] **Analysis details.**
  - The `resample` power-of-2 path has 2-3 samples of uncompensated latency, and `target_sr <= 0` is not validated.
  - `chromagram` index 0 is A (`tuning_hz`), not C; this is undocumented.
  - `spectral_flatness` emits a divide-by-zero `RuntimeWarning` on silence.

- [ ] **Effect details.**
  - `stereo_delay(left_ms=0)` delays by the ring length.
  - The docstring calls `saturate(mode="tape")` asymmetric, but `x - x^3/3` is odd-symmetric.
  - The `spectral_gate` `noise_floor_db` doc says both "attenuated to" and "relative to the threshold"; the code uses the relative meaning.
  - `psola_pitch_shift` should note that pure sines are a weak input and that it preserves formants.

- [ ] **Tuning of STK voices.** `plucked`/`sitar`/`stifkarp` are about 15 cents flat and `flute` about 18 cents sharp at 440 Hz. Likely upstream.

- [ ] **Weak test assertions.** `tests/test_envelopes.py:85` passes at full level, contradicting its comment. Eight `pytest.raises(Exception)` blocks in `test_madronalib.py` and `test_iirdesign.py` should name `ValueError`.

- [ ] **Packaging.**
  - `nanobind` is unpinned in `build-system.requires`.
  - The 35 MB sdist carries about 40 MB of unpacked vendored test data (`thirdparty/GrainflowLib/lib/AudioFile/tests/`).
  - macOS x86_64 wheels ship untested (`test-skip`).

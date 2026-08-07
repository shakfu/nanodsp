# The `daisysp.chorus` anomaly

**Status:** partially resolved. A real defect was found and fixed; whether it is
the defect that produced the original observation is unproven.

**Found:** 2026-08-07, during the 0.2.0 correctness work.
**Affects:** `effects.daisysp.chorus`, `effects.daisysp.flanger`,
`effects.daisysp.phaser`.
**Patched in:** `thirdparty/DaisySP/Source/Effects/{chorus,flanger,phaser}.cpp`
(see `thirdparty/VERSIONS.md`).

---

## Symptom

While building the golden-output regression corpus, `daisysp.chorus` was
observed returning a materially different result for identical input, twice
within the same process:

```
max absolute difference : 0.30
samples affected        : 99.7%
```

on material peaking around 0.5 -- a difference comparable to the signal itself,
not a rounding artifact.

It could not be reproduced on demand. Running the identical script twice, the
first run diverged and the second did not.

## What was ruled out

Each of these was tested and eliminated before the source was read:

| Hypothesis | Test | Result |
|---|---|---|
| `rand()` state | Explicit `srand(1)` / `srand(2)` via ctypes around the call | No effect on output |
| Input mutation | Compared the input buffer before/after, across six effects | No effect mutates its input |
| Allocation churn | 40 renders interleaved with varying-size numpy allocations | One distinct result |
| Signed zero only | Compared arrays elementwise rather than by hashed bytes | Values genuinely differed, not just `-0.0` vs `0.0` |
| Ordinary nondeterminism | 40 consecutive in-process renders | One distinct result |

The signature that remained -- rare, large, varying between processes rather
than within one -- points at reading memory that was never initialised.

## Mechanism found

Not by sanitizer; AddressSanitizer detects out-of-bounds access, not
uninitialised reads, and MSan is not readily available on macOS. It was found by
reading `ChorusEngine::Init`:

```c
void ChorusEngine::Init(float sample_rate)
{
    ...
    lfo_phase_ = 0.f;
    SetLfoFreq(.3f);      // <-- reads lfo_freq_, which nothing has written
    SetLfoDepth(.9f);
}

void ChorusEngine::SetLfoFreq(float freq)
{
    freq = 4.f * freq / sample_rate_;
    freq *= lfo_freq_ < 0.f ? -1.f : 1.f;   // "if we're headed down, keep going"
    lfo_freq_ = fclamp(freq, -.25f, .25f);
}
```

`lfo_freq_` is read to decide the LFO's direction before it has ever been
assigned. `Flanger::Init` and `PhaserEngine::Init` contain the identical
pattern.

### What a reversed LFO does to the audio

The LFO is a triangle that bounces between +/-1, flipping the sign of
`lfo_freq_` at each turn, and its output scales the delay-line length:

```c
del_.SetDelay(lfo_sig + delay_);
```

Starting negative makes the delay sweep *down* first rather than up -- roughly a
half-cycle phase offset in the modulation. Once latched it persists for the
object's whole life, because every later `SetLfoFreq` call preserves the
existing sign. No wrapper can correct it after construction.

`Chorus` holds **two** `ChorusEngine`s at different pan positions
(`SetPan(.25f, .75f)`), each reading its own uninitialised `lfo_freq_`. If the
two disagree in sign, the voices modulate in opposite directions, changing both
the interference pattern between them and the stereo image. The effect still
works and still sounds like a chorus; it is a different chorus.

### Why it was rare

Freshly mapped pages read as zero, and `0.0 < 0.0` is false, so a cold
allocation always took the correct forward path. The defect could only bite when
the object landed on recycled heap whose bytes happened to form a negative float
at that offset.

## The fix

One line per file, before the `SetLfoFreq` call:

```c
lfo_freq_ = 0.f;
```

### It does not change normal output

Verified directly: the chorus patch was reverted, the extension rebuilt, and 60
renders taken with and without it. Both produced the same fingerprint
(`5b1c7010...`). The patch makes every render take the path that virgin memory
already took -- it removes a rare wrong branch rather than altering the sound.

## What is still unproven

**The fix has not been shown to explain the original observation**, and one
piece of evidence argues against it.

An attempt to force the bad path deliberately -- filling freed heap blocks with
`-12345.0`, a negative float, then constructing `Chorus`, 60 trials -- failed to
reproduce the divergence *either with or without the patch*. The experiment
therefore had no discriminating power. The objects evidently do not land on the
poisoned blocks; nanobind allocates the holder through Python's own allocator,
not the general heap the test was poisoning.

Two separate claims, only one settled:

- **The uninitialised read is real and fixed.** Plain in the source, affects
  three shipped effects, needed no measurement to confirm.
- **That it caused the 0.30 divergence is a hypothesis.** Something else in
  `Chorus` may still be wrong.

## Related defects

Three sibling bugs of the same class were found in the same sweep, and unlike
this one, all three were confirmed empirically:

- `SyntheticBassDrum::Init` never initialises `transient_env_lp_`, which is only
  ever written through `fonepole()` -- a read-modify-write. Confirmed by a
  cold-versus-warm render comparison: the first render in a process differed
  from every later one.
- `Drip::Process` reads the members `inputs1_` / `inputs2_`, which are never
  assigned anywhere in the class, where it means the locals `inputs1` /
  `inputs2` -- a stray trailing underscore. Two of three resonator bands were
  driven by stale heap.
- Two STK heap buffer overflows (`PitShift`, `LentPitShift`), found with
  AddressSanitizer after an intermittent SIGTRAP inside `malloc`.

All are tabulated in `thirdparty/VERSIONS.md`.

## Safety net

`chorus`, `flanger` and `phaser` are pinned in `tests/GOLDEN.json`. If the
hypothesis is wrong and the real cause is still live, a moved fingerprint will
say so.

## If it recurs

1. **Do not regenerate the golden fixtures.** The moved fingerprint is the
   evidence; capture it first.
2. Record the failing fingerprint and whether it is stable within the process
   (`test_first_call_matches_later_calls` in `tests/test_stk_determinism.py`
   covers the cold/warm axis for the synthesis voices and is the pattern to
   copy).
3. Run `make asan`. It will not catch an uninitialised read, but it will catch
   an out-of-bounds write into the object, which is the other explanation for
   the observed magnitude.
4. For uninitialised reads specifically, a MemorySanitizer build on Linux
   (`-fsanitize=memory`, requires an instrumented libc++) is the tool that
   would settle it. Valgrind's memcheck on Linux is the lower-effort
   alternative.
5. The scan that found the sibling defects is worth re-running: compare each
   DaisySP class's private members against what its `Init` assigns, following
   `Set*()` calls. It reported 37 candidate classes, most of them false
   positives because members are set through setters called from `Init`.

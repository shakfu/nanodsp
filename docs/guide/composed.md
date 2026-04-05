# Composed Effects

Higher-level effects built by combining multiple primitives.

## Mastering chains

```python
from nanodsp.buffer import AudioBuffer
from nanodsp.effects import composed

buf = AudioBuffer.from_file("mix.wav")

# Full mastering: DC block -> EQ -> compress -> limit -> loudness normalize
mastered = composed.master(buf, target_lufs=-14.0)

# Genre-specific mastering presets
pop = composed.master_pop(buf)
hiphop = composed.master_hiphop(buf)
edm = composed.master_edm(buf)
classical = composed.master_classical(buf)
podcast = composed.master_podcast(buf)
```

## Vocal processing

```python
# Full vocal chain: de-ess -> EQ -> compress -> limit
vocals = composed.vocal_chain(buf, de_ess_freq=6000.0)

# With loudness target
vocals = composed.vocal_chain(buf, target_lufs=-16.0)
```

## Mixing tools

```python
# Add brightness above 3 kHz
bright = composed.exciter(buf, freq=3000.0, amount=0.3)

# Tame sibilance around 6 kHz
deessed = composed.de_esser(buf, freq=6000.0, threshold_db=-20.0, ratio=4.0)

# New York compression (heavy compress blended with dry)
punchy = composed.parallel_compress(buf, mix=0.5, ratio=8.0, threshold_db=-30.0)

# Multiband compression
multi = composed.multiband_compress(buf,
    crossover_freqs=[200.0, 2000.0, 8000.0],
    ratios=[2.0, 4.0, 3.0, 2.0],
)

# Vowel formant filtering
vowel = composed.formant_filter(buf, vowel="a")
```

## Delay effects

```python
# Stereo delay with independent L/R times
delayed = composed.stereo_delay(buf, left_ms=250.0, right_ms=375.0,
                                 feedback=0.3, mix=0.4)

# Ping-pong delay (signal bounces between L/R)
pp = composed.ping_pong_delay(buf, delay_ms=375.0, feedback=0.5, mix=0.5)
```

## Pitch and modulation

```python
# PSOLA pitch shift (time-domain, best for monophonic)
up_octave = composed.psola_pitch_shift(buf, semitones=12.0)
down_fifth = composed.psola_pitch_shift(buf, semitones=-7.0)

# Frequency shifter (Bode-style, shifts all frequencies by fixed Hz)
shifted = composed.freq_shift(buf, shift_hz=100.0)

# Ring modulator (produces sum and difference tones)
ring = composed.ring_mod(buf, carrier_freq=300.0)
ring_lfo = composed.ring_mod(buf, carrier_freq=300.0, lfo_freq=5.0, lfo_width=20.0)

# Auto-pan (LFO-driven stereo panning)
panned = composed.auto_pan(buf, rate=2.0, depth=1.0)
```

## Creative effects

```python
# Shimmer reverb -- reverb + octave-up pitch shift blended back
shimmer = composed.shimmer_reverb(buf, mix=0.4, shimmer=0.3, preset="cathedral")

# Tape echo -- darkening repeats with saturation
tape = composed.tape_echo(buf, delay_ms=300.0, feedback=0.5, tone=3000.0, drive=0.3)

# Lo-fi degradation (bitcrush + sample rate reduce + saturation)
lofi = composed.lo_fi(buf, bit_depth=6, reduce=0.5, tone=3000.0)

# Telephone simulation (300-3400 Hz bandpass + saturation)
phone = composed.telephone(buf)

# Gated reverb (80s production -- reverb + noise gate)
gated = composed.gated_reverb(buf, preset="plate", mix=0.5,
                               gate_threshold_db=-20.0, gate_hold_ms=30.0)
```

## Chaining composed effects

```python
from nanodsp.effects import filters, dynamics

# Full production chain using pipe()
result = (
    buf
    .pipe(composed.de_esser, freq=6000.0)
    .pipe(filters.highpass, cutoff_hz=80.0)
    .pipe(dynamics.compress, ratio=3.0, threshold=-20.0)
    .pipe(composed.shimmer_reverb, mix=0.2, shimmer=0.2)
    .pipe(dynamics.limit)
)
result.write("output.wav")
```

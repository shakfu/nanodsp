# Synthesis

Sound generators using DaisySP, STK, PolyBLEP, and fxdsp backends.

## Usage examples

### Basic oscillators

```python
from nanodsp import synthesis

# Sine wave at 440 Hz, 1 second
tone = synthesis.oscillator(frames=48000, freq=440.0, waveform="sine")

# Sawtooth with reduced amplitude
saw = synthesis.oscillator(frames=48000, freq=220.0, amp=0.5, waveform="saw")

# Square wave with pulse width
sq = synthesis.oscillator(frames=48000, freq=330.0, waveform="square", pw=0.3)

# Band-limited oscillator (DaisySP)
bl = synthesis.bl_oscillator(frames=48000, freq=440.0, waveform="saw")
```

### FM synthesis

```python
# Two-operator FM: carrier at 440 Hz, modulator at 2x carrier, mod index 1.5
fm = synthesis.fm2(frames=48000, freq=440.0, ratio=2.0, index=1.5)

# Formant oscillator
formant = synthesis.formant_oscillator(
    frames=48000, carrier_freq=440.0, formant_freq=1000.0
)
```

### Band-limited oscillators

```python
# PolyBLEP -- efficient, 14 waveforms
saw = synthesis.polyblep(frames=48000, freq=440.0, waveform="sawtooth")
sq = synthesis.polyblep(frames=48000, freq=440.0, waveform="square")

# BLIT -- configurable harmonics
blit = synthesis.blit_saw(frames=48000, freq=220.0, harmonics=20)
blit_sq = synthesis.blit_square(frames=48000, freq=220.0)

# DPW -- differentiated parabolic wave
dpw = synthesis.dpw_saw(frames=48000, freq=440.0)
pulse = synthesis.dpw_pulse(frames=48000, freq=440.0, duty=0.3)

# MinBLEP -- highest antialiasing quality
mb = synthesis.minblep(frames=48000, freq=440.0, waveform="saw")
mb_sq = synthesis.minblep(frames=48000, freq=440.0, waveform="square", pulse_width=0.3)
```

### Noise generators

```python
noise = synthesis.white_noise(frames=48000, amp=0.5)
clocked = synthesis.clocked_noise(frames=48000, freq=1000.0)
impulses = synthesis.dust(frames=48000, density=100.0)
```

### Drum synthesis

```python
kick = synthesis.analog_bass_drum(frames=48000, freq=60.0, decay=0.5, accent=0.8)
snare = synthesis.analog_snare_drum(frames=48000, freq=200.0, snappy=0.7)
hat = synthesis.hihat(frames=48000, freq=3000.0, decay=0.3, noisiness=0.8)

# Synthesis-focused variants with extra controls
syn_kick = synthesis.synthetic_bass_drum(
    frames=48000, freq=60.0, dirtiness=0.3, fm_env_amount=0.5
)
syn_snare = synthesis.synthetic_snare_drum(
    frames=48000, freq=200.0, fm_amount=0.3
)
```

### Physical modeling

```python
from nanodsp.buffer import AudioBuffer

# Karplus-Strong plucked string (excites a noise burst through a filtered delay)
excitation = AudioBuffer.noise(frames=48000, seed=42)
plucked = synthesis.karplus_strong(excitation, freq_hz=440.0, brightness=0.5)

# Simple pluck
p = synthesis.pluck(frames=48000, freq=440.0, decay=0.95, damp=0.9)

# Modal voice (resonant body)
modal = synthesis.modal_voice(frames=48000, freq=440.0, structure=0.5)

# Bowed string
bowed = synthesis.string_voice(frames=48000, freq=220.0, brightness=0.6)

# Water drop
drop = synthesis.drip(frames=48000, dettack=0.01)
```

### STK instruments

```python
# Single note
clarinet = synthesis.synth_note("clarinet", freq=440.0, duration=1.0, velocity=0.8)
flute = synthesis.synth_note("flute", freq=880.0, duration=0.5)

# Sequence of notes
melody = synthesis.synth_sequence("flute", notes=[
    (440.0, 0.0, 0.5),    # (freq, start_time, duration)
    (554.37, 0.5, 0.5),
    (659.26, 1.0, 1.0),
])
```

Available instruments: `clarinet`, `flute`, `brass`, `bowed`, `plucked`, `sitar`, `stifkarp`, `saxofony`, `recorder`, `blowbotl`, `blowhole`, `whistle`.

## API reference

::: nanodsp.synthesis
    options:
      show_if_no_docstring: false

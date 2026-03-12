# Reverb

FDN reverb (madronalib), classic Schroeder/Moorer reverbs (fxdsp), and STK reverbs/effects.

## Usage examples

### FDN reverb with presets

```python
from nanodsp.effects import reverb
from nanodsp.buffer import AudioBuffer

buf = AudioBuffer.from_file("input.wav")

# Hall reverb with 30% wet mix
hall = reverb.reverb(buf, preset="hall", mix=0.3)

# Room with short decay
room = reverb.reverb(buf, preset="room", decay=0.3, mix=0.2)

# Plate reverb (bright and dense)
plate = reverb.reverb(buf, preset="plate", mix=0.4)

# Cathedral with pre-delay
cathedral = reverb.reverb(
    buf, preset="cathedral", decay=0.9, damping=0.3, pre_delay_ms=40.0, mix=0.5
)

# Chamber (intimate)
chamber = reverb.reverb(buf, preset="chamber", mix=0.25)
```

Available presets: `room`, `hall`, `plate`, `chamber`, `cathedral`

### Classic reverbs

```python
# Schroeder reverb (4 combs + 2 allpasses)
schroeder = reverb.schroeder_reverb(buf, feedback=0.7, diffusion=0.5)

# With LFO modulation to reduce metallic ringing
mod_schroeder = reverb.schroeder_reverb(buf, feedback=0.7, mod_depth=0.1)

# Moorer reverb (adds early reflections network)
moorer = reverb.moorer_reverb(buf, feedback=0.7, diffusion=0.7, mod_depth=0.1)
```

### STK reverbs

```python
# Freeverb (Jezar's algorithm)
freeverb = reverb.stk_reverb(buf, algorithm="freeverb", room_size=0.7, mix=0.3)

# JCRev (John Chowning)
jcrev = reverb.stk_reverb(buf, algorithm="jcrev", t60=2.0, mix=0.3)

# NRev (CCRMA)
nrev = reverb.stk_reverb(buf, algorithm="nrev", t60=1.5, mix=0.4)

# PRCRev (Perry Cook)
prcrev = reverb.stk_reverb(buf, algorithm="prcrev", t60=1.0, mix=0.3)
```

### STK chorus and echo

```python
# Chorus effect
chorused = reverb.stk_chorus(buf, mod_depth=0.05, mod_freq=0.25, mix=0.5)

# Echo / delay
echoed = reverb.stk_echo(buf, delay_ms=250.0, mix=0.5)
```

## API reference

::: nanodsp.effects.reverb
    options:
      show_if_no_docstring: false

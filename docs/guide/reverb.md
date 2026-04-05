# Reverb

Reverb simulates the acoustic reflections of a physical space.

## FDN reverb (madronalib)

The primary reverb algorithm. Uses a matrix of 8 delay lines with feedback through a Hadamard mixing matrix and per-line damping. Five presets configure delay lengths and damping for different spaces.

```python
from nanodsp.buffer import AudioBuffer
from nanodsp.effects.reverb import (
    reverb, schroeder_reverb, moorer_reverb,
    stk_reverb, stk_chorus, stk_echo,
)

buf = AudioBuffer.from_file("input.wav")

# Presets: "room", "hall", "plate", "chamber", "cathedral"
room = reverb(buf, preset="room", mix=0.2, decay=0.6)
hall = reverb(buf, preset="hall", mix=0.3, decay=0.85)
plate = reverb(buf, preset="plate", mix=0.4, decay=0.7)
cathedral = reverb(buf, preset="cathedral", mix=0.5, decay=0.9, damping=0.7)

# With pre-delay (separates direct sound from reverb tail)
spacious = reverb(buf, preset="hall", mix=0.3, pre_delay_ms=30.0)
```

## Schroeder reverb

The classic reverb topology: 4 parallel feedback comb filters summed into 2 series allpass filters. Optional LFO modulation reduces metallic ringing.

```python
# Classic Schroeder
sch = schroeder_reverb(buf, feedback=0.7, diffusion=0.5)

# With LFO modulation for smoother tail
modulated = schroeder_reverb(buf, feedback=0.75, diffusion=0.6, mod_depth=0.1)
```

## Moorer reverb

Extends Schroeder with an 18-tap early reflections network before the combs. Separates early reflections (which convey room size) from the late diffuse tail.

```python
moorer = moorer_reverb(buf, feedback=0.7, diffusion=0.7, mod_depth=0.1)
```

## STK reverbs

Four reverb algorithms from the Synthesis ToolKit:

```python
# Freeverb -- Jezar's algorithm (8 combs + 4 allpasses)
free = stk_reverb(buf, algorithm="freeverb", mix=0.3, room_size=0.7, damping=0.5)

# John Chowning's reverb
jc = stk_reverb(buf, algorithm="jcrev", mix=0.3, t60=2.0)

# CCRMA NRev
nr = stk_reverb(buf, algorithm="nrev", mix=0.4, t60=1.5)

# Perry Cook's simple reverb
prc = stk_reverb(buf, algorithm="prcrev", mix=0.3, t60=1.0)
```

## STK chorus and echo

```python
# Stereo chorus
chorused = stk_chorus(buf, mod_depth=0.05, mod_freq=0.25, mix=0.5)

# Simple echo
echoed = stk_echo(buf, delay_ms=300.0, mix=0.4)
```

## ReverbSc (DaisySP)

Sean Costello's high-quality stereo reverb with lowpass damping.

```python
from nanodsp.effects.daisysp import reverb_sc

wet = reverb_sc(buf, feedback=0.85, lp_freq=8000.0)
```

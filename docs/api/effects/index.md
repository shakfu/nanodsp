# Effects

The `nanodsp.effects` package contains 70+ DSP functions organized into submodules.

Import from the specific submodule you need:

```python
from nanodsp.effects.filters import lowpass, highpass
from nanodsp.effects.dynamics import compress, limit
from nanodsp.effects.reverb import reverb, stk_reverb
from nanodsp.effects.daisysp import chorus, flanger
from nanodsp.effects.saturation import saturate, aa_hard_clip
from nanodsp.effects.composed import master, vocal_chain
```

| Submodule | Description |
|-----------|-------------|
| [`filters`](filters.md) | Signalsmith biquads, DaisySP SVF/ladder/moog/tone/modal/comb, virtual analog, IIR |
| [`daisysp`](daisysp.md) | Autowah, chorus, decimator, flanger, overdrive, phaser, pitch shift, tremolo, wavefold, bitcrush, reverb_sc |
| [`dynamics`](dynamics.md) | Compressor, limiter, noise gate, AGC |
| [`saturation`](saturation.md) | Soft/hard/tape saturation, antialiased waveshaping |
| [`reverb`](reverb.md) | FDN reverb, Schroeder, Moorer, STK reverbs, STK chorus/echo |
| [`composed`](composed.md) | Exciter, de-esser, parallel compress, stereo delay, multiband compress, mastering, vocal chain |

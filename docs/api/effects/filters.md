# Filters

Signalsmith biquad filters, DaisySP SVF/ladder/moog/tone/modal/comb, virtual analog (Faust), and multi-order IIR filters (DspFilters).

## Usage examples

### Biquad filters (signalsmith)

```python
from nanodsp.effects import filters
from nanodsp.buffer import AudioBuffer

buf = AudioBuffer.from_file("input.wav")

# Basic filters
lp = filters.lowpass(buf, cutoff_hz=1000.0)
hp = filters.highpass(buf, cutoff_hz=80.0)
bp = filters.bandpass(buf, center_hz=1000.0)
nt = filters.notch(buf, center_hz=50.0)         # remove 50 Hz hum

# EQ: boost 3 kHz by 6 dB with 1-octave bandwidth
eq = filters.peak_db(buf, center_hz=3000.0, db=6.0, octaves=1.0)

# Shelving filters
bright = filters.high_shelf_db(buf, cutoff_hz=8000.0, db=3.0)
warm = filters.low_shelf_db(buf, cutoff_hz=200.0, db=2.0)

# Allpass (phase shift without magnitude change)
ap = filters.allpass(buf, freq_hz=1000.0)

# Choose a filter design method
lp_vicanek = filters.lowpass(buf, cutoff_hz=5000.0, design="vicanek")
lp_cookbook = filters.lowpass(buf, cutoff_hz=5000.0, design="cookbook")
```

### DaisySP filters

```python
# State variable filter with resonance
lp = filters.svf_lowpass(buf, freq_hz=1000.0, resonance=0.5)
hp = filters.svf_highpass(buf, freq_hz=200.0, resonance=0.3)
bp = filters.svf_bandpass(buf, freq_hz=1000.0, resonance=0.7)
nt = filters.svf_notch(buf, freq_hz=1000.0)
pk = filters.svf_peak(buf, freq_hz=1000.0, resonance=0.8)

# 4-pole ladder filter (Moog-inspired, multiple modes)
lp24 = filters.ladder_filter(buf, freq_hz=800.0, resonance=0.6, mode="lp24")
bp12 = filters.ladder_filter(buf, freq_hz=1200.0, mode="bp12")
hp24 = filters.ladder_filter(buf, freq_hz=200.0, mode="hp24")

# Moog ladder (simplified: cutoff + resonance only)
moog = filters.moog_ladder(buf, freq_hz=1000.0, resonance=0.7)

# Simple 1-pole filters
gentle_lp = filters.tone_lowpass(buf, freq_hz=2000.0)
gentle_hp = filters.tone_highpass(buf, freq_hz=100.0)

# Resonant bandpass for modal synthesis
modal = filters.modal_bandpass(buf, freq_hz=440.0, q=50.0)

# Comb filter
comb = filters.comb_filter(buf, freq_hz=500.0, rev_time=0.5)
```

### Virtual analog filters (Faust)

```python
# Moog transistor ladder -- warm, classic
va_moog = filters.va_moog_ladder(buf, cutoff_hz=1000.0, q=2.0)

# Diode ladder -- acidic, TB-303 style
va_diode = filters.va_diode_ladder(buf, cutoff_hz=800.0, q=3.0)

# Korg MS-20 filters
va_korg_lp = filters.va_korg35_lpf(buf, cutoff_hz=1500.0, q=2.0)
va_korg_hp = filters.va_korg35_hpf(buf, cutoff_hz=200.0, q=1.5)

# Oberheim SEM multi-mode
va_ob_lp = filters.va_oberheim(buf, cutoff_hz=2000.0, q=1.0, mode="lpf")
va_ob_hp = filters.va_oberheim(buf, cutoff_hz=500.0, mode="hpf")
va_ob_bp = filters.va_oberheim(buf, cutoff_hz=1000.0, mode="bpf")
va_ob_bs = filters.va_oberheim(buf, cutoff_hz=1000.0, mode="bsf")
```

### Multi-order IIR filters (DspFilters)

```python
# Butterworth lowpass, 8th order
steep_lp = filters.iir_filter(
    buf, family="butterworth", filter_type="lowpass", order=8, freq=1000.0
)

# Chebyshev Type I highpass, 6th order with 1 dB ripple
cheby = filters.iir_filter(
    buf, family="chebyshev1", filter_type="highpass", order=6,
    freq=200.0, ripple_db=1.0
)

# Elliptic bandpass, 4th order
ellip = filters.iir_filter(
    buf, family="elliptic", filter_type="bandpass", order=4,
    freq=1000.0, width=500.0, ripple_db=1.0, rolloff=40.0
)

# Bessel lowpass (best phase linearity)
bessel = filters.iir_filter(
    buf, family="bessel", filter_type="lowpass", order=8, freq=5000.0
)

# Get SOS coefficients without applying
import numpy as np
sos = filters.iir_design(
    "butterworth", "lowpass", order=4, sample_rate=44100, freq=1000.0
)
```

## API reference

::: nanodsp.effects.filters
    options:
      show_if_no_docstring: false

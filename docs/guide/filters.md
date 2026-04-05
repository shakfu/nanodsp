# Filters

Filters selectively pass or reject frequencies. nanodsp provides four filter families at different abstraction levels.

## Biquad filters (signalsmith)

The biquad is a second-order IIR filter -- two poles and two zeros give enough flexibility for lowpass, highpass, bandpass, notch, peak EQ, and shelving responses with minimal computation.

Four design methods trade off different characteristics:

| Design | Best for |
|--------|----------|
| `"bilinear"` | General-purpose LP/HP, matches analog response |
| `"cookbook"` | Classic EQ curves (Robert Bristow-Johnson) |
| `"one_sided"` | Bandpass, notch, peak, allpass |
| `"vicanek"` | Accurate magnitude at high frequencies |

```python
from nanodsp.buffer import AudioBuffer
from nanodsp.effects.filters import (
    lowpass, highpass, bandpass, notch,
    peak_db, high_shelf_db, low_shelf_db, allpass,
)

buf = AudioBuffer.from_file("input.wav")

# Basic filtering
lp = lowpass(buf, cutoff_hz=2000.0)
hp = highpass(buf, cutoff_hz=80.0)
bp = bandpass(buf, cutoff_hz=1000.0, octaves=1.0)
nch = notch(buf, cutoff_hz=60.0)  # remove hum

# EQ: boost 3 kHz by 4 dB, cut lows, add air
eq = (
    buf
    .pipe(peak_db, cutoff_hz=3000.0, db=4.0)
    .pipe(low_shelf_db, cutoff_hz=200.0, db=-2.0)
    .pipe(high_shelf_db, cutoff_hz=10000.0, db=2.0)
)

# Choose a design method
lp_vicanek = lowpass(buf, cutoff_hz=5000.0, design="vicanek")

# Narrow Q via octave bandwidth
narrow_peak = peak_db(buf, cutoff_hz=1000.0, db=6.0, octaves=0.5)
```

## DaisySP filters

Several filter topologies beyond the basic biquad:

```python
from nanodsp.effects.filters import (
    svf_lowpass, svf_highpass, svf_bandpass, svf_notch, svf_peak,
    ladder_filter, moog_ladder,
    tone_lowpass, tone_highpass,
    modal_bandpass, comb_filter,
)

buf = AudioBuffer.from_file("input.wav")

# State Variable Filter -- resonant, with optional drive
svf = svf_lowpass(buf, freq_hz=1000.0, resonance=0.5, drive=0.3)
svf_hp = svf_highpass(buf, freq_hz=500.0, resonance=0.7)

# Ladder filter -- 24 dB/oct with mode selection
lad = ladder_filter(buf, freq_hz=800.0, resonance=0.6, mode="lp24")
lad_bp = ladder_filter(buf, freq_hz=1200.0, mode="bp24", drive=2.0)

# Moog ladder -- simpler interface
moog = moog_ladder(buf, freq_hz=600.0, resonance=0.8)

# Tone filters -- cheap 1-pole LP/HP
gentle_lp = tone_lowpass(buf, freq_hz=3000.0)
dc_block = tone_highpass(buf, freq_hz=20.0)

# Modal bandpass -- high-Q resonator for resonant bodies
resonance = modal_bandpass(buf, freq_hz=440.0, q=500.0)

# Comb filter -- evenly-spaced peaks/notches
comb = comb_filter(buf, freq_hz=200.0, rev_time=0.3)
```

## Virtual analog filters (Faust)

Zero-delay feedback filter models derived from analog circuit schematics. These accurately model the nonlinear behavior of classic synthesizer filters.

```python
from nanodsp.effects.filters import (
    va_moog_ladder, va_moog_half_ladder, va_diode_ladder,
    va_korg35_lpf, va_korg35_hpf, va_oberheim,
)

buf = AudioBuffer.from_file("input.wav")

# Classic Moog -- warm, fat
moog = va_moog_ladder(buf, cutoff_hz=800.0, q=3.0)

# Half ladder -- gentler 12 dB/oct rolloff
half = va_moog_half_ladder(buf, cutoff_hz=1200.0, q=2.0)

# Diode ladder -- acidic TB-303 character
acid = va_diode_ladder(buf, cutoff_hz=600.0, q=5.0)

# Korg MS-20 -- aggressive, screamy at high resonance
korg_lp = va_korg35_lpf(buf, cutoff_hz=1000.0, q=8.0)
korg_hp = va_korg35_hpf(buf, cutoff_hz=500.0, q=4.0)

# Oberheim SEM -- clean, precise, multi-mode
ob_lp = va_oberheim(buf, cutoff_hz=2000.0, q=2.0, mode="lpf")
ob_bp = va_oberheim(buf, cutoff_hz=1500.0, q=3.0, mode="bpf")
ob_notch = va_oberheim(buf, cutoff_hz=1000.0, q=1.5, mode="bsf")
```

## Multi-order IIR filters (DspFilters)

For sharper transitions than a biquad can provide. Orders 1--16, implemented as cascaded second-order sections (SOS).

| Family | Character |
|--------|-----------|
| Butterworth | Maximally flat passband, no ripple |
| Chebyshev I | Passband ripple, sharper transition |
| Chebyshev II | Flat passband, stopband ripple |
| Elliptic | Sharpest transition, ripple in both bands |
| Bessel | Maximally flat group delay (best phase linearity) |

```python
from nanodsp.effects.filters import iir_filter, iir_design

buf = AudioBuffer.from_file("input.wav")

# 8th-order Butterworth lowpass at 2 kHz
butter = iir_filter(buf, family="butterworth", filter_type="lowpass",
                    order=8, freq_hz=2000.0)

# 4th-order Chebyshev Type I highpass with 1 dB ripple
cheby = iir_filter(buf, family="chebyshev1", filter_type="highpass",
                   order=4, freq_hz=200.0, ripple_db=1.0)

# 6th-order Elliptic bandpass
ellip = iir_filter(buf, family="elliptic", filter_type="bandpass",
                   order=6, freq_hz=1000.0, bandwidth_hz=500.0, ripple_db=0.5)

# Get raw SOS coefficients for external use
sos = iir_design(family="butterworth", filter_type="lowpass",
                 order=4, freq_hz=1000.0, sample_rate=48000.0)
# sos shape: [n_sections, 6] -- each row is [b0, b1, b2, a0, a1, a2]
```

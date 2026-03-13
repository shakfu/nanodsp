// Ring modulator with carrier oscillator and optional LFO frequency modulation
// Based on: FX/AudioEffectRingMod.hpp
// Rewritten as clean, self-contained header for nanodsp.
#pragma once

#include <cmath>
#include <algorithm>

namespace fxdsp {

class RingMod {
    // Ring modulator: multiplies input by a sine-wave carrier oscillator.
    // An optional LFO modulates the carrier frequency for vibrato/wobble.
    //
    //   carrier(t) = sin(2*pi * (carrier_freq + lfo_width * lfo(t)) * t)
    //   output = input * carrier
public:
    RingMod() = default;

    void init(float sample_rate) {
        sr_ = sample_rate;
        inv_sr_ = 1.0f / sample_rate;
        carrier_phase_ = 0.0f;
        lfo_phase_ = 0.0f;
    }

    void reset() {
        carrier_phase_ = 0.0f;
        lfo_phase_ = 0.0f;
    }

    // ---- Parameters ----

    void set_carrier_freq(float hz) { carrier_freq_ = hz; }
    float get_carrier_freq() const { return carrier_freq_; }

    void set_lfo_freq(float hz) { lfo_freq_ = hz; }
    float get_lfo_freq() const { return lfo_freq_; }

    void set_lfo_width(float hz) { lfo_width_ = hz; }
    float get_lfo_width() const { return lfo_width_; }

    void set_mix(float m) { mix_ = std::max(0.0f, std::min(1.0f, m)); }
    float get_mix() const { return mix_; }

    // ---- Processing ----

    float tick(float x) {
        // Compute carrier with FM from LFO
        float carrier = std::sin(2.0f * pi_ * carrier_phase_);
        float wet = x * carrier;

        // Update LFO
        float lfo = std::sin(2.0f * pi_ * lfo_phase_);
        float delta_carrier = carrier_freq_ + lfo_width_ * lfo;

        // Advance phases
        carrier_phase_ += delta_carrier * inv_sr_;
        // Wrap to avoid precision loss
        while (carrier_phase_ >= 1.0f) carrier_phase_ -= 1.0f;
        while (carrier_phase_ < 0.0f) carrier_phase_ += 1.0f;

        lfo_phase_ += lfo_freq_ * inv_sr_;
        while (lfo_phase_ >= 1.0f) lfo_phase_ -= 1.0f;

        return (1.0f - mix_) * x + mix_ * wet;
    }

    void process(const float* in, float* out, unsigned count) {
        for (unsigned i = 0; i < count; ++i)
            out[i] = tick(in[i]);
    }

private:
    float sr_ = 44100.0f;
    float inv_sr_ = 1.0f / 44100.0f;
    float carrier_freq_ = 440.0f;
    float lfo_freq_ = 0.0f;
    float lfo_width_ = 0.0f;
    float mix_ = 1.0f;
    float carrier_phase_ = 0.0f;
    float lfo_phase_ = 0.0f;

    static constexpr float pi_ = 3.14159265358979f;
};

}  // namespace fxdsp

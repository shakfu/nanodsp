// Bode-style frequency shifter using allpass Hilbert transform approximation
// Based on: FX/BodeShifter.hpp (rewritten from scratch -- original had bugs)
// Uses 4-stage allpass pairs for wideband 90-degree phase split, then
// single-sideband modulation via quadrature oscillator.
#pragma once

#include <cmath>

namespace fxdsp {

class FreqShifter {
    // Frequency shifter using Hilbert transform (allpass network) and
    // quadrature oscillator.  Shift can be positive (up) or negative (down).
    //
    // Algorithm:
    //   1. Split input into I/Q via two allpass chains with ~90-degree
    //      phase difference (Hilbert approximation).
    //   2. Multiply by quadrature oscillator at shift frequency.
    //   3. Combine: out = I*cos(wt) - Q*sin(wt)  (single-sideband)
public:
    FreqShifter() = default;

    void init(float sample_rate) {
        sr_ = sample_rate;
        osc_phase_ = 0.0f;
        // Reset allpass states
        for (int i = 0; i < num_stages_; ++i) {
            state_a_[i] = state_b_[i] = 0.0f;
        }
    }

    void reset() {
        osc_phase_ = 0.0f;
        for (int i = 0; i < num_stages_; ++i) {
            state_a_[i] = state_b_[i] = 0.0f;
        }
    }

    void set_shift_hz(float hz) { shift_hz_ = hz; }
    float get_shift_hz() const { return shift_hz_; }

    float tick(float x) {
        // Hilbert transform via two allpass chains
        float a_out = x;
        for (int i = 0; i < num_stages_; ++i) {
            float tmp = coeff_a_[i] * (a_out - state_a_[i]) + state_a_[i];
            // Wait -- this is a first-order allpass: y = a*(x - y_prev) + x_prev
            // Correct form: y[n] = a * x[n] + x[n-1] - a * y[n-1]
            // Using transposed direct form:
            float y = coeff_a_[i] * a_out + state_a_[i];
            state_a_[i] = a_out - coeff_a_[i] * y;
            a_out = y;
        }

        float b_out = x;
        for (int i = 0; i < num_stages_; ++i) {
            float y = coeff_b_[i] * b_out + state_b_[i];
            state_b_[i] = b_out - coeff_b_[i] * y;
            b_out = y;
        }

        // Quadrature oscillator
        float phase_inc = 2.0f * pi_ * shift_hz_ / sr_;
        float cos_w = std::cos(osc_phase_);
        float sin_w = std::sin(osc_phase_);
        osc_phase_ += phase_inc;
        // Keep phase in [0, 2*pi) to avoid float precision loss
        if (osc_phase_ > 2.0f * pi_) osc_phase_ -= 2.0f * pi_;
        if (osc_phase_ < 0.0f) osc_phase_ += 2.0f * pi_;

        // Single-sideband modulation
        // a_out ~ in-phase, b_out ~ quadrature (90 degrees shifted)
        return a_out * cos_w - b_out * sin_w;
    }

    void process(const float* in, float* out, unsigned count) {
        for (unsigned i = 0; i < count; ++i)
            out[i] = tick(in[i]);
    }

private:
    float sr_ = 44100.0f;
    float shift_hz_ = 0.0f;
    float osc_phase_ = 0.0f;

    // Allpass coefficients for Hilbert transform approximation.
    // These give ~90-degree phase difference from ~15 Hz to ~20 kHz at 44.1 kHz.
    // From Wardle (1998) / Haynsworth and Smith.
    static constexpr int num_stages_ = 4;
    static constexpr float coeff_a_[4] = {
        0.6923878f, 0.9360654322959f, 0.9882295226860f, 0.9987488452737f
    };
    static constexpr float coeff_b_[4] = {
        0.4021921162426f, 0.8561710882420f, 0.9722909545651f, 0.9952884791278f
    };
    float state_a_[4] = {};
    float state_b_[4] = {};

    static constexpr float pi_ = 3.14159265358979f;
};

}  // namespace fxdsp

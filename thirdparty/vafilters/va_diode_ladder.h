// Diode Ladder 24 dB/oct lowpass filter
// Generated from Faust DSP (diodeladder.dsp) by Eric Tarr / Christopher Arndt
// License: MIT-style STK-4.3 license
// Cleaned up for nanodsp integration: no external dependencies
#pragma once
#include <cmath>
#include <algorithm>

namespace vafilters {

class DiodeLadder {
public:
    DiodeLadder() { reset(); }

    void init(float sample_rate) {
        fConst0_ = 3.14159274f / std::min(192000.0f, std::max(1.0f, sample_rate));
        reset();
    }

    void reset() {
        cutoff_ = 20000.0f;
        q_ = 1.0f;
        for (auto &v : fRec5_) v = cutoff_;
        for (auto &v : fRec1_) v = 0.0f;
        for (auto &v : fRec2_) v = 0.0f;
        for (auto &v : fRec3_) v = 0.0f;
        for (auto &v : fRec4_) v = 0.0f;
    }

    void set_cutoff(float hz) { cutoff_ = hz; fRec5_[0] = fRec5_[1] = hz; }
    float get_cutoff() const { return cutoff_; }
    void set_q(float q) { q_ = q; }
    float get_q() const { return q_; }

    void process(const float *in, float *out, unsigned count) {
        float fSlow0 = 0.00100000005f * cutoff_;
        float fSlow1 = q_ - 0.707000017f;
        float fSlow2 = 0.00514551532f * fSlow1;
        for (unsigned i = 0; i < count; ++i) {
            fRec5_[0] = fSlow0 + 0.999000013f * fRec5_[1];
            float t0 = std::tan(fConst0_ * fRec5_[0]);
            float inp = std::max(-1.0f, std::min(1.0f, 100.0f * in[i]));
            float t2 = 17.0f - 9.69999981f * pow10f(0.0f - 0.333333343f * (1.0f - (std::log10(fRec5_[0]) + -0.30103001f)));
            float t3 = t0 + 1.0f;
            float t0_2 = t0 * t0;
            float t0_3 = t0_2 * t0;
            float t0_4 = t0_3 * t0;
            float t4 = (0.5f * ((fRec1_[1] * t0) / t3)) + fRec2_[1];
            float t5 = (t0 * (1.0f - 0.25f * (t0 / t3))) + 1.0f;
            float t6 = (t0 * t4) / t5;
            float t7 = 0.5f * t6;
            float t8 = t7 + fRec3_[1];
            float t9 = (t0 * (1.0f - 0.25f * (t0 / t5))) + 1.0f;
            float t10 = (t0 * t8) / t9;
            float t11 = t10 + fRec4_[1];
            float t12 = t5 * t9;
            float t13 = (t0 * (1.0f - 0.5f * (t0 / t9))) + 1.0f;
            float t15 = t3 * t5;
            // Main feedback computation (matches Faust diodeladder.dsp compute kernel)
            float soft_in = 1.5f * (inp * (1.0f - 0.333333343f * inp * inp));
            float fb_inner = (0.0f - (0.0205820613f * t6 + 0.0411641225f * fRec1_[1]))
                             - 0.0205820613f * t10
                             - 0.00514551532f * ((t0_3 * t11) / (t12 * t13));
            float fb_scaled = (fSlow1 * ((t2 * fb_inner))) / t3;
            float stage_a = (soft_in + fb_scaled) * (0.5f * (t0_2 / (t9 * t13)) + 1.0f);
            float denom = fSlow2 * ((t0_4 * t2) / ((t15 * t9) * t13)) + 1.0f;
            float stage_b = stage_a / denom;
            float pass_through = (t8 + 0.5f * ((t0 * t11) / t13)) / t9;
            float t16 = (t0 * ((stage_b + pass_through) - fRec4_[1])) / t3;
            float t17 = (t0 * ((0.5f * (((fRec4_[1] + t16) * (0.25f * (t0_2 / t12) + 1.0f)) + ((t4 + 0.5f * t10) / t5))) - fRec3_[1])) / t3;
            float t18 = (t0 * ((0.5f * (((fRec3_[1] + t17) * (0.25f * (t0_2 / t15) + 1.0f)) + ((fRec1_[1] + t7) / t3))) - fRec2_[1])) / t3;
            float t19 = (t0 * (0.5f * (fRec2_[1] + t18) - fRec1_[1])) / t3;
            out[i] = fRec1_[1] + t19;
            fRec1_[0] = fRec1_[1] + 2.0f * t19;
            fRec2_[0] = fRec2_[1] + 2.0f * t18;
            fRec3_[0] = fRec3_[1] + 2.0f * t17;
            fRec4_[0] = fRec4_[1] + 2.0f * t16;
            fRec5_[1] = fRec5_[0];
            fRec1_[1] = fRec1_[0];
            fRec2_[1] = fRec2_[0];
            fRec3_[1] = fRec3_[0];
            fRec4_[1] = fRec4_[0];
        }
    }

private:
    static float pow10f(float x) {
        // x^10 = ((x^2)^2 * x)^2
        float x2 = x * x;
        float x4 = x2 * x2;
        float x5 = x4 * x;
        return x5 * x5;
    }

    float fConst0_ = 3.14159274f / 44100.0f;
    float cutoff_, q_;
    float fRec5_[2], fRec1_[2], fRec2_[2], fRec3_[2], fRec4_[2];
};

} // namespace vafilters

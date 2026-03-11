// Moog Ladder 24 dB/oct lowpass filter
// Generated from Faust DSP (moogladder.dsp) by Eric Tarr / Christopher Arndt
// License: MIT-style STK-4.3 license
// Cleaned up for nanodsp integration: no external dependencies
#pragma once
#include <cmath>
#include <algorithm>

namespace vafilters {

class MoogLadder {
public:
    MoogLadder() { reset(); }

    void init(float sample_rate) {
        fConst0_ = 3.14159274f / std::min(192000.0f, std::max(1.0f, sample_rate));
        reset();
    }

    void reset() {
        cutoff_ = 20000.0f;
        q_ = 1.0f;
        // Seed smoothing register with cutoff to avoid NaN from log10(~0)
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
        float fSlow1 = 0.0411641225f * (q_ - 0.707000017f);
        for (unsigned i = 0; i < count; ++i) {
            fRec5_[0] = fSlow0 + 0.999000013f * fRec5_[1];
            float t0 = std::tan(fConst0_ * fRec5_[0]);
            float t1 = 3.9000001f - 0.899999976f * std::pow(
                0.0f - 0.333333343f * (1.0f - (std::log10(fRec5_[0]) + -0.30103001f)), 0.200000003f);
            float t2 = t0 + 1.0f;
            float t0_2 = t0 * t0;
            float t0_3 = t0_2 * t0;
            float t0_4 = t0_3 * t0;
            float t3 = (t0 * ((((in[i] - fSlow1 * (t1 * ((fRec1_[1] + fRec2_[1] * t0) + t0_2 * fRec3_[1] + t0_3 * fRec4_[1])))
                / (fSlow1 * (t1 * t0_4) + 1.0f)) - fRec4_[1]))) / t2;
            float t4 = (t0 * (fRec4_[1] + t3 - fRec3_[1])) / t2;
            float t5 = (t0 * (fRec3_[1] + t4 - fRec2_[1])) / t2;
            float t6 = (t0 * (fRec2_[1] + t5 - fRec1_[1])) / t2;
            out[i] = fRec1_[1] + t6;
            fRec1_[0] = fRec1_[1] + 2.0f * t6;
            fRec2_[0] = fRec2_[1] + 2.0f * t5;
            fRec3_[0] = fRec3_[1] + 2.0f * t4;
            fRec4_[0] = fRec4_[1] + 2.0f * t3;
            fRec5_[1] = fRec5_[0];
            fRec1_[1] = fRec1_[0];
            fRec2_[1] = fRec2_[0];
            fRec3_[1] = fRec3_[0];
            fRec4_[1] = fRec4_[0];
        }
    }

private:
    float fConst0_ = 3.14159274f / 44100.0f;
    float cutoff_, q_;
    float fRec5_[2], fRec1_[2], fRec2_[2], fRec3_[2], fRec4_[2];
};

} // namespace vafilters

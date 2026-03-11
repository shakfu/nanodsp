// Korg 35 24 dB/oct highpass filter
// Generated from Faust DSP (korg35hpf.dsp) by Eric Tarr
// License: MIT-style STK-4.3 license
// Cleaned up for nanodsp integration: no external dependencies
#pragma once
#include <cmath>
#include <algorithm>

namespace vafilters {

class Korg35HPF {
public:
    Korg35HPF() { reset(); }

    void init(float sample_rate) {
        fConst0_ = 3.14159274f / std::min(192000.0f, std::max(1.0f, sample_rate));
        reset();
    }

    void reset() {
        cutoff_ = 20000.0f;
        q_ = 1.0f;
        for (auto &v : fRec4_) v = 0.0f;
        for (auto &v : fRec1_) v = 0.0f;
        for (auto &v : fRec2_) v = 0.0f;
        for (auto &v : fRec3_) v = 0.0f;
    }

    void set_cutoff(float hz) { cutoff_ = hz; }
    float get_cutoff() const { return cutoff_; }
    void set_q(float q) { q_ = q; }
    float get_q() const { return q_; }

    void process(const float *in, float *out, unsigned count) {
        float fSlow0 = 0.00100000005f * cutoff_;
        float fSlow1 = 0.215215757f * (q_ - 0.707000017f);
        for (unsigned i = 0; i < count; ++i) {
            float inp = in[i];
            fRec4_[0] = fSlow0 + 0.999000013f * fRec4_[1];
            float t1 = std::tan(fConst0_ * fRec4_[0]);
            float t2 = (inp - fRec3_[1]) * t1;
            float t3 = t1 + 1.0f;
            float t4 = t1 / t3;
            float t5 = (inp - (fRec3_[1] + (((t2 - fRec1_[1]) - fRec2_[1] * (0.0f - t4)) / t3)))
                / (1.0f - fSlow1 * ((t1 * (1.0f - t4)) / t3));
            out[i] = t5;
            float t6 = fSlow1 * t5;
            float t7 = (t1 * (t6 - fRec2_[1])) / t3;
            fRec1_[0] = fRec1_[1] + 2.0f * ((t1 * (t6 - (t7 + fRec1_[1] + fRec2_[1]))) / t3);
            fRec2_[0] = fRec2_[1] + 2.0f * t7;
            fRec3_[0] = fRec3_[1] + 2.0f * (t2 / t3);
            fRec4_[1] = fRec4_[0];
            fRec1_[1] = fRec1_[0];
            fRec2_[1] = fRec2_[0];
            fRec3_[1] = fRec3_[0];
        }
    }

private:
    float fConst0_ = 3.14159274f / 44100.0f;
    float cutoff_, q_;
    float fRec4_[2], fRec1_[2], fRec2_[2], fRec3_[2];
};

} // namespace vafilters

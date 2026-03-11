// Korg 35 24 dB/oct lowpass filter
// Generated from Faust DSP (korg35lpf.dsp) by Eric Tarr / Christopher Arndt
// License: MIT-style STK-4.3 license
// Cleaned up for nanodsp integration: no external dependencies
#pragma once
#include <cmath>
#include <algorithm>

namespace vafilters {

class Korg35LPF {
public:
    Korg35LPF() { reset(); }

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
            fRec4_[0] = fSlow0 + 0.999000013f * fRec4_[1];
            float t0 = std::tan(fConst0_ * fRec4_[0]);
            float t1 = (in[i] - fRec3_[1]) * t0;
            float t2 = t0 + 1.0f;
            float t3 = 1.0f - t0 / t2;
            float t4 = (t0 * ((((fRec3_[1] + ((t1 + fSlow1 * (fRec1_[1] * t3)) / t2))
                + fRec2_[1] * (0.0f - 1.0f / t2))
                / (1.0f - fSlow1 * ((t0 * t3) / t2))) - fRec1_[1])) / t2;
            float t5 = fRec1_[1] + t4;
            out[i] = t5;
            fRec1_[0] = fRec1_[1] + 2.0f * t4;
            fRec2_[0] = fRec2_[1] + 2.0f * ((t0 * (fSlow1 * t5 - fRec2_[1])) / t2);
            fRec3_[0] = fRec3_[1] + 2.0f * (t1 / t2);
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

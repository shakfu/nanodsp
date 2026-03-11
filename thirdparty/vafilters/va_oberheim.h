// Oberheim multi-mode state-variable filter
// Generated from Faust DSP (oberheim.dsp) by Eric Tarr / Christopher Arndt
// License: MIT-style STK-4.3 license
// 4 simultaneous outputs: BSF (notch), BPF, HPF, LPF
// Cleaned up for nanodsp integration: no external dependencies
#pragma once
#include <cmath>
#include <algorithm>

namespace vafilters {

class OberheimSVF {
public:
    OberheimSVF() { reset(); }

    void init(float sample_rate) {
        fConst0_ = 3.14159274f / std::min(192000.0f, std::max(1.0f, sample_rate));
        reset();
    }

    void reset() {
        cutoff_ = 20000.0f;
        q_ = 1.0f;
        for (auto &v : fRec6_) v = 0.0f;
        for (auto &v : fRec4_) v = 0.0f;
        for (auto &v : fRec5_) v = 0.0f;
    }

    void set_cutoff(float hz) { cutoff_ = hz; }
    float get_cutoff() const { return cutoff_; }
    void set_q(float q) { q_ = q; }
    float get_q() const { return q_; }

    // Process with all 4 outputs: bsf (notch), bpf, hpf, lpf
    void process_multi(const float *in,
                       float *out_bsf, float *out_bpf,
                       float *out_hpf, float *out_lpf,
                       unsigned count) {
        float fSlow0 = 0.00100000005f * cutoff_;
        float fSlow1 = 1.0f / q_;
        for (unsigned i = 0; i < count; ++i) {
            fRec6_[0] = fSlow0 + 0.999000013f * fRec6_[1];
            float t0 = std::tan(fConst0_ * fRec6_[0]);
            float t1 = fSlow1 + t0;
            float t2 = in[i] - (fRec4_[1] + fRec5_[1] * t1);
            float t3 = t0 * t1 + 1.0f;
            float t4 = (t0 * t2) / t3;
            float t5 = std::max(-1.0f, std::min(1.0f, fRec5_[1] + t4));
            float t6 = 1.0f - 0.333333343f * t5 * t5;
            float t7 = t0 * t5 * t6;
            float lpf = fRec4_[1] + t7;
            float hpf = t2 / t3;
            float bpf = t5 * t6;
            float bsf = t7 + fRec4_[1] + hpf;
            fRec4_[0] = fRec4_[1] + 2.0f * t7;
            fRec5_[0] = t4 + bpf;
            if (out_bsf) out_bsf[i] = bsf;
            if (out_bpf) out_bpf[i] = bpf;
            if (out_hpf) out_hpf[i] = hpf;
            if (out_lpf) out_lpf[i] = lpf;
            fRec6_[1] = fRec6_[0];
            fRec4_[1] = fRec4_[0];
            fRec5_[1] = fRec5_[0];
        }
    }

    // Single-output convenience: process with selected mode
    enum Mode { LPF = 0, HPF = 1, BPF = 2, BSF = 3 };

    void process(const float *in, float *out, unsigned count, Mode mode = LPF) {
        float *lpf = nullptr, *hpf = nullptr, *bpf = nullptr, *bsf = nullptr;
        switch (mode) {
            case LPF: lpf = out; break;
            case HPF: hpf = out; break;
            case BPF: bpf = out; break;
            case BSF: bsf = out; break;
        }
        process_multi(in, bsf, bpf, hpf, lpf, count);
    }

private:
    float fConst0_ = 3.14159274f / 44100.0f;
    float cutoff_, q_;
    float fRec6_[2], fRec4_[2], fRec5_[2];
};

} // namespace vafilters

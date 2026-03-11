// Antialiased waveshaping using antiderivative method
// Original: FX/Waveshaping.hpp
// Algorithms: 1st-order antialiased hard clipper, soft clipper, and
//             2nd-order antialiased wavefolder (Buchla 259 style)
#pragma once

#include <cmath>

namespace fxdsp {

class HardClipper {
    // Antialiased hard clipping using first-order antiderivative method.
public:
    HardClipper() = default;

    void reset() {
        xn1_ = 0.0f;
        Fn1_ = 0.0f;
    }

    float tick(float x) {
        float Fn = antideriv(x);
        float out;
        if (std::abs(x - xn1_) < thresh_) {
            out = clip(0.5f * (x + xn1_));
        } else {
            out = (Fn - Fn1_) / (x - xn1_);
        }
        xn1_ = x;
        Fn1_ = Fn;
        return out;
    }

    void process(const float* in, float* out, unsigned count) {
        for (unsigned i = 0; i < count; ++i)
            out[i] = tick(in[i]);
    }

    // Raw (non-antialiased) clip for reference
    static float clip(float x) {
        // clamp to [-1, 1]
        return 0.5f * (sign(x + 1.0f) * (x + 1.0f)
                      - sign(x - 1.0f) * (x - 1.0f));
    }

private:
    static float sign(float x) {
        return (x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f);
    }

    static float antideriv(float x) {
        return 0.25f * (sign(x + 1.0f) * (x + 1.0f) * (x + 1.0f)
                       - sign(x - 1.0f) * (x - 1.0f) * (x - 1.0f) - 2.0f);
    }

    float xn1_ = 0.0f;
    float Fn1_ = 0.0f;
    static constexpr float thresh_ = 0.1f;
};


class SoftClipper {
    // Antialiased piecewise soft saturation (sin-based) using
    // first-order antiderivative method.
public:
    SoftClipper() = default;

    void reset() {
        xn1_ = 0.0f;
        Fn1_ = 0.0f;
    }

    float tick(float x) {
        float Fn = antideriv(x);
        float out;
        if (std::abs(x - xn1_) < thresh_) {
            out = saturate(0.5f * (x + xn1_));
        } else {
            out = (Fn - Fn1_) / (x - xn1_);
        }
        xn1_ = x;
        Fn1_ = Fn;
        return out;
    }

    void process(const float* in, float* out, unsigned count) {
        for (unsigned i = 0; i < count; ++i)
            out[i] = tick(in[i]);
    }

    // Raw (non-antialiased) saturation for reference
    static float saturate(float x) {
        if (std::abs(x) < 1.0f)
            return std::sin(0.5f * pi_ * x);
        return (x > 0.0f) ? 1.0f : -1.0f;
    }

private:
    static float sign(float x) {
        return (x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f);
    }

    static float antideriv(float x) {
        if (std::abs(x) < 1.0f)
            return 1.0f - (2.0f / pi_) * std::cos(x * 0.5f * pi_);
        return sign(x) * x;
    }

    float xn1_ = 0.0f;
    float Fn1_ = 0.0f;
    static constexpr float thresh_ = 0.1f;
    static constexpr float pi_ = 3.14159265358979f;
};


class Wavefolder {
    // Antialiased wavefolder using second-order antiderivative method.
    // Produces Buchla 259-style folding distortion.
public:
    Wavefolder() = default;

    void reset() {
        xn1_ = 0.0f;
        xn2_ = 0.0f;
        Fn1_ = 0.0f;
        Gn1_ = 0.0f;
        clipper_.reset();
    }

    float tick(float x) {
        float Fn = fold_n2(x);
        float Gn;
        if (std::abs(x - xn1_) < thresh_) {
            Gn = fold_n1(0.5f * (x + xn1_));
        } else {
            Gn = (Fn - Fn1_) / (x - xn1_);
        }

        float out;
        if (std::abs(x - xn2_) < thresh_) {
            float delta = 0.5f * (x - 2.0f * xn1_ + xn2_);
            if (std::abs(delta) < thresh_) {
                out = fold_n0(0.25f * (x + 2.0f * xn1_ + xn2_));
            } else {
                float tmp1 = fold_n1(0.5f * (x + xn2_));
                float tmp2 = fold_n2(0.5f * (x + xn2_));
                out = (2.0f / delta) * (tmp1 + (Fn1_ - tmp2) / delta);
            }
        } else {
            out = 2.0f * (Gn - Gn1_) / (x - xn2_);
        }

        Fn1_ = Fn;
        Gn1_ = Gn;
        xn2_ = xn1_;
        xn1_ = x;
        return out;
    }

    void process(const float* in, float* out, unsigned count) {
        for (unsigned i = 0; i < count; ++i)
            out[i] = tick(in[i]);
    }

    // Raw fold: 2*clip(x) - x
    static float fold(float x) {
        return 2.0f * HardClipper::clip(x) - x;
    }

private:
    float fold_n0(float x) { return fold(x); }
    float fold_n1(float x) {
        return 2.0f * clip_n1(x) - 0.5f * x * x;
    }
    float fold_n2(float x) {
        return 2.0f * clip_n2(x) - (1.0f / 6.0f) * x * x * x;
    }

    static float sign(float x) {
        return (x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f);
    }

    // HardClipper antiderivatives (duplicated to avoid state coupling)
    static float clip_n1(float x) {
        return 0.25f * (sign(x + 1.0f) * (x + 1.0f) * (x + 1.0f)
                       - sign(x - 1.0f) * (x - 1.0f) * (x - 1.0f) - 2.0f);
    }
    static float clip_n2(float x) {
        return (1.0f / 12.0f) * (sign(x + 1.0f) * (x + 1.0f) * (x + 1.0f) * (x + 1.0f)
                                - sign(x - 1.0f) * (x - 1.0f) * (x - 1.0f) * (x - 1.0f)
                                - 6.0f * x);
    }

    HardClipper clipper_;  // unused but kept for potential future use
    float xn1_ = 0.0f;
    float xn2_ = 0.0f;
    float Fn1_ = 0.0f;
    float Gn1_ = 0.0f;
    static constexpr float thresh_ = 0.1f;
};

}  // namespace fxdsp

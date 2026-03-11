// Formant filter: vowel synthesis using cascaded biquad bandpass filters
// New implementation for nanodsp (not derived from RackFX which has heavy deps)
#pragma once

#include <cmath>
#include <algorithm>

namespace fxdsp {

class FormantFilter {
public:
    // Vowel indices
    enum Vowel { A = 0, E, I, O, U, NUM_VOWELS };

    FormantFilter() = default;

    void init(float sample_rate) {
        sr_ = sample_rate;
        set_vowel(A);
    }

    void reset() {
        for (int i = 0; i < NUM_FORMANTS; ++i) {
            for (int j = 0; j < 2; ++j) {
                x1_[i][j] = x2_[i][j] = 0.0f;
                y1_[i][j] = y2_[i][j] = 0.0f;
            }
        }
    }

    void set_vowel(int v) {
        vowel_ = std::max(0, std::min((int)NUM_VOWELS - 1, v));
        update_coefficients();
    }
    int get_vowel() const { return vowel_; }

    // Interpolate between two vowels (0.0 = vowel_a, 1.0 = vowel_b)
    void set_vowel_blend(int vowel_a, int vowel_b, float mix) {
        mix = std::max(0.0f, std::min(1.0f, mix));
        vowel_a = std::max(0, std::min((int)NUM_VOWELS - 1, vowel_a));
        vowel_b = std::max(0, std::min((int)NUM_VOWELS - 1, vowel_b));

        for (int i = 0; i < NUM_FORMANTS; ++i) {
            float freq = formants_[vowel_a][i][0] * (1.0f - mix)
                       + formants_[vowel_b][i][0] * mix;
            float bw   = formants_[vowel_a][i][1] * (1.0f - mix)
                       + formants_[vowel_b][i][1] * mix;
            float gain = formants_[vowel_a][i][2] * (1.0f - mix)
                       + formants_[vowel_b][i][2] * mix;
            compute_bandpass(i, freq, bw, gain);
        }
    }

    float tick(float x) {
        float out = 0.0f;
        for (int i = 0; i < NUM_FORMANTS; ++i) {
            float y = b0_[i] * x + b1_[i] * x1_[i][0] + b2_[i] * x2_[i][0]
                                  - a1_[i] * y1_[i][0] - a2_[i] * y2_[i][0];
            x2_[i][0] = x1_[i][0];
            x1_[i][0] = x;
            y2_[i][0] = y1_[i][0];
            y1_[i][0] = y;
            out += y;
        }
        return out;
    }

    void process(const float* in, float* out, unsigned count) {
        for (unsigned n = 0; n < count; ++n)
            out[n] = tick(in[n]);
    }

    // Process stereo (independent filter state per channel)
    void process_stereo(const float* in_l, const float* in_r,
                        float* out_l, float* out_r, unsigned count) {
        for (unsigned n = 0; n < count; ++n) {
            float yl = 0.0f, yr = 0.0f;
            for (int i = 0; i < NUM_FORMANTS; ++i) {
                // Left
                float y = b0_[i] * in_l[n] + b1_[i] * x1_[i][0] + b2_[i] * x2_[i][0]
                                             - a1_[i] * y1_[i][0] - a2_[i] * y2_[i][0];
                x2_[i][0] = x1_[i][0]; x1_[i][0] = in_l[n];
                y2_[i][0] = y1_[i][0]; y1_[i][0] = y;
                yl += y;

                // Right
                y = b0_[i] * in_r[n] + b1_[i] * x1_[i][1] + b2_[i] * x2_[i][1]
                                       - a1_[i] * y1_[i][1] - a2_[i] * y2_[i][1];
                x2_[i][1] = x1_[i][1]; x1_[i][1] = in_r[n];
                y2_[i][1] = y1_[i][1]; y1_[i][1] = y;
                yr += y;
            }
            out_l[n] = yl;
            out_r[n] = yr;
        }
    }

private:
    static constexpr int NUM_FORMANTS = 3;
    static constexpr float pi_ = 3.14159265358979f;

    // Formant data: [vowel][formant][freq_hz, bandwidth_hz, gain_linear]
    // Based on standard male vocal formant measurements
    static constexpr float formants_[NUM_VOWELS][NUM_FORMANTS][3] = {
        // A (as in "father")
        {{ 800.0f,  80.0f, 1.0f}, {1150.0f,  90.0f, 0.50f}, {2900.0f, 120.0f, 0.25f}},
        // E (as in "bed")
        {{ 350.0f,  60.0f, 1.0f}, {2000.0f, 100.0f, 0.50f}, {2800.0f, 120.0f, 0.25f}},
        // I (as in "heed")
        {{ 270.0f,  60.0f, 1.0f}, {2140.0f,  90.0f, 0.50f}, {2950.0f, 100.0f, 0.25f}},
        // O (as in "boat")
        {{ 450.0f,  70.0f, 1.0f}, { 800.0f,  80.0f, 0.50f}, {2830.0f, 100.0f, 0.25f}},
        // U (as in "boot")
        {{ 325.0f,  50.0f, 1.0f}, { 700.0f,  60.0f, 0.50f}, {2530.0f, 170.0f, 0.25f}},
    };

    void update_coefficients() {
        for (int i = 0; i < NUM_FORMANTS; ++i) {
            compute_bandpass(i, formants_[vowel_][i][0],
                               formants_[vowel_][i][1],
                               formants_[vowel_][i][2]);
        }
    }

    void compute_bandpass(int idx, float freq, float bw, float gain) {
        float w0 = 2.0f * pi_ * freq / sr_;
        float alpha = std::sin(w0) * std::sinh(std::log(2.0f) / 2.0f * bw / freq * w0 / std::sin(w0));
        float cos_w0 = std::cos(w0);
        float a0_inv = 1.0f / (1.0f + alpha);

        b0_[idx] = gain * alpha * a0_inv;
        b1_[idx] = 0.0f;
        b2_[idx] = -gain * alpha * a0_inv;
        a1_[idx] = -2.0f * cos_w0 * a0_inv;
        a2_[idx] = (1.0f - alpha) * a0_inv;
    }

    float sr_ = 44100.0f;
    int vowel_ = A;

    // Biquad coefficients per formant
    float b0_[NUM_FORMANTS] = {};
    float b1_[NUM_FORMANTS] = {};
    float b2_[NUM_FORMANTS] = {};
    float a1_[NUM_FORMANTS] = {};
    float a2_[NUM_FORMANTS] = {};

    // State per formant, per channel (0=left/mono, 1=right)
    float x1_[NUM_FORMANTS][2] = {};
    float x2_[NUM_FORMANTS][2] = {};
    float y1_[NUM_FORMANTS][2] = {};
    float y2_[NUM_FORMANTS][2] = {};
};

}  // namespace fxdsp

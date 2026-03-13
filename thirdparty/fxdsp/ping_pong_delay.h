// Stereo ping-pong delay with crossed feedback and linear interpolation
// Based on: FX/KHPingPongDelay.hpp (Keith Hearne)
// Rewritten as clean, self-contained header for nanodsp.
#pragma once

#include <cmath>
#include <vector>
#include <algorithm>

namespace fxdsp {

class PingPongDelay {
    // Stereo ping-pong delay that bounces the signal between left and right
    // channels.  Each channel has its own delay line; feedback is crossed so
    // the left output feeds back into the right delay and vice versa.
public:
    PingPongDelay() = default;

    void init(float sample_rate, float max_delay_ms = 2000.0f) {
        sr_ = sample_rate;
        int max_samples = (int)std::ceil(sample_rate * max_delay_ms * 0.001f) + 2;
        buf_l_.assign(max_samples, 0.0f);
        buf_r_.assign(max_samples, 0.0f);
        buf_size_ = max_samples;
        write_l_ = write_r_ = 0;
        set_delay_ms(500.0f);
    }

    void reset() {
        std::fill(buf_l_.begin(), buf_l_.end(), 0.0f);
        std::fill(buf_r_.begin(), buf_r_.end(), 0.0f);
        write_l_ = write_r_ = 0;
        fb_l_ = fb_r_ = 0.0f;
    }

    // ---- Parameters ----

    void set_delay_ms(float ms) {
        delay_ms_ = std::max(0.0f, ms);
        float d = sr_ * delay_ms_ * 0.001f;
        delay_int_ = (int)d;
        delay_frac_ = d - (float)delay_int_;
        if (delay_int_ >= buf_size_ - 1)
            delay_int_ = buf_size_ - 2;
    }
    float get_delay_ms() const { return delay_ms_; }

    void set_feedback(float fb) { feedback_ = std::max(-0.99f, std::min(0.99f, fb)); }
    float get_feedback() const { return feedback_; }

    void set_mix(float m) { mix_ = std::max(0.0f, std::min(1.0f, m)); }
    float get_mix() const { return mix_; }

    // ---- Processing ----

    // Process one stereo sample pair.  Returns (left, right).
    std::pair<float, float> tick(float in_l, float in_r) {
        // Read from delay lines with linear interpolation
        float del_l = read_interp(buf_l_, write_l_);
        float del_r = read_interp(buf_r_, write_r_);

        // Write input + crossed feedback into delay lines
        buf_l_[write_l_] = in_l + feedback_ * fb_r_;
        buf_r_[write_r_] = in_r + feedback_ * fb_l_;

        // Store feedback outputs for next sample (crossed)
        fb_l_ = del_l;
        fb_r_ = del_r;

        // Advance write positions
        write_l_ = (write_l_ + 1) % buf_size_;
        write_r_ = (write_r_ + 1) % buf_size_;

        // Mix dry and wet
        float out_l = (1.0f - mix_) * in_l + mix_ * del_l;
        float out_r = (1.0f - mix_) * in_r + mix_ * del_r;
        return {out_l, out_r};
    }

    // Process interleaved stereo arrays: in[2, n] -> out[2, n]
    void process(const float* in_l, const float* in_r,
                 float* out_l, float* out_r, unsigned count) {
        for (unsigned i = 0; i < count; ++i) {
            std::pair<float, float> s = tick(in_l[i], in_r[i]);
            out_l[i] = s.first;
            out_r[i] = s.second;
        }
    }

private:
    float read_interp(const std::vector<float>& buf, int wp) const {
        int i1 = wp - delay_int_ - 1;
        if (i1 < 0) i1 += buf_size_;
        int i2 = i1 - 1;
        if (i2 < 0) i2 += buf_size_;
        return buf[i1] * (1.0f - delay_frac_) + buf[i2] * delay_frac_;
    }

    float sr_ = 44100.0f;
    std::vector<float> buf_l_, buf_r_;
    int buf_size_ = 0;
    int write_l_ = 0, write_r_ = 0;
    float delay_ms_ = 500.0f;
    int delay_int_ = 0;
    float delay_frac_ = 0.0f;
    float feedback_ = 0.5f;
    float mix_ = 0.5f;
    float fb_l_ = 0.0f, fb_r_ = 0.0f;
};

}  // namespace fxdsp

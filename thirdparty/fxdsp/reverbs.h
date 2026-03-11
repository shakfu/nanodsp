// Classic reverb algorithms: Schroeder and Moorer
// Based on: Reverbs/SchroederReverb.hpp, Reverbs/MoorerReverb.hpp
// Rewritten with dynamic allocation, bug fixes, and configurable parameters.
#pragma once

#include <cmath>
#include <vector>
#include <algorithm>

namespace fxdsp {

// Fractional delay line with linear interpolation and optional sine LFO
class DelayLine {
public:
    explicit DelayLine(int max_samples = 96000)
        : buffer_(max_samples, 0.0f), max_size_(max_samples) {}

    void set_delay(float samples) {
        delay_ = std::max(0.0f, std::min((float)(max_size_ - 2), samples));
    }

    void set_lfo(float depth, float rate_hz, float sample_rate) {
        lfo_depth_ = depth;
        lfo_inc_ = (sample_rate > 0.0f) ? 2.0f * pi_ * rate_hz / sample_rate : 0.0f;
    }

    float read() const {
        float d = delay_ + lfo_depth_ * std::sin(lfo_phase_);
        d = std::max(0.0f, std::min((float)(max_size_ - 2), d));
        int d1 = (int)d;
        float frac = d - (float)d1;
        int i1 = write_pos_ - d1 - 1;
        if (i1 < 0) i1 += max_size_;
        int i2 = i1 - 1;
        if (i2 < 0) i2 += max_size_;
        return buffer_[i1] * (1.0f - frac) + buffer_[i2] * frac;
    }

    void write(float x) {
        buffer_[write_pos_] = x;
        write_pos_ = (write_pos_ + 1) % max_size_;
        lfo_phase_ += lfo_inc_;
        if (lfo_phase_ > 2.0f * pi_) lfo_phase_ -= 2.0f * pi_;
    }

    float process(float x) {
        float y = read();
        write(x);
        return y;
    }

    void reset() {
        std::fill(buffer_.begin(), buffer_.end(), 0.0f);
        write_pos_ = 0;
        lfo_phase_ = 0.0f;
    }

private:
    std::vector<float> buffer_;
    int max_size_;
    int write_pos_ = 0;
    float delay_ = 0.0f;
    float lfo_depth_ = 0.0f;
    float lfo_inc_ = 0.0f;
    mutable float lfo_phase_ = 0.0f;
    static constexpr float pi_ = 3.14159265358979f;
};

// Allpass filter for diffusion
class AllpassSection {
public:
    explicit AllpassSection(int max_samples = 4096)
        : delay_(max_samples) {}

    void set_delay(float samples) { delay_.set_delay(samples); }
    void set_gain(float g) { gain_ = g; }
    void set_lfo(float depth, float rate, float sr) { delay_.set_lfo(depth, rate, sr); }

    float process(float x) {
        float delayed = delay_.read();
        float v = x + (-gain_) * delayed;
        delay_.write(v);
        return delayed + gain_ * v;
    }

    void reset() { delay_.reset(); fb_ = 0.0f; }

private:
    DelayLine delay_;
    float gain_ = 0.5f;
    float fb_ = 0.0f;
};

// Feedback comb filter
class FeedbackComb {
public:
    explicit FeedbackComb(int max_samples = 4096)
        : delay_(max_samples) {}

    void set_delay(float samples) { delay_.set_delay(samples); }
    void set_feedback(float g) { feedback_ = std::max(-0.999f, std::min(0.999f, g)); }
    void set_lfo(float depth, float rate, float sr) { delay_.set_lfo(depth, rate, sr); }

    float process(float x) {
        float delayed = delay_.read();
        float v = x + (-feedback_) * delayed;
        delay_.write(v);
        return delayed;
    }

    void reset() { delay_.reset(); }

private:
    DelayLine delay_;
    float feedback_ = 0.7f;
};

// Early reflections using fixed tap delay
class EarlyReflections {
public:
    EarlyReflections() : buffer_(4096, 0.0f) {}

    void set_sample_rate(float sr) {
        // Scale tap times from 44100 Hz reference
        float scale = sr / 44100.0f;
        for (int i = 0; i < num_taps_; ++i) {
            scaled_taps_[i] = (int)(ref_taps_[i] * scale);
        }
        buf_size_ = (int)(3520 * scale) + 1;
        buf_size_ = std::min(buf_size_, (int)buffer_.size());
    }

    float process(float x) {
        buffer_[write_pos_] = x;
        float y = 0.0f;
        for (int i = 0; i < num_taps_; ++i) {
            int idx = write_pos_ - scaled_taps_[i];
            if (idx < 0) idx += buf_size_;
            y += tap_gains_[i] * buffer_[idx];
        }
        write_pos_ = (write_pos_ + 1) % buf_size_;
        return y;
    }

    void reset() {
        std::fill(buffer_.begin(), buffer_.end(), 0.0f);
        write_pos_ = 0;
    }

private:
    static constexpr int num_taps_ = 18;
    int ref_taps_[18] = {190, 949, 993, 1183, 1192, 1315, 2021, 2140,
                         2524, 2590, 2625, 2700, 3119, 3123, 3202, 3268, 3321, 3515};
    float tap_gains_[18] = {.841f, .504f, .49f, .379f, .38f, .346f, .289f, .272f,
                            .192f, .193f, .217f, .181f, .18f, .181f, .176f, .142f, .167f, .134f};
    int scaled_taps_[18] = {};
    std::vector<float> buffer_;
    int buf_size_ = 3520;
    int write_pos_ = 0;
};


// Schroeder reverberator: 4 parallel combs + 2 series allpasses
class SchroederReverb {
public:
    SchroederReverb() {
        for (auto& c : comb_) c = FeedbackComb(4096);
        for (auto& a : apf_)  a = AllpassSection(4096);
    }

    void init(float sample_rate) {
        sr_ = sample_rate;
        float scale = sample_rate / 48000.0f;

        comb_[0].set_delay(1426.0f * scale);
        comb_[1].set_delay(1781.0f * scale);
        comb_[2].set_delay(1973.0f * scale);
        comb_[3].set_delay(2098.0f * scale);

        apf_[0].set_delay(240.0f * scale);
        apf_[1].set_delay(82.0f * scale);

        set_feedback(0.7f);
        set_diffusion(0.5f);
        set_mod_depth(0.0f);
    }

    void set_feedback(float g) {
        feedback_ = g;
        for (auto& c : comb_) c.set_feedback(g);
    }

    void set_diffusion(float g) {
        diffusion_ = g;
        apf_[0].set_gain(g);
        apf_[1].set_gain(g);
    }

    void set_mod_depth(float d) {
        float rates[4] = {0.723f, 1.257f, 0.893f, 1.111f};
        for (int i = 0; i < 4; ++i)
            comb_[i].set_lfo(d, rates[i], sr_);
        apf_[0].set_lfo(d, 0.832f, sr_);
        apf_[1].set_lfo(d, 0.964f, sr_);
    }

    float get_feedback() const { return feedback_; }
    float get_diffusion() const { return diffusion_; }

    float tick(float x) {
        float sum = 0.0f;
        for (auto& c : comb_)
            sum += c.process(x);
        sum *= 0.25f;
        float y = apf_[0].process(sum);
        y = apf_[1].process(y);
        return y;
    }

    void process(const float* in, float* out, unsigned count) {
        for (unsigned i = 0; i < count; ++i)
            out[i] = tick(in[i]);
    }

    void reset() {
        for (auto& c : comb_) c.reset();
        for (auto& a : apf_)  a.reset();
    }

private:
    FeedbackComb comb_[4];
    AllpassSection apf_[2];
    float sr_ = 48000.0f;
    float feedback_ = 0.7f;
    float diffusion_ = 0.5f;
};


// Moorer reverberator: early reflections + 4 parallel combs + 2 series allpasses
class MoorerReverb {
public:
    MoorerReverb() {
        for (auto& c : comb_) c = FeedbackComb(4096);
        for (auto& a : apf_)  a = AllpassSection(4096);
    }

    void init(float sample_rate) {
        sr_ = sample_rate;
        float scale = sample_rate / 48000.0f;

        er_.set_sample_rate(sample_rate);

        comb_[0].set_delay(1426.0f * scale);
        comb_[1].set_delay(1781.0f * scale);
        comb_[2].set_delay(1973.0f * scale);
        comb_[3].set_delay(2098.0f * scale);

        apf_[0].set_delay(240.0f * scale);
        apf_[1].set_delay(82.0f * scale);

        set_feedback(0.7f);
        set_diffusion(0.7f);
        set_mod_depth(0.1f);
    }

    void set_feedback(float g) {
        feedback_ = g;
        for (auto& c : comb_) c.set_feedback(g);
    }

    void set_diffusion(float g) {
        diffusion_ = g;
        apf_[0].set_gain(g - 0.01f);
        apf_[1].set_gain(g - 0.03f);
    }

    void set_mod_depth(float d) {
        float rates[4] = {0.99f, 0.91f, 0.93f, 0.97f};
        for (int i = 0; i < 4; ++i)
            comb_[i].set_lfo(d, rates[i], sr_);
        apf_[0].set_lfo(d, 1.1f, sr_);
        apf_[1].set_lfo(d, 1.2f, sr_);
    }

    float get_feedback() const { return feedback_; }
    float get_diffusion() const { return diffusion_; }

    float tick(float x) {
        float early = er_.process(x);
        // Late reverb: combs fed by input (not early reflections)
        float sum = 0.0f;
        for (auto& c : comb_)
            sum += c.process(x);
        sum *= 0.5f;
        float late = apf_[0].process(sum);
        late = apf_[1].process(late);
        // Mix early reflections + late reverb
        return early + late;
    }

    void process(const float* in, float* out, unsigned count) {
        for (unsigned i = 0; i < count; ++i)
            out[i] = tick(in[i]);
    }

    void reset() {
        er_.reset();
        for (auto& c : comb_) c.reset();
        for (auto& a : apf_)  a.reset();
    }

private:
    EarlyReflections er_;
    FeedbackComb comb_[4];
    AllpassSection apf_[2];
    float sr_ = 48000.0f;
    float feedback_ = 0.7f;
    float diffusion_ = 0.7f;
};

}  // namespace fxdsp

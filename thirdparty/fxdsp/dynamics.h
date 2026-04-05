// Dynamics processors: sidechain compressor, transient shaper, lookahead limiter
// All operate on mono float buffers via process() methods.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace fxdsp {

// ---------------------------------------------------------------------------
// Sidechain Compressor
// ---------------------------------------------------------------------------

class SidechainCompressor {
public:
    SidechainCompressor() = default;

    void init(float sample_rate) {
        sr_ = sample_rate;
        smoothed_env_db_ = threshold_;
    }

    void set_ratio(float r) { ratio_ = r; }
    void set_threshold(float t) { threshold_ = t; smoothed_env_db_ = t; }
    void set_attack(float a) {
        attack_coeff_ = (a > 0.0f) ? 1.0f - std::exp(-1.0f / (sr_ * a)) : 1.0f;
    }
    void set_release(float r) {
        release_coeff_ = (r > 0.0f) ? 1.0f - std::exp(-1.0f / (sr_ * r)) : 1.0f;
    }

    void reset() { smoothed_env_db_ = threshold_; }

    // Process: apply gain reduction to `buf` based on envelope of `sidechain`.
    // Both buffers must have length `n`. Output written to `out`.
    void process(const float* buf, const float* sidechain, float* out, unsigned n) {
        for (unsigned i = 0; i < n; ++i) {
            float sc_abs = std::abs(sidechain[i]);
            float env_db = 20.0f * std::log10(sc_abs + eps_);

            // Smooth envelope in dB domain
            if (env_db > smoothed_env_db_)
                smoothed_env_db_ += attack_coeff_ * (env_db - smoothed_env_db_);
            else
                smoothed_env_db_ += release_coeff_ * (env_db - smoothed_env_db_);

            // Compute gain reduction
            float gain = 1.0f;
            if (smoothed_env_db_ > threshold_) {
                float over = smoothed_env_db_ - threshold_;
                float reduction = over * (1.0f - 1.0f / ratio_);
                gain = std::pow(10.0f, -reduction / 20.0f);
            }
            out[i] = buf[i] * gain;
        }
    }

private:
    float sr_ = 48000.0f;
    float ratio_ = 4.0f;
    float threshold_ = -20.0f;
    float attack_coeff_ = 0.0f;
    float release_coeff_ = 0.0f;
    float smoothed_env_db_ = -120.0f;
    static constexpr float eps_ = 1e-20f;
};

// ---------------------------------------------------------------------------
// Transient Shaper
// ---------------------------------------------------------------------------

class TransientShaper {
public:
    TransientShaper() = default;

    void init(float sample_rate) {
        sr_ = sample_rate;
        fast_env_ = 0.0f;
        slow_env_ = 0.0f;
    }

    void set_attack_gain(float g) { attack_gain_ = g; }
    void set_sustain_gain(float g) { sustain_gain_ = g; }

    void set_fast_attack(float t) {
        fa_coeff_ = (t > 0.0f) ? 1.0f - std::exp(-1.0f / (sr_ * t)) : 1.0f;
    }
    void set_fast_release(float t) {
        fr_coeff_ = (t > 0.0f) ? 1.0f - std::exp(-1.0f / (sr_ * t)) : 1.0f;
    }
    void set_slow_attack(float t) {
        sa_coeff_ = (t > 0.0f) ? 1.0f - std::exp(-1.0f / (sr_ * t)) : 1.0f;
    }
    void set_slow_release(float t) {
        sr_coeff_ = (t > 0.0f) ? 1.0f - std::exp(-1.0f / (sr_ * t)) : 1.0f;
    }

    void reset() {
        fast_env_ = 0.0f;
        slow_env_ = 0.0f;
    }

    void process(const float* in, float* out, unsigned n) {
        for (unsigned i = 0; i < n; ++i) {
            float level = std::abs(in[i]);

            // Fast envelope (tracks transients)
            if (level > fast_env_)
                fast_env_ += fa_coeff_ * (level - fast_env_);
            else
                fast_env_ += fr_coeff_ * (level - fast_env_);

            // Slow envelope (tracks sustain)
            if (level > slow_env_)
                slow_env_ += sa_coeff_ * (level - slow_env_);
            else
                slow_env_ += sr_coeff_ * (level - slow_env_);

            float transient = std::max(0.0f, fast_env_ - slow_env_);
            float sustain = slow_env_;
            float total = fast_env_ + eps_;

            float gain = (attack_gain_ * transient + sustain_gain_ * sustain) / total;
            out[i] = in[i] * gain;
        }
    }

private:
    float sr_ = 48000.0f;
    float attack_gain_ = 1.0f;
    float sustain_gain_ = 1.0f;
    float fa_coeff_ = 0.0f;
    float fr_coeff_ = 0.0f;
    float sa_coeff_ = 0.0f;
    float sr_coeff_ = 0.0f;
    float fast_env_ = 0.0f;
    float slow_env_ = 0.0f;
    static constexpr float eps_ = 1e-10f;
};

// ---------------------------------------------------------------------------
// Lookahead Limiter
// ---------------------------------------------------------------------------

class LookaheadLimiter {
public:
    LookaheadLimiter() = default;

    void init(float sample_rate) {
        sr_ = sample_rate;
    }

    void set_threshold_db(float db) {
        threshold_lin_ = std::pow(10.0f, db / 20.0f);
    }
    void set_lookahead_ms(float ms) {
        lookahead_ = std::max(1u, (unsigned)(sr_ * ms / 1000.0f));
    }
    void set_release(float r) {
        release_coeff_ = (r > 0.0f) ? 1.0f - std::exp(-1.0f / (sr_ * r)) : 1.0f;
    }

    // Process mono: applies delay + gain curve. Output length = input length.
    // First `lookahead` output samples are silence (delay compensation).
    void process(const float* in, float* out, unsigned n) {
        if (n == 0) return;

        unsigned la = std::min(lookahead_, n);

        // Build delayed signal
        std::vector<float> delayed(n, 0.0f);
        if (la < n)
            std::memcpy(delayed.data() + la, in, (n - la) * sizeof(float));

        // Raw gain for delayed signal
        std::vector<float> raw_gain(n, 1.0f);
        for (unsigned i = 0; i < n; ++i) {
            float peak = std::abs(delayed[i]);
            if (peak > threshold_lin_)
                raw_gain[i] = threshold_lin_ / (peak + eps_);
        }

        // Forward-looking minimum over lookahead window (backward scan)
        std::vector<float> ahead_gain(n, 1.0f);
        float running_min = 1.0f;
        for (int i = (int)n - 1; i >= 0; --i) {
            running_min = std::min(running_min, raw_gain[i]);
            ahead_gain[i] = running_min;
            // When the sample exiting the window was the min, rescan
            if ((unsigned)i + la < n &&
                raw_gain[(unsigned)i + la] <= running_min + 1e-15f) {
                running_min = 1.0f;
                for (unsigned j = (unsigned)i; j < std::min((unsigned)i + la, n); ++j)
                    running_min = std::min(running_min, raw_gain[j]);
            }
        }

        // Smooth: snap downward, release upward
        float current = 1.0f;
        for (unsigned i = 0; i < n; ++i) {
            float target = ahead_gain[i];
            if (target < current)
                current = target;
            else
                current += release_coeff_ * (target - current);
            out[i] = delayed[i] * current;
        }
    }

private:
    float sr_ = 48000.0f;
    float threshold_lin_ = 1.0f;
    unsigned lookahead_ = 240;  // ~5ms at 48kHz
    float release_coeff_ = 0.0f;
    static constexpr float eps_ = 1e-20f;
};

} // namespace fxdsp

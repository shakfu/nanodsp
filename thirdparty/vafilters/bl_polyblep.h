// PolyBLEP oscillator with 14 waveform types
// Based on "Phaseshaping Oscillator Algorithms for Musical Sound Synthesis"
// by Kleimola, Lazzarini, Timoney, Valimaki (SMC 2010)
// Modified by macho charlie 1993
// Cleaned up for nanodsp integration: no external dependencies
#pragma once
#include <cmath>
#include <cstdint>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace bloscillators {

namespace detail {
    template<typename T>
    inline T square_number(T x) { return x * x; }

    inline float blep(float t, float dt) {
        if (t < dt) {
            return -square_number(t / dt - 1.0f);
        } else if (t > 1.0f - dt) {
            return square_number((t - 1.0f) / dt + 1.0f);
        }
        return 0.0f;
    }

    inline float blamp(float t, float dt) {
        if (t < dt) {
            t = t / dt - 1.0f;
            return -1.0f / 3.0f * square_number(t) * t;
        } else if (t > 1.0f - dt) {
            t = (t - 1.0f) / dt + 1.0f;
            return 1.0f / 3.0f * square_number(t) * t;
        }
        return 0.0f;
    }

    inline int64_t floor_int(float x) {
        return static_cast<int64_t>(x) | 0;
    }
} // namespace detail

class PolyBLEP {
public:
    enum Waveform {
        SINE = 0,
        COSINE,
        TRIANGLE,
        SQUARE,
        RECTANGLE,
        SAWTOOTH,
        RAMP,
        MODIFIED_TRIANGLE,
        MODIFIED_SQUARE,
        HALF_WAVE_RECTIFIED_SINE,
        FULL_WAVE_RECTIFIED_SINE,
        TRIANGULAR_PULSE,
        TRAPEZOID_FIXED,
        TRAPEZOID_VARIABLE,
        NUM_WAVEFORMS
    };

    PolyBLEP(float sample_rate = 44100.0f, Waveform wave = SAWTOOTH)
        : sample_rate_(sample_rate), amplitude_(1.0f), t_(0.0f),
          pulse_width_(0.5f), waveform_(wave) {
        set_frequency(440.0f);
    }

    void set_sample_rate(float sr) {
        float freq = get_frequency();
        sample_rate_ = sr;
        set_frequency(freq);
    }
    float get_sample_rate() const { return sample_rate_; }

    void set_frequency(float hz) { dt_ = hz / sample_rate_; }
    float get_frequency() const { return dt_ * sample_rate_; }

    void set_waveform(Waveform w) { waveform_ = w; }
    Waveform get_waveform() const { return waveform_; }

    void set_pulse_width(float pw) { pulse_width_ = pw; }
    float get_pulse_width() const { return pulse_width_; }

    void set_phase(float p) { t_ = p; }
    float get_phase() const { return t_; }

    void reset() { t_ = 0.0f; }

    void sync(float phase) {
        t_ = phase;
        if (t_ >= 0.0f)
            t_ -= detail::floor_int(t_);
        else
            t_ += 1.0f - detail::floor_int(t_);
    }

    float tick() {
        float sample = get_sample();
        t_ += dt_;
        t_ -= detail::floor_int(t_);
        return sample;
    }

    void generate(float *out, unsigned count) {
        for (unsigned i = 0; i < count; ++i)
            out[i] = tick();
    }

private:
    float get_sample() {
        if (dt_ >= 0.5f) return gen_sin();
        switch (waveform_) {
            case SINE:                      return gen_sin();
            case COSINE:                    return gen_cos();
            case TRIANGLE:                  return gen_tri();
            case SQUARE:                    return gen_sqr();
            case RECTANGLE:                 return gen_rect();
            case SAWTOOTH:                  return gen_saw();
            case RAMP:                      return gen_ramp();
            case MODIFIED_TRIANGLE:         return gen_tri2();
            case MODIFIED_SQUARE:           return gen_sqr2();
            case HALF_WAVE_RECTIFIED_SINE:  return gen_half();
            case FULL_WAVE_RECTIFIED_SINE:  return gen_full();
            case TRIANGULAR_PULSE:          return gen_trip();
            case TRAPEZOID_FIXED:           return gen_trap();
            case TRAPEZOID_VARIABLE:        return gen_trap2();
            default:                        return 0.0f;
        }
    }

    float gen_sin() { return amplitude_ * std::sin(2.0f * (float)M_PI * t_); }
    float gen_cos() { return amplitude_ * std::cos(2.0f * (float)M_PI * t_); }

    float gen_saw() {
        float _t = t_ + 0.5f;
        _t -= detail::floor_int(_t);
        float y = 2.0f * _t - 1.0f;
        y -= detail::blep(_t, dt_);
        return amplitude_ * y;
    }

    float gen_ramp() {
        float _t = t_;
        _t -= detail::floor_int(_t);
        float y = 1.0f - 2.0f * _t;
        y += detail::blep(_t, dt_);
        return amplitude_ * y;
    }

    float gen_sqr() {
        float t2 = t_ + 0.5f;
        t2 -= detail::floor_int(t2);
        float y = t_ < 0.5f ? 1.0f : -1.0f;
        y += detail::blep(t_, dt_) - detail::blep(t2, dt_);
        return amplitude_ * y;
    }

    float gen_rect() {
        float pw = pulse_width_;
        float t2 = t_ + 1.0f - pw;
        t2 -= detail::floor_int(t2);
        float y = -2.0f * pw;
        if (t_ < pw) y += 2.0f;
        y += detail::blep(t_, dt_) - detail::blep(t2, dt_);
        return amplitude_ * y;
    }

    float gen_tri() {
        float t1 = t_ + 0.25f;
        t1 -= detail::floor_int(t1);
        float t2 = t_ + 0.75f;
        t2 -= detail::floor_int(t2);
        float y = t_ * 4.0f;
        if (y >= 3.0f) y -= 4.0f;
        else if (y > 1.0f) y = 2.0f - y;
        y += 4.0f * dt_ * (detail::blamp(t1, dt_) - detail::blamp(t2, dt_));
        return amplitude_ * y;
    }

    float gen_tri2() {
        float pw = std::fmax(0.0001f, std::fmin(0.9999f, pulse_width_));
        float t1 = t_ + 0.5f * pw;
        t1 -= detail::floor_int(t1);
        float t2 = t_ + 1.0f - 0.5f * pw;
        t2 -= detail::floor_int(t2);
        float y = t_ * 2.0f;
        if (y >= 2.0f - pw) y = (y - 2.0f) / pw;
        else if (y >= pw) y = 1.0f - (y - pw) / (1.0f - pw);
        else y /= pw;
        y += dt_ / (pw - pw * pw) * (detail::blamp(t1, dt_) - detail::blamp(t2, dt_));
        return amplitude_ * y;
    }

    float gen_sqr2() {
        float pw = pulse_width_;
        float t1 = t_ + 0.875f + 0.25f * (pw - 0.5f);
        t1 -= detail::floor_int(t1);
        float t2 = t_ + 0.375f + 0.25f * (pw - 0.5f);
        t2 -= detail::floor_int(t2);
        float y = t1 < 0.5f ? 1.0f : -1.0f;
        y += detail::blep(t1, dt_) - detail::blep(t2, dt_);
        t1 += 0.5f * (1.0f - pw);
        t1 -= detail::floor_int(t1);
        t2 += 0.5f * (1.0f - pw);
        t2 -= detail::floor_int(t2);
        y += t1 < 0.5f ? 1.0f : -1.0f;
        y += detail::blep(t1, dt_) - detail::blep(t2, dt_);
        return amplitude_ * 0.5f * y;
    }

    float gen_half() {
        float t2 = t_ + 0.5f;
        t2 -= detail::floor_int(t2);
        float y = (t_ < 0.5f ? 2.0f * std::sin(2.0f * (float)M_PI * t_) - 2.0f / (float)M_PI : -2.0f / (float)M_PI);
        y += 2.0f * (float)M_PI * dt_ * (detail::blamp(t_, dt_) + detail::blamp(t2, dt_));
        return amplitude_ * y;
    }

    float gen_full() {
        float _t = t_ + 0.25f;
        _t -= detail::floor_int(_t);
        float y = 2.0f * std::sin((float)M_PI * _t) - 4.0f / (float)M_PI;
        y += 2.0f * (float)M_PI * dt_ * detail::blamp(_t, dt_);
        return amplitude_ * y;
    }

    float gen_trip() {
        float pw = pulse_width_;
        float t1 = t_ + 0.75f + 0.5f * pw;
        t1 -= detail::floor_int(t1);
        float y;
        if (t1 >= pw) {
            y = -pw;
        } else {
            y = 4.0f * t1;
            y = (y >= 2.0f * pw ? 4.0f - y / pw - pw : y / pw - pw);
        }
        if (pw > 0.0f) {
            float t2 = t1 + 1.0f - 0.5f * pw;
            t2 -= detail::floor_int(t2);
            float t3 = t1 + 1.0f - pw;
            t3 -= detail::floor_int(t3);
            y += 2.0f * dt_ / pw * (detail::blamp(t1, dt_) - 2.0f * detail::blamp(t2, dt_) + detail::blamp(t3, dt_));
        }
        return amplitude_ * y;
    }

    float gen_trap() {
        float y = 4.0f * t_;
        if (y >= 3.0f) y -= 4.0f;
        else if (y > 1.0f) y = 2.0f - y;
        y = std::fmax(-1.0f, std::fmin(1.0f, 2.0f * y));
        float t1 = t_ + 0.125f;
        t1 -= detail::floor_int(t1);
        float t2 = t1 + 0.5f;
        t2 -= detail::floor_int(t2);
        y += 4.0f * dt_ * (detail::blamp(t1, dt_) - detail::blamp(t2, dt_));
        t1 = t_ + 0.375f;
        t1 -= detail::floor_int(t1);
        t2 = t1 + 0.5f;
        t2 -= detail::floor_int(t2);
        y += 4.0f * dt_ * (detail::blamp(t1, dt_) - detail::blamp(t2, dt_));
        return amplitude_ * y;
    }

    float gen_trap2() {
        float pw = std::fmin(0.9999f, pulse_width_);
        float scale = 1.0f / (1.0f - pw);
        float y = 4.0f * t_;
        if (y >= 3.0f) y -= 4.0f;
        else if (y > 1.0f) y = 2.0f - y;
        y = std::fmax(-1.0f, std::fmin(1.0f, scale * y));
        float t1 = t_ + 0.25f - 0.25f * pw;
        t1 -= detail::floor_int(t1);
        float t2 = t1 + 0.5f;
        t2 -= detail::floor_int(t2);
        y += scale * 2.0f * dt_ * (detail::blamp(t1, dt_) - detail::blamp(t2, dt_));
        t1 = t_ + 0.25f + 0.25f * pw;
        t1 -= detail::floor_int(t1);
        t2 = t1 + 0.5f;
        t2 -= detail::floor_int(t2);
        y += scale * 2.0f * dt_ * (detail::blamp(t1, dt_) - detail::blamp(t2, dt_));
        return amplitude_ * y;
    }

    float sample_rate_;
    float amplitude_;
    float t_;        // phase [0, 1)
    float dt_;       // freq / sample_rate
    float pulse_width_;
    Waveform waveform_;
};

// BLIT-based sawtooth oscillator (STK-style sinc train with leaky integration)
class BlitSaw {
public:
    BlitSaw(float sample_rate = 44100.0f, float frequency = 220.0f)
        : sample_rate_(sample_rate) {
        reset();
        set_frequency(frequency);
    }

    void set_sample_rate(float sr) { sample_rate_ = sr; set_frequency(get_frequency()); }
    void set_frequency(float hz) {
        p_ = sample_rate_ / hz;
        C2_ = 1.0f / p_;
        rate_ = (float)M_PI * C2_;
        update_harmonics();
    }
    float get_frequency() const { return sample_rate_ / p_; }

    void set_harmonics(unsigned n) { n_harmonics_ = n; update_harmonics(); state_ = -0.5f * a_; }

    void reset() {
        phase_ = 0.0f;
        state_ = 0.0f;
        n_harmonics_ = 0;
    }

    float tick() {
        float tmp, denominator = std::sin(phase_);
        if (std::fabs(denominator) <= 1e-7f)
            tmp = a_;
        else {
            tmp = std::sin(m_ * phase_);
            tmp /= p_ * denominator;
        }
        tmp += state_ - C2_;
        state_ = tmp * 0.995f;
        phase_ += rate_;
        if (phase_ >= (float)M_PI) phase_ -= (float)M_PI;
        return tmp;
    }

    void generate(float *out, unsigned count) {
        for (unsigned i = 0; i < count; ++i) out[i] = tick();
    }

private:
    void update_harmonics() {
        if (n_harmonics_ <= 0) {
            unsigned max_h = (unsigned)std::floor(0.5f * p_);
            m_ = 2 * max_h + 1;
        } else {
            m_ = 2 * n_harmonics_ + 1;
        }
        a_ = (float)m_ / p_;
    }

    float sample_rate_;
    unsigned n_harmonics_ = 0;
    unsigned m_ = 1;
    float rate_ = 0.0f;
    float phase_ = 0.0f;
    float p_ = 100.0f;
    float C2_ = 0.01f;
    float a_ = 0.0f;
    float state_ = 0.0f;
};

// BLIT-based square wave oscillator (bipolar sinc train with DC blocker)
class BlitSquare {
public:
    BlitSquare(float sample_rate = 44100.0f, float frequency = 220.0f)
        : sample_rate_(sample_rate) {
        reset();
        set_frequency(frequency);
    }

    void set_sample_rate(float sr) { sample_rate_ = sr; set_frequency(get_frequency()); }
    void set_frequency(float hz) {
        p_ = 0.5f * sample_rate_ / hz;
        rate_ = (float)M_PI / p_;
        update_harmonics();
    }
    float get_frequency() const { return 0.5f * sample_rate_ / p_; }

    void set_harmonics(unsigned n) { n_harmonics_ = n; update_harmonics(); }

    void reset() {
        phase_ = 0.0f;
        y_ = 0.0f;
        dcb_state_ = 0.0f;
        last_blit_ = 0.0f;
        n_harmonics_ = 0;
    }

    float tick() {
        float temp = last_blit_;
        float denominator = std::sin(phase_);
        if (std::fabs(denominator) < 1e-7f) {
            if (phase_ < 0.1f || phase_ > 2.0f * (float)M_PI - 0.1f)
                last_blit_ = a_;
            else
                last_blit_ = -a_;
        } else {
            last_blit_ = std::sin(m_ * phase_);
            last_blit_ /= p_ * denominator;
        }
        last_blit_ += temp;
        // DC blocker
        y_ = last_blit_ - dcb_state_ + 0.999f * y_;
        dcb_state_ = last_blit_;
        phase_ += rate_;
        if (phase_ >= 2.0f * (float)M_PI) phase_ -= 2.0f * (float)M_PI;
        return y_;
    }

    void generate(float *out, unsigned count) {
        for (unsigned i = 0; i < count; ++i) out[i] = tick();
    }

private:
    void update_harmonics() {
        if (n_harmonics_ <= 0) {
            unsigned max_h = (unsigned)std::floor(0.5f * p_);
            m_ = 2 * (max_h + 1);
        } else {
            m_ = 2 * (n_harmonics_ + 1);
        }
        a_ = (float)m_ / p_;
    }

    float sample_rate_;
    unsigned n_harmonics_ = 0;
    unsigned m_ = 2;
    float rate_ = 0.0f;
    float phase_ = 0.0f;
    float p_ = 100.0f;
    float a_ = 0.0f;
    float last_blit_ = 0.0f;
    float dcb_state_ = 0.0f;
    float y_ = 0.0f;
};

// DPW (Differentiated Parabolic Wave) sawtooth oscillator
class DPWSaw {
public:
    DPWSaw(float sample_rate = 44100.0f, float frequency = 440.0f)
        : fs_(sample_rate) {
        reset();
        set_frequency(frequency);
    }

    void set_sample_rate(float sr) { fs_ = sr; set_frequency(freq_); }
    void set_frequency(float hz) {
        freq_ = hz;
        inc_ = hz / fs_;
        scale_ = fs_ / (4.0f * hz);
    }
    float get_frequency() const { return freq_; }

    void reset() {
        phase_ = 0.0f;
        // Seed last_value_ to parabolic value at phase=0 to avoid
        // first-sample transient from the differentiator
        float v = phase_ * 2.0f - 1.0f;
        last_value_ = v * v;
    }

    float tick() {
        float value = phase_ * 2.0f - 1.0f;
        value = value * value;
        float out = scale_ * (value - last_value_);
        last_value_ = value;
        phase_ = std::fmod(phase_ + inc_, 1.0f);
        return out;
    }

    void generate(float *out, unsigned count) {
        for (unsigned i = 0; i < count; ++i) out[i] = tick();
    }

private:
    float fs_, freq_ = 440.0f;
    float inc_ = 0.0f;
    float scale_ = 1.0f;
    float phase_ = 0.0f;
    float last_value_ = 0.0f;
};

// DPW pulse oscillator (two saws subtracted for variable-width pulse)
class DPWPulse {
public:
    DPWPulse(float sample_rate = 44100.0f, float frequency = 440.0f)
        : fs_(sample_rate) {
        reset();
        set_frequency(frequency);
    }

    void set_sample_rate(float sr) { fs_ = sr; set_frequency(freq_); }
    void set_frequency(float hz) {
        freq_ = hz;
        inc_ = hz / fs_;
        scale_ = 0.5f * fs_ / (4.0f * hz);
    }
    float get_frequency() const { return freq_; }

    void set_duty(float d) {
        duty_ = std::fmax(0.01f, std::fmin(0.99f, d));
    }
    float get_duty() const { return duty_; }

    void reset() {
        pos_a_ = 0.0f;
        pos_b_ = 0.5f;
        duty_ = 0.5f;
        // Seed differentiator state to avoid first-sample transient
        float va = pos_a_ * 2.0f - 1.0f;
        float vb = pos_b_ * 2.0f - 1.0f;
        last_a_ = va * va;
        last_b_ = vb * vb;
    }

    float tick() {
        pos_a_ = std::fmod(pos_a_, 1.0f);
        pos_b_ = std::fmod(pos_b_, 1.0f);
        float va = pos_a_ * 2.0f - 1.0f;
        float vb = pos_b_ * 2.0f - 1.0f;
        va *= va;
        vb *= vb;
        float out = ((va - last_a_) - (vb - last_b_)) * scale_;
        last_a_ = va;
        last_b_ = vb;
        pos_a_ += inc_;
        pos_b_ += inc_;
        return out;
    }

    void generate(float *out, unsigned count) {
        for (unsigned i = 0; i < count; ++i) out[i] = tick();
    }

private:
    float fs_, freq_ = 440.0f;
    float inc_ = 0.0f;
    float scale_ = 1.0f;
    float duty_ = 0.5f;
    float pos_a_ = 0.0f, pos_b_ = 0.5f;
    float last_a_ = 0.0f, last_b_ = 0.0f;
};

} // namespace bloscillators

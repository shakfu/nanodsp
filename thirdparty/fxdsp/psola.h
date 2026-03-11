// PSOLA (Pitch-Synchronous Overlap-Add) pitch shifter
// Based on: FX/FXPsola.hpp by Xinyuan Lai
// Cleaned: removed static members, debug output, raw pointers -> vectors.
#pragma once

#include <cmath>
#include <vector>
#include <algorithm>
#include <cstring>

namespace fxdsp {

class PsolaShifter {
public:
    // Process an entire buffer, shifting pitch by semitones.
    // Returns a new buffer of the same length.
    static std::vector<float> process(const float* input, int length,
                                       float sample_rate, float semitones) {
        if (length <= 0 || semitones == 0.0f) {
            return std::vector<float>(input, input + std::max(0, length));
        }

        const int block_size = 2048;
        const int hop_size = 1024;
        const float freq_upper = 500.0f;
        const int median_len = 6;
        int n_blocks = (length - block_size) / hop_size + 1;
        if (n_blocks <= 0) {
            return std::vector<float>(input, input + length);
        }

        // 1. Pitch detection via half-autocorrelation
        std::vector<float> f0(n_blocks, 0.0f);
        std::vector<float> block_rms(n_blocks, 0.0f);
        std::vector<float> acf(block_size, 0.0f);
        int min_lag = (int)std::floor(sample_rate / freq_upper);
        float rms_max = 0.0f;

        for (int n = 0; n < n_blocks; ++n) {
            int start = n * hop_size;

            // RMS
            float rms = 0.0f;
            for (int i = 0; i < block_size && start + i < length; ++i) {
                rms += input[start + i] * input[start + i];
            }
            block_rms[n] = std::sqrt(rms / block_size);
            rms_max = std::max(rms_max, block_rms[n]);

            // Half autocorrelation
            for (int i = 0; i < block_size; ++i) {
                acf[i] = 0.0f;
                for (int j = 0; j < block_size - i && start + i + j < length; ++j) {
                    acf[i] += input[start + j] * input[start + i + j];
                }
            }

            // Find zero crossing
            int zero_idx = 0;
            for (int i = 0; i < block_size - 1; ++i) {
                if (acf[i] * acf[i + 1] <= 0.0f) {
                    zero_idx = i;
                    break;
                }
            }

            int init_lag = std::max(min_lag, zero_idx);

            // Find peak after initial lag
            int max_idx = init_lag;
            for (int i = init_lag; i < block_size; ++i) {
                if (acf[i] > acf[max_idx]) max_idx = i;
            }

            f0[n] = (max_idx > 0) ? sample_rate / (float)max_idx : 0.0f;
        }

        // Median filter f0
        median_filter_f(f0.data(), n_blocks, median_len);

        // Suppress unvoiced
        for (int n = 0; n < n_blocks; ++n) {
            if (block_rms[n] < 0.05f * rms_max)
                f0[n] = 0.0f;
        }

        // 2. Get pitch marks
        std::vector<int> p0(n_blocks);
        std::vector<int> pitch_marks;
        std::vector<int> mark_intervals;

        for (int n = 0; n < n_blocks; ++n) {
            if (f0[n] <= 0.0f) {
                f0[n] = (n > 0) ? f0[n - 1] : 150.0f;
            }
            if (f0[n] <= 0.0f) f0[n] = 150.0f;

            p0[n] = (int)std::round(sample_rate / f0[n]);
            if (p0[n] <= 0) p0[n] = 1;

            if (n == 0) {
                int local_mark = 0;
                for (int i = 0; i < p0[n] && i < length; ++i) {
                    if (std::abs(input[local_mark]) < std::abs(input[i]))
                        local_mark = i;
                }
                pitch_marks.push_back(local_mark);
            }

            int local_mark = pitch_marks.back();
            while (local_mark + p0[n] <= block_size + n * hop_size && local_mark + p0[n] < length) {
                local_mark += p0[n];
                pitch_marks.push_back(local_mark);
                mark_intervals.push_back(p0[n]);
            }
        }

        if (pitch_marks.size() < 2 || mark_intervals.empty()) {
            return std::vector<float>(input, input + length);
        }

        // 3. Resynthesis with pitch shift
        float shift_ratio = std::pow(2.0f, semitones / 12.0f);
        std::vector<float> output(length, 0.0f);

        // Ensure mark_intervals is same length as pitch_marks
        while (mark_intervals.size() < pitch_marks.size()) {
            mark_intervals.push_back(mark_intervals.back());
        }

        // Trim edge marks
        if (pitch_marks.front() < mark_intervals.front()) {
            pitch_marks.erase(pitch_marks.begin());
            mark_intervals.erase(mark_intervals.begin());
        }
        if (pitch_marks.empty() || mark_intervals.empty()) {
            return std::vector<float>(input, input + length);
        }
        if (pitch_marks.back() + mark_intervals.back() >= length) {
            pitch_marks.pop_back();
            mark_intervals.pop_back();
        }
        if (pitch_marks.empty() || mark_intervals.empty()) {
            return std::vector<float>(input, input + length);
        }

        int out_ptr = mark_intervals.front();
        std::vector<float> hann;

        while (out_ptr < length) {
            // Find nearest pitch mark
            int sel = 0;
            int min_dist = std::abs(pitch_marks[0] - out_ptr);
            for (int i = 1; i < (int)pitch_marks.size(); ++i) {
                int d = std::abs(pitch_marks[i] - out_ptr);
                if (d < min_dist) { min_dist = d; sel = i; }
            }

            int interval = mark_intervals[sel];
            if (interval <= 0) interval = 1;
            int win_len = 2 * interval + 1;

            // Generate Hann window
            hann.resize(win_len);
            for (int i = 0; i < win_len; ++i) {
                hann[i] = 0.5f * (1.0f - std::cos(2.0f * pi_ * i / (win_len - 1)));
            }

            // Overlap-add grain
            for (int i = -interval; i <= interval; ++i) {
                int in_idx = pitch_marks[sel] + i;
                int out_idx = out_ptr + i;
                int h_idx = interval + i;
                if (in_idx >= 0 && in_idx < length && out_idx >= 0 && out_idx < length) {
                    output[out_idx] += input[in_idx] * hann[h_idx];
                }
            }

            out_ptr += std::max(1, (int)std::round(interval / shift_ratio));
        }

        return output;
    }

private:
    static constexpr float pi_ = 3.14159265358979f;

    static void median_filter_f(float* arr, int len, int filt_len) {
        if (len <= filt_len) return;
        std::vector<float> win(filt_len);
        int half = filt_len / 2;
        for (int n = half; n < len - half; ++n) {
            for (int i = 0; i < filt_len; ++i)
                win[i] = arr[n - half + i];
            std::sort(win.begin(), win.end());
            arr[n] = win[half];
        }
    }
};

}  // namespace fxdsp

#include "_core_common.h"

#include "mldsp.h"

using namespace ml;

static constexpr size_t kBlock = kFloatsPerDSPVector;  // 64

// Helper: process arbitrary-length ArrayF through a DSPVector-based functor
template <typename Fn>
NpF1 ml_process(ArrayF input, Fn fn) {
    size_t n = input.shape(0);
    auto *out = new float[n];
    const float *in = input.data();
    { nb::gil_scoped_release rel;
      size_t pos = 0;
      while (pos + kBlock <= n) {
          DSPVector vIn(in + pos);
          DSPVector vOut = fn(vIn);
          std::copy(vOut.getConstBuffer(), vOut.getConstBuffer() + kBlock, out + pos);
          pos += kBlock;
      }
      if (pos < n) {
          size_t rem = n - pos;
          float padded[kBlock] = {};
          std::copy(in + pos, in + n, padded);
          DSPVector vIn(padded);
          DSPVector vOut = fn(vIn);
          std::copy(vOut.getConstBuffer(), vOut.getConstBuffer() + rem, out + pos);
      }
    }
    return make_f1(out, n);
}

// Helper: process arbitrary-length mono input -> stereo output [2, n] via DSPVector functor
template <typename Fn>
NpF2 ml_process_stereo(ArrayF input, Fn fn) {
    size_t n = input.shape(0);
    auto *out = new float[2 * n];
    const float *in = input.data();
    { nb::gil_scoped_release rel;
      size_t pos = 0;
      while (pos + kBlock <= n) {
          DSPVector vIn(in + pos);
          DSPVectorArray<2> vOut = fn(vIn);
          const float *left = vOut.getConstBuffer();
          const float *right = vOut.getConstBuffer() + kBlock;
          std::copy(left, left + kBlock, out + pos);
          std::copy(right, right + kBlock, out + n + pos);
          pos += kBlock;
      }
      if (pos < n) {
          size_t rem = n - pos;
          float padded[kBlock] = {};
          std::copy(in + pos, in + n, padded);
          DSPVector vIn(padded);
          DSPVectorArray<2> vOut = fn(vIn);
          const float *left = vOut.getConstBuffer();
          const float *right = vOut.getConstBuffer() + kBlock;
          std::copy(left, left + rem, out + pos);
          std::copy(right, right + rem, out + n + pos);
      }
    }
    return make_f2(out, 2, n);
}

// Helper: process two ArrayF inputs through a DSPVector-based functor
template <typename Fn>
NpF1 ml_process2(ArrayF input1, ArrayF input2, Fn fn) {
    size_t n = input1.shape(0);
    auto *out = new float[n];
    const float *in1 = input1.data();
    const float *in2 = input2.data();
    { nb::gil_scoped_release rel;
      size_t pos = 0;
      while (pos + kBlock <= n) {
          DSPVector v1(in1 + pos);
          DSPVector v2(in2 + pos);
          DSPVector vOut = fn(v1, v2);
          std::copy(vOut.getConstBuffer(), vOut.getConstBuffer() + kBlock, out + pos);
          pos += kBlock;
      }
      if (pos < n) {
          size_t rem = n - pos;
          float p1[kBlock] = {};
          float p2[kBlock] = {};
          std::copy(in1 + pos, in1 + n, p1);
          std::copy(in2 + pos, in2 + n, p2);
          DSPVector v1(p1);
          DSPVector v2(p2);
          DSPVector vOut = fn(v1, v2);
          std::copy(vOut.getConstBuffer(), vOut.getConstBuffer() + rem, out + pos);
      }
    }
    return make_f1(out, n);
}

// ============================================================================
// Projections
// ============================================================================

static void bind_projections(nb::module_ &mod) {
    // Each projection: scalar + vectorized overloads
#define BIND_PROJ(name, mlname) \
    mod.def(#name, [](float x) { return ml::projections::mlname(x); }, "x"_a); \
    mod.def(#name, [](ArrayF input) -> NpF1 { \
        size_t n = input.shape(0); \
        auto *out = new float[n]; \
        const float *in = input.data(); \
        { nb::gil_scoped_release rel; \
          for (size_t i = 0; i < n; ++i) out[i] = ml::projections::mlname(in[i]); \
        } \
        return make_f1(out, n); \
    }, "input"_a);

    BIND_PROJ(smoothstep, smoothstep)
    BIND_PROJ(bell, bell)
    BIND_PROJ(ease_in, easeIn)
    BIND_PROJ(ease_out, easeOut)
    BIND_PROJ(ease_in_out, easeInOut)
    BIND_PROJ(ease_in_cubic, easeInCubic)
    BIND_PROJ(ease_out_cubic, easeOutCubic)
    BIND_PROJ(ease_in_out_cubic, easeInOutCubic)
    BIND_PROJ(ease_in_quartic, easeInQuartic)
    BIND_PROJ(ease_out_quartic, easeOutQuartic)
    BIND_PROJ(ease_in_out_quartic, easeInOutQuartic)
    BIND_PROJ(overshoot, overshoot)
    BIND_PROJ(flip, flip)
    BIND_PROJ(squared, squared)
    BIND_PROJ(flatcenter, flatcenter)
    BIND_PROJ(bisquared, bisquared)
    BIND_PROJ(inv_bisquared, invBisquared)
    BIND_PROJ(clip, clip)

#undef BIND_PROJ
}

// ============================================================================
// Windows
// ============================================================================

static void bind_windows(nb::module_ &mod) {
#define BIND_WIN(name, mlname) \
    mod.def(#name, [](int size) -> NpF1 { \
        if (size < 1) throw std::invalid_argument("size must be >= 1"); \
        auto *out = new float[size]; \
        ml::makeWindow(out, (size_t)size, ml::dspwindows::mlname); \
        return make_f1(out, (size_t)size); \
    }, "size"_a);

    BIND_WIN(hamming, hamming)
    BIND_WIN(blackman, blackman)
    BIND_WIN(flat_top, flatTop)
    BIND_WIN(triangle, triangle)
    BIND_WIN(raised_cosine, raisedCosine)
    BIND_WIN(rectangle, rectangle)

#undef BIND_WIN
}

// ============================================================================
// FDN (Feedback Delay Network)
// ============================================================================

// Wrapper for FDN that properly manages delay buffer allocation.
// ml::FDN::setDelaysInSamples() only sets the read offset on the internal
// IntegerDelay objects but doesn't allocate their buffers (which requires
// setMaxDelayInSamples). Since mDelays is private, we use a wrapper that
// reimplements FDN with proper initialization.
template <int SIZE>
struct FDNWrapper {
    std::array<ml::IntegerDelay, SIZE> mDelays;
    std::array<ml::OnePole, SIZE> mFilters;
    std::array<ml::DSPVector, SIZE> mDelayInputVectors{};
    std::array<float, SIZE> mFeedbackGains{};

    void setDelaysInSamples(std::array<float, SIZE> times) {
        for (int n = 0; n < SIZE; ++n) {
            int len = (int)times[n] - (int)kFloatsPerDSPVector;
            if (len < 1) len = 1;
            mDelays[n].setMaxDelayInSamples(times[n]);
            mDelays[n].setDelayInSamples(len);
        }
    }

    void setFilterCutoffs(std::array<float, SIZE> omegas) {
        for (int n = 0; n < SIZE; ++n) {
            mFilters[n].coeffs = ml::OnePole::makeCoeffs(omegas[n]);
        }
    }

    DSPVectorArray<2> operator()(const DSPVector x) {
        // run delays
        for (int n = 0; n < SIZE; ++n) {
            mDelayInputVectors[n] = mDelays[n](mDelayInputVectors[n]);
        }

        // get stereo output sum
        DSPVector sumR, sumL;
        for (int n = 0; n < (SIZE & (~1)); ++n) {
            if (n & 1) sumL += mDelayInputVectors[n];
            else       sumR += mDelayInputVectors[n];
        }

        // Householder feedback matrix
        DSPVector sumOfDelays;
        for (int n = 0; n < SIZE; ++n) {
            sumOfDelays += mDelayInputVectors[n];
        }
        sumOfDelays *= DSPVector(2.0f / SIZE);

        for (int n = 0; n < SIZE; ++n) {
            mDelayInputVectors[n] -= sumOfDelays;
            mDelayInputVectors[n] = mFilters[n](mDelayInputVectors[n]) * DSPVector(mFeedbackGains[n]);
            mDelayInputVectors[n] += x;
        }

        return concatRows(sumL, sumR);
    }
};

template <int SIZE>
static void bind_fdn(nb::module_ &mod, const char *name, const char *doc) {
    using FDNType = FDNWrapper<SIZE>;

    nb::class_<FDNType>(mod, name, doc)
        .def(nb::init<>())
        .def("set_delays_in_samples", [](FDNType &self, std::vector<float> times) {
            if ((int)times.size() != SIZE)
                throw std::invalid_argument(
                    "Expected " + std::to_string(SIZE) + " delay times, got " +
                    std::to_string(times.size()));
            std::array<float, SIZE> arr;
            std::copy(times.begin(), times.end(), arr.begin());
            self.setDelaysInSamples(arr);
        }, "times"_a)
        .def("set_filter_cutoffs", [](FDNType &self, std::vector<float> omegas) {
            if ((int)omegas.size() != SIZE)
                throw std::invalid_argument(
                    "Expected " + std::to_string(SIZE) + " cutoffs, got " +
                    std::to_string(omegas.size()));
            std::array<float, SIZE> arr;
            std::copy(omegas.begin(), omegas.end(), arr.begin());
            self.setFilterCutoffs(arr);
        }, "omegas"_a)
        .def("set_feedback_gains", [](FDNType &self, std::vector<float> gains) {
            if ((int)gains.size() != SIZE)
                throw std::invalid_argument(
                    "Expected " + std::to_string(SIZE) + " gains, got " +
                    std::to_string(gains.size()));
            std::copy(gains.begin(), gains.end(), self.mFeedbackGains.begin());
        }, "gains"_a)
        .def("get_feedback_gains", [](FDNType &self) {
            return std::vector<float>(self.mFeedbackGains.begin(), self.mFeedbackGains.end());
        })
        .def("process", [](FDNType &self, ArrayF input) -> NpF2 {
            return ml_process_stereo(input, [&](const DSPVector &v) {
                return self(v);
            });
        }, "input"_a);
}

// ============================================================================
// PitchbendableDelay
// ============================================================================

static void bind_pitchbendable_delay(nb::module_ &mod) {
    nb::class_<ml::PitchbendableDelay>(mod, "PitchbendableDelay",
        "Click-free modulating delay with internal crossfading")
        .def(nb::init<>())
        .def("set_max_delay_in_samples",
             &ml::PitchbendableDelay::setMaxDelayInSamples, "max_delay"_a)
        .def("clear", &ml::PitchbendableDelay::clear)
        .def("process", [](ml::PitchbendableDelay &self,
                           ArrayF input, ArrayF delay_samples) -> NpF1 {
            if (input.shape(0) != delay_samples.shape(0))
                throw std::invalid_argument(
                    "input and delay_samples must have the same length");
            return ml_process2(input, delay_samples,
                [&](const DSPVector &vIn, const DSPVector &vDelay) {
                    return self(vIn, vDelay);
                });
        }, "input"_a, "delay_samples"_a);
}

// ============================================================================
// Downsampler / Upsampler
// ============================================================================

static void bind_resampling(nb::module_ &mod) {
    nb::class_<ml::Downsampler>(mod, "Downsampler",
        "Octave downsampling via cascaded half-band filters")
        .def(nb::init<int>(), "octaves"_a)
        .def("clear", &ml::Downsampler::clear)
        .def("process", [](ml::Downsampler &self, ArrayF input) -> NpF1 {
            size_t n = input.shape(0);
            if (n % kBlock != 0)
                throw std::invalid_argument(
                    "Input length must be a multiple of 64 (BLOCK_SIZE)");

            const float *in = input.data();
            size_t blocks_in = n / kBlock;
            std::vector<float> result;
            result.reserve(n);

            { nb::gil_scoped_release rel;
              for (size_t i = 0; i < blocks_in; ++i) {
                  DSPVector v(in + i * kBlock);
                  if (self.write(v)) {
                      DSPVector out = self.read();
                      const float *buf = out.getConstBuffer();
                      result.insert(result.end(), buf, buf + kBlock);
                  }
              }
            }

            size_t out_n = result.size();
            auto *out = new float[out_n];
            std::copy(result.begin(), result.end(), out);
            return make_f1(out, out_n);
        }, "input"_a);

    nb::class_<ml::Upsampler>(mod, "Upsampler",
        "Octave upsampling via cascaded half-band filters")
        .def(nb::init<int>(), "octaves"_a)
        .def("clear", &ml::Upsampler::clear)
        .def("process", [](ml::Upsampler &self, ArrayF input, int octaves) -> NpF1 {
            size_t n = input.shape(0);
            if (n % kBlock != 0)
                throw std::invalid_argument(
                    "Input length must be a multiple of 64 (BLOCK_SIZE)");

            const float *in = input.data();
            size_t blocks_in = n / kBlock;
            int reads_per_write = 1 << octaves;
            size_t out_n = n * reads_per_write;
            auto *out = new float[out_n];
            size_t out_pos = 0;

            { nb::gil_scoped_release rel;
              for (size_t i = 0; i < blocks_in; ++i) {
                  DSPVector v(in + i * kBlock);
                  self.write(v);
                  for (int r = 0; r < reads_per_write; ++r) {
                      DSPVector rd = self.read();
                      const float *buf = rd.getConstBuffer();
                      std::copy(buf, buf + kBlock, out + out_pos);
                      out_pos += kBlock;
                  }
              }
            }

            return make_f1(out, out_n);
        }, "input"_a, "octaves"_a);
}

// ============================================================================
// Generators
// ============================================================================

static void bind_generators(nb::module_ &mod) {
    // OneShotGen
    nb::class_<ml::OneShotGen>(mod, "OneShotGen",
        "Trigger-based 0-to-1 ramp generator")
        .def(nb::init<>())
        .def("trigger", &ml::OneShotGen::trigger)
        .def("process_sample",
             &ml::OneShotGen::nextSample, "cycles_per_sample"_a)
        .def("process", [](ml::OneShotGen &self, ArrayF cps) -> NpF1 {
            return ml_process(cps, [&](const DSPVector &v) {
                return self(v);
            });
        }, "cycles_per_sample"_a);

    // LinearGlide
    nb::class_<ml::LinearGlide>(mod, "LinearGlide",
        "Vector-quantized linear parameter smoothing")
        .def(nb::init<>())
        .def("set_glide_time_in_samples",
             &ml::LinearGlide::setGlideTimeInSamples, "time"_a)
        .def("set_value", &ml::LinearGlide::setValue, "value"_a)
        .def("clear", &ml::LinearGlide::clear)
        .def("process", [](ml::LinearGlide &self, float target, int n) -> NpF1 {
            if (n < 0) throw std::invalid_argument("n must be >= 0");
            auto *out = new float[n];
            { nb::gil_scoped_release rel;
              size_t pos = 0;
              while (pos + kBlock <= (size_t)n) {
                  DSPVector v = self(target);
                  std::copy(v.getConstBuffer(), v.getConstBuffer() + kBlock, out + pos);
                  pos += kBlock;
              }
              if (pos < (size_t)n) {
                  DSPVector v = self(target);
                  std::copy(v.getConstBuffer(), v.getConstBuffer() + ((size_t)n - pos), out + pos);
              }
            }
            return make_f1(out, (size_t)n);
        }, "target"_a, "n"_a);

    // SampleAccurateLinearGlide
    nb::class_<ml::SampleAccurateLinearGlide>(mod, "SampleAccurateLinearGlide",
        "Sample-accurate linear parameter smoothing")
        .def(nb::init<>())
        .def("set_glide_time_in_samples",
             &ml::SampleAccurateLinearGlide::setGlideTimeInSamples, "time"_a)
        .def("set_value", &ml::SampleAccurateLinearGlide::setValue, "value"_a)
        .def("clear", &ml::SampleAccurateLinearGlide::clear)
        .def("process_sample",
             &ml::SampleAccurateLinearGlide::nextSample, "target"_a)
        .def("process", [](ml::SampleAccurateLinearGlide &self,
                           float target, int n) -> NpF1 {
            if (n < 0) throw std::invalid_argument("n must be >= 0");
            auto *out = new float[n];
            { nb::gil_scoped_release rel;
              for (int i = 0; i < n; ++i)
                  out[i] = self.nextSample(target);
            }
            return make_f1(out, (size_t)n);
        }, "target"_a, "n"_a);

    // TempoLock
    nb::class_<ml::TempoLock>(mod, "TempoLock",
        "PLL-based phase-coherent clock follower")
        .def(nb::init<>())
        .def("clear", &ml::TempoLock::clear)
        .def("process", [](ml::TempoLock &self, ArrayF phasor,
                           float ratio, float inv_sr) -> NpF1 {
            return ml_process(phasor, [&](const DSPVector &v) {
                return self(v, ratio, inv_sr);
            });
        }, "phasor"_a, "ratio"_a, "inv_sample_rate"_a);
}

// ============================================================================
// Module entry point
// ============================================================================

void bind_madronalib(nb::module_ &m) {
    auto mod = m.def_submodule("madronalib", "Madronalib DSP bindings");

    // Constants
    mod.attr("BLOCK_SIZE") = (int)kBlock;

    // Scalar math: amp <-> dB (scalar + vectorized)
    mod.def("amp_to_db", [](float a) { return ml::ampTodB(a); }, "amplitude"_a);
    mod.def("amp_to_db", [](ArrayF input) -> NpF1 {
        size_t n = input.shape(0);
        auto *out = new float[n];
        const float *in = input.data();
        { nb::gil_scoped_release rel;
          for (size_t i = 0; i < n; ++i) out[i] = ml::ampTodB(in[i]);
        }
        return make_f1(out, n);
    }, "amplitude"_a);

    mod.def("db_to_amp", [](float db) { return ml::dBToAmp(db); }, "db"_a);
    mod.def("db_to_amp", [](ArrayF input) -> NpF1 {
        size_t n = input.shape(0);
        auto *out = new float[n];
        const float *in = input.data();
        { nb::gil_scoped_release rel;
          for (size_t i = 0; i < n; ++i) out[i] = ml::dBToAmp(in[i]);
        }
        return make_f1(out, n);
    }, "db"_a);

    // Submodules
    auto reverbs = mod.def_submodule("reverbs", "Feedback Delay Network reverbs");
    bind_fdn<4>(reverbs, "FDN4", "4-line Feedback Delay Network (mono in, stereo out)");
    bind_fdn<8>(reverbs, "FDN8", "8-line Feedback Delay Network (mono in, stereo out)");
    bind_fdn<16>(reverbs, "FDN16", "16-line Feedback Delay Network (mono in, stereo out)");

    auto delays = mod.def_submodule("delays", "Madronalib delay processors");
    bind_pitchbendable_delay(delays);

    auto resampling = mod.def_submodule("resampling", "Octave resampling");
    bind_resampling(resampling);

    auto generators = mod.def_submodule("generators", "Signal generators and parameter smoothing");
    bind_generators(generators);

    auto projections = mod.def_submodule("projections", "Easing and shaping functions");
    bind_projections(projections);

    auto windows = mod.def_submodule("windows", "Window functions");
    bind_windows(windows);
}

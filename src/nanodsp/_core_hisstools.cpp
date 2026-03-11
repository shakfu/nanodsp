#include "_core_common.h"

#include "WindowFunctions.hpp"
#include "Statistics.hpp"
#include "PartialTracker.hpp"
#include "SpectralProcessor.hpp"
#include "KernelSmoother.hpp"
#include "MonoConvolve.h"
#include "Convolver.h"
#include "ConvolveErrors.h"

// ---------------------------------------------------------------------------
// Helper: check ConvolveError and throw Python exception
// ---------------------------------------------------------------------------
static void check_convolve_error(ConvolveError err) {
    switch (err) {
        case CONVOLVE_ERR_NONE: return;
        case CONVOLVE_ERR_IN_CHAN_OUT_OF_RANGE:
            throw std::out_of_range("input channel out of range");
        case CONVOLVE_ERR_OUT_CHAN_OUT_OF_RANGE:
            throw std::out_of_range("output channel out of range");
        case CONVOLVE_ERR_MEM_UNAVAILABLE:
            throw std::runtime_error("convolver memory unavailable");
        case CONVOLVE_ERR_MEM_ALLOC_TOO_SMALL:
            throw std::runtime_error("convolver memory allocation too small");
        case CONVOLVE_ERR_TIME_IMPULSE_TOO_LONG:
            throw std::runtime_error("time-domain impulse too long");
        case CONVOLVE_ERR_TIME_LENGTH_OUT_OF_RANGE:
            throw std::runtime_error("time-domain length out of range");
        default:
            throw std::runtime_error("convolver error (code " + std::to_string(err) + ")");
    }
}

// ---------------------------------------------------------------------------
// Wrapper: MonoConvolve -- manages temp buffer internally
// ---------------------------------------------------------------------------
struct MonoConvolveWrapper {
    HISSTools::MonoConvolve conv;
    std::vector<float> temp_buf;

    MonoConvolveWrapper(size_t maxLength, int latency)
        : conv(maxLength, static_cast<LatencyMode>(latency)) {}

    void set_ir(ArrayF ir, bool resize) {
        auto err = conv.set(ir.data(), ir.shape(0), resize);
        check_convolve_error(err);
    }

    void reset() { conv.reset(); }

    NpF1 process(ArrayF input) {
        size_t n = input.shape(0);
        const float *in_data = input.data();
        temp_buf.resize(n);
        auto *out = new float[n];
        { nb::gil_scoped_release rel;
          conv.process(in_data, temp_buf.data(), out, n);
        }
        return make_f1(out, n);
    }
};

// ---------------------------------------------------------------------------
// Wrapper: Convolver -- stores numIns/numOuts, converts 2D numpy
// ---------------------------------------------------------------------------
struct ConvolverWrapper {
    HISSTools::Convolver conv;
    uint32_t numIns, numOuts;

    ConvolverWrapper(uint32_t ins, uint32_t outs, int latency)
        : conv(ins, outs, static_cast<LatencyMode>(latency)), numIns(ins), numOuts(outs) {}

    void set_ir(int in_chan, int out_chan, ArrayF ir, bool resize) {
        auto err = conv.set(static_cast<uint32_t>(in_chan),
                            static_cast<uint32_t>(out_chan),
                            ir.data(), ir.shape(0), resize);
        check_convolve_error(err);
    }

    void clear(bool resize) { conv.clear(resize); }
    void reset() { conv.reset(); }

    uint32_t get_num_ins() const { return numIns; }
    uint32_t get_num_outs() const { return numOuts; }

    NpF2 process(Array2F input) {
        size_t in_chans = input.shape(0);
        size_t n = input.shape(1);
        if (in_chans != numIns)
            throw std::invalid_argument(
                "input has " + std::to_string(in_chans) +
                " channels, expected " + std::to_string(numIns));

        // Build input pointer array
        std::vector<const float*> in_ptrs(numIns);
        for (uint32_t c = 0; c < numIns; c++)
            in_ptrs[c] = input.data() + c * n;

        // Allocate output
        auto *out = new float[numOuts * n];
        std::vector<float*> out_ptrs(numOuts);
        for (uint32_t c = 0; c < numOuts; c++)
            out_ptrs[c] = out + c * n;

        { nb::gil_scoped_release rel;
          conv.process(in_ptrs.data(), out_ptrs.data(), numIns, numOuts, n);
        }
        return make_f2(out, numOuts, n);
    }
};

// ---------------------------------------------------------------------------
// Window generation macro
// ---------------------------------------------------------------------------
#define BIND_WINDOW(sub, name) \
    sub.def(#name, [](int size) { \
        if (size < 1) throw std::invalid_argument("size must be >= 1"); \
        auto *data = new float[size]; \
        window_functions::name<float>(data, size - 1, 0, size, window_functions::params()); \
        return make_f1(data, size); \
    }, nb::arg("size"))

#define BIND_WINDOW_PARAM1(sub, name, p0name) \
    sub.def(#name, [](int size, double p0) { \
        if (size < 1) throw std::invalid_argument("size must be >= 1"); \
        auto *data = new float[size]; \
        window_functions::name<float>(data, size - 1, 0, size, window_functions::params(p0)); \
        return make_f1(data, size); \
    }, nb::arg("size"), nb::arg(p0name))

#define BIND_WINDOW_PARAM2(sub, name, p0name, p1name) \
    sub.def(#name, [](int size, double p0, double p1) { \
        if (size < 1) throw std::invalid_argument("size must be >= 1"); \
        auto *data = new float[size]; \
        window_functions::name<float>(data, size - 1, 0, size, window_functions::params(p0, p1)); \
        return make_f1(data, size); \
    }, nb::arg("size"), nb::arg(p0name), nb::arg(p1name))

// ---------------------------------------------------------------------------
// Statistics binding macro
// ---------------------------------------------------------------------------
#define BIND_STAT(sub, name) \
    sub.def(#name, [](ArrayF input) { \
        return name(input.data(), input.shape(0)); \
    }, nb::arg("input"))

// ---------------------------------------------------------------------------
// bind_hisstools
// ---------------------------------------------------------------------------
void bind_hisstools(nb::module_ &m) {
    auto ht = m.def_submodule("hisstools", "HISSTools Library: SIMD-optimized DSP");

    // ===== convolution submodule =====
    auto conv_sub = ht.def_submodule("convolution", "Multichannel partitioned convolution");

    nb::enum_<LatencyMode>(conv_sub, "LatencyMode")
        .value("zero", kLatencyZero)
        .value("short_", kLatencyShort)
        .value("medium", kLatencyMedium);

    nb::class_<MonoConvolveWrapper>(conv_sub, "MonoConvolve")
        .def(nb::init<size_t, int>(), nb::arg("max_length"), nb::arg("latency") = 0)
        .def("set_ir", &MonoConvolveWrapper::set_ir, nb::arg("ir"), nb::arg("resize") = true)
        .def("reset", &MonoConvolveWrapper::reset)
        .def("process", &MonoConvolveWrapper::process, nb::arg("input"));

    nb::class_<ConvolverWrapper>(conv_sub, "Convolver")
        .def(nb::init<uint32_t, uint32_t, int>(),
             nb::arg("num_ins"), nb::arg("num_outs"), nb::arg("latency") = 0)
        .def("set_ir", &ConvolverWrapper::set_ir,
             nb::arg("in_chan"), nb::arg("out_chan"), nb::arg("ir"), nb::arg("resize") = true)
        .def("clear", &ConvolverWrapper::clear, nb::arg("resize") = false)
        .def("reset", &ConvolverWrapper::reset)
        .def_prop_ro("num_ins", &ConvolverWrapper::get_num_ins)
        .def_prop_ro("num_outs", &ConvolverWrapper::get_num_outs)
        .def("process", &ConvolverWrapper::process, nb::arg("input"));

    // ===== spectral submodule =====
    auto spec_sub = ht.def_submodule("spectral", "Spectral processing");

    using SP = spectral_processor<double>;
    using SPEdge = SP::EdgeMode;

    nb::enum_<SPEdge>(spec_sub, "EdgeMode")
        .value("linear", SPEdge::Linear)
        .value("wrap", SPEdge::Wrap)
        .value("wrap_centre", SPEdge::WrapCentre)
        .value("fold", SPEdge::Fold)
        .value("fold_repeat", SPEdge::FoldRepeat);

    nb::class_<SP>(spec_sub, "SpectralProcessor")
        .def(nb::init<uintptr_t>(), nb::arg("max_fft_size") = 32768)
        .def("set_max_fft_size", &SP::set_max_fft_size, nb::arg("size"))
        .def("max_fft_size", &SP::max_fft_size)
        .def("convolve", [](SP &self, ArrayF in1, ArrayF in2, SPEdge mode) {
            size_t n1 = in1.shape(0), n2 = in2.shape(0);
            uintptr_t out_size = self.convolved_size(n1, n2, mode);
            if (!out_size) throw std::runtime_error("convolved_size is 0 (check max_fft_size)");
            const float *p1 = in1.data(), *p2 = in2.data();
            auto *out = new float[out_size];
            { nb::gil_scoped_release rel;
              std::vector<double> d1(n1), d2(n2);
              for (size_t i = 0; i < n1; i++) d1[i] = p1[i];
              for (size_t i = 0; i < n2; i++) d2[i] = p2[i];
              std::vector<double> dout(out_size);
              SP::in_ptr dp1(d1.data(), n1);
              SP::in_ptr dp2(d2.data(), n2);
              self.convolve(dout.data(), dp1, dp2, mode);
              for (size_t i = 0; i < out_size; i++) out[i] = static_cast<float>(dout[i]);
            }
            return make_f1(out, out_size);
        }, nb::arg("in1"), nb::arg("in2"), nb::arg("mode") = SPEdge::Linear)
        .def("correlate", [](SP &self, ArrayF in1, ArrayF in2, SPEdge mode) {
            size_t n1 = in1.shape(0), n2 = in2.shape(0);
            uintptr_t out_size = self.correlated_size(n1, n2, mode);
            if (!out_size) throw std::runtime_error("correlated_size is 0 (check max_fft_size)");
            const float *p1 = in1.data(), *p2 = in2.data();
            auto *out = new float[out_size];
            { nb::gil_scoped_release rel;
              std::vector<double> d1(n1), d2(n2);
              for (size_t i = 0; i < n1; i++) d1[i] = p1[i];
              for (size_t i = 0; i < n2; i++) d2[i] = p2[i];
              std::vector<double> dout(out_size);
              SP::in_ptr dp1(d1.data(), n1);
              SP::in_ptr dp2(d2.data(), n2);
              self.correlate(dout.data(), dp1, dp2, mode);
              for (size_t i = 0; i < out_size; i++) out[i] = static_cast<float>(dout[i]);
            }
            return make_f1(out, out_size);
        }, nb::arg("in1"), nb::arg("in2"), nb::arg("mode") = SPEdge::Linear)
        .def("change_phase", [](SP &self, ArrayF input, double phase, double time_mul) {
            size_t n = input.shape(0);
            const float *in_data = input.data();
            auto *out = new float[n];
            { nb::gil_scoped_release rel;
              std::vector<double> din(n);
              for (size_t i = 0; i < n; i++) din[i] = in_data[i];
              size_t out_size = static_cast<size_t>(std::round(n * time_mul));
              if (out_size < 1) out_size = 1;
              uintptr_t fft_log2 = SP::calc_fft_size_log2(out_size);
              uintptr_t fft_size = uintptr_t(1) << fft_log2;
              std::vector<double> dout(fft_size, 0.0);
              self.change_phase(dout.data(), din.data(), n, phase, time_mul);
              for (size_t i = 0; i < n; i++) out[i] = static_cast<float>(dout[i]);
            }
            return make_f1(out, n);
        }, nb::arg("input"), nb::arg("phase"), nb::arg("time_multiplier") = 1.0)
        .def("convolved_size", &SP::convolved_size,
             nb::arg("size1"), nb::arg("size2"), nb::arg("mode") = SPEdge::Linear)
        .def("correlated_size", &SP::correlated_size,
             nb::arg("size1"), nb::arg("size2"), nb::arg("mode") = SPEdge::Linear);

    // KernelSmoother (uses double internally for SIMD)
    using KS = kernel_smoother<double>;
    using KSEdge = KS::EdgeMode;

    nb::enum_<KSEdge>(spec_sub, "SmoothEdgeMode")
        .value("zero_pad", KSEdge::ZeroPad)
        .value("extend", KSEdge::Extend)
        .value("wrap", KSEdge::Wrap)
        .value("fold", KSEdge::Fold)
        .value("mirror", KSEdge::Mirror);

    nb::class_<KS>(spec_sub, "KernelSmoother")
        .def(nb::init<uintptr_t>(), nb::arg("max_fft_size") = (1 << 18))
        .def("smooth", [](KS &self, ArrayF input, ArrayF kernel,
                          double width_lo, double width_hi, bool symmetric, KSEdge edges) {
            size_t n = input.shape(0);
            size_t kn = kernel.shape(0);
            const float *in_data = input.data();
            const float *k_data = kernel.data();
            auto *out = new float[n];
            { nb::gil_scoped_release rel;
              std::vector<double> din(n), dkernel(kn);
              for (size_t i = 0; i < n; i++) din[i] = in_data[i];
              for (size_t i = 0; i < kn; i++) dkernel[i] = k_data[i];
              std::vector<double> dout(n);
              self.smooth(dout.data(), din.data(), dkernel.data(), n, kn, width_lo, width_hi, symmetric, edges);
              for (size_t i = 0; i < n; i++) out[i] = static_cast<float>(dout[i]);
            }
            return make_f1(out, n);
        }, nb::arg("input"), nb::arg("kernel"),
           nb::arg("width_lo"), nb::arg("width_hi"),
           nb::arg("symmetric") = true, nb::arg("edges") = KSEdge::Fold);

    // ===== analysis submodule =====
    auto ana_sub = ht.def_submodule("analysis", "Statistical analysis and partial tracking");

    // Statistics
    BIND_STAT(ana_sub, stat_mean);
    BIND_STAT(ana_sub, stat_rms);
    BIND_STAT(ana_sub, stat_sum);
    BIND_STAT(ana_sub, stat_min);
    BIND_STAT(ana_sub, stat_max);
    BIND_STAT(ana_sub, stat_min_position);
    BIND_STAT(ana_sub, stat_max_position);
    BIND_STAT(ana_sub, stat_variance);
    BIND_STAT(ana_sub, stat_standard_deviation);
    BIND_STAT(ana_sub, stat_centroid);
    BIND_STAT(ana_sub, stat_spread);
    BIND_STAT(ana_sub, stat_skewness);
    BIND_STAT(ana_sub, stat_kurtosis);
    BIND_STAT(ana_sub, stat_flatness);
    BIND_STAT(ana_sub, stat_crest);
    BIND_STAT(ana_sub, stat_product);
    BIND_STAT(ana_sub, stat_geometric_mean);
    BIND_STAT(ana_sub, stat_sum_abs);
    BIND_STAT(ana_sub, stat_sum_squares);
    BIND_STAT(ana_sub, stat_mean_squares);
    BIND_STAT(ana_sub, stat_length);

    ana_sub.def("stat_pdf_percentile", [](ArrayF input, double centile) {
        return stat_pdf_percentile(input.data(), centile, input.shape(0));
    }, nb::arg("input"), nb::arg("centile"));

    ana_sub.def("stat_count_above", [](ArrayF input, float threshold) {
        return stat_count_above(input.data(), threshold, input.shape(0));
    }, nb::arg("input"), nb::arg("threshold"));

    ana_sub.def("stat_count_below", [](ArrayF input, float threshold) {
        return stat_count_below(input.data(), threshold, input.shape(0));
    }, nb::arg("input"), nb::arg("threshold"));

    // Peak / Track / PartialTracker
    using PeakF = peak<float>;
    using TrackF = track<float>;
    using TrackState = TrackF::State;
    using PT = partial_tracker<float>;

    nb::class_<PeakF>(ana_sub, "Peak")
        .def(nb::init<float, float>(), nb::arg("freq"), nb::arg("amp"))
        .def(nb::init<>())
        .def("freq", &PeakF::freq)
        .def("amp", &PeakF::amp)
        .def("pitch", &PeakF::pitch)
        .def("db", &PeakF::db);

    nb::enum_<TrackState>(ana_sub, "TrackState")
        .value("off", TrackState::Off)
        .value("start", TrackState::Start)
        .value("continue_", TrackState::Continue)
        .value("switch", TrackState::Switch);

    nb::class_<TrackF>(ana_sub, "Track")
        .def(nb::init<>())
        .def("active", &TrackF::active)
        .def_prop_ro("peak", [](const TrackF &t) { return t.m_peak; })
        .def_prop_ro("state", [](const TrackF &t) { return t.m_state; });

    nb::class_<PT>(ana_sub, "PartialTracker")
        .def(nb::init<size_t, size_t>(), nb::arg("max_tracks"), nb::arg("max_peaks"))
        .def("set_cost_calculation", &PT::set_cost_calculation,
             nb::arg("square_cost"), nb::arg("use_pitch"), nb::arg("use_db"))
        .def("set_cost_scaling", &PT::set_cost_scaling,
             nb::arg("freq_unit"), nb::arg("amp_unit"), nb::arg("max_cost"))
        .def("reset", &PT::reset)
        .def("process", [](PT &self, nb::list peaks_list, float start_threshold) {
            size_t n = nb::len(peaks_list);
            std::vector<PeakF> peaks(n);
            for (size_t i = 0; i < n; i++)
                peaks[i] = nb::cast<PeakF>(peaks_list[i]);
            self.process(peaks.data(), n, start_threshold);
        }, nb::arg("peaks"), nb::arg("start_threshold"))
        .def("get_track", &PT::get_track, nb::arg("index"), nb::rv_policy::reference_internal)
        .def("max_peaks", &PT::max_peaks)
        .def("max_tracks", &PT::max_tracks);

    // ===== windows submodule =====
    auto win_sub = ht.def_submodule("windows", "Window function generation");

    // Simple windows (no extra params)
    BIND_WINDOW(win_sub, rect);
    BIND_WINDOW(win_sub, triangle);
    BIND_WINDOW(win_sub, welch);
    BIND_WINDOW(win_sub, parzen);
    BIND_WINDOW(win_sub, sine);
    BIND_WINDOW(win_sub, hann);
    BIND_WINDOW(win_sub, hamming);
    BIND_WINDOW(win_sub, blackman);
    BIND_WINDOW(win_sub, exact_blackman);
    BIND_WINDOW(win_sub, blackman_harris_62dB);
    BIND_WINDOW(win_sub, blackman_harris_71dB);
    BIND_WINDOW(win_sub, blackman_harris_74dB);
    BIND_WINDOW(win_sub, blackman_harris_92dB);
    BIND_WINDOW(win_sub, nuttall_1st_64dB);
    BIND_WINDOW(win_sub, nuttall_1st_93dB);
    BIND_WINDOW(win_sub, nuttall_3rd_47dB);
    BIND_WINDOW(win_sub, nuttall_3rd_83dB);
    BIND_WINDOW(win_sub, nuttall_5th_61dB);
    BIND_WINDOW(win_sub, nuttall_minimal_71dB);
    BIND_WINDOW(win_sub, nuttall_minimal_98dB);
    BIND_WINDOW(win_sub, ni_flat_top);
    BIND_WINDOW(win_sub, hp_flat_top);
    BIND_WINDOW(win_sub, stanford_flat_top);
    BIND_WINDOW(win_sub, heinzel_flat_top_70dB);
    BIND_WINDOW(win_sub, heinzel_flat_top_90dB);
    BIND_WINDOW(win_sub, heinzel_flat_top_95dB);

    // Parameterized windows
    BIND_WINDOW_PARAM1(win_sub, tukey, "alpha");
    BIND_WINDOW_PARAM1(win_sub, kaiser, "beta");
    BIND_WINDOW_PARAM1(win_sub, sine_taper, "order");
    BIND_WINDOW_PARAM2(win_sub, trapezoid, "a", "b");

    // Cosine-sum windows (with user-supplied coefficients)
    win_sub.def("cosine_2_term", [](int size, double a0) {
        if (size < 1) throw std::invalid_argument("size must be >= 1");
        auto *data = new float[size];
        window_functions::cosine_2_term<float>(data, size - 1, 0, size, window_functions::params(a0));
        return make_f1(data, size);
    }, nb::arg("size"), nb::arg("a0"));

    win_sub.def("cosine_3_term", [](int size, double a0, double a1, double a2) {
        if (size < 1) throw std::invalid_argument("size must be >= 1");
        auto *data = new float[size];
        window_functions::cosine_3_term<float>(data, size - 1, 0, size, window_functions::params(a0, a1, a2));
        return make_f1(data, size);
    }, nb::arg("size"), nb::arg("a0"), nb::arg("a1"), nb::arg("a2"));

    win_sub.def("cosine_4_term", [](int size, double a0, double a1, double a2, double a3) {
        if (size < 1) throw std::invalid_argument("size must be >= 1");
        auto *data = new float[size];
        window_functions::cosine_4_term<float>(data, size - 1, 0, size, window_functions::params(a0, a1, a2, a3));
        return make_f1(data, size);
    }, nb::arg("size"), nb::arg("a0"), nb::arg("a1"), nb::arg("a2"), nb::arg("a3"));

    win_sub.def("cosine_5_term", [](int size, double a0, double a1, double a2, double a3, double a4) {
        if (size < 1) throw std::invalid_argument("size must be >= 1");
        auto *data = new float[size];
        window_functions::cosine_5_term<float>(data, size - 1, 0, size, window_functions::params(a0, a1, a2, a3, a4));
        return make_f1(data, size);
    }, nb::arg("size"), nb::arg("a0"), nb::arg("a1"), nb::arg("a2"), nb::arg("a3"), nb::arg("a4"));
}

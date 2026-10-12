#include "_core_common.h"

#include <signalsmith-dsp/filters.h>
#include <signalsmith-dsp/fft.h>
#include <signalsmith-dsp/delay.h>
#include <signalsmith-dsp/envelopes.h>
#include <signalsmith-dsp/spectral.h>
#include <signalsmith-dsp/rates.h>
#include <signalsmith-dsp/mix.h>

// ============================================================================
// Filters
// ============================================================================

static void bind_filters(nb::module_ &m) {
    auto filters = m.def_submodule("filters", "Biquad IIR filters");

    nb::enum_<signalsmith::filters::BiquadDesign>(filters, "BiquadDesign")
        .value("bilinear", signalsmith::filters::BiquadDesign::bilinear)
        .value("cookbook", signalsmith::filters::BiquadDesign::cookbook)
        .value("one_sided", signalsmith::filters::BiquadDesign::oneSided)
        .value("vicanek", signalsmith::filters::BiquadDesign::vicanek);

    using Biquad = signalsmith::filters::BiquadStatic<float>;

    nb::class_<Biquad>(filters, "Biquad",
        "A standard biquad filter. Frequencies are normalized (0 to 0.5, "
        "where 0.5 = Nyquist).")
        .def(nb::init<>())
        .def("lowpass", [](Biquad &s, double f, double o, signalsmith::filters::BiquadDesign d) -> Biquad& { s.lowpass(f,o,d); return s; },
            "freq"_a, "octaves"_a = Biquad::defaultBandwidth, "design"_a = signalsmith::filters::BiquadDesign::bilinear, nb::rv_policy::reference)
        .def("lowpass_q", [](Biquad &s, double f, double q, signalsmith::filters::BiquadDesign d) -> Biquad& { s.lowpassQ(f,q,d); return s; },
            "freq"_a, "q"_a, "design"_a = signalsmith::filters::BiquadDesign::bilinear, nb::rv_policy::reference)
        .def("highpass", [](Biquad &s, double f, double o, signalsmith::filters::BiquadDesign d) -> Biquad& { s.highpass(f,o,d); return s; },
            "freq"_a, "octaves"_a = Biquad::defaultBandwidth, "design"_a = signalsmith::filters::BiquadDesign::bilinear, nb::rv_policy::reference)
        .def("highpass_q", [](Biquad &s, double f, double q, signalsmith::filters::BiquadDesign d) -> Biquad& { s.highpassQ(f,q,d); return s; },
            "freq"_a, "q"_a, "design"_a = signalsmith::filters::BiquadDesign::bilinear, nb::rv_policy::reference)
        .def("bandpass", [](Biquad &s, double f, double o, signalsmith::filters::BiquadDesign d) -> Biquad& { s.bandpass(f,o,d); return s; },
            "freq"_a, "octaves"_a = Biquad::defaultBandwidth, "design"_a = signalsmith::filters::BiquadDesign::oneSided, nb::rv_policy::reference)
        .def("bandpass_q", [](Biquad &s, double f, double q, signalsmith::filters::BiquadDesign d) -> Biquad& { s.bandpassQ(f,q,d); return s; },
            "freq"_a, "q"_a, "design"_a = signalsmith::filters::BiquadDesign::oneSided, nb::rv_policy::reference)
        .def("notch", [](Biquad &s, double f, double o, signalsmith::filters::BiquadDesign d) -> Biquad& { s.notch(f,o,d); return s; },
            "freq"_a, "octaves"_a = Biquad::defaultBandwidth, "design"_a = signalsmith::filters::BiquadDesign::oneSided, nb::rv_policy::reference)
        .def("notch_q", [](Biquad &s, double f, double q, signalsmith::filters::BiquadDesign d) -> Biquad& { s.notchQ(f,q,d); return s; },
            "freq"_a, "q"_a, "design"_a = signalsmith::filters::BiquadDesign::oneSided, nb::rv_policy::reference)
        .def("peak", [](Biquad &s, double f, double g, double o, signalsmith::filters::BiquadDesign d) -> Biquad& { s.peak(f,g,o,d); return s; },
            "freq"_a, "gain"_a, "octaves"_a = 1.0, "design"_a = signalsmith::filters::BiquadDesign::oneSided, nb::rv_policy::reference)
        .def("peak_db", [](Biquad &s, double f, double db, double o, signalsmith::filters::BiquadDesign d) -> Biquad& { s.peakDb(f,db,o,d); return s; },
            "freq"_a, "db"_a, "octaves"_a = 1.0, "design"_a = signalsmith::filters::BiquadDesign::oneSided, nb::rv_policy::reference)
        .def("high_shelf", [](Biquad &s, double f, double g, double o, signalsmith::filters::BiquadDesign d) -> Biquad& { s.highShelf(f,g,o,d); return s; },
            "freq"_a, "gain"_a, "octaves"_a = Biquad::defaultBandwidth, "design"_a = signalsmith::filters::BiquadDesign::oneSided, nb::rv_policy::reference)
        .def("high_shelf_db", [](Biquad &s, double f, double db, double o, signalsmith::filters::BiquadDesign d) -> Biquad& { s.highShelfDb(f,db,o,d); return s; },
            "freq"_a, "db"_a, "octaves"_a = Biquad::defaultBandwidth, "design"_a = signalsmith::filters::BiquadDesign::oneSided, nb::rv_policy::reference)
        .def("low_shelf", [](Biquad &s, double f, double g, double o, signalsmith::filters::BiquadDesign d) -> Biquad& { s.lowShelf(f,g,o,d); return s; },
            "freq"_a, "gain"_a, "octaves"_a = 2.0, "design"_a = signalsmith::filters::BiquadDesign::oneSided, nb::rv_policy::reference)
        .def("low_shelf_db", [](Biquad &s, double f, double db, double o, signalsmith::filters::BiquadDesign d) -> Biquad& { s.lowShelfDb(f,db,o,d); return s; },
            "freq"_a, "db"_a, "octaves"_a = 2.0, "design"_a = signalsmith::filters::BiquadDesign::oneSided, nb::rv_policy::reference)
        .def("allpass", [](Biquad &s, double f, double o, signalsmith::filters::BiquadDesign d) -> Biquad& { s.allpass(f,o,d); return s; },
            "freq"_a, "octaves"_a = 1.0, "design"_a = signalsmith::filters::BiquadDesign::oneSided, nb::rv_policy::reference)
        .def("allpass_q", [](Biquad &s, double f, double q, signalsmith::filters::BiquadDesign d) -> Biquad& { s.allpassQ(f,q,d); return s; },
            "freq"_a, "q"_a, "design"_a = signalsmith::filters::BiquadDesign::oneSided, nb::rv_policy::reference)
        .def("add_gain", [](Biquad &s, double f) -> Biquad& { s.addGain(f); return s; }, "factor"_a, nb::rv_policy::reference)
        .def("add_gain_db", [](Biquad &s, double db) -> Biquad& { s.addGainDb(db); return s; }, "db"_a, nb::rv_policy::reference)
        .def("reset", &Biquad::reset)
        .def("response", &Biquad::response, "freq"_a)
        .def("response_db", &Biquad::responseDb, "freq"_a)
        .def("process", [](Biquad &self, ArrayF input) {
            size_t n = input.shape(0);
            auto *out = new float[n];
            const float *in = input.data();
            { nb::gil_scoped_release rel;
              for (size_t i = 0; i < n; ++i) out[i] = self(in[i]);
            }
            return make_f1(out, n);
        }, "input"_a, "Process a 1-D float32 buffer, returning filtered output.")
        .def("process_inplace", [](Biquad &self, ArrayF buf) {
            float *d = buf.data();
            size_t n = buf.shape(0);
            { nb::gil_scoped_release rel;
              for (size_t i = 0; i < n; ++i) d[i] = self(d[i]);
            }
        }, "buffer"_a, "Process a 1-D float32 buffer in-place.");
}

// ============================================================================
// FFT
// ============================================================================

static void bind_fft(nb::module_ &m) {
    auto mod = m.def_submodule("fft", "FFT (complex and real)");

    using CFFT = signalsmith::fft::FFT<float>;
    using RFFT = signalsmith::fft::RealFFT<float>;

    nb::class_<CFFT>(mod, "FFT", "Complex-to-complex FFT. Fast for sizes 2^a * 3^b.")
        .def(nb::init<size_t, int>(), "size"_a, "fast_direction"_a = 0)
        .def("size", &CFFT::size)
        .def("set_size", &CFFT::setSize, "size"_a)
        .def_static("fast_size_above", &CFFT::fastSizeAbove, "size"_a)
        .def_static("fast_size_below", &CFFT::fastSizeBelow, "size"_a)
        .def("fft", [](CFFT &self, ArrayCF input) {
            size_t n = self.size();
            if (input.shape(0) != n) throw std::invalid_argument("Input size must match FFT size");
            auto *out = new std::complex<float>[n];
            { nb::gil_scoped_release rel;
              self.fft(input.data(), out);
            }
            return make_cf1(out, n);
        }, "input"_a, "Forward complex FFT.")
        .def("ifft", [](CFFT &self, ArrayCF input) {
            size_t n = self.size();
            if (input.shape(0) != n) throw std::invalid_argument("Input size must match FFT size");
            auto *out = new std::complex<float>[n];
            { nb::gil_scoped_release rel;
              self.ifft(input.data(), out);
            }
            return make_cf1(out, n);
        }, "input"_a, "Inverse complex FFT (unscaled).");

    nb::class_<RFFT>(mod, "RealFFT", "Real-input FFT. Output has N/2 complex bins (modified real FFT).")
        .def(nb::init<size_t, int>(), "size"_a, "fast_direction"_a = 0)
        .def("size", &RFFT::size)
        .def("set_size", &RFFT::setSize, "size"_a)
        .def_static("fast_size_above", &RFFT::fastSizeAbove, "size"_a)
        .def_static("fast_size_below", &RFFT::fastSizeBelow, "size"_a)
        .def("fft", [](RFFT &self, ArrayF input) {
            size_t n = self.size();
            if (input.shape(0) != n) throw std::invalid_argument("Input size must match FFT size");
            size_t bins = n / 2;
            auto *out = new std::complex<float>[bins];
            { nb::gil_scoped_release rel;
              self.fft(input.data(), out);
            }
            return make_cf1(out, bins);
        }, "input"_a, "Forward real FFT.")
        .def("ifft", [](RFFT &self, ArrayCF input) {
            size_t n = self.size();
            size_t bins = n / 2;
            if (input.shape(0) != bins) throw std::invalid_argument("Input size must match N/2 bins");
            auto *out = new float[n];
            { nb::gil_scoped_release rel;
              self.ifft(input.data(), out);
            }
            return make_f1(out, n);
        }, "input"_a, "Inverse real FFT (unscaled).");
}

// ============================================================================
// Delay
// ============================================================================

static void bind_delay(nb::module_ &m) {
    auto mod = m.def_submodule("delay", "Delay line utilities");

    using DelayLinear = signalsmith::delay::Delay<float, signalsmith::delay::InterpolatorLinear>;
    using DelayCubic = signalsmith::delay::Delay<float, signalsmith::delay::InterpolatorCubic>;

    auto bind_delay_class = [&](auto *dummy, const char *name, const char *doc) {
        using D = std::remove_pointer_t<decltype(dummy)>;
        nb::class_<D>(mod, name, doc)
            .def(nb::init<int>(), "capacity"_a = 0)
            .def("reset", &D::reset, "value"_a = 0.0f)
            .def("resize", &D::resize, "capacity"_a, "value"_a = 0.0f)
            .def_prop_ro_static("latency", [](nb::handle) { return D::latency; })
            .def("process", [](D &self, ArrayF input, float delay_samples) {
                size_t n = input.shape(0);
                auto *out = new float[n];
                const float *in = input.data();
                { nb::gil_scoped_release rel;
                  for (size_t i = 0; i < n; ++i) { self.write(in[i]); out[i] = self.read(delay_samples); }
                }
                return make_f1(out, n);
            }, "input"_a, "delay_samples"_a, "Write input and read with fixed delay.")
            .def("process_varying", [](D &self, ArrayF input, ArrayF delays) {
                size_t n = input.shape(0);
                if (delays.shape(0) != n) throw std::invalid_argument("Input and delays must have same length");
                auto *out = new float[n];
                const float *in = input.data();
                const float *d = delays.data();
                { nb::gil_scoped_release rel;
                  for (size_t i = 0; i < n; ++i) { self.write(in[i]); out[i] = self.read(d[i]); }
                }
                return make_f1(out, n);
            }, "input"_a, "delays"_a, "Write input and read with per-sample varying delay.");
    };

    bind_delay_class(static_cast<DelayLinear*>(nullptr), "Delay", "Delay line with linear interpolation.");
    bind_delay_class(static_cast<DelayCubic*>(nullptr), "DelayCubic", "Delay line with cubic interpolation.");
}

// ============================================================================
// Envelopes
// ============================================================================

static void bind_envelopes(nb::module_ &m) {
    auto mod = m.def_submodule("envelopes", "Envelopes, LFOs, and smoothing filters");

    nb::class_<signalsmith::envelopes::CubicLfo>(mod, "CubicLfo",
        "An LFO based on cubic segments with optional rate/depth randomization.")
        .def(nb::init<>())
        .def(nb::init<long>(), "seed"_a)
        .def("reset", &signalsmith::envelopes::CubicLfo::reset)
        .def("set", &signalsmith::envelopes::CubicLfo::set,
            "low"_a, "high"_a, "rate"_a, "rate_variation"_a = 0.0f, "depth_variation"_a = 0.0f)
        .def("next", &signalsmith::envelopes::CubicLfo::next)
        .def("process", [](signalsmith::envelopes::CubicLfo &self, int n) {
            auto *out = new float[n];
            { nb::gil_scoped_release rel;
              for (int i = 0; i < n; ++i) out[i] = self.next();
            }
            return make_f1(out, (size_t)n);
        }, "n"_a, "Generate n LFO samples.");

    // Lengths below 1 index the ring buffers out of bounds (negative) or divide
    // by zero (0). PeakHold and BoxStackFilter also cannot grow in set(), so
    // their maximum is kept to bound it.
    auto check_len = [](long n, const char *what, long max = -1) {
        if (n < 1 || (max >= 1 && n > max))
            throw std::invalid_argument(std::string(what) + " must be in [1, "
                + (max >= 1 ? std::to_string(max) : std::string("inf")) + "], got "
                + std::to_string(n));
    };
    auto process_env = [](auto &self, ArrayF input) {
        size_t n = input.shape(0);
        auto *out = new float[n];
        const float *in = input.data();
        { nb::gil_scoped_release rel;
          for (size_t i = 0; i < n; ++i) out[i] = self(in[i]);
        }
        return make_f1(out, n);
    };

    using BoxF = signalsmith::envelopes::BoxFilter<float>;
    nb::class_<BoxF>(mod, "BoxFilter", "Rectangular moving average filter (FIR).")
        .def("__init__", [check_len](BoxF *self, int max_length) {
            check_len(max_length, "max_length");
            new (self) BoxF(max_length);
        }, "max_length"_a)
        .def("resize", [check_len](BoxF &self, int n) { check_len(n, "max_length"); self.resize(n); }, "max_length"_a)
        .def("set", [check_len](BoxF &self, int n) { check_len(n, "length"); self.set(n); }, "length"_a)
        .def("reset", &BoxF::reset, "fill"_a = 0.0f)
        .def("process", [process_env](BoxF &self, ArrayF input) { return process_env(self, input); }, "input"_a);

    struct BoxStack : signalsmith::envelopes::BoxStackFilter<float> {
        int max_size;
        BoxStack(int m, int layers) : BoxStackFilter(m, layers), max_size(m) {}
    };
    nb::class_<BoxStack>(mod, "BoxStackFilter", "FIR filter from stacked box filters.")
        .def("__init__", [check_len](BoxStack *self, int max_size, int layers) {
            check_len(max_size, "max_size");
            check_len(layers, "layers");
            new (self) BoxStack(max_size, layers);
        }, "max_size"_a, "layers"_a = 4)
        .def("set", [check_len](BoxStack &self, int n) {
            check_len(n, "size", self.max_size);
            self.set(n);
        }, "size"_a)
        .def("reset", [](BoxStack &self, float fill) { self.reset(fill); }, "fill"_a = 0.0f)
        .def("process", [process_env](BoxStack &self, ArrayF input) { return process_env(self, input); }, "input"_a);

    struct PH : signalsmith::envelopes::PeakHold<float> {
        int max_length;
        explicit PH(int m) : PeakHold(m), max_length(m) {}
    };
    nb::class_<PH>(mod, "PeakHold", "Variable-size peak-hold filter.")
        .def("__init__", [check_len](PH *self, int max_length) {
            check_len(max_length, "max_length");
            new (self) PH(max_length);
        }, "max_length"_a)
        .def("resize", [check_len](PH &self, int n) {
            check_len(n, "max_length");
            self.resize(n);
            self.max_length = n;
        }, "max_length"_a)
        .def("set", [check_len](PH &self, int n, bool preserve) {
            check_len(n, "new_size", self.max_length);
            self.set(n, preserve);
        }, "new_size"_a, "preserve_current_peak"_a = false)
        .def("reset", [](PH &self, float fill) { self.reset(fill); }, "fill"_a = std::numeric_limits<float>::lowest())
        .def("process", [process_env](PH &self, ArrayF input) { return process_env(self, input); }, "input"_a);

    struct PDL : signalsmith::envelopes::PeakDecayLinear<float> {
        int max_length;
        explicit PDL(int m) : PeakDecayLinear(m), max_length(m) {}
    };
    nb::class_<PDL>(mod, "PeakDecayLinear", "Peak-hold with linear decay.")
        .def("__init__", [check_len](PDL *self, int max_length) {
            check_len(max_length, "max_length");
            new (self) PDL(max_length);
        }, "max_length"_a)
        .def("resize", [check_len](PDL &self, int n) {
            check_len(n, "max_length");
            self.resize(n);
            self.max_length = n;
        }, "max_length"_a)
        .def("set", [check_len](PDL &self, double n) {
            check_len((long)std::ceil(n), "length", self.max_length);
            self.set(n);
        }, "length"_a)
        .def("reset", [](PDL &self, float start) { self.reset(start); }, "start"_a = std::numeric_limits<float>::lowest())
        .def("process", [process_env](PDL &self, ArrayF input) { return process_env(self, input); }, "input"_a);
}

// ============================================================================
// Spectral
// ============================================================================

static void bind_spectral(nb::module_ &m) {
    auto mod = m.def_submodule("spectral", "STFT and spectral processing");
    using STFT = signalsmith::spectral::STFT<float>;
    using Complex = std::complex<float>;

    nb::class_<STFT>(mod, "STFT", "Short-Time Fourier Transform with overlap-add synthesis.")
        .def(nb::init<int, int, int, int, int>(),
            "channels"_a, "window_size"_a, "interval"_a, "history_length"_a = 0, "zero_padding"_a = 0)
        .def("window_size", &STFT::windowSize)
        .def("fft_size", &STFT::fftSize)
        .def("interval", &STFT::interval)
        .def("bands", &STFT::bands)
        .def("latency", &STFT::latency)
        .def("reset", &STFT::reset)
        .def("analyse", [](STFT &self, Array2F input) {
            int frames = (int)input.shape(1);
            struct CD { const float *base; int stride;
                struct Ch { const float *d; float operator[](int i) const { return d[i]; } };
                Ch operator[](int c) const { return {base + c * stride}; }
            };
            CD cd{input.data(), frames};
            { nb::gil_scoped_release rel;
              self.analyse(cd);
            }
        }, "input"_a, "Analyse multi-channel input [channels, frames] into spectrum.")
        .def("analyse_channel", [](STFT &self, int channel, ArrayF input) {
            struct D { const float *p; float operator[](int i) const { return p[i]; } };
            D d{input.data()};
            { nb::gil_scoped_release rel;
              self.analyse(channel, d);
            }
        }, "channel"_a, "input"_a, "Analyse a single channel from 1-D float32.")
        .def("get_spectrum", [](STFT &self) {
            int bands = self.bands();
            auto *out = new Complex[bands];
            std::copy(self.spectrum[0], self.spectrum[0] + bands, out);
            return make_cf1(out, (size_t)bands);
        }, "Get spectrum for channel 0.")
        .def("get_spectrum_channel", [](STFT &self, int ch) {
            int bands = self.bands();
            auto *out = new Complex[bands];
            std::copy(self.spectrum[ch], self.spectrum[ch] + bands, out);
            return make_cf1(out, (size_t)bands);
        }, "channel"_a)
        .def("set_spectrum_channel", [](STFT &self, int ch, ArrayCF data) {
            int bands = self.bands();
            if ((int)data.shape(0) != bands) throw std::invalid_argument("Spectrum size must match bands()");
            std::copy(data.data(), data.data() + bands, self.spectrum[ch]);
        }, "channel"_a, "data"_a);
}

// ============================================================================
// Rates
// ============================================================================

static void bind_rates(nb::module_ &m) {
    auto mod = m.def_submodule("rates", "Multi-rate processing (oversampling)");
    // The vendored class holds 2 * max_block samples per channel and does not
    // check either bound, so blocks are fed in max_block chunks here.
    struct OS : signalsmith::rates::Oversampler2xFIR<float> {
        int channels, max_block;
        OS(int c, int m, int h, double p) : Oversampler2xFIR(c, m, h, p), channels(c), max_block(m) {}
        void check_channels(size_t ch) const {
            if ((int)ch != channels)
                throw std::invalid_argument("Oversampler2x: expected " + std::to_string(channels)
                    + " channels, got " + std::to_string(ch));
        }
    };
    struct Ptrs {
        std::vector<float*> p;
        float* operator[](int c) const { return p[c]; }
    };

    nb::class_<OS>(mod, "Oversampler2x", "2x FIR oversampling.")
        .def("__init__", [](OS *self, int channels, int max_block, int half_latency, double pass_freq) {
            if (channels < 1 || max_block < 1 || half_latency < 1)
                throw std::invalid_argument("Oversampler2x: channels, max_block and half_latency must be >= 1");
            new (self) OS(channels, max_block, half_latency, pass_freq);
        }, "channels"_a, "max_block"_a, "half_latency"_a = 16, "pass_freq"_a = 0.43)
        .def("reset", [](OS &self) { self.reset(); })
        .def("latency", [](OS &self) { return self.latency(); })
        .def("up", [](OS &self, Array2F input) {
            self.check_channels(input.shape(0));
            int ch = (int)input.shape(0), fr = (int)input.shape(1);
            size_t ofr = (size_t)fr * 2;
            auto *out = new float[ch * ofr];
            const float *in = input.data();
            { nb::gil_scoped_release rel;
              Ptrs ip; ip.p.resize(ch);
              for (int off = 0; off < fr; off += self.max_block) {
                  int n = std::min(self.max_block, fr - off);
                  for (int c = 0; c < ch; ++c) ip.p[c] = const_cast<float*>(in) + (size_t)c * fr + off;
                  self.up(ip, n);
                  for (int c = 0; c < ch; ++c)
                      std::copy(self[c], self[c] + 2 * n, out + c * ofr + 2 * (size_t)off);
              }
            }
            return make_f2(out, (size_t)ch, ofr);
        }, "input"_a, "Upsample [channels, frames] -> [channels, frames*2].")
        .def("down", [](OS &self, Array2F input, int low) {
            self.check_channels(input.shape(0));
            if (low < 0 || input.shape(1) != 2 * (size_t)low)
                throw std::invalid_argument("Oversampler2x.down: input must have 2 * low_samples frames");
            int ch = (int)input.shape(0);
            size_t ifr = input.shape(1);
            auto *out = new float[ch * (size_t)low];
            const float *in = input.data();
            { nb::gil_scoped_release rel;
              Ptrs op; op.p.resize(ch);
              for (int off = 0; off < low; off += self.max_block) {
                  int n = std::min(self.max_block, low - off);
                  for (int c = 0; c < ch; ++c) {
                      const float *src = in + c * ifr + 2 * (size_t)off;
                      std::copy(src, src + 2 * n, self[c]);
                      op.p[c] = out + (size_t)c * low + off;
                  }
                  self.down(op, n);
              }
            }
            return make_f2(out, (size_t)ch, (size_t)low);
        }, "input"_a, "low_samples"_a)
        .def("process", [](OS &self, Array2F input) {
            self.check_channels(input.shape(0));
            int ch = (int)input.shape(0), fr = (int)input.shape(1);
            auto *out = new float[ch * (size_t)fr];
            const float *in = input.data();
            { nb::gil_scoped_release rel;
              Ptrs ip, op; ip.p.resize(ch); op.p.resize(ch);
              for (int off = 0; off < fr; off += self.max_block) {
                  int n = std::min(self.max_block, fr - off);
                  for (int c = 0; c < ch; ++c) {
                      ip.p[c] = const_cast<float*>(in) + (size_t)c * fr + off;
                      op.p[c] = out + (size_t)c * fr + off;
                  }
                  self.up(ip, n);
                  self.down(op, n);
              }
            }
            return make_f2(out, (size_t)ch, (size_t)fr);
        }, "input"_a, "Round-trip upsample+downsample. Input: [channels, frames].");
}

// ============================================================================
// Mix
// ============================================================================

static void bind_mix(nb::module_ &m) {
    auto mod = m.def_submodule("mix", "Multichannel mixing utilities");

    using DynHad = signalsmith::mix::Hadamard<float, -1>;
    nb::class_<DynHad>(mod, "Hadamard", "Hadamard mixing matrix (dynamic size, power of 2).")
        .def(nb::init<int>(), "size"_a)
        .def("in_place", [](const DynHad &self, ArrayF data) {
            size_t n = data.shape(0);
            auto *out = new float[n];
            std::copy(data.data(), data.data() + n, out);
            self.inPlace(out);
            return make_f1(out, n);
        }, "data"_a, "Apply scaled orthogonal Hadamard, returning new array.")
        .def("scaling_factor", &DynHad::scalingFactor);

    using DynHouse = signalsmith::mix::Householder<float, -1>;
    nb::class_<DynHouse>(mod, "Householder", "Householder reflection matrix (dynamic size).")
        .def(nb::init<int>(), "size"_a)
        .def("in_place", [](const DynHouse &self, ArrayF data) {
            size_t n = data.shape(0);
            auto *out = new float[n];
            std::copy(data.data(), data.data() + n, out);
            self.inPlace(out);
            return make_f1(out, n);
        }, "data"_a, "Apply Householder reflection, returning new array.");

    mod.def("cheap_energy_crossfade", [](float x) {
        float to_c, from_c;
        signalsmith::mix::cheapEnergyCrossfade(x, to_c, from_c);
        return nb::make_tuple(to_c, from_c);
    }, "x"_a, "Cheap energy-preserving crossfade. Returns (to_coeff, from_coeff).");
}

// ============================================================================
// Public entry point
// ============================================================================

void bind_signalsmith(nb::module_ &m) {
    bind_filters(m);
    bind_fft(m);
    bind_delay(m);
    bind_envelopes(m);
    bind_spectral(m);
    bind_rates(m);
    bind_mix(m);
}

#include "_core_common.h"
#include "vafilters.h"

using namespace vafilters;

// Helper: process a 1D float array through a filter, return new array
template <typename Filter>
static NpF1 va_process(Filter &f, ArrayF input) {
    size_t n = input.shape(0);
    float *out = new float[n];
    {
        nb::gil_scoped_release rel;
        f.process(input.data(), out, static_cast<unsigned>(n));
    }
    return make_f1(out, n);
}

// Macro for binding the common cutoff/q/init/reset/process interface
#define BIND_VA_FILTER(cls, py_name, doc)                                      \
    nb::class_<cls>(sub, py_name, doc)                                         \
        .def(nb::init<>())                                                     \
        .def("init", &cls::init, "sample_rate"_a,                              \
             "Initialize with sample rate (Hz)")                               \
        .def("reset", &cls::reset, "Reset filter state")                       \
        .def_prop_rw("cutoff",                                                 \
                     &cls::get_cutoff, &cls::set_cutoff,                        \
                     "Cutoff frequency in Hz (20-20000)")                       \
        .def_prop_rw("q",                                                      \
                     &cls::get_q, &cls::set_q,                                  \
                     "Resonance/Q factor")                                      \
        .def("process", [](cls &self, ArrayF input) {                          \
            return va_process(self, input);                                     \
        }, "input"_a, "Process audio samples, return filtered output")

void bind_vafilters(nb::module_ &m) {
    auto sub = m.def_submodule("vafilters",
        "Virtual analog filter models (Faust-generated, MIT-style STK-4.3 license)");

    BIND_VA_FILTER(MoogLadder, "MoogLadder",
        "Moog Ladder 24 dB/oct lowpass filter");

    BIND_VA_FILTER(MoogHalfLadder, "MoogHalfLadder",
        "Moog Half-Ladder 12 dB/oct lowpass filter");

    BIND_VA_FILTER(DiodeLadder, "DiodeLadder",
        "Diode Ladder 24 dB/oct lowpass filter");

    BIND_VA_FILTER(Korg35LPF, "Korg35LPF",
        "Korg 35 24 dB/oct lowpass filter");

    BIND_VA_FILTER(Korg35HPF, "Korg35HPF",
        "Korg 35 24 dB/oct highpass filter");

    // OberheimSVF has extra multi-output + mode support
    nb::class_<OberheimSVF>(sub, "OberheimSVF",
        "Oberheim multi-mode state-variable filter (LP/HP/BP/BSF)")
        .def(nb::init<>())
        .def("init", &OberheimSVF::init, "sample_rate"_a,
             "Initialize with sample rate (Hz)")
        .def("reset", &OberheimSVF::reset, "Reset filter state")
        .def_prop_rw("cutoff",
                     &OberheimSVF::get_cutoff, &OberheimSVF::set_cutoff,
                     "Cutoff frequency in Hz (20-20000)")
        .def_prop_rw("q",
                     &OberheimSVF::get_q, &OberheimSVF::set_q,
                     "Resonance/Q factor (0.5-10)")
        .def("process", [](OberheimSVF &self, ArrayF input, int mode) {
            size_t n = input.shape(0);
            float *out = new float[n];
            {
                nb::gil_scoped_release rel;
                self.process(input.data(), out, static_cast<unsigned>(n),
                             static_cast<OberheimSVF::Mode>(mode));
            }
            return make_f1(out, n);
        }, "input"_a, "mode"_a = 0,
           "Process with selected mode: 0=LPF, 1=HPF, 2=BPF, 3=BSF")
        .def("process_multi", [](OberheimSVF &self, ArrayF input) {
            size_t n = input.shape(0);
            float *lpf = new float[n];
            float *hpf = new float[n];
            float *bpf = new float[n];
            float *bsf = new float[n];
            {
                nb::gil_scoped_release rel;
                self.process_multi(input.data(), bsf, bpf, hpf, lpf,
                                   static_cast<unsigned>(n));
            }
            return nb::make_tuple(
                make_f1(lpf, n), make_f1(hpf, n),
                make_f1(bpf, n), make_f1(bsf, n));
        }, "input"_a,
           "Process and return all 4 outputs: (lpf, hpf, bpf, bsf)");
}

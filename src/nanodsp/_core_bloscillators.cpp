#include "_core_common.h"
#include "bl_polyblep.h"

using namespace bloscillators;

template <typename Osc>
static NpF1 osc_generate(Osc &o, unsigned count) {
    float *out = new float[count];
    {
        nb::gil_scoped_release rel;
        o.generate(out, count);
    }
    return make_f1(out, count);
}

void bind_bloscillators(nb::module_ &m) {
    auto sub = m.def_submodule("bloscillators",
        "Band-limited oscillators (PolyBLEP, BLIT, DPW)");

    // PolyBLEP -- 14 waveforms
    auto pb = nb::class_<PolyBLEP>(sub, "PolyBLEP",
        "PolyBLEP oscillator with 14 anti-aliased waveforms");
    nb::enum_<PolyBLEP::Waveform>(pb, "Waveform")
        .value("SINE", PolyBLEP::SINE)
        .value("COSINE", PolyBLEP::COSINE)
        .value("TRIANGLE", PolyBLEP::TRIANGLE)
        .value("SQUARE", PolyBLEP::SQUARE)
        .value("RECTANGLE", PolyBLEP::RECTANGLE)
        .value("SAWTOOTH", PolyBLEP::SAWTOOTH)
        .value("RAMP", PolyBLEP::RAMP)
        .value("MODIFIED_TRIANGLE", PolyBLEP::MODIFIED_TRIANGLE)
        .value("MODIFIED_SQUARE", PolyBLEP::MODIFIED_SQUARE)
        .value("HALF_WAVE_RECTIFIED_SINE", PolyBLEP::HALF_WAVE_RECTIFIED_SINE)
        .value("FULL_WAVE_RECTIFIED_SINE", PolyBLEP::FULL_WAVE_RECTIFIED_SINE)
        .value("TRIANGULAR_PULSE", PolyBLEP::TRIANGULAR_PULSE)
        .value("TRAPEZOID_FIXED", PolyBLEP::TRAPEZOID_FIXED)
        .value("TRAPEZOID_VARIABLE", PolyBLEP::TRAPEZOID_VARIABLE);
    pb.def(nb::init<float, PolyBLEP::Waveform>(),
           "sample_rate"_a = 44100.0f, "waveform"_a = PolyBLEP::SAWTOOTH)
      .def("reset", &PolyBLEP::reset)
      .def("sync", &PolyBLEP::sync, "phase"_a)
      .def_prop_rw("frequency",
                   &PolyBLEP::get_frequency, &PolyBLEP::set_frequency)
      .def_prop_rw("waveform",
                   &PolyBLEP::get_waveform, &PolyBLEP::set_waveform)
      .def_prop_rw("pulse_width",
                   &PolyBLEP::get_pulse_width, &PolyBLEP::set_pulse_width)
      .def_prop_rw("phase",
                   &PolyBLEP::get_phase, &PolyBLEP::set_phase)
      .def("tick", &PolyBLEP::tick, "Generate one sample")
      .def("generate", [](PolyBLEP &self, unsigned count) {
          return osc_generate(self, count);
      }, "count"_a, "Generate count samples");

    // BLIT Sawtooth
    nb::class_<BlitSaw>(sub, "BlitSaw",
        "BLIT sawtooth oscillator with configurable harmonics")
        .def(nb::init<float, float>(),
             "sample_rate"_a = 44100.0f, "frequency"_a = 220.0f)
        .def("reset", &BlitSaw::reset)
        .def_prop_rw("frequency",
                     &BlitSaw::get_frequency, &BlitSaw::set_frequency)
        .def("set_harmonics", &BlitSaw::set_harmonics, "n"_a,
             "Set number of harmonics (0 = maximum up to Nyquist)")
        .def("tick", &BlitSaw::tick, "Generate one sample")
        .def("generate", [](BlitSaw &self, unsigned count) {
            return osc_generate(self, count);
        }, "count"_a, "Generate count samples");

    // BLIT Square
    nb::class_<BlitSquare>(sub, "BlitSquare",
        "BLIT square wave oscillator with DC blocker")
        .def(nb::init<float, float>(),
             "sample_rate"_a = 44100.0f, "frequency"_a = 220.0f)
        .def("reset", &BlitSquare::reset)
        .def_prop_rw("frequency",
                     &BlitSquare::get_frequency, &BlitSquare::set_frequency)
        .def("set_harmonics", &BlitSquare::set_harmonics, "n"_a,
             "Set number of harmonics (0 = maximum up to Nyquist)")
        .def("tick", &BlitSquare::tick, "Generate one sample")
        .def("generate", [](BlitSquare &self, unsigned count) {
            return osc_generate(self, count);
        }, "count"_a, "Generate count samples");

    // DPW Sawtooth
    nb::class_<DPWSaw>(sub, "DPWSaw",
        "DPW (Differentiated Parabolic Wave) sawtooth oscillator")
        .def(nb::init<float, float>(),
             "sample_rate"_a = 44100.0f, "frequency"_a = 440.0f)
        .def("reset", &DPWSaw::reset)
        .def_prop_rw("frequency",
                     &DPWSaw::get_frequency, &DPWSaw::set_frequency)
        .def("tick", &DPWSaw::tick, "Generate one sample")
        .def("generate", [](DPWSaw &self, unsigned count) {
            return osc_generate(self, count);
        }, "count"_a, "Generate count samples");

    // DPW Pulse
    nb::class_<DPWPulse>(sub, "DPWPulse",
        "DPW pulse oscillator with variable duty cycle")
        .def(nb::init<float, float>(),
             "sample_rate"_a = 44100.0f, "frequency"_a = 440.0f)
        .def("reset", &DPWPulse::reset)
        .def_prop_rw("frequency",
                     &DPWPulse::get_frequency, &DPWPulse::set_frequency)
        .def_prop_rw("duty",
                     &DPWPulse::get_duty, &DPWPulse::set_duty)
        .def("tick", &DPWPulse::tick, "Generate one sample")
        .def("generate", [](DPWPulse &self, unsigned count) {
            return osc_generate(self, count);
        }, "count"_a, "Generate count samples");
}

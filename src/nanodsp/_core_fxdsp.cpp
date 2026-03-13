#include "_core_common.h"
#include "fxdsp.h"

using namespace fxdsp;

// Helper: process float array through a processor with tick()
template <typename Proc>
static NpF1 process_array(Proc &proc, ArrayF input) {
    size_t n = input.shape(0);
    float *out = new float[n];
    const float *in = input.data();
    {
        nb::gil_scoped_release rel;
        proc.process(in, out, (unsigned)n);
    }
    return make_f1(out, n);
}

// Helper: generate from an oscillator
template <typename Osc>
static NpF1 osc_gen(Osc &o, unsigned count) {
    float *out = new float[count];
    {
        nb::gil_scoped_release rel;
        o.generate(out, count);
    }
    return make_f1(out, count);
}

void bind_fxdsp(nb::module_ &m) {
    auto sub = m.def_submodule("fxdsp",
        "Additional DSP algorithms (waveshaping, reverbs, minBLEP, PSOLA, formant)");

    // --- Antialiased Waveshaping ---

    nb::class_<HardClipper>(sub, "HardClipper",
        "Antialiased hard clipper using 1st-order antiderivative method")
        .def(nb::init<>())
        .def("reset", &HardClipper::reset)
        .def("tick", &HardClipper::tick, "x"_a, "Process one sample")
        .def("process", [](HardClipper &self, ArrayF input) {
            return process_array(self, input);
        }, "input"_a, "Process array of samples");

    nb::class_<SoftClipper>(sub, "SoftClipper",
        "Antialiased soft clipper (sin-based saturation) with 1st-order AA")
        .def(nb::init<>())
        .def("reset", &SoftClipper::reset)
        .def("tick", &SoftClipper::tick, "x"_a, "Process one sample")
        .def("process", [](SoftClipper &self, ArrayF input) {
            return process_array(self, input);
        }, "input"_a, "Process array of samples");

    nb::class_<Wavefolder>(sub, "Wavefolder",
        "Antialiased wavefolder (Buchla 259 style) with 2nd-order AA")
        .def(nb::init<>())
        .def("reset", &Wavefolder::reset)
        .def("tick", &Wavefolder::tick, "x"_a, "Process one sample")
        .def("process", [](Wavefolder &self, ArrayF input) {
            return process_array(self, input);
        }, "input"_a, "Process array of samples");

    // --- Reverbs ---

    nb::class_<SchroederReverb>(sub, "SchroederReverb",
        "Schroeder reverberator (4 parallel combs + 2 series allpasses)")
        .def(nb::init<>())
        .def("init", &SchroederReverb::init, "sample_rate"_a)
        .def("reset", &SchroederReverb::reset)
        .def_prop_rw("feedback",
                     &SchroederReverb::get_feedback, &SchroederReverb::set_feedback)
        .def_prop_rw("diffusion",
                     &SchroederReverb::get_diffusion, &SchroederReverb::set_diffusion)
        .def("set_mod_depth", &SchroederReverb::set_mod_depth, "depth"_a)
        .def("tick", &SchroederReverb::tick, "x"_a)
        .def("process", [](SchroederReverb &self, ArrayF input) {
            size_t n = input.shape(0);
            float *out = new float[n];
            const float *in = input.data();
            {
                nb::gil_scoped_release rel;
                self.process(in, out, (unsigned)n);
            }
            return make_f1(out, n);
        }, "input"_a);

    nb::class_<MoorerReverb>(sub, "MoorerReverb",
        "Moorer reverberator (early reflections + 4 combs + 2 allpasses)")
        .def(nb::init<>())
        .def("init", &MoorerReverb::init, "sample_rate"_a)
        .def("reset", &MoorerReverb::reset)
        .def_prop_rw("feedback",
                     &MoorerReverb::get_feedback, &MoorerReverb::set_feedback)
        .def_prop_rw("diffusion",
                     &MoorerReverb::get_diffusion, &MoorerReverb::set_diffusion)
        .def("set_mod_depth", &MoorerReverb::set_mod_depth, "depth"_a)
        .def("tick", &MoorerReverb::tick, "x"_a)
        .def("process", [](MoorerReverb &self, ArrayF input) {
            size_t n = input.shape(0);
            float *out = new float[n];
            const float *in = input.data();
            {
                nb::gil_scoped_release rel;
                self.process(in, out, (unsigned)n);
            }
            return make_f1(out, n);
        }, "input"_a);

    // --- MinBLEP Oscillator ---

    auto mb = nb::class_<MinBLEP>(sub, "MinBLEP",
        "MinBLEP oscillator with 4 waveforms (saw, rsaw, square, triangle)");
    nb::enum_<MinBLEP::Waveform>(mb, "Waveform")
        .value("SAW", MinBLEP::SAW)
        .value("RSAW", MinBLEP::RSAW)
        .value("SQUARE", MinBLEP::SQUARE)
        .value("TRIANGLE", MinBLEP::TRIANGLE);
    mb.def(nb::init<float, float>(),
           "sample_rate"_a = 44100.0f, "frequency"_a = 440.0f)
      .def("reset", &MinBLEP::reset)
      .def_prop_rw("frequency",
                   &MinBLEP::get_frequency, &MinBLEP::set_frequency)
      .def_prop_rw("waveform",
                   &MinBLEP::get_waveform, &MinBLEP::set_waveform)
      .def_prop_rw("pulse_width",
                   &MinBLEP::get_pulse_width, &MinBLEP::set_pulse_width)
      .def("tick", &MinBLEP::tick, "Generate one sample")
      .def("generate", [](MinBLEP &self, unsigned count) {
          return osc_gen(self, count);
      }, "count"_a, "Generate count samples");

    // --- PSOLA Pitch Shifter ---

    sub.def("psola_pitch_shift",
        [](ArrayF input, float sample_rate, float semitones) {
            size_t n = input.shape(0);
            std::vector<float> result;
            {
                nb::gil_scoped_release rel;
                result = PsolaShifter::process(input.data(), (int)n,
                                               sample_rate, semitones);
            }
            float *out = new float[result.size()];
            std::memcpy(out, result.data(), result.size() * sizeof(float));
            return make_f1(out, result.size());
        },
        "input"_a, "sample_rate"_a, "semitones"_a,
        "PSOLA pitch shift by semitones (positive = up, negative = down)");

    // --- Formant Filter ---

    auto ff = nb::class_<FormantFilter>(sub, "FormantFilter",
        "Vowel formant filter using cascaded bandpass biquads");
    nb::enum_<FormantFilter::Vowel>(ff, "Vowel")
        .value("A", FormantFilter::A)
        .value("E", FormantFilter::E)
        .value("I", FormantFilter::I)
        .value("O", FormantFilter::O)
        .value("U", FormantFilter::U);
    ff.def(nb::init<>())
      .def("init", &FormantFilter::init, "sample_rate"_a)
      .def("reset", &FormantFilter::reset)
      .def_prop_rw("vowel",
                   &FormantFilter::get_vowel, &FormantFilter::set_vowel)
      .def("set_vowel_blend", &FormantFilter::set_vowel_blend,
           "vowel_a"_a, "vowel_b"_a, "mix"_a,
           "Interpolate between two vowels (0=a, 1=b)")
      .def("tick", &FormantFilter::tick, "x"_a)
      .def("process", [](FormantFilter &self, ArrayF input) {
          return process_array(self, input);
      }, "input"_a);

    // --- Ping-Pong Delay ---

    nb::class_<PingPongDelay>(sub, "PingPongDelay",
        "Stereo ping-pong delay with crossed feedback and linear interpolation")
        .def(nb::init<>())
        .def("init", &PingPongDelay::init,
             "sample_rate"_a, "max_delay_ms"_a = 2000.0f)
        .def("reset", &PingPongDelay::reset)
        .def_prop_rw("delay_ms",
                     &PingPongDelay::get_delay_ms, &PingPongDelay::set_delay_ms)
        .def_prop_rw("feedback",
                     &PingPongDelay::get_feedback, &PingPongDelay::set_feedback)
        .def_prop_rw("mix",
                     &PingPongDelay::get_mix, &PingPongDelay::set_mix)
        .def("tick", [](PingPongDelay &self, float in_l, float in_r) {
            auto p = self.tick(in_l, in_r);
            return nb::make_tuple(p.first, p.second);
        }, "in_l"_a, "in_r"_a, "Process one stereo sample, returns (left, right)")
        .def("process", [](PingPongDelay &self, Array2F input) {
            if (input.shape(0) != 2)
                throw std::invalid_argument(
                    "PingPongDelay.process expects shape [2, N]");
            size_t n = input.shape(1);
            float *out_l = new float[n];
            float *out_r = new float[n];
            const float *in_l = input.data();
            const float *in_r = input.data() + n;
            {
                nb::gil_scoped_release rel;
                self.process(in_l, in_r, out_l, out_r, (unsigned)n);
            }
            float *out = new float[2 * n];
            std::memcpy(out, out_l, n * sizeof(float));
            std::memcpy(out + n, out_r, n * sizeof(float));
            delete[] out_l;
            delete[] out_r;
            return make_f2(out, 2, n);
        }, "input"_a, "Process stereo array [2, N] -> [2, N]");

    // --- Frequency Shifter ---

    nb::class_<FreqShifter>(sub, "FreqShifter",
        "Bode-style frequency shifter using allpass Hilbert transform")
        .def(nb::init<>())
        .def("init", &FreqShifter::init, "sample_rate"_a)
        .def("reset", &FreqShifter::reset)
        .def_prop_rw("shift_hz",
                     &FreqShifter::get_shift_hz, &FreqShifter::set_shift_hz)
        .def("tick", &FreqShifter::tick, "x"_a, "Process one sample")
        .def("process", [](FreqShifter &self, ArrayF input) {
            return process_array(self, input);
        }, "input"_a, "Process array of samples");

    // --- Ring Modulator ---

    nb::class_<RingMod>(sub, "RingMod",
        "Ring modulator with carrier oscillator and optional LFO FM")
        .def(nb::init<>())
        .def("init", &RingMod::init, "sample_rate"_a)
        .def("reset", &RingMod::reset)
        .def_prop_rw("carrier_freq",
                     &RingMod::get_carrier_freq, &RingMod::set_carrier_freq)
        .def_prop_rw("lfo_freq",
                     &RingMod::get_lfo_freq, &RingMod::set_lfo_freq)
        .def_prop_rw("lfo_width",
                     &RingMod::get_lfo_width, &RingMod::set_lfo_width)
        .def_prop_rw("mix",
                     &RingMod::get_mix, &RingMod::set_mix)
        .def("tick", &RingMod::tick, "x"_a, "Process one sample")
        .def("process", [](RingMod &self, ArrayF input) {
            return process_array(self, input);
        }, "input"_a, "Process array of samples");
}

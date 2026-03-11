#include "_core_common.h"

#include <daisysp.h>
#include <daisysp-lgpl.h>

// ============================================================================
// Oscillators
// ============================================================================

static void bind_daisysp_oscillators(nb::module_ &m) {
    auto mod = m.def_submodule("oscillators", "DaisySP oscillator modules");

    // Waveform constants for Oscillator (uint8_t, not C++ enum)
    mod.attr("WAVE_SIN") = (int)daisysp::Oscillator::WAVE_SIN;
    mod.attr("WAVE_TRI") = (int)daisysp::Oscillator::WAVE_TRI;
    mod.attr("WAVE_SAW") = (int)daisysp::Oscillator::WAVE_SAW;
    mod.attr("WAVE_RAMP") = (int)daisysp::Oscillator::WAVE_RAMP;
    mod.attr("WAVE_SQUARE") = (int)daisysp::Oscillator::WAVE_SQUARE;
    mod.attr("WAVE_POLYBLEP_TRI") = (int)daisysp::Oscillator::WAVE_POLYBLEP_TRI;
    mod.attr("WAVE_POLYBLEP_SAW") = (int)daisysp::Oscillator::WAVE_POLYBLEP_SAW;
    mod.attr("WAVE_POLYBLEP_SQUARE") = (int)daisysp::Oscillator::WAVE_POLYBLEP_SQUARE;

    // BlOsc waveform constants
    mod.attr("BLOSC_WAVE_TRIANGLE") = (int)daisysp::BlOsc::WAVE_TRIANGLE;
    mod.attr("BLOSC_WAVE_SAW") = (int)daisysp::BlOsc::WAVE_SAW;
    mod.attr("BLOSC_WAVE_SQUARE") = (int)daisysp::BlOsc::WAVE_SQUARE;
    mod.attr("BLOSC_WAVE_OFF") = (int)daisysp::BlOsc::WAVE_OFF;

    using Osc = daisysp::Oscillator;
    nb::class_<Osc>(mod, "Oscillator", "Multi-waveform oscillator")
        .def(nb::init<>())
        .def("init", &Osc::Init, "sample_rate"_a)
        .def("set_freq", &Osc::SetFreq, "freq"_a)
        .def("set_amp", &Osc::SetAmp, "amp"_a)
        .def("set_waveform", [](Osc &self, int wf) { self.SetWaveform((uint8_t)wf); }, "waveform"_a)
        .def("set_pw", &Osc::SetPw, "pw"_a)
        .def("phase_add", &Osc::PhaseAdd, "phase"_a)
        .def("reset", &Osc::Reset, "phase"_a = 0.0f)
        .def("is_eor", &Osc::IsEOR)
        .def("is_eoc", &Osc::IsEOC)
        .def("is_rising", &Osc::IsRising)
        .def("is_falling", &Osc::IsFalling)
        .def("process_sample", &Osc::Process)
        .def("process", [](Osc &self, int n) {
            auto *out = new float[n];
            { nb::gil_scoped_release rel;
              for (int i = 0; i < n; ++i) out[i] = self.Process();
            }
            return make_f1(out, (size_t)n);
        }, "n"_a, "Generate n samples.");

    using Fm = daisysp::Fm2;
    nb::class_<Fm>(mod, "Fm2", "2-operator FM oscillator")
        .def(nb::init<>())
        .def("init", &Fm::Init, "sample_rate"_a)
        .def("set_frequency", &Fm::SetFrequency, "freq"_a)
        .def("set_ratio", &Fm::SetRatio, "ratio"_a)
        .def("set_index", &Fm::SetIndex, "index"_a)
        .def("get_index", &Fm::GetIndex)
        .def("reset", &Fm::Reset)
        .def("process_sample", &Fm::Process)
        .def("process", [](Fm &self, int n) {
            auto *out = new float[n];
            { nb::gil_scoped_release rel;
              for (int i = 0; i < n; ++i) out[i] = self.Process();
            }
            return make_f1(out, (size_t)n);
        }, "n"_a);

    using FO = daisysp::FormantOscillator;
    nb::class_<FO>(mod, "FormantOscillator", "Formant synthesis oscillator")
        .def(nb::init<>())
        .def("init", &FO::Init, "sample_rate"_a)
        .def("set_formant_freq", &FO::SetFormantFreq, "freq"_a)
        .def("set_carrier_freq", &FO::SetCarrierFreq, "freq"_a)
        .def("set_phase_shift", &FO::SetPhaseShift, "ps"_a)
        .def("process_sample", &FO::Process)
        .def("process", [](FO &self, int n) {
            auto *out = new float[n];
            { nb::gil_scoped_release rel;
              for (int i = 0; i < n; ++i) out[i] = self.Process();
            }
            return make_f1(out, (size_t)n);
        }, "n"_a);

    using HO = daisysp::HarmonicOscillator<16>;
    nb::class_<HO>(mod, "HarmonicOscillator", "Additive harmonic oscillator (16 harmonics)")
        .def(nb::init<>())
        .def("init", &HO::Init, "sample_rate"_a)
        .def("set_freq", &HO::SetFreq, "freq"_a)
        .def("set_first_harm_idx", &HO::SetFirstHarmIdx, "idx"_a)
        .def("set_amplitudes", [](HO &self, ArrayF amps) {
            self.SetAmplitudes(amps.data());
        }, "amplitudes"_a, "Set all harmonic amplitudes from array.")
        .def("set_single_amp", &HO::SetSingleAmp, "amp"_a, "idx"_a)
        .def("process_sample", &HO::Process)
        .def("process", [](HO &self, int n) {
            auto *out = new float[n];
            { nb::gil_scoped_release rel;
              for (int i = 0; i < n; ++i) out[i] = self.Process();
            }
            return make_f1(out, (size_t)n);
        }, "n"_a);

    using OB = daisysp::OscillatorBank;
    nb::class_<OB>(mod, "OscillatorBank", "Divide-down organ style oscillator bank (7 oscillators)")
        .def(nb::init<>())
        .def("init", &OB::Init, "sample_rate"_a)
        .def("set_freq", &OB::SetFreq, "freq"_a)
        .def("set_amplitudes", [](OB &self, ArrayF amps) {
            if (amps.shape(0) != 7) throw std::invalid_argument("Expected 7 amplitudes");
            self.SetAmplitudes(amps.data());
        }, "amplitudes"_a, "Set amplitudes for all 7 oscillators.")
        .def("set_single_amp", &OB::SetSingleAmp, "amp"_a, "idx"_a)
        .def("set_gain", &OB::SetGain, "gain"_a)
        .def("process_sample", &OB::Process)
        .def("process", [](OB &self, int n) {
            auto *out = new float[n];
            { nb::gil_scoped_release rel;
              for (int i = 0; i < n; ++i) out[i] = self.Process();
            }
            return make_f1(out, (size_t)n);
        }, "n"_a);

    using VSaw = daisysp::VariableSawOscillator;
    nb::class_<VSaw>(mod, "VariableSawOscillator", "Variable-shape saw oscillator")
        .def(nb::init<>())
        .def("init", &VSaw::Init, "sample_rate"_a)
        .def("set_freq", &VSaw::SetFreq, "freq"_a)
        .def("set_pw", &VSaw::SetPW, "pw"_a)
        .def("set_waveshape", &VSaw::SetWaveshape, "waveshape"_a)
        .def("process_sample", &VSaw::Process)
        .def("process", [](VSaw &self, int n) {
            auto *out = new float[n];
            { nb::gil_scoped_release rel;
              for (int i = 0; i < n; ++i) out[i] = self.Process();
            }
            return make_f1(out, (size_t)n);
        }, "n"_a);

    using VShape = daisysp::VariableShapeOscillator;
    nb::class_<VShape>(mod, "VariableShapeOscillator", "Continuously variable shape oscillator")
        .def(nb::init<>())
        .def("init", &VShape::Init, "sample_rate"_a)
        .def("set_freq", &VShape::SetFreq, "freq"_a)
        .def("set_pw", &VShape::SetPW, "pw"_a)
        .def("set_waveshape", &VShape::SetWaveshape, "waveshape"_a)
        .def("set_sync", &VShape::SetSync, "enable_sync"_a)
        .def("set_sync_freq", &VShape::SetSyncFreq, "freq"_a)
        .def("process_sample", &VShape::Process)
        .def("process", [](VShape &self, int n) {
            auto *out = new float[n];
            { nb::gil_scoped_release rel;
              for (int i = 0; i < n; ++i) out[i] = self.Process();
            }
            return make_f1(out, (size_t)n);
        }, "n"_a);

    using Vos = daisysp::VosimOscillator;
    nb::class_<Vos>(mod, "VosimOscillator", "VOSIM synthesis oscillator")
        .def(nb::init<>())
        .def("init", &Vos::Init, "sample_rate"_a)
        .def("set_freq", &Vos::SetFreq, "freq"_a)
        .def("set_form1_freq", &Vos::SetForm1Freq, "freq"_a)
        .def("set_form2_freq", &Vos::SetForm2Freq, "freq"_a)
        .def("set_shape", &Vos::SetShape, "shape"_a)
        .def("process_sample", &Vos::Process)
        .def("process", [](Vos &self, int n) {
            auto *out = new float[n];
            { nb::gil_scoped_release rel;
              for (int i = 0; i < n; ++i) out[i] = self.Process();
            }
            return make_f1(out, (size_t)n);
        }, "n"_a);

    using ZOsc = daisysp::ZOscillator;
    nb::class_<ZOsc>(mod, "ZOscillator", "Z-plane oscillator")
        .def(nb::init<>())
        .def("init", &ZOsc::Init, "sample_rate"_a)
        .def("set_freq", &ZOsc::SetFreq, "freq"_a)
        .def("set_formant_freq", &ZOsc::SetFormantFreq, "freq"_a)
        .def("set_shape", &ZOsc::SetShape, "shape"_a)
        .def("set_mode", &ZOsc::SetMode, "mode"_a)
        .def("process_sample", &ZOsc::Process)
        .def("process", [](ZOsc &self, int n) {
            auto *out = new float[n];
            { nb::gil_scoped_release rel;
              for (int i = 0; i < n; ++i) out[i] = self.Process();
            }
            return make_f1(out, (size_t)n);
        }, "n"_a);

    // LGPL
    using Bl = daisysp::BlOsc;
    nb::class_<Bl>(mod, "BlOsc", "Band-limited oscillator (LGPL)")
        .def(nb::init<>())
        .def("init", &Bl::Init, "sample_rate"_a)
        .def("set_freq", &Bl::SetFreq, "freq"_a)
        .def("set_amp", &Bl::SetAmp, "amp"_a)
        .def("set_pw", &Bl::SetPw, "pw"_a)
        .def("set_waveform", [](Bl &self, int wf) { self.SetWaveform((uint8_t)wf); }, "waveform"_a)
        .def("reset", &Bl::Reset)
        .def("process_sample", &Bl::Process)
        .def("process", [](Bl &self, int n) {
            auto *out = new float[n];
            { nb::gil_scoped_release rel;
              for (int i = 0; i < n; ++i) out[i] = self.Process();
            }
            return make_f1(out, (size_t)n);
        }, "n"_a);
}

// ============================================================================
// Filters
// ============================================================================

static void bind_daisysp_filters(nb::module_ &m) {
    auto mod = m.def_submodule("filters", "DaisySP filter modules");

    // --- SVF (multi-output) ---
    using SVF = daisysp::Svf;
    nb::class_<SVF>(mod, "Svf", "State variable filter (multi-output)")
        .def(nb::init<>())
        .def("init", &SVF::Init, "sample_rate"_a)
        .def("set_freq", &SVF::SetFreq, "freq"_a)
        .def("set_res", &SVF::SetRes, "res"_a)
        .def("set_drive", &SVF::SetDrive, "drive"_a)
        .def("process_sample", [](SVF &self, float in) {
            self.Process(in);
            return nb::make_tuple(self.Low(), self.High(), self.Band(), self.Notch(), self.Peak());
        }, "in"_a, "Process one sample. Returns (low, high, band, notch, peak).")
        .def("low", &SVF::Low)
        .def("high", &SVF::High)
        .def("band", &SVF::Band)
        .def("notch", &SVF::Notch)
        .def("peak", &SVF::Peak)
        .def("process_low", [](SVF &self, ArrayF input) {
            size_t n = input.shape(0);
            auto *out = new float[n];
            const float *in = input.data();
            { nb::gil_scoped_release rel;
              for (size_t i = 0; i < n; ++i) { self.Process(in[i]); out[i] = self.Low(); }
            }
            return make_f1(out, n);
        }, "input"_a, "Process buffer returning lowpass output.")
        .def("process_high", [](SVF &self, ArrayF input) {
            size_t n = input.shape(0);
            auto *out = new float[n];
            const float *in = input.data();
            { nb::gil_scoped_release rel;
              for (size_t i = 0; i < n; ++i) { self.Process(in[i]); out[i] = self.High(); }
            }
            return make_f1(out, n);
        }, "input"_a, "Process buffer returning highpass output.")
        .def("process_band", [](SVF &self, ArrayF input) {
            size_t n = input.shape(0);
            auto *out = new float[n];
            const float *in = input.data();
            { nb::gil_scoped_release rel;
              for (size_t i = 0; i < n; ++i) { self.Process(in[i]); out[i] = self.Band(); }
            }
            return make_f1(out, n);
        }, "input"_a, "Process buffer returning bandpass output.")
        .def("process_notch", [](SVF &self, ArrayF input) {
            size_t n = input.shape(0);
            auto *out = new float[n];
            const float *in = input.data();
            { nb::gil_scoped_release rel;
              for (size_t i = 0; i < n; ++i) { self.Process(in[i]); out[i] = self.Notch(); }
            }
            return make_f1(out, n);
        }, "input"_a, "Process buffer returning notch output.")
        .def("process_peak", [](SVF &self, ArrayF input) {
            size_t n = input.shape(0);
            auto *out = new float[n];
            const float *in = input.data();
            { nb::gil_scoped_release rel;
              for (size_t i = 0; i < n; ++i) { self.Process(in[i]); out[i] = self.Peak(); }
            }
            return make_f1(out, n);
        }, "input"_a, "Process buffer returning peak output.");

    // --- OnePole ---
    using OP = daisysp::OnePole;
    nb::enum_<OP::FilterMode>(mod, "OnePoleFM")
        .value("LOW_PASS", OP::FILTER_MODE_LOW_PASS)
        .value("HIGH_PASS", OP::FILTER_MODE_HIGH_PASS);

    nb::class_<OP>(mod, "OnePole", "One-pole lowpass/highpass filter")
        .def(nb::init<>())
        .def("init", &OP::Init)
        .def("reset", &OP::Reset)
        .def("set_frequency", &OP::SetFrequency, "freq"_a)
        .def("set_filter_mode", &OP::SetFilterMode, "mode"_a)
        .def("process_sample", &OP::Process, "in"_a)
        .def("process", [](OP &self, ArrayF input) {
            size_t n = input.shape(0);
            auto *out = new float[n];
            const float *in = input.data();
            { nb::gil_scoped_release rel;
              for (size_t i = 0; i < n; ++i) out[i] = self.Process(in[i]);
            }
            return make_f1(out, n);
        }, "input"_a);

    // --- LadderFilter ---
    using LF = daisysp::LadderFilter;
    nb::enum_<LF::FilterMode>(mod, "LadderFilterMode")
        .value("LP24", LF::FilterMode::LP24)
        .value("LP12", LF::FilterMode::LP12)
        .value("BP24", LF::FilterMode::BP24)
        .value("BP12", LF::FilterMode::BP12)
        .value("HP24", LF::FilterMode::HP24)
        .value("HP12", LF::FilterMode::HP12);

    nb::class_<LF>(mod, "LadderFilter", "Huovilainen transistor ladder filter")
        .def(nb::init<>())
        .def("init", &LF::Init, "sample_rate"_a)
        .def("set_freq", &LF::SetFreq, "freq"_a)
        .def("set_res", &LF::SetRes, "res"_a)
        .def("set_passband_gain", &LF::SetPassbandGain, "pbg"_a)
        .def("set_input_drive", &LF::SetInputDrive, "drv"_a)
        .def("set_filter_mode", &LF::SetFilterMode, "mode"_a)
        .def("process_sample", &LF::Process, "in"_a)
        .def("process", [](LF &self, ArrayF input) {
            size_t n = input.shape(0);
            auto *out = new float[n];
            const float *in = input.data();
            { nb::gil_scoped_release rel;
              for (size_t i = 0; i < n; ++i) out[i] = self.Process(in[i]);
            }
            return make_f1(out, n);
        }, "input"_a);

    // --- Soap (multi-output) ---
    using SP = daisysp::Soap;
    nb::class_<SP>(mod, "Soap", "Second-order allpass filter (bandpass + bandreject outputs)")
        .def(nb::init<>())
        .def("init", &SP::Init, "sample_rate"_a)
        .def("set_center_freq", &SP::SetCenterFreq, "freq"_a)
        .def("set_filter_bandwidth", &SP::SetFilterBandwidth, "bandwidth"_a)
        .def("process_sample", [](SP &self, float in) {
            self.Process(in);
            return nb::make_tuple(self.Bandpass(), self.Bandreject());
        }, "in"_a, "Process one sample. Returns (bandpass, bandreject).")
        .def("bandpass", &SP::Bandpass)
        .def("bandreject", &SP::Bandreject)
        .def("process_bandpass", [](SP &self, ArrayF input) {
            size_t n = input.shape(0);
            auto *out = new float[n];
            const float *in = input.data();
            { nb::gil_scoped_release rel;
              for (size_t i = 0; i < n; ++i) { self.Process(in[i]); out[i] = self.Bandpass(); }
            }
            return make_f1(out, n);
        }, "input"_a)
        .def("process_bandreject", [](SP &self, ArrayF input) {
            size_t n = input.shape(0);
            auto *out = new float[n];
            const float *in = input.data();
            { nb::gil_scoped_release rel;
              for (size_t i = 0; i < n; ++i) { self.Process(in[i]); out[i] = self.Bandreject(); }
            }
            return make_f1(out, n);
        }, "input"_a);

    // --- LGPL Filters ---

    // Allpass (needs external buffer -- managed internally)
    using AP = daisysp::Allpass;
    nb::class_<AP>(mod, "Allpass", "Allpass filter (LGPL)")
        .def("__init__", [](AP *self, float sample_rate, size_t max_size) {
            new (self) AP();
            auto *buf = new float[max_size]();
            self->Init(sample_rate, buf, max_size);
        }, "sample_rate"_a, "max_size"_a = 4096)
        .def("set_freq", &AP::SetFreq, "looptime"_a)
        .def("set_rev_time", &AP::SetRevTime, "revtime"_a)
        .def("process_sample", &AP::Process, "in"_a)
        .def("process", [](AP &self, ArrayF input) {
            size_t n = input.shape(0);
            auto *out = new float[n];
            const float *in = input.data();
            { nb::gil_scoped_release rel;
              for (size_t i = 0; i < n; ++i) out[i] = self.Process(in[i]);
            }
            return make_f1(out, n);
        }, "input"_a);

    // ATone (high-pass, takes float& args)
    using AT = daisysp::ATone;
    nb::class_<AT>(mod, "ATone", "First-order recursive high-pass filter (LGPL)")
        .def(nb::init<>())
        .def("init", &AT::Init, "sample_rate"_a)
        .def("set_freq", [](AT &self, float freq) { self.SetFreq(freq); }, "freq"_a)
        .def("get_freq", &AT::GetFreq)
        .def("process_sample", [](AT &self, float in) { return self.Process(in); }, "in"_a)
        .def("process", [](AT &self, ArrayF input) {
            size_t n = input.shape(0);
            auto *out = new float[n];
            float val;
            { nb::gil_scoped_release rel;
              for (size_t i = 0; i < n; ++i) { val = input.data()[i]; out[i] = self.Process(val); }
            }
            return make_f1(out, n);
        }, "input"_a);

    // Biquad (LGPL) -- named DaisyBiquad to avoid collision with signalsmith Biquad
    using BQ = daisysp::Biquad;
    nb::class_<BQ>(mod, "DaisyBiquad", "Two-pole recursive biquad filter (LGPL)")
        .def(nb::init<>())
        .def("init", &BQ::Init, "sample_rate"_a)
        .def("set_res", &BQ::SetRes, "res"_a)
        .def("set_cutoff", &BQ::SetCutoff, "cutoff"_a)
        .def("process_sample", &BQ::Process, "in"_a)
        .def("process", [](BQ &self, ArrayF input) {
            size_t n = input.shape(0);
            auto *out = new float[n];
            const float *in = input.data();
            { nb::gil_scoped_release rel;
              for (size_t i = 0; i < n; ++i) out[i] = self.Process(in[i]);
            }
            return make_f1(out, n);
        }, "input"_a);

    // Comb (needs external buffer)
    using CB = daisysp::Comb;
    nb::class_<CB>(mod, "Comb", "Comb filter (LGPL)")
        .def("__init__", [](CB *self, float sample_rate, size_t max_size) {
            new (self) CB();
            auto *buf = new float[max_size]();
            self->Init(sample_rate, buf, max_size);
        }, "sample_rate"_a, "max_size"_a = 4096)
        .def("set_freq", &CB::SetFreq, "freq"_a)
        .def("set_period", &CB::SetPeriod, "looptime"_a)
        .def("set_rev_time", &CB::SetRevTime, "revtime"_a)
        .def("process_sample", &CB::Process, "in"_a)
        .def("process", [](CB &self, ArrayF input) {
            size_t n = input.shape(0);
            auto *out = new float[n];
            const float *in = input.data();
            { nb::gil_scoped_release rel;
              for (size_t i = 0; i < n; ++i) out[i] = self.Process(in[i]);
            }
            return make_f1(out, n);
        }, "input"_a);

    // Mode
    using MD = daisysp::Mode;
    nb::class_<MD>(mod, "Mode", "Resonant modal bandpass filter (LGPL)")
        .def(nb::init<>())
        .def("init", &MD::Init, "sample_rate"_a)
        .def("set_freq", &MD::SetFreq, "freq"_a)
        .def("set_q", &MD::SetQ, "q"_a)
        .def("clear", &MD::Clear)
        .def("process_sample", &MD::Process, "in"_a)
        .def("process", [](MD &self, ArrayF input) {
            size_t n = input.shape(0);
            auto *out = new float[n];
            const float *in = input.data();
            { nb::gil_scoped_release rel;
              for (size_t i = 0; i < n; ++i) out[i] = self.Process(in[i]);
            }
            return make_f1(out, n);
        }, "input"_a);

    // MoogLadder
    using ML = daisysp::MoogLadder;
    nb::class_<ML>(mod, "MoogLadder", "Classic Moog ladder filter (LGPL)")
        .def(nb::init<>())
        .def("init", &ML::Init, "sample_rate"_a)
        .def("set_freq", &ML::SetFreq, "freq"_a)
        .def("set_res", &ML::SetRes, "res"_a)
        .def("process_sample", &ML::Process, "in"_a)
        .def("process", [](ML &self, ArrayF input) {
            size_t n = input.shape(0);
            auto *out = new float[n];
            const float *in = input.data();
            { nb::gil_scoped_release rel;
              for (size_t i = 0; i < n; ++i) out[i] = self.Process(in[i]);
            }
            return make_f1(out, n);
        }, "input"_a);

    // NlFilt (block-based)
    using NL = daisysp::NlFilt;
    nb::class_<NL>(mod, "NlFilt", "Nonlinear filter (LGPL)")
        .def(nb::init<>())
        .def("init", &NL::Init)
        .def("set_coefficients", &NL::SetCoefficients, "a"_a, "b"_a, "d"_a, "C"_a, "L"_a)
        .def("set_a", &NL::SetA, "a"_a)
        .def("set_b", &NL::SetB, "b"_a)
        .def("set_d", &NL::SetD, "d"_a)
        .def("set_c", &NL::SetC, "C"_a)
        .def("set_l", &NL::SetL, "L"_a)
        .def("process", [](NL &self, ArrayF input) {
            size_t n = input.shape(0);
            auto *out = new float[n];
            // NlFilt::ProcessBlock takes float* in, float* out, size_t
            std::vector<float> in_copy(input.data(), input.data() + n);
            { nb::gil_scoped_release rel;
              self.ProcessBlock(in_copy.data(), out, n);
            }
            return make_f1(out, n);
        }, "input"_a);

    // Tone (low-pass, takes float& in Process on some versions but current is float)
    using TN = daisysp::Tone;
    nb::class_<TN>(mod, "Tone", "First-order recursive low-pass filter (LGPL)")
        .def(nb::init<>())
        .def("init", &TN::Init, "sample_rate"_a)
        .def("set_freq", [](TN &self, float freq) { self.SetFreq(freq); }, "freq"_a)
        .def("get_freq", &TN::GetFreq)
        .def("process_sample", [](TN &self, float in) { return self.Process(in); }, "in"_a)
        .def("process", [](TN &self, ArrayF input) {
            size_t n = input.shape(0);
            auto *out = new float[n];
            float val;
            { nb::gil_scoped_release rel;
              for (size_t i = 0; i < n; ++i) { val = input.data()[i]; out[i] = self.Process(val); }
            }
            return make_f1(out, n);
        }, "input"_a);
}

// ============================================================================
// Effects
// ============================================================================

static void bind_daisysp_effects(nb::module_ &m) {
    auto mod = m.def_submodule("effects", "DaisySP effect modules");

    using AW = daisysp::Autowah;
    nb::class_<AW>(mod, "Autowah", "Auto-wah effect")
        .def(nb::init<>())
        .def("init", &AW::Init, "sample_rate"_a)
        .def("set_wah", &AW::SetWah, "wah"_a)
        .def("set_dry_wet", &AW::SetDryWet, "drywet"_a)
        .def("set_level", &AW::SetLevel, "level"_a)
        .def("process_sample", &AW::Process, "in"_a)
        .def("process", [](AW &self, ArrayF input) {
            size_t n = input.shape(0);
            auto *out = new float[n];
            const float *in = input.data();
            { nb::gil_scoped_release rel;
              for (size_t i = 0; i < n; ++i) out[i] = self.Process(in[i]);
            }
            return make_f1(out, n);
        }, "input"_a);

    // Chorus (stereo output)
    using CH = daisysp::Chorus;
    nb::class_<CH>(mod, "Chorus", "Stereo chorus effect")
        .def(nb::init<>())
        .def("init", &CH::Init, "sample_rate"_a)
        .def("set_lfo_depth", [](CH &self, float d) { self.SetLfoDepth(d); }, "depth"_a)
        .def("set_lfo_freq", [](CH &self, float f) { self.SetLfoFreq(f); }, "freq"_a)
        .def("set_delay", [](CH &self, float d) { self.SetDelay(d); }, "delay"_a)
        .def("set_delay_ms", [](CH &self, float ms) { self.SetDelayMs(ms); }, "ms"_a)
        .def("set_feedback", [](CH &self, float f) { self.SetFeedback(f); }, "feedback"_a)
        .def("get_left", &CH::GetLeft)
        .def("get_right", &CH::GetRight)
        .def("process_sample", &CH::Process, "in"_a)
        .def("process", [](CH &self, ArrayF input) {
            size_t n = input.shape(0);
            auto *out = new float[n];
            const float *in = input.data();
            { nb::gil_scoped_release rel;
              for (size_t i = 0; i < n; ++i) out[i] = self.Process(in[i]);
            }
            return make_f1(out, n);
        }, "input"_a, "Process mono input, returns mono (left channel).")
        .def("process_stereo", [](CH &self, ArrayF input) {
            size_t n = input.shape(0);
            auto *out = new float[2 * n];
            const float *in = input.data();
            { nb::gil_scoped_release rel;
              for (size_t i = 0; i < n; ++i) {
                  self.Process(in[i]);
                  out[i] = self.GetLeft();
                  out[n + i] = self.GetRight();
              }
            }
            return make_f2(out, 2, n);
        }, "input"_a, "Process mono input, returns stereo [2, frames].");

    using DC = daisysp::Decimator;
    nb::class_<DC>(mod, "Decimator", "Bit-crushing / sample rate reduction")
        .def(nb::init<>())
        .def("init", &DC::Init)
        .def("set_downsample_factor", &DC::SetDownsampleFactor, "factor"_a)
        .def("set_bitcrush_factor", &DC::SetBitcrushFactor, "factor"_a)
        .def("set_bits_to_crush", [](DC &self, int bits) { uint8_t b = (uint8_t)bits; self.SetBitsToCrush(b); }, "bits"_a)
        .def("set_smooth_crushing", &DC::SetSmoothCrushing, "enable"_a)
        .def("process_sample", &DC::Process, "in"_a)
        .def("process", [](DC &self, ArrayF input) {
            size_t n = input.shape(0);
            auto *out = new float[n];
            const float *in = input.data();
            { nb::gil_scoped_release rel;
              for (size_t i = 0; i < n; ++i) out[i] = self.Process(in[i]);
            }
            return make_f1(out, n);
        }, "input"_a);

    using FL = daisysp::Flanger;
    nb::class_<FL>(mod, "Flanger", "Flanger effect")
        .def(nb::init<>())
        .def("init", &FL::Init, "sample_rate"_a)
        .def("set_feedback", &FL::SetFeedback, "feedback"_a)
        .def("set_lfo_depth", &FL::SetLfoDepth, "depth"_a)
        .def("set_lfo_freq", &FL::SetLfoFreq, "freq"_a)
        .def("set_delay", &FL::SetDelay, "delay"_a)
        .def("set_delay_ms", &FL::SetDelayMs, "ms"_a)
        .def("process_sample", &FL::Process, "in"_a)
        .def("process", [](FL &self, ArrayF input) {
            size_t n = input.shape(0);
            auto *out = new float[n];
            const float *in = input.data();
            { nb::gil_scoped_release rel;
              for (size_t i = 0; i < n; ++i) out[i] = self.Process(in[i]);
            }
            return make_f1(out, n);
        }, "input"_a);

    using OD = daisysp::Overdrive;
    nb::class_<OD>(mod, "Overdrive", "Soft-clipping overdrive")
        .def(nb::init<>())
        .def("init", &OD::Init)
        .def("set_drive", &OD::SetDrive, "drive"_a)
        .def("process_sample", &OD::Process, "in"_a)
        .def("process", [](OD &self, ArrayF input) {
            size_t n = input.shape(0);
            auto *out = new float[n];
            const float *in = input.data();
            { nb::gil_scoped_release rel;
              for (size_t i = 0; i < n; ++i) out[i] = self.Process(in[i]);
            }
            return make_f1(out, n);
        }, "input"_a);

    using PH = daisysp::Phaser;
    nb::class_<PH>(mod, "Phaser", "Multi-pole phaser effect")
        .def(nb::init<>())
        .def("init", &PH::Init, "sample_rate"_a)
        .def("set_poles", &PH::SetPoles, "poles"_a)
        .def("set_lfo_depth", &PH::SetLfoDepth, "depth"_a)
        .def("set_lfo_freq", &PH::SetLfoFreq, "freq"_a)
        .def("set_freq", &PH::SetFreq, "freq"_a)
        .def("set_feedback", &PH::SetFeedback, "feedback"_a)
        .def("process_sample", &PH::Process, "in"_a)
        .def("process", [](PH &self, ArrayF input) {
            size_t n = input.shape(0);
            auto *out = new float[n];
            const float *in = input.data();
            { nb::gil_scoped_release rel;
              for (size_t i = 0; i < n; ++i) out[i] = self.Process(in[i]);
            }
            return make_f1(out, n);
        }, "input"_a);

    // PitchShifter (Process takes float&)
    using PS = daisysp::PitchShifter;
    nb::class_<PS>(mod, "PitchShifter", "Time-domain pitch shifter")
        .def(nb::init<>())
        .def("init", &PS::Init, "sample_rate"_a)
        .def("set_transposition", [](PS &self, float t) { self.SetTransposition(t); }, "semitones"_a)
        .def("set_del_size", &PS::SetDelSize, "size"_a)
        .def("set_fun", &PS::SetFun, "fun"_a)
        .def("process_sample", [](PS &self, float in) { return self.Process(in); }, "in"_a)
        .def("process", [](PS &self, ArrayF input) {
            size_t n = input.shape(0);
            auto *out = new float[n];
            float val;
            { nb::gil_scoped_release rel;
              for (size_t i = 0; i < n; ++i) { val = input.data()[i]; out[i] = self.Process(val); }
            }
            return make_f1(out, n);
        }, "input"_a);

    using SRR = daisysp::SampleRateReducer;
    nb::class_<SRR>(mod, "SampleRateReducer", "Sample rate reduction effect")
        .def(nb::init<>())
        .def("init", &SRR::Init)
        .def("set_freq", &SRR::SetFreq, "freq"_a)
        .def("process_sample", &SRR::Process, "in"_a)
        .def("process", [](SRR &self, ArrayF input) {
            size_t n = input.shape(0);
            auto *out = new float[n];
            const float *in = input.data();
            { nb::gil_scoped_release rel;
              for (size_t i = 0; i < n; ++i) out[i] = self.Process(in[i]);
            }
            return make_f1(out, n);
        }, "input"_a);

    using TR = daisysp::Tremolo;
    nb::class_<TR>(mod, "Tremolo", "Tremolo effect")
        .def(nb::init<>())
        .def("init", &TR::Init, "sample_rate"_a)
        .def("set_freq", &TR::SetFreq, "freq"_a)
        .def("set_waveform", &TR::SetWaveform, "waveform"_a)
        .def("set_depth", &TR::SetDepth, "depth"_a)
        .def("process_sample", &TR::Process, "in"_a)
        .def("process", [](TR &self, ArrayF input) {
            size_t n = input.shape(0);
            auto *out = new float[n];
            const float *in = input.data();
            { nb::gil_scoped_release rel;
              for (size_t i = 0; i < n; ++i) out[i] = self.Process(in[i]);
            }
            return make_f1(out, n);
        }, "input"_a);

    using WF = daisysp::Wavefolder;
    nb::class_<WF>(mod, "Wavefolder", "Wave folding distortion")
        .def(nb::init<>())
        .def("init", &WF::Init)
        .def("set_gain", &WF::SetGain, "gain"_a)
        .def("set_offset", &WF::SetOffset, "offset"_a)
        .def("process_sample", &WF::Process, "in"_a)
        .def("process", [](WF &self, ArrayF input) {
            size_t n = input.shape(0);
            auto *out = new float[n];
            const float *in = input.data();
            { nb::gil_scoped_release rel;
              for (size_t i = 0; i < n; ++i) out[i] = self.Process(in[i]);
            }
            return make_f1(out, n);
        }, "input"_a);

    // LGPL effects

    using BC = daisysp::Bitcrush;
    nb::class_<BC>(mod, "Bitcrush", "Bitcrusher with downsample (LGPL)")
        .def(nb::init<>())
        .def("init", &BC::Init, "sample_rate"_a)
        .def("set_bit_depth", &BC::SetBitDepth, "depth"_a)
        .def("set_crush_rate", &BC::SetCrushRate, "rate"_a)
        .def("process_sample", &BC::Process, "in"_a)
        .def("process", [](BC &self, ArrayF input) {
            size_t n = input.shape(0);
            auto *out = new float[n];
            const float *in = input.data();
            { nb::gil_scoped_release rel;
              for (size_t i = 0; i < n; ++i) out[i] = self.Process(in[i]);
            }
            return make_f1(out, n);
        }, "input"_a);

    using FD = daisysp::Fold;
    nb::class_<FD>(mod, "Fold", "Fold distortion (LGPL)")
        .def(nb::init<>())
        .def("init", &FD::Init)
        .def("set_increment", &FD::SetIncrement, "incr"_a)
        .def("process_sample", &FD::Process, "in"_a)
        .def("process", [](FD &self, ArrayF input) {
            size_t n = input.shape(0);
            auto *out = new float[n];
            const float *in = input.data();
            { nb::gil_scoped_release rel;
              for (size_t i = 0; i < n; ++i) out[i] = self.Process(in[i]);
            }
            return make_f1(out, n);
        }, "input"_a);

    // ReverbSc (stereo I/O)
    using RV = daisysp::ReverbSc;
    nb::class_<RV>(mod, "ReverbSc", "Stereo reverb (LGPL)")
        .def(nb::init<>())
        .def("init", &RV::Init, "sample_rate"_a)
        .def("set_feedback", [](RV &self, float fb) { self.SetFeedback(fb); }, "feedback"_a)
        .def("set_lp_freq", [](RV &self, float f) { self.SetLpFreq(f); }, "freq"_a)
        .def("process_sample", [](RV &self, float in1, float in2) {
            float out1, out2;
            self.Process(in1, in2, &out1, &out2);
            return nb::make_tuple(out1, out2);
        }, "in1"_a, "in2"_a, "Process one stereo sample pair. Returns (out1, out2).")
        .def("process", [](RV &self, Array2F input) {
            if (input.shape(0) != 2) throw std::invalid_argument("Expected [2, frames] input");
            size_t n = input.shape(1);
            auto *out = new float[2 * n];
            const float *l = input.data();
            const float *r = input.data() + n;
            float o1, o2;
            { nb::gil_scoped_release rel;
              for (size_t i = 0; i < n; ++i) {
                  self.Process(l[i], r[i], &o1, &o2);
                  out[i] = o1;
                  out[n + i] = o2;
              }
            }
            return make_f2(out, 2, n);
        }, "input"_a, "Process stereo buffer [2, frames] -> [2, frames].");
}

// ============================================================================
// Dynamics
// ============================================================================

static void bind_daisysp_dynamics(nb::module_ &m) {
    auto mod = m.def_submodule("dynamics", "DaisySP dynamics modules");

    // CrossFade curve constants
    mod.attr("CROSSFADE_LIN") = (int)daisysp::CROSSFADE_LIN;
    mod.attr("CROSSFADE_CPOW") = (int)daisysp::CROSSFADE_CPOW;
    mod.attr("CROSSFADE_LOG") = (int)daisysp::CROSSFADE_LOG;
    mod.attr("CROSSFADE_EXP") = (int)daisysp::CROSSFADE_EXP;

    using CF = daisysp::CrossFade;
    nb::class_<CF>(mod, "CrossFade", "Crossfade between two signals")
        .def(nb::init<>())
        .def("init", [](CF &self, int curve) { self.Init(curve); }, "curve"_a = (int)daisysp::CROSSFADE_LIN)
        .def("set_pos", &CF::SetPos, "pos"_a)
        .def("set_curve", [](CF &self, int curve) { self.SetCurve((uint8_t)curve); }, "curve"_a)
        .def("process_sample", [](CF &self, float in1, float in2) {
            return self.Process(in1, in2);
        }, "in1"_a, "in2"_a)
        .def("process", [](CF &self, ArrayF input1, ArrayF input2) {
            size_t n = input1.shape(0);
            if (input2.shape(0) != n) throw std::invalid_argument("Inputs must have same length");
            auto *out = new float[n];
            float a, b;
            { nb::gil_scoped_release rel;
              for (size_t i = 0; i < n; ++i) {
                  a = input1.data()[i]; b = input2.data()[i];
                  out[i] = self.Process(a, b);
              }
            }
            return make_f1(out, n);
        }, "input1"_a, "input2"_a);

    using LM = daisysp::Limiter;
    nb::class_<LM>(mod, "Limiter", "Peak limiter (block-based)")
        .def(nb::init<>())
        .def("init", &LM::Init)
        .def("process", [](LM &self, ArrayF input, float pre_gain) {
            size_t n = input.shape(0);
            auto *out = new float[n];
            std::copy(input.data(), input.data() + n, out);
            { nb::gil_scoped_release rel;
              self.ProcessBlock(out, n, pre_gain);
            }
            return make_f1(out, n);
        }, "input"_a, "pre_gain"_a = 1.0f);

    // LGPL dynamics

    using BL = daisysp::Balance;
    nb::class_<BL>(mod, "Balance", "Balance signal level to match comparator (LGPL)")
        .def(nb::init<>())
        .def("init", &BL::Init, "sample_rate"_a)
        .def("set_cutoff", &BL::SetCutoff, "cutoff"_a)
        .def("process_sample", &BL::Process, "sig"_a, "comp"_a)
        .def("process", [](BL &self, ArrayF sig, ArrayF comp) {
            size_t n = sig.shape(0);
            if (comp.shape(0) != n) throw std::invalid_argument("Inputs must have same length");
            auto *out = new float[n];
            const float *s = sig.data();
            const float *c = comp.data();
            { nb::gil_scoped_release rel;
              for (size_t i = 0; i < n; ++i) out[i] = self.Process(s[i], c[i]);
            }
            return make_f1(out, n);
        }, "sig"_a, "comp"_a);

    using CP = daisysp::Compressor;
    nb::class_<CP>(mod, "Compressor", "Dynamics compressor (LGPL)")
        .def(nb::init<>())
        .def("init", &CP::Init, "sample_rate"_a)
        .def("set_ratio", &CP::SetRatio, "ratio"_a)
        .def("set_threshold", &CP::SetThreshold, "threshold"_a)
        .def("set_attack", &CP::SetAttack, "attack"_a)
        .def("set_release", &CP::SetRelease, "release"_a)
        .def("set_makeup", &CP::SetMakeup, "gain"_a)
        .def("auto_makeup", [](CP &self, bool enable) { self.AutoMakeup(enable); }, "enable"_a)
        .def("get_ratio", &CP::GetRatio)
        .def("get_threshold", &CP::GetThreshold)
        .def("get_attack", &CP::GetAttack)
        .def("get_release", &CP::GetRelease)
        .def("get_makeup", &CP::GetMakeup)
        .def("get_gain", &CP::GetGain)
        .def("process_sample", [](CP &self, float in) { return self.Process(in); }, "in"_a)
        .def("apply", &CP::Apply, "in"_a)
        .def("process", [](CP &self, ArrayF input) {
            size_t n = input.shape(0);
            auto *out = new float[n];
            const float *in = input.data();
            { nb::gil_scoped_release rel;
              for (size_t i = 0; i < n; ++i) out[i] = self.Process(in[i]);
            }
            return make_f1(out, n);
        }, "input"_a);
}

// ============================================================================
// Control (Envelopes)
// ============================================================================

static void bind_daisysp_control(nb::module_ &m) {
    auto mod = m.def_submodule("control", "DaisySP control/envelope modules");

    // AdEnv segment constants
    mod.attr("ADENV_SEG_IDLE") = (int)daisysp::ADENV_SEG_IDLE;
    mod.attr("ADENV_SEG_ATTACK") = (int)daisysp::ADENV_SEG_ATTACK;
    mod.attr("ADENV_SEG_DECAY") = (int)daisysp::ADENV_SEG_DECAY;

    using AE = daisysp::AdEnv;
    nb::class_<AE>(mod, "AdEnv", "Attack-decay envelope")
        .def(nb::init<>())
        .def("init", &AE::Init, "sample_rate"_a)
        .def("trigger", &AE::Trigger)
        .def("set_time", [](AE &self, int seg, float t) { self.SetTime((uint8_t)seg, t); }, "seg"_a, "time"_a)
        .def("set_curve", &AE::SetCurve, "scalar"_a)
        .def("set_min", &AE::SetMin, "min"_a)
        .def("set_max", &AE::SetMax, "max"_a)
        .def("get_value", &AE::GetValue)
        .def("get_current_segment", [](AE &self) { return (int)self.GetCurrentSegment(); })
        .def("is_running", &AE::IsRunning)
        .def("process_sample", &AE::Process)
        .def("process", [](AE &self, int n) {
            auto *out = new float[n];
            { nb::gil_scoped_release rel;
              for (int i = 0; i < n; ++i) out[i] = self.Process();
            }
            return make_f1(out, (size_t)n);
        }, "n"_a, "Generate n envelope samples.");

    // ADSR segment constants
    mod.attr("ADSR_SEG_IDLE") = (int)daisysp::ADSR_SEG_IDLE;
    mod.attr("ADSR_SEG_ATTACK") = (int)daisysp::ADSR_SEG_ATTACK;
    mod.attr("ADSR_SEG_DECAY") = (int)daisysp::ADSR_SEG_DECAY;
    mod.attr("ADSR_SEG_RELEASE") = (int)daisysp::ADSR_SEG_RELEASE;

    using AD = daisysp::Adsr;
    nb::class_<AD>(mod, "Adsr", "ADSR envelope")
        .def(nb::init<>())
        .def("init", &AD::Init, "sample_rate"_a, "block_size"_a = 1)
        .def("retrigger", &AD::Retrigger, "hard"_a)
        .def("set_attack_time", [](AD &self, float t) { self.SetAttackTime(t); }, "time"_a)
        .def("set_decay_time", &AD::SetDecayTime, "time"_a)
        .def("set_release_time", &AD::SetReleaseTime, "time"_a)
        .def("set_sustain_level", &AD::SetSustainLevel, "level"_a)
        .def("set_time", [](AD &self, int seg, float t) { self.SetTime((uint8_t)seg, t); }, "seg"_a, "time"_a)
        .def("get_current_segment", [](AD &self) { return (int)self.GetCurrentSegment(); })
        .def("is_running", &AD::IsRunning)
        .def("process_sample", &AD::Process, "gate"_a)
        .def("process", [](AD &self, ArrayF gate) {
            size_t n = gate.shape(0);
            auto *out = new float[n];
            const float *g = gate.data();
            { nb::gil_scoped_release rel;
              for (size_t i = 0; i < n; ++i) out[i] = self.Process(g[i] > 0.5f);
            }
            return make_f1(out, n);
        }, "gate"_a, "Process gate signal (>0.5 = on).");

    using PHS = daisysp::Phasor;
    nb::class_<PHS>(mod, "Phasor", "Phase accumulator (0-1 ramp)")
        .def(nb::init<>())
        .def("init", [](PHS &self, float sr, float freq) { self.Init(sr, freq); },
             "sample_rate"_a, "freq"_a = 1.0f)
        .def("set_freq", &PHS::SetFreq, "freq"_a)
        .def("get_freq", &PHS::GetFreq)
        .def("process_sample", &PHS::Process)
        .def("process", [](PHS &self, int n) {
            auto *out = new float[n];
            { nb::gil_scoped_release rel;
              for (int i = 0; i < n; ++i) out[i] = self.Process();
            }
            return make_f1(out, (size_t)n);
        }, "n"_a);

    // LGPL
    using LN = daisysp::Line;
    nb::class_<LN>(mod, "Line", "Linear ramp generator (LGPL)")
        .def(nb::init<>())
        .def("init", &LN::Init, "sample_rate"_a)
        .def("start", &LN::Start, "start"_a, "end"_a, "dur"_a)
        .def("process", [](LN &self, int n) {
            auto *out = new float[n];
            uint8_t finished = 0;
            { nb::gil_scoped_release rel;
              for (int i = 0; i < n; ++i) {
                  out[i] = self.Process(&finished);
              }
            }
            return nb::make_tuple(make_f1(out, (size_t)n), (bool)finished);
        }, "n"_a, "Generate n samples. Returns (samples, finished).");
}

// ============================================================================
// Noise
// ============================================================================

static void bind_daisysp_noise(nb::module_ &m) {
    auto mod = m.def_submodule("noise", "DaisySP noise modules");

    using WN = daisysp::WhiteNoise;
    nb::class_<WN>(mod, "WhiteNoise", "White noise generator")
        .def(nb::init<>())
        .def("init", &WN::Init)
        .def("set_amp", &WN::SetAmp, "amp"_a)
        .def("process_sample", &WN::Process)
        .def("process", [](WN &self, int n) {
            auto *out = new float[n];
            { nb::gil_scoped_release rel;
              for (int i = 0; i < n; ++i) out[i] = self.Process();
            }
            return make_f1(out, (size_t)n);
        }, "n"_a);

    using DU = daisysp::Dust;
    nb::class_<DU>(mod, "Dust", "Random impulse generator")
        .def(nb::init<>())
        .def("init", &DU::Init)
        .def("set_density", &DU::SetDensity, "density"_a)
        .def("process_sample", &DU::Process)
        .def("process", [](DU &self, int n) {
            auto *out = new float[n];
            { nb::gil_scoped_release rel;
              for (int i = 0; i < n; ++i) out[i] = self.Process();
            }
            return make_f1(out, (size_t)n);
        }, "n"_a);

    using CN = daisysp::ClockedNoise;
    nb::class_<CN>(mod, "ClockedNoise", "Sample-and-hold noise with variable clock")
        .def(nb::init<>())
        .def("init", &CN::Init, "sample_rate"_a)
        .def("set_freq", &CN::SetFreq, "freq"_a)
        .def("sync", &CN::Sync)
        .def("process_sample", &CN::Process)
        .def("process", [](CN &self, int n) {
            auto *out = new float[n];
            { nb::gil_scoped_release rel;
              for (int i = 0; i < n; ++i) out[i] = self.Process();
            }
            return make_f1(out, (size_t)n);
        }, "n"_a);

    // FractalRandomGenerator instantiated with ClockedNoise, order 3
    using FRG = daisysp::FractalRandomGenerator<daisysp::ClockedNoise, 3>;
    nb::class_<FRG>(mod, "FractalRandomGenerator", "Fractal noise (stacked ClockedNoise, 3 octaves)")
        .def(nb::init<>())
        .def("init", &FRG::Init, "sample_rate"_a)
        .def("set_freq", &FRG::SetFreq, "freq"_a)
        .def("set_color", &FRG::SetColor, "color"_a)
        .def("process_sample", &FRG::Process)
        .def("process", [](FRG &self, int n) {
            auto *out = new float[n];
            { nb::gil_scoped_release rel;
              for (int i = 0; i < n; ++i) out[i] = self.Process();
            }
            return make_f1(out, (size_t)n);
        }, "n"_a);

    using GL = daisysp::GrainletOscillator;
    nb::class_<GL>(mod, "GrainletOscillator", "Grainlet oscillator")
        .def(nb::init<>())
        .def("init", &GL::Init, "sample_rate"_a)
        .def("set_freq", &GL::SetFreq, "freq"_a)
        .def("set_formant_freq", &GL::SetFormantFreq, "freq"_a)
        .def("set_shape", &GL::SetShape, "shape"_a)
        .def("set_bleed", &GL::SetBleed, "bleed"_a)
        .def("process_sample", &GL::Process)
        .def("process", [](GL &self, int n) {
            auto *out = new float[n];
            { nb::gil_scoped_release rel;
              for (int i = 0; i < n; ++i) out[i] = self.Process();
            }
            return make_f1(out, (size_t)n);
        }, "n"_a);

    using PT = daisysp::Particle;
    nb::class_<PT>(mod, "Particle", "Random impulse train through resonant filter")
        .def(nb::init<>())
        .def("init", &PT::Init, "sample_rate"_a)
        .def("set_freq", &PT::SetFreq, "freq"_a)
        .def("set_resonance", &PT::SetResonance, "res"_a)
        .def("set_random_freq", &PT::SetRandomFreq, "freq"_a)
        .def("set_density", &PT::SetDensity, "density"_a)
        .def("set_gain", &PT::SetGain, "gain"_a)
        .def("set_spread", &PT::SetSpread, "spread"_a)
        .def("set_sync", &PT::SetSync, "sync"_a)
        .def("process_sample", &PT::Process)
        .def("process", [](PT &self, int n) {
            auto *out = new float[n];
            { nb::gil_scoped_release rel;
              for (int i = 0; i < n; ++i) out[i] = self.Process();
            }
            return make_f1(out, (size_t)n);
        }, "n"_a);

    using SR = daisysp::SmoothRandomGenerator;
    nb::class_<SR>(mod, "SmoothRandomGenerator", "Hermite-interpolated smooth random")
        .def(nb::init<>())
        .def("init", &SR::Init, "sample_rate"_a)
        .def("set_freq", &SR::SetFreq, "freq"_a)
        .def("process_sample", &SR::Process)
        .def("process", [](SR &self, int n) {
            auto *out = new float[n];
            { nb::gil_scoped_release rel;
              for (int i = 0; i < n; ++i) out[i] = self.Process();
            }
            return make_f1(out, (size_t)n);
        }, "n"_a);
}

// ============================================================================
// Drums
// ============================================================================

static void bind_daisysp_drums(nb::module_ &m) {
    auto mod = m.def_submodule("drums", "DaisySP drum modules");

    using ABD = daisysp::AnalogBassDrum;
    nb::class_<ABD>(mod, "AnalogBassDrum", "808-style analog bass drum")
        .def(nb::init<>())
        .def("init", &ABD::Init, "sample_rate"_a)
        .def("trig", &ABD::Trig)
        .def("set_sustain", &ABD::SetSustain, "sustain"_a)
        .def("set_accent", &ABD::SetAccent, "accent"_a)
        .def("set_freq", &ABD::SetFreq, "freq"_a)
        .def("set_tone", &ABD::SetTone, "tone"_a)
        .def("set_decay", &ABD::SetDecay, "decay"_a)
        .def("set_attack_fm_amount", &ABD::SetAttackFmAmount, "amount"_a)
        .def("set_self_fm_amount", &ABD::SetSelfFmAmount, "amount"_a)
        .def("process_sample", [](ABD &self, bool trigger) { return self.Process(trigger); },
             "trigger"_a = false)
        .def("process", [](ABD &self, int n) {
            auto *out = new float[n];
            { nb::gil_scoped_release rel;
              out[0] = self.Process(true);
              for (int i = 1; i < n; ++i) out[i] = self.Process(false);
            }
            return make_f1(out, (size_t)n);
        }, "n"_a, "Trigger and generate n samples.");

    using ASD = daisysp::AnalogSnareDrum;
    nb::class_<ASD>(mod, "AnalogSnareDrum", "808-style analog snare drum")
        .def(nb::init<>())
        .def("init", &ASD::Init, "sample_rate"_a)
        .def("trig", &ASD::Trig)
        .def("set_sustain", &ASD::SetSustain, "sustain"_a)
        .def("set_accent", &ASD::SetAccent, "accent"_a)
        .def("set_freq", &ASD::SetFreq, "freq"_a)
        .def("set_tone", &ASD::SetTone, "tone"_a)
        .def("set_decay", &ASD::SetDecay, "decay"_a)
        .def("set_snappy", &ASD::SetSnappy, "snappy"_a)
        .def("process_sample", [](ASD &self, bool trigger) { return self.Process(trigger); },
             "trigger"_a = false)
        .def("process", [](ASD &self, int n) {
            auto *out = new float[n];
            { nb::gil_scoped_release rel;
              out[0] = self.Process(true);
              for (int i = 1; i < n; ++i) out[i] = self.Process(false);
            }
            return make_f1(out, (size_t)n);
        }, "n"_a, "Trigger and generate n samples.");

    // HiHat -- instantiate with default template params
    using HH = daisysp::HiHat<>;
    nb::class_<HH>(mod, "HiHat", "808-style hi-hat")
        .def(nb::init<>())
        .def("init", &HH::Init, "sample_rate"_a)
        .def("trig", &HH::Trig)
        .def("set_sustain", &HH::SetSustain, "sustain"_a)
        .def("set_accent", &HH::SetAccent, "accent"_a)
        .def("set_freq", &HH::SetFreq, "freq"_a)
        .def("set_tone", &HH::SetTone, "tone"_a)
        .def("set_decay", &HH::SetDecay, "decay"_a)
        .def("set_noisiness", &HH::SetNoisiness, "noisiness"_a)
        .def("process_sample", [](HH &self, bool trigger) { return self.Process(trigger); },
             "trigger"_a = false)
        .def("process", [](HH &self, int n) {
            auto *out = new float[n];
            { nb::gil_scoped_release rel;
              out[0] = self.Process(true);
              for (int i = 1; i < n; ++i) out[i] = self.Process(false);
            }
            return make_f1(out, (size_t)n);
        }, "n"_a, "Trigger and generate n samples.");

    using SBD = daisysp::SyntheticBassDrum;
    nb::class_<SBD>(mod, "SyntheticBassDrum", "Synthetic bass drum")
        .def(nb::init<>())
        .def("init", &SBD::Init, "sample_rate"_a)
        .def("trig", &SBD::Trig)
        .def("set_sustain", &SBD::SetSustain, "sustain"_a)
        .def("set_accent", &SBD::SetAccent, "accent"_a)
        .def("set_freq", &SBD::SetFreq, "freq"_a)
        .def("set_tone", &SBD::SetTone, "tone"_a)
        .def("set_decay", &SBD::SetDecay, "decay"_a)
        .def("set_dirtiness", &SBD::SetDirtiness, "dirtiness"_a)
        .def("set_fm_envelope_amount", &SBD::SetFmEnvelopeAmount, "amount"_a)
        .def("set_fm_envelope_decay", &SBD::SetFmEnvelopeDecay, "decay"_a)
        .def("process_sample", [](SBD &self, bool trigger) { return self.Process(trigger); },
             "trigger"_a = false)
        .def("process", [](SBD &self, int n) {
            auto *out = new float[n];
            { nb::gil_scoped_release rel;
              out[0] = self.Process(true);
              for (int i = 1; i < n; ++i) out[i] = self.Process(false);
            }
            return make_f1(out, (size_t)n);
        }, "n"_a, "Trigger and generate n samples.");

    using SSD = daisysp::SyntheticSnareDrum;
    nb::class_<SSD>(mod, "SyntheticSnareDrum", "Synthetic snare drum")
        .def(nb::init<>())
        .def("init", &SSD::Init, "sample_rate"_a)
        .def("trig", &SSD::Trig)
        .def("set_sustain", &SSD::SetSustain, "sustain"_a)
        .def("set_accent", &SSD::SetAccent, "accent"_a)
        .def("set_freq", &SSD::SetFreq, "freq"_a)
        .def("set_fm_amount", &SSD::SetFmAmount, "amount"_a)
        .def("set_decay", &SSD::SetDecay, "decay"_a)
        .def("set_snappy", &SSD::SetSnappy, "snappy"_a)
        .def("process_sample", [](SSD &self, bool trigger) { return self.Process(trigger); },
             "trigger"_a = false)
        .def("process", [](SSD &self, int n) {
            auto *out = new float[n];
            { nb::gil_scoped_release rel;
              out[0] = self.Process(true);
              for (int i = 1; i < n; ++i) out[i] = self.Process(false);
            }
            return make_f1(out, (size_t)n);
        }, "n"_a, "Trigger and generate n samples.");
}

// ============================================================================
// Physical Modeling
// ============================================================================

static void bind_daisysp_physical_modeling(nb::module_ &m) {
    auto mod = m.def_submodule("physical_modeling", "DaisySP physical modeling modules");

    using DR = daisysp::Drip;
    nb::class_<DR>(mod, "Drip", "Water drip physical model")
        .def(nb::init<>())
        .def("init", &DR::Init, "sample_rate"_a, "dettack"_a)
        .def("process_sample", &DR::Process, "trig"_a)
        .def("process", [](DR &self, int n) {
            auto *out = new float[n];
            { nb::gil_scoped_release rel;
              out[0] = self.Process(true);
              for (int i = 1; i < n; ++i) out[i] = self.Process(false);
            }
            return make_f1(out, (size_t)n);
        }, "n"_a, "Trigger and generate n samples.");

    // String (Karplus-Strong) -- Process takes const float, so direct bind is fine
    using ST = daisysp::String;
    nb::class_<ST>(mod, "String", "Karplus-Strong string model")
        .def(nb::init<>())
        .def("init", &ST::Init, "sample_rate"_a)
        .def("reset", &ST::Reset)
        .def("set_freq", &ST::SetFreq, "freq"_a)
        .def("set_non_linearity", &ST::SetNonLinearity, "amount"_a)
        .def("set_brightness", &ST::SetBrightness, "brightness"_a)
        .def("set_damping", &ST::SetDamping, "damping"_a)
        .def("process_sample", &ST::Process, "in"_a)
        .def("process", [](ST &self, ArrayF input) {
            size_t n = input.shape(0);
            auto *out = new float[n];
            const float *in = input.data();
            { nb::gil_scoped_release rel;
              for (size_t i = 0; i < n; ++i) out[i] = self.Process(in[i]);
            }
            return make_f1(out, n);
        }, "input"_a, "Process excitation input.");

    using MV = daisysp::ModalVoice;
    nb::class_<MV>(mod, "ModalVoice", "Modal percussion voice")
        .def(nb::init<>())
        .def("init", &MV::Init, "sample_rate"_a)
        .def("trig", &MV::Trig)
        .def("set_sustain", &MV::SetSustain, "sustain"_a)
        .def("set_freq", &MV::SetFreq, "freq"_a)
        .def("set_accent", &MV::SetAccent, "accent"_a)
        .def("set_structure", &MV::SetStructure, "structure"_a)
        .def("set_brightness", &MV::SetBrightness, "brightness"_a)
        .def("set_damping", &MV::SetDamping, "damping"_a)
        .def("get_aux", &MV::GetAux)
        .def("process_sample", [](MV &self, bool trigger) { return self.Process(trigger); },
             "trigger"_a = false)
        .def("process", [](MV &self, int n) {
            auto *out = new float[n];
            { nb::gil_scoped_release rel;
              out[0] = self.Process(true);
              for (int i = 1; i < n; ++i) out[i] = self.Process(false);
            }
            return make_f1(out, (size_t)n);
        }, "n"_a, "Trigger and generate n samples.");

    using RS = daisysp::Resonator;
    nb::class_<RS>(mod, "Resonator", "Multi-mode resonant body")
        .def(nb::init<>())
        .def("init", &RS::Init, "position"_a, "resolution"_a, "sample_rate"_a)
        .def("set_freq", &RS::SetFreq, "freq"_a)
        .def("set_structure", &RS::SetStructure, "structure"_a)
        .def("set_brightness", &RS::SetBrightness, "brightness"_a)
        .def("set_damping", &RS::SetDamping, "damping"_a)
        .def("process_sample", &RS::Process, "in"_a)
        .def("process", [](RS &self, ArrayF input) {
            size_t n = input.shape(0);
            auto *out = new float[n];
            const float *in = input.data();
            { nb::gil_scoped_release rel;
              for (size_t i = 0; i < n; ++i) out[i] = self.Process(in[i]);
            }
            return make_f1(out, n);
        }, "input"_a);

    using SV = daisysp::StringVoice;
    nb::class_<SV>(mod, "StringVoice", "Extended Karplus-Strong string voice")
        .def(nb::init<>())
        .def("init", &SV::Init, "sample_rate"_a)
        .def("trig", &SV::Trig)
        .def("reset", &SV::Reset)
        .def("set_sustain", &SV::SetSustain, "sustain"_a)
        .def("set_freq", &SV::SetFreq, "freq"_a)
        .def("set_accent", &SV::SetAccent, "accent"_a)
        .def("set_structure", &SV::SetStructure, "structure"_a)
        .def("set_brightness", &SV::SetBrightness, "brightness"_a)
        .def("set_damping", &SV::SetDamping, "damping"_a)
        .def("get_aux", &SV::GetAux)
        .def("process_sample", [](SV &self, bool trigger) { return self.Process(trigger); },
             "trigger"_a = false)
        .def("process", [](SV &self, int n) {
            auto *out = new float[n];
            { nb::gil_scoped_release rel;
              out[0] = self.Process(true);
              for (int i = 1; i < n; ++i) out[i] = self.Process(false);
            }
            return make_f1(out, (size_t)n);
        }, "n"_a, "Trigger and generate n samples.");

    // LGPL: Pluck (needs external buffer)
    using PL = daisysp::Pluck;
    mod.attr("PLUCK_MODE_RECURSIVE") = (int)daisysp::PLUCK_MODE_RECURSIVE;
    mod.attr("PLUCK_MODE_WEIGHTED_AVERAGE") = (int)daisysp::PLUCK_MODE_WEIGHTED_AVERAGE;

    nb::class_<PL>(mod, "Pluck", "Karplus-Strong plucked string (LGPL)")
        .def("__init__", [](PL *self, float sample_rate, int npt, int mode) {
            new (self) PL();
            auto *buf = new float[npt]();
            self->Init(sample_rate, buf, npt, mode);
        }, "sample_rate"_a, "npt"_a = 256, "mode"_a = 0)
        .def("set_amp", &PL::SetAmp, "amp"_a)
        .def("set_freq", &PL::SetFreq, "freq"_a)
        .def("set_decay", &PL::SetDecay, "decay"_a)
        .def("set_damp", &PL::SetDamp, "damp"_a)
        .def("set_mode", &PL::SetMode, "mode"_a)
        .def("process_sample", [](PL &self, float trig) { return self.Process(trig); }, "trig"_a)
        .def("process", [](PL &self, int n) {
            auto *out = new float[n];
            float trig = 1.0f;
            { nb::gil_scoped_release rel;
              out[0] = self.Process(trig);
              trig = 0.0f;
              for (int i = 1; i < n; ++i) out[i] = self.Process(trig);
            }
            return make_f1(out, (size_t)n);
        }, "n"_a, "Trigger and generate n samples.");
}

// ============================================================================
// Utility
// ============================================================================

static void bind_daisysp_utility(nb::module_ &m) {
    auto mod = m.def_submodule("utility", "DaisySP utility modules");

    using DCB = daisysp::DcBlock;
    nb::class_<DCB>(mod, "DcBlock", "DC blocking filter")
        .def(nb::init<>())
        .def("init", &DCB::Init, "sample_rate"_a)
        .def("process_sample", &DCB::Process, "in"_a)
        .def("process", [](DCB &self, ArrayF input) {
            size_t n = input.shape(0);
            auto *out = new float[n];
            const float *in = input.data();
            { nb::gil_scoped_release rel;
              for (size_t i = 0; i < n; ++i) out[i] = self.Process(in[i]);
            }
            return make_f1(out, n);
        }, "input"_a);

    // DelayLine: instantiate at 96000 (2 seconds at 48kHz)
    using DL = daisysp::DelayLine<float, 96000>;
    nb::class_<DL>(mod, "DelayLine", "Delay line (max 96000 samples / 2s at 48kHz)")
        .def(nb::init<>())
        .def("init", &DL::Init)
        .def("reset", &DL::Reset)
        .def("set_delay", [](DL &self, float delay) { self.SetDelay(delay); }, "delay"_a)
        .def("write", &DL::Write, "sample"_a)
        .def("read", [](const DL &self) { return self.Read(); },
             "Read at set delay.")
        .def("read_at", [](DL &self, float delay) { return self.Read(delay); }, "delay"_a,
             "Read at specific delay.")
        .def("read_hermite", &DL::ReadHermite, "delay"_a)
        .def("process", [](DL &self, ArrayF input, float delay_samples) {
            size_t n = input.shape(0);
            auto *out = new float[n];
            const float *in = input.data();
            { nb::gil_scoped_release rel;
              for (size_t i = 0; i < n; ++i) {
                  self.Write(in[i]);
                  out[i] = self.Read(delay_samples);
              }
            }
            return make_f1(out, n);
        }, "input"_a, "delay_samples"_a, "Write input and read with fixed delay.");

    // Looper (needs external buffer)
    using LO = daisysp::Looper;
    nb::enum_<LO::Mode>(mod, "LooperMode")
        .value("NORMAL", LO::Mode::NORMAL)
        .value("ONETIME_DUB", LO::Mode::ONETIME_DUB)
        .value("REPLACE", LO::Mode::REPLACE)
        .value("FRIPPERTRONICS", LO::Mode::FRIPPERTRONICS);

    nb::class_<LO>(mod, "Looper", "Audio looper")
        .def("__init__", [](LO *self, size_t size) {
            new (self) LO();
            auto *buf = new float[size]();
            self->Init(buf, size);
        }, "size"_a = 48000)
        .def("trig_record", &LO::TrigRecord)
        .def("set_mode", &LO::SetMode, "mode"_a)
        .def("get_mode", &LO::GetMode)
        .def("set_reverse", &LO::SetReverse, "state"_a)
        .def("toggle_reverse", &LO::ToggleReverse)
        .def("get_reverse", &LO::GetReverse)
        .def("set_half_speed", &LO::SetHalfSpeed, "state"_a)
        .def("toggle_half_speed", &LO::ToggleHalfSpeed)
        .def("get_half_speed", &LO::GetHalfSpeed)
        .def("clear", &LO::Clear)
        .def("recording", &LO::Recording)
        .def("is_near_beginning", &LO::IsNearBeginning)
        .def("process_sample", &LO::Process, "input"_a)
        .def("process", [](LO &self, ArrayF input) {
            size_t n = input.shape(0);
            auto *out = new float[n];
            const float *in = input.data();
            { nb::gil_scoped_release rel;
              for (size_t i = 0; i < n; ++i) out[i] = self.Process(in[i]);
            }
            return make_f1(out, n);
        }, "input"_a);

    using MT = daisysp::Maytrig;
    nb::class_<MT>(mod, "Maytrig", "Probabilistic trigger")
        .def(nb::init<>())
        .def("process", &MT::Process, "prob"_a);

    using ME = daisysp::Metro;
    nb::class_<ME>(mod, "Metro", "Metronome / clock generator")
        .def(nb::init<>())
        .def("init", &ME::Init, "freq"_a, "sample_rate"_a)
        .def("set_freq", &ME::SetFreq, "freq"_a)
        .def("get_freq", &ME::GetFreq)
        .def("reset", &ME::Reset)
        .def("process_sample", [](ME &self) { return (bool)self.Process(); })
        .def("process", [](ME &self, int n) {
            auto *out = new float[n];
            { nb::gil_scoped_release rel;
              for (int i = 0; i < n; ++i) out[i] = (float)self.Process();
            }
            return make_f1(out, (size_t)n);
        }, "n"_a);

    using SH = daisysp::SampleHold;
    nb::enum_<SH::Mode>(mod, "SampleHoldMode")
        .value("SAMPLE_HOLD", SH::MODE_SAMPLE_HOLD)
        .value("TRACK_HOLD", SH::MODE_TRACK_HOLD);

    nb::class_<SH>(mod, "SampleHold", "Sample-and-hold / track-and-hold")
        .def(nb::init<>())
        .def("process", &SH::Process, "trigger"_a, "input"_a, "mode"_a = SH::MODE_SAMPLE_HOLD);

    // LGPL utility

    using JT = daisysp::Jitter;
    nb::class_<JT>(mod, "Jitter", "Random line segment generator (LGPL)")
        .def(nb::init<>())
        .def("init", &JT::Init, "sample_rate"_a)
        .def("set_cps_min", &JT::SetCpsMin, "cps_min"_a)
        .def("set_cps_max", &JT::SetCpsMax, "cps_max"_a)
        .def("set_amp", &JT::SetAmp, "amp"_a)
        .def("process_sample", &JT::Process)
        .def("process", [](JT &self, int n) {
            auto *out = new float[n];
            { nb::gil_scoped_release rel;
              for (int i = 0; i < n; ++i) out[i] = self.Process();
            }
            return make_f1(out, (size_t)n);
        }, "n"_a);

    using PO = daisysp::Port;
    nb::class_<PO>(mod, "Port", "Portamento / glide filter (LGPL)")
        .def(nb::init<>())
        .def("init", &PO::Init, "sample_rate"_a, "htime"_a)
        .def("set_htime", &PO::SetHtime, "htime"_a)
        .def("get_htime", &PO::GetHtime)
        .def("process_sample", &PO::Process, "in"_a)
        .def("process", [](PO &self, ArrayF input) {
            size_t n = input.shape(0);
            auto *out = new float[n];
            const float *in = input.data();
            { nb::gil_scoped_release rel;
              for (size_t i = 0; i < n; ++i) out[i] = self.Process(in[i]);
            }
            return make_f1(out, n);
        }, "input"_a);
}

// ============================================================================
// Public entry point
// ============================================================================

void bind_daisysp(nb::module_ &m) {
    auto dsp = m.def_submodule("daisysp", "DaisySP bindings");
    bind_daisysp_oscillators(dsp);
    bind_daisysp_filters(dsp);
    bind_daisysp_effects(dsp);
    bind_daisysp_dynamics(dsp);
    bind_daisysp_control(dsp);
    bind_daisysp_noise(dsp);
    bind_daisysp_drums(dsp);
    bind_daisysp_physical_modeling(dsp);
    bind_daisysp_utility(dsp);
}

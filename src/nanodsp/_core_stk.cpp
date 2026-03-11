#include "_core_common.h"

#include <Stk.h>
#include <SineWave.h>
#include <Noise.h>
#include <Blit.h>
#include <BlitSaw.h>
#include <BlitSquare.h>
#include <ADSR.h>
#include <Asymp.h>
#include <Envelope.h>
#include <Modulate.h>

#include <BiQuad.h>
#include <OnePole.h>
#include <OneZero.h>
#include <TwoPole.h>
#include <TwoZero.h>
#include <PoleZero.h>
#include <Resonate.h>
#include <FormSwep.h>

#include <Delay.h>
#include <DelayA.h>
#include <DelayL.h>
#include <TapDelay.h>

#include <FreeVerb.h>
#include <JCRev.h>
#include <NRev.h>
#include <PRCRev.h>
#include <Echo.h>
#include <Chorus.h>
#include <PitShift.h>
#include <LentPitShift.h>

#include <Clarinet.h>
#include <Flute.h>
#include <Brass.h>
#include <Bowed.h>
#include <Plucked.h>
#include <Sitar.h>
#include <StifKarp.h>
#include <Saxofony.h>
#include <Recorder.h>
#include <BlowBotl.h>
#include <BlowHole.h>
#include <Whistle.h>
#include <Guitar.h>
#include <Twang.h>

using namespace stk;

// ============================================================================
// Generators
// ============================================================================

static void bind_stk_generators(nb::module_ &m) {
    auto mod = m.def_submodule("generators", "STK generators: oscillators, envelopes, noise");

    // --- SineWave ---
    nb::class_<SineWave>(mod, "SineWave", "Sinusoid oscillator using table lookup")
        .def(nb::init<>())
        .def("reset", &SineWave::reset)
        .def("set_rate", [](SineWave &s, float r) { s.setRate((StkFloat)r); }, "rate"_a)
        .def("set_frequency", [](SineWave &s, float f) { s.setFrequency((StkFloat)f); }, "frequency"_a)
        .def("add_time", [](SineWave &s, float t) { s.addTime((StkFloat)t); }, "time"_a)
        .def("add_phase", [](SineWave &s, float p) { s.addPhase((StkFloat)p); }, "phase"_a)
        .def("add_phase_offset", [](SineWave &s, float p) { s.addPhaseOffset((StkFloat)p); }, "phase_offset"_a)
        .def("process_sample", [](SineWave &s) { return (float)s.tick(); })
        .def("process", [](SineWave &s, int n) {
            auto *out = new float[n];
            { nb::gil_scoped_release rel;
              for (int i = 0; i < n; ++i) out[i] = (float)s.tick();
            }
            return make_f1(out, (size_t)n);
        }, "n"_a);

    // --- Noise ---
    nb::class_<Noise>(mod, "Noise", "White noise generator")
        .def(nb::init<unsigned int>(), "seed"_a = 0)
        .def("set_seed", &Noise::setSeed, "seed"_a = 0)
        .def("process_sample", [](Noise &s) { return (float)s.tick(); })
        .def("process", [](Noise &s, int n) {
            auto *out = new float[n];
            { nb::gil_scoped_release rel;
              for (int i = 0; i < n; ++i) out[i] = (float)s.tick();
            }
            return make_f1(out, (size_t)n);
        }, "n"_a);

    // --- Blit ---
    nb::class_<Blit>(mod, "Blit", "Band-limited impulse train")
        .def(nb::init<StkFloat>(), "frequency"_a = 220.0)
        .def("reset", &Blit::reset)
        .def("set_phase", [](Blit &s, float p) { s.setPhase((StkFloat)p); }, "phase"_a)
        .def("get_phase", [](Blit &s) { return (float)s.getPhase(); })
        .def("set_frequency", [](Blit &s, float f) { s.setFrequency((StkFloat)f); }, "frequency"_a)
        .def("set_harmonics", &Blit::setHarmonics, "n_harmonics"_a = 0)
        .def("process_sample", [](Blit &s) { return (float)s.tick(); })
        .def("process", [](Blit &s, int n) {
            auto *out = new float[n];
            { nb::gil_scoped_release rel;
              for (int i = 0; i < n; ++i) out[i] = (float)s.tick();
            }
            return make_f1(out, (size_t)n);
        }, "n"_a);

    // --- BlitSaw ---
    nb::class_<BlitSaw>(mod, "BlitSaw", "Band-limited sawtooth wave")
        .def(nb::init<StkFloat>(), "frequency"_a = 220.0)
        .def("reset", &BlitSaw::reset)
        .def("set_frequency", [](BlitSaw &s, float f) { s.setFrequency((StkFloat)f); }, "frequency"_a)
        .def("set_harmonics", &BlitSaw::setHarmonics, "n_harmonics"_a = 0)
        .def("process_sample", [](BlitSaw &s) { return (float)s.tick(); })
        .def("process", [](BlitSaw &s, int n) {
            auto *out = new float[n];
            { nb::gil_scoped_release rel;
              for (int i = 0; i < n; ++i) out[i] = (float)s.tick();
            }
            return make_f1(out, (size_t)n);
        }, "n"_a);

    // --- BlitSquare ---
    nb::class_<BlitSquare>(mod, "BlitSquare", "Band-limited square wave")
        .def(nb::init<StkFloat>(), "frequency"_a = 220.0)
        .def("reset", &BlitSquare::reset)
        .def("set_phase", [](BlitSquare &s, float p) { s.setPhase((StkFloat)p); }, "phase"_a)
        .def("get_phase", [](BlitSquare &s) { return (float)s.getPhase(); })
        .def("set_frequency", [](BlitSquare &s, float f) { s.setFrequency((StkFloat)f); }, "frequency"_a)
        .def("set_harmonics", &BlitSquare::setHarmonics, "n_harmonics"_a = 0)
        .def("process_sample", [](BlitSquare &s) { return (float)s.tick(); })
        .def("process", [](BlitSquare &s, int n) {
            auto *out = new float[n];
            { nb::gil_scoped_release rel;
              for (int i = 0; i < n; ++i) out[i] = (float)s.tick();
            }
            return make_f1(out, (size_t)n);
        }, "n"_a);

    // ADSR states
    mod.attr("ADSR_ATTACK") = (int)ADSR::ATTACK;
    mod.attr("ADSR_DECAY") = (int)ADSR::DECAY;
    mod.attr("ADSR_SUSTAIN") = (int)ADSR::SUSTAIN;
    mod.attr("ADSR_RELEASE") = (int)ADSR::RELEASE;
    mod.attr("ADSR_IDLE") = (int)ADSR::IDLE;

    // --- ADSR ---
    nb::class_<ADSR>(mod, "ADSR", "ADSR envelope generator")
        .def(nb::init<>())
        .def("key_on", &ADSR::keyOn)
        .def("key_off", &ADSR::keyOff)
        .def("set_attack_rate", [](ADSR &s, float r) { s.setAttackRate((StkFloat)r); }, "rate"_a)
        .def("set_attack_target", [](ADSR &s, float t) { s.setAttackTarget((StkFloat)t); }, "target"_a)
        .def("set_decay_rate", [](ADSR &s, float r) { s.setDecayRate((StkFloat)r); }, "rate"_a)
        .def("set_sustain_level", [](ADSR &s, float l) { s.setSustainLevel((StkFloat)l); }, "level"_a)
        .def("set_release_rate", [](ADSR &s, float r) { s.setReleaseRate((StkFloat)r); }, "rate"_a)
        .def("set_attack_time", [](ADSR &s, float t) { s.setAttackTime((StkFloat)t); }, "time"_a)
        .def("set_decay_time", [](ADSR &s, float t) { s.setDecayTime((StkFloat)t); }, "time"_a)
        .def("set_release_time", [](ADSR &s, float t) { s.setReleaseTime((StkFloat)t); }, "time"_a)
        .def("set_all_times", [](ADSR &s, float a, float d, float sl, float r) {
            s.setAllTimes((StkFloat)a, (StkFloat)d, (StkFloat)sl, (StkFloat)r);
        }, "attack_time"_a, "decay_time"_a, "sustain_level"_a, "release_time"_a)
        .def("set_target", [](ADSR &s, float t) { s.setTarget((StkFloat)t); }, "target"_a)
        .def("set_value", [](ADSR &s, float v) { s.setValue((StkFloat)v); }, "value"_a)
        .def("get_state", &ADSR::getState)
        .def("process_sample", [](ADSR &s) { return (float)s.tick(); })
        .def("process", [](ADSR &s, int n) {
            auto *out = new float[n];
            { nb::gil_scoped_release rel;
              for (int i = 0; i < n; ++i) out[i] = (float)s.tick();
            }
            return make_f1(out, (size_t)n);
        }, "n"_a);

    // --- Asymp ---
    nb::class_<Asymp>(mod, "Asymp", "Asymptotic curve envelope")
        .def(nb::init<>())
        .def("key_on", &Asymp::keyOn)
        .def("key_off", &Asymp::keyOff)
        .def("set_tau", [](Asymp &s, float t) { s.setTau((StkFloat)t); }, "tau"_a)
        .def("set_time", [](Asymp &s, float t) { s.setTime((StkFloat)t); }, "time"_a)
        .def("set_t60", [](Asymp &s, float t) { s.setT60((StkFloat)t); }, "t60"_a)
        .def("set_target", [](Asymp &s, float t) { s.setTarget((StkFloat)t); }, "target"_a)
        .def("set_value", [](Asymp &s, float v) { s.setValue((StkFloat)v); }, "value"_a)
        .def("get_state", &Asymp::getState)
        .def("process_sample", [](Asymp &s) { return (float)s.tick(); })
        .def("process", [](Asymp &s, int n) {
            auto *out = new float[n];
            { nb::gil_scoped_release rel;
              for (int i = 0; i < n; ++i) out[i] = (float)s.tick();
            }
            return make_f1(out, (size_t)n);
        }, "n"_a);

    // --- Envelope ---
    nb::class_<Envelope>(mod, "Envelope", "Linear ramp envelope")
        .def(nb::init<>())
        .def("key_on", [](Envelope &s, float t) { s.keyOn((StkFloat)t); }, "target"_a = 1.0f)
        .def("key_off", [](Envelope &s, float t) { s.keyOff((StkFloat)t); }, "target"_a = 0.0f)
        .def("set_rate", [](Envelope &s, float r) { s.setRate((StkFloat)r); }, "rate"_a)
        .def("set_time", [](Envelope &s, float t) { s.setTime((StkFloat)t); }, "time"_a)
        .def("set_target", [](Envelope &s, float t) { s.setTarget((StkFloat)t); }, "target"_a)
        .def("set_value", [](Envelope &s, float v) { s.setValue((StkFloat)v); }, "value"_a)
        .def("get_state", &Envelope::getState)
        .def("process_sample", [](Envelope &s) { return (float)s.tick(); })
        .def("process", [](Envelope &s, int n) {
            auto *out = new float[n];
            { nb::gil_scoped_release rel;
              for (int i = 0; i < n; ++i) out[i] = (float)s.tick();
            }
            return make_f1(out, (size_t)n);
        }, "n"_a);

    // --- Modulate ---
    nb::class_<Modulate>(mod, "Modulate", "Periodic/random modulation source")
        .def(nb::init<>())
        .def("reset", &Modulate::reset)
        .def("set_vibrato_rate", [](Modulate &s, float r) { s.setVibratoRate((StkFloat)r); }, "rate"_a)
        .def("set_vibrato_gain", [](Modulate &s, float g) { s.setVibratoGain((StkFloat)g); }, "gain"_a)
        .def("set_random_rate", [](Modulate &s, float r) { s.setRandomRate((StkFloat)r); }, "rate"_a)
        .def("set_random_gain", [](Modulate &s, float g) { s.setRandomGain((StkFloat)g); }, "gain"_a)
        .def("process_sample", [](Modulate &s) { return (float)s.tick(); })
        .def("process", [](Modulate &s, int n) {
            auto *out = new float[n];
            { nb::gil_scoped_release rel;
              for (int i = 0; i < n; ++i) out[i] = (float)s.tick();
            }
            return make_f1(out, (size_t)n);
        }, "n"_a);
}

// ============================================================================
// Filters
// ============================================================================

// Macro for filter process_sample/process (tick with input)
#define FILTER_PROCESS(CLS) \
    .def("process_sample", [](CLS &s, float in) { \
        return (float)s.tick((StkFloat)in); \
    }, "input"_a) \
    .def("process", [](CLS &s, ArrayF input) { \
        size_t n = input.shape(0); \
        auto *out = new float[n]; \
        const float *in = input.data(); \
        { nb::gil_scoped_release rel; \
          for (size_t i = 0; i < n; ++i) out[i] = (float)s.tick((StkFloat)in[i]); \
        } \
        return make_f1(out, n); \
    }, "input"_a)

// Macro for common Filter base methods
#define FILTER_BASE(CLS) \
    .def("set_gain", [](CLS &s, float g) { s.setGain((StkFloat)g); }, "gain"_a) \
    .def("get_gain", [](CLS &s) { return (float)s.getGain(); }) \
    .def("clear", &CLS::clear) \
    .def("phase_delay", [](CLS &s, float f) { return (float)s.phaseDelay((StkFloat)f); }, "frequency"_a)

static void bind_stk_filters(nb::module_ &m) {
    auto mod = m.def_submodule("filters", "STK digital filters");

    // --- BiQuad ---
    nb::class_<BiQuad>(mod, "BiQuad", "Two-pole, two-zero biquad filter with design methods")
        .def(nb::init<>())
        FILTER_BASE(BiQuad)
        .def("set_coefficients", [](BiQuad &s, float b0, float b1, float b2, float a1, float a2, bool clear) {
            s.setCoefficients((StkFloat)b0, (StkFloat)b1, (StkFloat)b2, (StkFloat)a1, (StkFloat)a2, clear);
        }, "b0"_a, "b1"_a, "b2"_a, "a1"_a, "a2"_a, "clear_state"_a = false)
        .def("set_b0", [](BiQuad &s, float v) { s.setB0((StkFloat)v); }, "b0"_a)
        .def("set_b1", [](BiQuad &s, float v) { s.setB1((StkFloat)v); }, "b1"_a)
        .def("set_b2", [](BiQuad &s, float v) { s.setB2((StkFloat)v); }, "b2"_a)
        .def("set_a1", [](BiQuad &s, float v) { s.setA1((StkFloat)v); }, "a1"_a)
        .def("set_a2", [](BiQuad &s, float v) { s.setA2((StkFloat)v); }, "a2"_a)
        .def("set_resonance", [](BiQuad &s, float f, float r, bool norm) {
            s.setResonance((StkFloat)f, (StkFloat)r, norm);
        }, "frequency"_a, "radius"_a, "normalize"_a = false)
        .def("set_notch", [](BiQuad &s, float f, float r) {
            s.setNotch((StkFloat)f, (StkFloat)r);
        }, "frequency"_a, "radius"_a)
        .def("set_low_pass", [](BiQuad &s, float fc, float q) {
            s.setLowPass((StkFloat)fc, (StkFloat)q);
        }, "fc"_a, "q"_a = (float)M_SQRT1_2)
        .def("set_high_pass", [](BiQuad &s, float fc, float q) {
            s.setHighPass((StkFloat)fc, (StkFloat)q);
        }, "fc"_a, "q"_a = (float)M_SQRT1_2)
        .def("set_band_pass", [](BiQuad &s, float fc, float q) {
            s.setBandPass((StkFloat)fc, (StkFloat)q);
        }, "fc"_a, "q"_a)
        .def("set_band_reject", [](BiQuad &s, float fc, float q) {
            s.setBandReject((StkFloat)fc, (StkFloat)q);
        }, "fc"_a, "q"_a)
        .def("set_all_pass", [](BiQuad &s, float fc, float q) {
            s.setAllPass((StkFloat)fc, (StkFloat)q);
        }, "fc"_a, "q"_a)
        .def("set_equal_gain_zeroes", &BiQuad::setEqualGainZeroes)
        FILTER_PROCESS(BiQuad);

    // --- OnePole ---
    nb::class_<OnePole>(mod, "OnePole", "One-pole filter")
        .def(nb::init<StkFloat>(), "pole"_a = 0.9)
        FILTER_BASE(OnePole)
        .def("set_b0", [](OnePole &s, float v) { s.setB0((StkFloat)v); }, "b0"_a)
        .def("set_a1", [](OnePole &s, float v) { s.setA1((StkFloat)v); }, "a1"_a)
        .def("set_coefficients", [](OnePole &s, float b0, float a1, bool clear) {
            s.setCoefficients((StkFloat)b0, (StkFloat)a1, clear);
        }, "b0"_a, "a1"_a, "clear_state"_a = false)
        .def("set_pole", [](OnePole &s, float p) { s.setPole((StkFloat)p); }, "pole"_a)
        FILTER_PROCESS(OnePole);

    // --- OneZero ---
    nb::class_<OneZero>(mod, "OneZero", "One-zero filter")
        .def(nb::init<StkFloat>(), "zero"_a = -1.0)
        FILTER_BASE(OneZero)
        .def("set_b0", [](OneZero &s, float v) { s.setB0((StkFloat)v); }, "b0"_a)
        .def("set_b1", [](OneZero &s, float v) { s.setB1((StkFloat)v); }, "b1"_a)
        .def("set_coefficients", [](OneZero &s, float b0, float b1, bool clear) {
            s.setCoefficients((StkFloat)b0, (StkFloat)b1, clear);
        }, "b0"_a, "b1"_a, "clear_state"_a = false)
        .def("set_zero", [](OneZero &s, float z) { s.setZero((StkFloat)z); }, "zero"_a)
        FILTER_PROCESS(OneZero);

    // --- TwoPole ---
    nb::class_<TwoPole>(mod, "TwoPole", "Two-pole filter")
        .def(nb::init<>())
        FILTER_BASE(TwoPole)
        .def("set_b0", [](TwoPole &s, float v) { s.setB0((StkFloat)v); }, "b0"_a)
        .def("set_a1", [](TwoPole &s, float v) { s.setA1((StkFloat)v); }, "a1"_a)
        .def("set_a2", [](TwoPole &s, float v) { s.setA2((StkFloat)v); }, "a2"_a)
        .def("set_coefficients", [](TwoPole &s, float b0, float a1, float a2, bool clear) {
            s.setCoefficients((StkFloat)b0, (StkFloat)a1, (StkFloat)a2, clear);
        }, "b0"_a, "a1"_a, "a2"_a, "clear_state"_a = false)
        .def("set_resonance", [](TwoPole &s, float f, float r, bool norm) {
            s.setResonance((StkFloat)f, (StkFloat)r, norm);
        }, "frequency"_a, "radius"_a, "normalize"_a = false)
        FILTER_PROCESS(TwoPole);

    // --- TwoZero ---
    nb::class_<TwoZero>(mod, "TwoZero", "Two-zero filter")
        .def(nb::init<>())
        FILTER_BASE(TwoZero)
        .def("set_b0", [](TwoZero &s, float v) { s.setB0((StkFloat)v); }, "b0"_a)
        .def("set_b1", [](TwoZero &s, float v) { s.setB1((StkFloat)v); }, "b1"_a)
        .def("set_b2", [](TwoZero &s, float v) { s.setB2((StkFloat)v); }, "b2"_a)
        .def("set_coefficients", [](TwoZero &s, float b0, float b1, float b2, bool clear) {
            s.setCoefficients((StkFloat)b0, (StkFloat)b1, (StkFloat)b2, clear);
        }, "b0"_a, "b1"_a, "b2"_a, "clear_state"_a = false)
        .def("set_notch", [](TwoZero &s, float f, float r) {
            s.setNotch((StkFloat)f, (StkFloat)r);
        }, "frequency"_a, "radius"_a)
        FILTER_PROCESS(TwoZero);

    // --- PoleZero ---
    nb::class_<PoleZero>(mod, "PoleZero", "One-pole, one-zero filter")
        .def(nb::init<>())
        FILTER_BASE(PoleZero)
        .def("set_b0", [](PoleZero &s, float v) { s.setB0((StkFloat)v); }, "b0"_a)
        .def("set_b1", [](PoleZero &s, float v) { s.setB1((StkFloat)v); }, "b1"_a)
        .def("set_a1", [](PoleZero &s, float v) { s.setA1((StkFloat)v); }, "a1"_a)
        .def("set_coefficients", [](PoleZero &s, float b0, float b1, float a1, bool clear) {
            s.setCoefficients((StkFloat)b0, (StkFloat)b1, (StkFloat)a1, clear);
        }, "b0"_a, "b1"_a, "a1"_a, "clear_state"_a = false)
        .def("set_allpass", [](PoleZero &s, float c) { s.setAllpass((StkFloat)c); }, "coefficient"_a)
        .def("set_block_zero", [](PoleZero &s, float p) { s.setBlockZero((StkFloat)p); }, "pole"_a = 0.99f)
        FILTER_PROCESS(PoleZero);

    // --- Resonate ---
    nb::class_<Resonate>(mod, "Resonate", "Noise-driven formant filter instrument")
        .def(nb::init<>())
        .def("set_resonance", [](Resonate &s, float f, float r) {
            s.setResonance((StkFloat)f, (StkFloat)r);
        }, "frequency"_a, "radius"_a)
        .def("set_notch", [](Resonate &s, float f, float r) {
            s.setNotch((StkFloat)f, (StkFloat)r);
        }, "frequency"_a, "radius"_a)
        .def("set_equal_gain_zeroes", &Resonate::setEqualGainZeroes)
        .def("key_on", &Resonate::keyOn)
        .def("key_off", &Resonate::keyOff)
        .def("note_on", [](Resonate &s, float f, float a) {
            s.noteOn((StkFloat)f, (StkFloat)a);
        }, "frequency"_a, "amplitude"_a)
        .def("note_off", [](Resonate &s, float a) { s.noteOff((StkFloat)a); }, "amplitude"_a)
        .def("control_change", [](Resonate &s, int n, float v) {
            s.controlChange(n, (StkFloat)v);
        }, "number"_a, "value"_a)
        .def("process_sample", [](Resonate &s) { return (float)s.tick(); })
        .def("process", [](Resonate &s, int n) {
            auto *out = new float[n];
            { nb::gil_scoped_release rel;
              for (int i = 0; i < n; ++i) out[i] = (float)s.tick();
            }
            return make_f1(out, (size_t)n);
        }, "n"_a);

    // --- FormSwep ---
    nb::class_<FormSwep>(mod, "FormSwep", "Sweepable formant filter")
        .def(nb::init<>())
        FILTER_BASE(FormSwep)
        .def("set_resonance", [](FormSwep &s, float f, float r) {
            s.setResonance((StkFloat)f, (StkFloat)r);
        }, "frequency"_a, "radius"_a)
        .def("set_states", [](FormSwep &s, float f, float r, float g) {
            s.setStates((StkFloat)f, (StkFloat)r, (StkFloat)g);
        }, "frequency"_a, "radius"_a, "gain"_a = 1.0f)
        .def("set_targets", [](FormSwep &s, float f, float r, float g) {
            s.setTargets((StkFloat)f, (StkFloat)r, (StkFloat)g);
        }, "frequency"_a, "radius"_a, "gain"_a = 1.0f)
        .def("set_sweep_rate", [](FormSwep &s, float r) { s.setSweepRate((StkFloat)r); }, "rate"_a)
        .def("set_sweep_time", [](FormSwep &s, float t) { s.setSweepTime((StkFloat)t); }, "time"_a)
        FILTER_PROCESS(FormSwep);
}

// ============================================================================
// Delays
// ============================================================================

static void bind_stk_delays(nb::module_ &m) {
    auto mod = m.def_submodule("delays", "STK delay lines");

    // --- Delay (integer) ---
    nb::class_<Delay>(mod, "Delay", "Non-interpolating delay line")
        .def(nb::init<unsigned long, unsigned long>(), "delay"_a = 0, "max_delay"_a = 4095)
        .def("clear", &Delay::clear)
        .def("set_maximum_delay", &Delay::setMaximumDelay, "delay"_a)
        .def("set_delay", &Delay::setDelay, "delay"_a)
        .def("get_delay", [](Delay &s) { return (unsigned long)s.getDelay(); })
        .def("tap_out", [](Delay &s, unsigned long d) { return (float)s.tapOut(d); }, "tap_delay"_a)
        .def("tap_in", [](Delay &s, float v, unsigned long d) { s.tapIn((StkFloat)v, d); }, "value"_a, "tap_delay"_a)
        .def("next_out", [](Delay &s) { return (float)s.nextOut(); })
        .def("energy", [](Delay &s) { return (float)s.energy(); })
        .def("set_gain", [](Delay &s, float g) { s.setGain((StkFloat)g); }, "gain"_a)
        .def("get_gain", [](Delay &s) { return (float)s.getGain(); })
        FILTER_PROCESS(Delay);

    // --- DelayA (allpass interpolation) ---
    nb::class_<DelayA>(mod, "DelayA", "Allpass interpolating delay line")
        .def(nb::init<StkFloat, unsigned long>(), "delay"_a = 0.5, "max_delay"_a = 4095)
        .def("clear", &DelayA::clear)
        .def("set_maximum_delay", &DelayA::setMaximumDelay, "delay"_a)
        .def("set_delay", [](DelayA &s, float d) { s.setDelay((StkFloat)d); }, "delay"_a)
        .def("get_delay", [](DelayA &s) { return (float)s.getDelay(); })
        .def("next_out", [](DelayA &s) { return (float)s.nextOut(); })
        .def("set_gain", [](DelayA &s, float g) { s.setGain((StkFloat)g); }, "gain"_a)
        .def("get_gain", [](DelayA &s) { return (float)s.getGain(); })
        FILTER_PROCESS(DelayA);

    // --- DelayL (linear interpolation) ---
    nb::class_<DelayL>(mod, "DelayL", "Linear interpolating delay line")
        .def(nb::init<StkFloat, unsigned long>(), "delay"_a = 0.0, "max_delay"_a = 4095)
        .def("clear", &DelayL::clear)
        .def("set_maximum_delay", &DelayL::setMaximumDelay, "delay"_a)
        .def("set_delay", [](DelayL &s, float d) { s.setDelay((StkFloat)d); }, "delay"_a)
        .def("get_delay", [](DelayL &s) { return (float)s.getDelay(); })
        .def("tap_out", [](DelayL &s, unsigned long d) { return (float)s.tapOut(d); }, "tap_delay"_a)
        .def("tap_in", [](DelayL &s, float v, unsigned long d) { s.tapIn((StkFloat)v, d); }, "value"_a, "tap_delay"_a)
        .def("next_out", [](DelayL &s) { return (float)s.nextOut(); })
        .def("set_gain", [](DelayL &s, float g) { s.setGain((StkFloat)g); }, "gain"_a)
        .def("get_gain", [](DelayL &s) { return (float)s.getGain(); })
        FILTER_PROCESS(DelayL);

    // --- TapDelay ---
    nb::class_<TapDelay>(mod, "TapDelay", "Multi-tap delay line")
        .def(nb::init<std::vector<unsigned long>, unsigned long>(),
             "taps"_a = std::vector<unsigned long>{0}, "max_delay"_a = 4095)
        .def("clear", &TapDelay::clear)
        .def("set_maximum_delay", &TapDelay::setMaximumDelay, "delay"_a)
        .def("set_tap_delays", &TapDelay::setTapDelays, "taps"_a)
        .def("get_tap_delays", &TapDelay::getTapDelays)
        .def("set_gain", [](TapDelay &s, float g) { s.setGain((StkFloat)g); }, "gain"_a)
        .def("get_gain", [](TapDelay &s) { return (float)s.getGain(); })
        .def("process_sample", [](TapDelay &s, float in) {
            // Returns list of tap outputs
            StkFrames out(1, (unsigned int)s.getTapDelays().size());
            s.tick((StkFloat)in, out);
            nb::list result;
            for (unsigned int i = 0; i < out.channels(); ++i)
                result.append((float)out(0, i));
            return result;
        }, "input"_a);
}

// ============================================================================
// Effects
// ============================================================================

// Macro for mono-in stereo-out reverb (JCRev, NRev, PRCRev)
#define MONO_STEREO_REVERB(CLS, DOC) \
    nb::class_<CLS>(mod, #CLS, DOC) \
        .def(nb::init<StkFloat>(), "t60"_a = 1.0) \
        .def("clear", &CLS::clear) \
        .def("set_t60", [](CLS &s, float t) { s.setT60((StkFloat)t); }, "t60"_a) \
        .def("set_effect_mix", [](CLS &s, float m) { s.setEffectMix((StkFloat)m); }, "mix"_a) \
        .def("process_sample", [](CLS &s, float in) { \
            s.tick((StkFloat)in); \
            return nb::make_tuple((float)s.lastOut(0), (float)s.lastOut(1)); \
        }, "input"_a) \
        .def("process", [](CLS &s, ArrayF input) { \
            size_t n = input.shape(0); \
            auto *out = new float[2 * n]; \
            const float *in = input.data(); \
            { nb::gil_scoped_release rel; \
              for (size_t i = 0; i < n; ++i) { \
                  s.tick((StkFloat)in[i]); \
                  out[i] = (float)s.lastOut(0); \
                  out[n + i] = (float)s.lastOut(1); \
              } \
            } \
            return make_f2(out, 2, n); \
        }, "input"_a)

static void bind_stk_effects(nb::module_ &m) {
    auto mod = m.def_submodule("effects", "STK audio effects");

    // --- FreeVerb (stereo in, stereo out) ---
    nb::class_<FreeVerb>(mod, "FreeVerb", "Schroeder reverberator (stereo)")
        .def(nb::init<>())
        .def("clear", &FreeVerb::clear)
        .def("set_effect_mix", [](FreeVerb &s, float m) { s.setEffectMix((StkFloat)m); }, "mix"_a)
        .def("set_room_size", [](FreeVerb &s, float v) { s.setRoomSize((StkFloat)v); }, "value"_a)
        .def("get_room_size", [](FreeVerb &s) { return (float)s.getRoomSize(); })
        .def("set_damping", [](FreeVerb &s, float v) { s.setDamping((StkFloat)v); }, "value"_a)
        .def("get_damping", [](FreeVerb &s) { return (float)s.getDamping(); })
        .def("set_width", [](FreeVerb &s, float v) { s.setWidth((StkFloat)v); }, "value"_a)
        .def("get_width", [](FreeVerb &s) { return (float)s.getWidth(); })
        .def("set_mode", &FreeVerb::setMode, "frozen"_a)
        .def("get_mode", [](FreeVerb &s) { return (float)s.getMode(); })
        // Stereo process_sample
        .def("process_sample", [](FreeVerb &s, float inL, float inR) {
            s.tick((StkFloat)inL, (StkFloat)inR);
            return nb::make_tuple((float)s.lastOut(0), (float)s.lastOut(1));
        }, "input_l"_a, "input_r"_a = 0.0f)
        // Stereo process (2D input [2, frames])
        .def("process", [](FreeVerb &s, Array2F input) {
            size_t n = input.shape(1);
            auto *out = new float[2 * n];
            const float *l = input.data();
            const float *r = input.data() + n;
            { nb::gil_scoped_release rel;
              for (size_t i = 0; i < n; ++i) {
                  s.tick((StkFloat)l[i], (StkFloat)r[i]);
                  out[i] = (float)s.lastOut(0);
                  out[n + i] = (float)s.lastOut(1);
              }
            }
            return make_f2(out, 2, n);
        }, "input"_a)
        // Mono process variant
        .def("process_mono", [](FreeVerb &s, ArrayF input) {
            size_t n = input.shape(0);
            auto *out = new float[2 * n];
            const float *in = input.data();
            { nb::gil_scoped_release rel;
              for (size_t i = 0; i < n; ++i) {
                  s.tick((StkFloat)in[i], 0.0);
                  out[i] = (float)s.lastOut(0);
                  out[n + i] = (float)s.lastOut(1);
              }
            }
            return make_f2(out, 2, n);
        }, "input"_a);

    // --- JCRev, NRev, PRCRev (mono->stereo reverbs) ---
    MONO_STEREO_REVERB(JCRev, "John Chowning reverberator (mono->stereo)");
    MONO_STEREO_REVERB(NRev, "CCRMA NRev reverberator (mono->stereo)");
    MONO_STEREO_REVERB(PRCRev, "Perry Cook reverberator (mono->stereo)");

    // --- Echo (mono->mono) ---
    nb::class_<Echo>(mod, "Echo", "Echo effect")
        .def(nb::init<unsigned long>(), "max_delay"_a = (unsigned long)Stk::sampleRate())
        .def("clear", &Echo::clear)
        .def("set_effect_mix", [](Echo &s, float m) { s.setEffectMix((StkFloat)m); }, "mix"_a)
        .def("set_maximum_delay", &Echo::setMaximumDelay, "delay"_a)
        .def("set_delay", &Echo::setDelay, "delay"_a)
        .def("process_sample", [](Echo &s, float in) {
            return (float)s.tick((StkFloat)in);
        }, "input"_a)
        .def("process", [](Echo &s, ArrayF input) {
            size_t n = input.shape(0);
            auto *out = new float[n];
            const float *in = input.data();
            { nb::gil_scoped_release rel;
              for (size_t i = 0; i < n; ++i) out[i] = (float)s.tick((StkFloat)in[i]);
            }
            return make_f1(out, n);
        }, "input"_a);

    // --- Chorus (mono->stereo) ---
    nb::class_<Chorus>(mod, "Chorus", "Chorus effect (mono->stereo)")
        .def(nb::init<StkFloat>(), "base_delay"_a = 6000.0)
        .def("clear", &Chorus::clear)
        .def("set_effect_mix", [](Chorus &s, float m) { s.setEffectMix((StkFloat)m); }, "mix"_a)
        .def("set_mod_depth", [](Chorus &s, float d) { s.setModDepth((StkFloat)d); }, "depth"_a)
        .def("set_mod_frequency", [](Chorus &s, float f) { s.setModFrequency((StkFloat)f); }, "frequency"_a)
        .def("process_sample", [](Chorus &s, float in) {
            s.tick((StkFloat)in);
            return nb::make_tuple((float)s.lastOut(0), (float)s.lastOut(1));
        }, "input"_a)
        .def("process", [](Chorus &s, ArrayF input) {
            size_t n = input.shape(0);
            auto *out = new float[2 * n];
            const float *in = input.data();
            { nb::gil_scoped_release rel;
              for (size_t i = 0; i < n; ++i) {
                  s.tick((StkFloat)in[i]);
                  out[i] = (float)s.lastOut(0);
                  out[n + i] = (float)s.lastOut(1);
              }
            }
            return make_f2(out, 2, n);
        }, "input"_a);

    // --- PitShift (mono->mono) ---
    nb::class_<PitShift>(mod, "PitShift", "Simple pitch shifter")
        .def(nb::init<>())
        .def("clear", &PitShift::clear)
        .def("set_effect_mix", [](PitShift &s, float m) { s.setEffectMix((StkFloat)m); }, "mix"_a)
        .def("set_shift", [](PitShift &s, float sh) { s.setShift((StkFloat)sh); }, "shift"_a)
        .def("process_sample", [](PitShift &s, float in) {
            return (float)s.tick((StkFloat)in);
        }, "input"_a)
        .def("process", [](PitShift &s, ArrayF input) {
            size_t n = input.shape(0);
            auto *out = new float[n];
            const float *in = input.data();
            { nb::gil_scoped_release rel;
              for (size_t i = 0; i < n; ++i) out[i] = (float)s.tick((StkFloat)in[i]);
            }
            return make_f1(out, n);
        }, "input"_a);

    // --- LentPitShift (mono->mono) ---
    nb::class_<LentPitShift>(mod, "LentPitShift", "Lent algorithm pitch shifter")
        .def(nb::init<StkFloat, int>(), "shift"_a = 1.0, "window_size"_a = 512)
        .def("clear", &LentPitShift::clear)
        .def("set_shift", [](LentPitShift &s, float sh) { s.setShift((StkFloat)sh); }, "shift"_a)
        .def("process_sample", [](LentPitShift &s, float in) {
            return (float)s.tick((StkFloat)in);
        }, "input"_a)
        .def("process", [](LentPitShift &s, ArrayF input) {
            size_t n = input.shape(0);
            auto *out = new float[n];
            const float *in = input.data();
            { nb::gil_scoped_release rel;
              for (size_t i = 0; i < n; ++i) out[i] = (float)s.tick((StkFloat)in[i]);
            }
            return make_f1(out, n);
        }, "input"_a);
}

// ============================================================================
// Instruments
// ============================================================================

// Macro for standard Instrmnt subclass bindings
#define INSTRUMENT_COMMON(CLS) \
    .def("clear", &CLS::clear) \
    .def("set_frequency", [](CLS &s, float f) { s.setFrequency((StkFloat)f); }, "frequency"_a) \
    .def("note_on", [](CLS &s, float f, float a) { s.noteOn((StkFloat)f, (StkFloat)a); }, \
         "frequency"_a, "amplitude"_a) \
    .def("note_off", [](CLS &s, float a) { s.noteOff((StkFloat)a); }, "amplitude"_a) \
    .def("control_change", [](CLS &s, int n, float v) { s.controlChange(n, (StkFloat)v); }, \
         "number"_a, "value"_a) \
    .def("process_sample", [](CLS &s) { return (float)s.tick(); }) \
    .def("process", [](CLS &s, int n) { \
        auto *out = new float[n]; \
        { nb::gil_scoped_release rel; \
          for (int i = 0; i < n; ++i) out[i] = (float)s.tick(); \
        } \
        return make_f1(out, (size_t)n); \
    }, "n"_a)

// Macro for wind instruments with start/stop blowing
#define BLOWING_METHODS(CLS) \
    .def("start_blowing", [](CLS &s, float a, float r) { s.startBlowing((StkFloat)a, (StkFloat)r); }, \
         "amplitude"_a, "rate"_a) \
    .def("stop_blowing", [](CLS &s, float r) { s.stopBlowing((StkFloat)r); }, "rate"_a)

static void bind_stk_instruments(nb::module_ &m) {
    auto mod = m.def_submodule("instruments", "STK physical model instruments");

    // --- Clarinet ---
    nb::class_<Clarinet>(mod, "Clarinet", "Digital waveguide clarinet")
        .def(nb::init<StkFloat>(), "lowest_frequency"_a = 8.0)
        INSTRUMENT_COMMON(Clarinet)
        BLOWING_METHODS(Clarinet);

    // --- Flute ---
    nb::class_<Flute>(mod, "Flute", "Digital waveguide flute")
        .def(nb::init<StkFloat>(), "lowest_frequency"_a = 8.0)
        INSTRUMENT_COMMON(Flute)
        .def("set_jet_reflection", [](Flute &s, float v) { s.setJetReflection((StkFloat)v); }, "coefficient"_a)
        .def("set_end_reflection", [](Flute &s, float v) { s.setEndReflection((StkFloat)v); }, "coefficient"_a)
        .def("set_jet_delay", [](Flute &s, float v) { s.setJetDelay((StkFloat)v); }, "delay"_a)
        .def("start_blowing", [](Flute &s, float a, float r) { s.startBlowing((StkFloat)a, (StkFloat)r); },
             "amplitude"_a, "rate"_a)
        .def("stop_blowing", [](Flute &s, float r) { s.stopBlowing((StkFloat)r); }, "rate"_a);

    // --- Brass ---
    nb::class_<Brass>(mod, "Brass", "Digital waveguide brass instrument")
        .def(nb::init<StkFloat>(), "lowest_frequency"_a = 8.0)
        INSTRUMENT_COMMON(Brass)
        .def("set_lip", [](Brass &s, float f) { s.setLip((StkFloat)f); }, "frequency"_a)
        BLOWING_METHODS(Brass);

    // --- Bowed ---
    nb::class_<Bowed>(mod, "Bowed", "Bowed string physical model")
        .def(nb::init<StkFloat>(), "lowest_frequency"_a = 8.0)
        INSTRUMENT_COMMON(Bowed)
        .def("set_vibrato", [](Bowed &s, float g) { s.setVibrato((StkFloat)g); }, "gain"_a)
        .def("start_bowing", [](Bowed &s, float a, float r) { s.startBowing((StkFloat)a, (StkFloat)r); },
             "amplitude"_a, "rate"_a)
        .def("stop_bowing", [](Bowed &s, float r) { s.stopBowing((StkFloat)r); }, "rate"_a);

    // --- Plucked ---
    nb::class_<Plucked>(mod, "Plucked", "Karplus-Strong plucked string")
        .def(nb::init<StkFloat>(), "lowest_frequency"_a = 10.0)
        INSTRUMENT_COMMON(Plucked);

    // --- Sitar ---
    nb::class_<Sitar>(mod, "Sitar", "Sitar physical model")
        .def(nb::init<StkFloat>(), "lowest_frequency"_a = 8.0)
        INSTRUMENT_COMMON(Sitar);

    // --- StifKarp ---
    nb::class_<StifKarp>(mod, "StifKarp", "Stiff string Karplus-Strong")
        .def(nb::init<StkFloat>(), "lowest_frequency"_a = 8.0)
        INSTRUMENT_COMMON(StifKarp)
        .def("set_stretch", [](StifKarp &s, float v) { s.setStretch((StkFloat)v); }, "stretch"_a)
        .def("set_base_loop_gain", [](StifKarp &s, float v) { s.setBaseLoopGain((StkFloat)v); }, "gain"_a);

    // --- Saxofony ---
    nb::class_<Saxofony>(mod, "Saxofony", "Saxophone physical model")
        .def(nb::init<StkFloat>(), "lowest_frequency"_a = 8.0)
        INSTRUMENT_COMMON(Saxofony)
        .def("set_blow_position", [](Saxofony &s, float p) { s.setBlowPosition((StkFloat)p); }, "position"_a)
        BLOWING_METHODS(Saxofony);

    // --- Recorder ---
    nb::class_<Recorder>(mod, "Recorder", "Recorder physical model")
        .def(nb::init<>())
        INSTRUMENT_COMMON(Recorder)
        .def("start_blowing", [](Recorder &s, float a, float r) { s.startBlowing((StkFloat)a, (StkFloat)r); },
             "amplitude"_a, "rate"_a)
        .def("stop_blowing", [](Recorder &s, float r) { s.stopBlowing((StkFloat)r); }, "rate"_a);

    // --- BlowBotl ---
    nb::class_<BlowBotl>(mod, "BlowBotl", "Blown bottle physical model")
        .def(nb::init<>())
        INSTRUMENT_COMMON(BlowBotl)
        BLOWING_METHODS(BlowBotl);

    // --- BlowHole ---
    nb::class_<BlowHole>(mod, "BlowHole", "Clarinet with tonehole and register vent")
        .def(nb::init<StkFloat>(), "lowest_frequency"_a)
        INSTRUMENT_COMMON(BlowHole)
        .def("set_tonehole", [](BlowHole &s, float v) { s.setTonehole((StkFloat)v); }, "value"_a)
        .def("set_vent", [](BlowHole &s, float v) { s.setVent((StkFloat)v); }, "value"_a)
        BLOWING_METHODS(BlowHole);

    // --- Whistle ---
    nb::class_<Whistle>(mod, "Whistle", "Police whistle physical model")
        .def(nb::init<>())
        INSTRUMENT_COMMON(Whistle)
        BLOWING_METHODS(Whistle);

    // --- Guitar (not Instrmnt -- different API) ---
    nb::class_<Guitar>(mod, "Guitar", "Multi-string guitar model")
        .def(nb::init<unsigned int, std::string>(), "n_strings"_a = 6, "body_file"_a = "")
        .def("clear", &Guitar::clear)
        .def("set_pluck_position", [](Guitar &s, float p, int str) {
            s.setPluckPosition((StkFloat)p, str);
        }, "position"_a, "string"_a = -1)
        .def("set_loop_gain", [](Guitar &s, float g, int str) {
            s.setLoopGain((StkFloat)g, str);
        }, "gain"_a, "string"_a = -1)
        .def("set_frequency", [](Guitar &s, float f, unsigned int str) {
            s.setFrequency((StkFloat)f, str);
        }, "frequency"_a, "string"_a = 0)
        .def("note_on", [](Guitar &s, float f, float a, unsigned int str) {
            s.noteOn((StkFloat)f, (StkFloat)a, str);
        }, "frequency"_a, "amplitude"_a, "string"_a = 0)
        .def("note_off", [](Guitar &s, float a, unsigned int str) {
            s.noteOff((StkFloat)a, str);
        }, "amplitude"_a, "string"_a = 0)
        .def("control_change", [](Guitar &s, int n, float v, int str) {
            s.controlChange(n, (StkFloat)v, str);
        }, "number"_a, "value"_a, "string"_a = -1)
        .def("process_sample", [](Guitar &s, float in) { return (float)s.tick((StkFloat)in); },
             "input"_a = 0.0f)
        .def("process", [](Guitar &s, int n) {
            auto *out = new float[n];
            { nb::gil_scoped_release rel;
              for (int i = 0; i < n; ++i) out[i] = (float)s.tick(0.0);
            }
            return make_f1(out, (size_t)n);
        }, "n"_a);

    // --- Twang (not Instrmnt -- different API) ---
    nb::class_<Twang>(mod, "Twang", "Enhanced Karplus-Strong plucked string")
        .def(nb::init<StkFloat>(), "lowest_frequency"_a = 50.0)
        .def("clear", &Twang::clear)
        .def("set_lowest_frequency", [](Twang &s, float f) { s.setLowestFrequency((StkFloat)f); }, "frequency"_a)
        .def("set_frequency", [](Twang &s, float f) { s.setFrequency((StkFloat)f); }, "frequency"_a)
        .def("set_pluck_position", [](Twang &s, float p) { s.setPluckPosition((StkFloat)p); }, "position"_a)
        .def("set_loop_gain", [](Twang &s, float g) { s.setLoopGain((StkFloat)g); }, "gain"_a)
        .def("set_loop_filter", [](Twang &s, std::vector<float> coeffs) {
            std::vector<StkFloat> c(coeffs.begin(), coeffs.end());
            s.setLoopFilter(c);
        }, "coefficients"_a)
        .def("process_sample", [](Twang &s, float in) { return (float)s.tick((StkFloat)in); }, "input"_a)
        .def("process", [](Twang &s, ArrayF input) {
            size_t n = input.shape(0);
            auto *out = new float[n];
            const float *in = input.data();
            { nb::gil_scoped_release rel;
              for (size_t i = 0; i < n; ++i) out[i] = (float)s.tick((StkFloat)in[i]);
            }
            return make_f1(out, n);
        }, "input"_a);
}

// ============================================================================
// Main entry point
// ============================================================================

void bind_stk(nb::module_ &m) {
    auto stk_mod = m.def_submodule("stk", "STK (Synthesis ToolKit) bindings");

    // Global sample rate
    stk_mod.def("set_sample_rate", [](float rate) {
        Stk::setSampleRate((StkFloat)rate);
    }, "rate"_a);
    stk_mod.def("sample_rate", []() {
        return (float)Stk::sampleRate();
    });

    bind_stk_generators(stk_mod);
    bind_stk_filters(stk_mod);
    bind_stk_delays(stk_mod);
    bind_stk_effects(stk_mod);
    bind_stk_instruments(stk_mod);
}

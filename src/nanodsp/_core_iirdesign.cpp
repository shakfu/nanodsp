// IIR filter design bindings using DspFilters (Vinnie Falco, MIT)
// Supports Butterworth, Chebyshev I/II, Elliptic, Bessel in orders 1-32.
// Exposes both coefficient extraction (design_iir -> SOS array) and
// direct processing (IIRFilter class with setup/process/reset).

#include "_core_common.h"

// Suppress warnings from DspFilters headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wsign-compare"
#ifdef __clang__
#pragma GCC diagnostic ignored "-Wdeprecated-copy"
#endif

#include "DspFilters/Dsp.h"

#pragma GCC diagnostic pop

#include <vector>
#include <stdexcept>
#include <cstring>
#include <memory>

namespace nb = nanobind;

// ---------------------------------------------------------------------------
// Enums for Python-facing API
// ---------------------------------------------------------------------------

enum class FilterFamily { BUTTERWORTH, CHEBYSHEV1, CHEBYSHEV2, ELLIPTIC, BESSEL };
enum class FilterType   { LOWPASS, HIGHPASS, BANDPASS, BANDSTOP };

// ---------------------------------------------------------------------------
// MaxOrder = 16 covers practical use (32 biquad stages for bandpass/bandstop)
// ---------------------------------------------------------------------------
static constexpr int MAX_ORDER = 16;

// ---------------------------------------------------------------------------
// Factory: design a filter and return it as a Cascade pointer.
// We use a type-erased wrapper since all filter types share the Cascade base.
// ---------------------------------------------------------------------------

// Wrapper that owns a concrete filter and exposes the Cascade interface
struct FilterHolder {
    virtual ~FilterHolder() = default;
    virtual Dsp::Cascade& cascade() = 0;
    virtual void reset_state() = 0;
    virtual float process_sample(float in) = 0;
};

// Concrete holder for a specific DspFilters type
template <class FilterType>
struct ConcreteHolder : FilterHolder {
    FilterType filter;
    typename Dsp::CascadeStages<MAX_ORDER * 2>::template State<Dsp::DirectFormII> state;

    Dsp::Cascade& cascade() override { return filter; }

    void reset_state() override { state.reset(); }

    float process_sample(float in) override {
        double d = in;
        filter.process(1, &d, state);
        return (float)d;
    }
};

// Factory function
static std::unique_ptr<FilterHolder> make_filter(
    FilterFamily family, FilterType type, int order,
    double sample_rate, double freq,
    double width, double ripple_db, double rolloff_db)
{
    if (order < 1 || order > MAX_ORDER)
        throw std::invalid_argument("order must be 1-" + std::to_string(MAX_ORDER));
    if (sample_rate <= 0)
        throw std::invalid_argument("sample_rate must be positive");
    if (freq <= 0 || freq >= sample_rate / 2)
        throw std::invalid_argument("freq must be in (0, sample_rate/2)");

    // Band filters need width
    bool is_band = (type == FilterType::BANDPASS || type == FilterType::BANDSTOP);
    if (is_band && width <= 0)
        throw std::invalid_argument("width must be positive for bandpass/bandstop");

    // Macro to reduce repetition
    #define MAKE_LP(NS) { \
        auto h = std::make_unique<ConcreteHolder<Dsp::NS::LowPass<MAX_ORDER>>>(); \
        h->filter.setup(order, sample_rate, freq); return h; }
    #define MAKE_HP(NS) { \
        auto h = std::make_unique<ConcreteHolder<Dsp::NS::HighPass<MAX_ORDER>>>(); \
        h->filter.setup(order, sample_rate, freq); return h; }
    #define MAKE_BP(NS) { \
        auto h = std::make_unique<ConcreteHolder<Dsp::NS::BandPass<MAX_ORDER>>>(); \
        h->filter.setup(order, sample_rate, freq, width); return h; }
    #define MAKE_BS(NS) { \
        auto h = std::make_unique<ConcreteHolder<Dsp::NS::BandStop<MAX_ORDER>>>(); \
        h->filter.setup(order, sample_rate, freq, width); return h; }

    // Chebyshev I: extra ripple_db param
    #define MAKE_LP_C1 { \
        auto h = std::make_unique<ConcreteHolder<Dsp::ChebyshevI::LowPass<MAX_ORDER>>>(); \
        h->filter.setup(order, sample_rate, freq, ripple_db); return h; }
    #define MAKE_HP_C1 { \
        auto h = std::make_unique<ConcreteHolder<Dsp::ChebyshevI::HighPass<MAX_ORDER>>>(); \
        h->filter.setup(order, sample_rate, freq, ripple_db); return h; }
    #define MAKE_BP_C1 { \
        auto h = std::make_unique<ConcreteHolder<Dsp::ChebyshevI::BandPass<MAX_ORDER>>>(); \
        h->filter.setup(order, sample_rate, freq, width, ripple_db); return h; }
    #define MAKE_BS_C1 { \
        auto h = std::make_unique<ConcreteHolder<Dsp::ChebyshevI::BandStop<MAX_ORDER>>>(); \
        h->filter.setup(order, sample_rate, freq, width, ripple_db); return h; }

    // Chebyshev II: extra stopband_db (reuse ripple_db param)
    #define MAKE_LP_C2 { \
        auto h = std::make_unique<ConcreteHolder<Dsp::ChebyshevII::LowPass<MAX_ORDER>>>(); \
        h->filter.setup(order, sample_rate, freq, ripple_db); return h; }
    #define MAKE_HP_C2 { \
        auto h = std::make_unique<ConcreteHolder<Dsp::ChebyshevII::HighPass<MAX_ORDER>>>(); \
        h->filter.setup(order, sample_rate, freq, ripple_db); return h; }
    #define MAKE_BP_C2 { \
        auto h = std::make_unique<ConcreteHolder<Dsp::ChebyshevII::BandPass<MAX_ORDER>>>(); \
        h->filter.setup(order, sample_rate, freq, width, ripple_db); return h; }
    #define MAKE_BS_C2 { \
        auto h = std::make_unique<ConcreteHolder<Dsp::ChebyshevII::BandStop<MAX_ORDER>>>(); \
        h->filter.setup(order, sample_rate, freq, width, ripple_db); return h; }

    // Elliptic: ripple_db + rolloff_db
    #define MAKE_LP_EL { \
        auto h = std::make_unique<ConcreteHolder<Dsp::Elliptic::LowPass<MAX_ORDER>>>(); \
        h->filter.setup(order, sample_rate, freq, ripple_db, rolloff_db); return h; }
    #define MAKE_HP_EL { \
        auto h = std::make_unique<ConcreteHolder<Dsp::Elliptic::HighPass<MAX_ORDER>>>(); \
        h->filter.setup(order, sample_rate, freq, ripple_db, rolloff_db); return h; }
    #define MAKE_BP_EL { \
        auto h = std::make_unique<ConcreteHolder<Dsp::Elliptic::BandPass<MAX_ORDER>>>(); \
        h->filter.setup(order, sample_rate, freq, width, ripple_db, rolloff_db); return h; }
    #define MAKE_BS_EL { \
        auto h = std::make_unique<ConcreteHolder<Dsp::Elliptic::BandStop<MAX_ORDER>>>(); \
        h->filter.setup(order, sample_rate, freq, width, ripple_db, rolloff_db); return h; }

    switch (family) {
    case FilterFamily::BUTTERWORTH:
        switch (type) {
        case FilterType::LOWPASS:  MAKE_LP(Butterworth)
        case FilterType::HIGHPASS: MAKE_HP(Butterworth)
        case FilterType::BANDPASS: MAKE_BP(Butterworth)
        case FilterType::BANDSTOP: MAKE_BS(Butterworth)
        }
        break;

    case FilterFamily::CHEBYSHEV1:
        if (ripple_db <= 0) ripple_db = 1.0;  // sensible default
        switch (type) {
        case FilterType::LOWPASS:  MAKE_LP_C1
        case FilterType::HIGHPASS: MAKE_HP_C1
        case FilterType::BANDPASS: MAKE_BP_C1
        case FilterType::BANDSTOP: MAKE_BS_C1
        }
        break;

    case FilterFamily::CHEBYSHEV2:
        if (ripple_db <= 0) ripple_db = 40.0;  // stopband attenuation default
        switch (type) {
        case FilterType::LOWPASS:  MAKE_LP_C2
        case FilterType::HIGHPASS: MAKE_HP_C2
        case FilterType::BANDPASS: MAKE_BP_C2
        case FilterType::BANDSTOP: MAKE_BS_C2
        }
        break;

    case FilterFamily::ELLIPTIC:
        if (ripple_db <= 0) ripple_db = 1.0;
        // rolloff is transition width, not dB; valid range approx [-16, 4]
        // 0 = default (moderate transition), higher = sharper transition
        // do NOT clamp -- let DspFilters handle it
        switch (type) {
        case FilterType::LOWPASS:  MAKE_LP_EL
        case FilterType::HIGHPASS: MAKE_HP_EL
        case FilterType::BANDPASS: MAKE_BP_EL
        case FilterType::BANDSTOP: MAKE_BS_EL
        }
        break;

    case FilterFamily::BESSEL:
        switch (type) {
        case FilterType::LOWPASS:  MAKE_LP(Bessel)
        case FilterType::HIGHPASS: MAKE_HP(Bessel)
        case FilterType::BANDPASS: MAKE_BP(Bessel)
        case FilterType::BANDSTOP: MAKE_BS(Bessel)
        }
        break;
    }

    #undef MAKE_LP
    #undef MAKE_HP
    #undef MAKE_BP
    #undef MAKE_BS
    #undef MAKE_LP_C1
    #undef MAKE_HP_C1
    #undef MAKE_BP_C1
    #undef MAKE_BS_C1
    #undef MAKE_LP_C2
    #undef MAKE_HP_C2
    #undef MAKE_BP_C2
    #undef MAKE_BS_C2
    #undef MAKE_LP_EL
    #undef MAKE_HP_EL
    #undef MAKE_BP_EL
    #undef MAKE_BS_EL

    throw std::logic_error("unreachable");
}

// ---------------------------------------------------------------------------
// Extract SOS coefficients from a designed filter
// Returns [n_sections, 6] array: each row is [b0, b1, b2, a0, a1, a2]
// All coefficients normalized so a0 = 1.0
// ---------------------------------------------------------------------------
static NpF2 extract_sos(Dsp::Cascade& cascade) {
    int n = cascade.getNumStages();
    float* data = new float[n * 6];
    for (int i = 0; i < n; ++i) {
        const auto& s = cascade[i];
        double a0 = s.getA0();
        // Normalize: DspFilters stores coefficients pre-multiplied by a0
        // getB0() returns m_b0*m_a0, etc.  We want b0/a0 form.
        data[i * 6 + 0] = (float)s.getB0();  // b0 * a0 (already normalized)
        data[i * 6 + 1] = (float)s.getB1();  // b1 * a0
        data[i * 6 + 2] = (float)s.getB2();  // b2 * a0
        data[i * 6 + 3] = (float)a0;         // a0
        data[i * 6 + 4] = (float)s.getA1();  // a1 * a0
        data[i * 6 + 5] = (float)s.getA2();  // a2 * a0
    }
    return make_f2(data, n, 6);
}

// ---------------------------------------------------------------------------
// Stateful IIR filter class for Python
// ---------------------------------------------------------------------------

class IIRFilter {
public:
    IIRFilter() = default;

    void setup(int family, int type, int order,
               double sample_rate, double freq,
               double width, double ripple_db, double rolloff_db) {
        holder_ = make_filter(
            (FilterFamily)family, (FilterType)type,
            order, sample_rate, freq, width, ripple_db, rolloff_db);
        holder_->reset_state();
    }

    NpF2 sos() {
        if (!holder_) throw std::runtime_error("filter not initialized -- call setup() first");
        return extract_sos(holder_->cascade());
    }

    NpF1 process(ArrayF input) {
        if (!holder_) throw std::runtime_error("filter not initialized -- call setup() first");
        unsigned n = input.shape(0);
        float* out = new float[n];
        const float* in = input.data();
        {
            nb::gil_scoped_release release;
            for (unsigned i = 0; i < n; ++i)
                out[i] = holder_->process_sample(in[i]);
        }
        return make_f1(out, n);
    }

    void reset() {
        if (holder_) holder_->reset_state();
    }

    int num_stages() const {
        if (!holder_) return 0;
        return holder_->cascade().getNumStages();
    }

private:
    std::unique_ptr<FilterHolder> holder_;
};

// ---------------------------------------------------------------------------
// Module-level design function (stateless, returns SOS coefficients)
// ---------------------------------------------------------------------------

static NpF2 design_iir(int family, int type, int order,
                        double sample_rate, double freq,
                        double width, double ripple_db, double rolloff_db) {
    auto holder = make_filter(
        (FilterFamily)family, (FilterType)type,
        order, sample_rate, freq, width, ripple_db, rolloff_db);
    return extract_sos(holder->cascade());
}

// ---------------------------------------------------------------------------
// Module-level filter function (stateless, designs + processes in one call)
// ---------------------------------------------------------------------------

static NpF1 apply_iir(ArrayF input, int family, int type, int order,
                       double sample_rate, double freq,
                       double width, double ripple_db, double rolloff_db) {
    auto holder = make_filter(
        (FilterFamily)family, (FilterType)type,
        order, sample_rate, freq, width, ripple_db, rolloff_db);
    unsigned n = input.shape(0);
    float* out = new float[n];
    const float* in = input.data();
    {
        nb::gil_scoped_release release;
        for (unsigned i = 0; i < n; ++i)
            out[i] = holder->process_sample(in[i]);
    }
    return make_f1(out, n);
}

// ---------------------------------------------------------------------------
// Bind
// ---------------------------------------------------------------------------

void bind_iirdesign(nb::module_& m) {
    auto sub = m.def_submodule("iirdesign", "IIR filter design (DspFilters)");

    nb::enum_<FilterFamily>(sub, "Family")
        .value("BUTTERWORTH", FilterFamily::BUTTERWORTH)
        .value("CHEBYSHEV1",  FilterFamily::CHEBYSHEV1)
        .value("CHEBYSHEV2",  FilterFamily::CHEBYSHEV2)
        .value("ELLIPTIC",    FilterFamily::ELLIPTIC)
        .value("BESSEL",      FilterFamily::BESSEL);

    nb::enum_<FilterType>(sub, "Type")
        .value("LOWPASS",  FilterType::LOWPASS)
        .value("HIGHPASS", FilterType::HIGHPASS)
        .value("BANDPASS", FilterType::BANDPASS)
        .value("BANDSTOP", FilterType::BANDSTOP);

    nb::class_<IIRFilter>(sub, "IIRFilter")
        .def(nb::init<>())
        .def("setup", &IIRFilter::setup,
             "family"_a, "type"_a, "order"_a,
             "sample_rate"_a, "freq"_a,
             "width"_a = 0.0, "ripple_db"_a = 0.0, "rolloff_db"_a = 0.0)
        .def("process", &IIRFilter::process, "input"_a)
        .def("sos", &IIRFilter::sos)
        .def("reset", &IIRFilter::reset)
        .def_prop_ro("num_stages", &IIRFilter::num_stages);

    sub.def("design", &design_iir,
            "family"_a, "type"_a, "order"_a,
            "sample_rate"_a, "freq"_a,
            "width"_a = 0.0, "ripple_db"_a = 0.0, "rolloff_db"_a = 0.0,
            "Design an IIR filter and return SOS coefficients [n_sections, 6].\n"
            "Each row: [b0, b1, b2, a0, a1, a2].");

    sub.def("apply", &apply_iir,
            "input"_a, "family"_a, "type"_a, "order"_a,
            "sample_rate"_a, "freq"_a,
            "width"_a = 0.0, "ripple_db"_a = 0.0, "rolloff_db"_a = 0.0,
            "Design an IIR filter and apply it to the input array.");
}

#include "_core_common.h"

NB_MODULE(_core, m) {
    m.doc() = "nanodsp: Python bindings for DSP libraries via nanobind";
    m.def("add", [](int a, int b) { return a + b; }, "Add two integers", nb::arg("a"), nb::arg("b"));
    m.def("greet", [](const std::string& name) { return "Hello, " + name + "!"; }, "Return a greeting string", nb::arg("name"));
    bind_signalsmith(m);
    bind_daisysp(m);
    bind_stk(m);
    bind_madronalib(m);
    bind_hisstools(m);
    bind_choc(m);
    bind_grainflow(m);
    bind_vafilters(m);
    bind_bloscillators(m);
    bind_fxdsp(m);
    bind_iirdesign(m);
}

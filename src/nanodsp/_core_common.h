#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/complex.h>

#include <string>
#include <cmath>
#include <complex>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <limits>

namespace nb = nanobind;
using namespace nb::literals;

// Input type aliases: accept any contiguous float32/complex64 array
using ArrayF = nb::ndarray<float, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using Array2F = nb::ndarray<float, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using ArrayCF = nb::ndarray<std::complex<float>, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

// Return type aliases: return as numpy arrays
using NpF1 = nb::ndarray<nb::numpy, float, nb::ndim<1>>;
using NpF2 = nb::ndarray<nb::numpy, float, nb::ndim<2>>;
using NpCF1 = nb::ndarray<nb::numpy, std::complex<float>, nb::ndim<1>>;

inline NpF1 make_f1(float *data, size_t n) {
    nb::capsule del(data, [](void *p) noexcept { delete[] static_cast<float*>(p); });
    return NpF1(data, {n}, del);
}
inline NpF2 make_f2(float *data, size_t r, size_t c) {
    nb::capsule del(data, [](void *p) noexcept { delete[] static_cast<float*>(p); });
    size_t shape[2] = {r, c};
    return NpF2(data, 2, shape, del);
}
inline NpCF1 make_cf1(std::complex<float> *data, size_t n) {
    nb::capsule del(data, [](void *p) noexcept { delete[] static_cast<std::complex<float>*>(p); });
    return NpCF1(data, {n}, del);
}

// ---------------------------------------------------------------------------
// Mono block-processing helpers
//
// These factor out the dominant per-sample binding patterns (allocate output,
// release the GIL, loop, hand ownership to numpy). They are templated on the
// DSP object type so any class exposing the expected Process() shape can reuse
// them. Multi-output / multi-channel / block APIs keep bespoke bindings.
// ---------------------------------------------------------------------------

// Process a mono buffer sample-by-sample: out[i] = self.Process(in[i]).
// A local lvalue is used so this works whether Process takes float,
// const float&, or a non-const float& (the legacy DaisySP signature).
template <typename T>
inline NpF1 util_process_mono(T &self, ArrayF input) {
    size_t n = input.shape(0);
    auto *out = new float[n];
    const float *in = input.data();
    { nb::gil_scoped_release rel;
      for (size_t i = 0; i < n; ++i) { float v = in[i]; out[i] = self.Process(v); }
    }
    return make_f1(out, n);
}

// Generate n samples from a nullary generator: out[i] = self.Process().
template <typename T>
inline NpF1 util_generate_mono(T &self, int n) {
    auto *out = new float[(size_t)n];
    { nb::gil_scoped_release rel;
      for (int i = 0; i < n; ++i) out[i] = self.Process();
    }
    return make_f1(out, (size_t)n);
}

// Trigger on the first sample, then free-run: out[0] = Process(true),
// out[i>0] = Process(false). Used by drums and excited voices.
template <typename T>
inline NpF1 util_trigger_generate_mono(T &self, int n) {
    auto *out = new float[(size_t)n];
    { nb::gil_scoped_release rel;
      out[0] = self.Process(true);
      for (int i = 1; i < n; ++i) out[i] = self.Process(false);
    }
    return make_f1(out, (size_t)n);
}

// Library binding entry points
void bind_signalsmith(nb::module_ &m);
void bind_daisysp(nb::module_ &m);
void bind_stk(nb::module_ &m);
void bind_madronalib(nb::module_ &m);
void bind_hisstools(nb::module_ &m);
void bind_choc(nb::module_ &m);
void bind_grainflow(nb::module_ &m);
void bind_vafilters(nb::module_ &m);
void bind_bloscillators(nb::module_ &m);
void bind_fxdsp(nb::module_ &m);
void bind_iirdesign(nb::module_ &m);
void bind_paulstretch(nb::module_ &m);
void bind_signalsmith_stretch(nb::module_ &m);

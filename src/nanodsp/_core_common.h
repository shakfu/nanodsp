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

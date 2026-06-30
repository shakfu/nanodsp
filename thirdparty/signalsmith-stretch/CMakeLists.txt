cmake_minimum_required(VERSION 3.24)

# signalsmith-stretch (MIT) -- high-quality time-stretch / pitch-shift.
#
# Vendored at release 1.1.1, the last version that builds on the signalsmith
# DSP library (newer releases depend on the separate signalsmith-linear FFT
# library). The upstream header is unmodified; its `dsp/...` includes resolve
# through the forwarding shims in dsp/ to the signalsmith-dsp target already
# vendored at thirdparty/signalsmith.
add_library(signalsmith-stretch INTERFACE)
target_include_directories(signalsmith-stretch INTERFACE ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(signalsmith-stretch INTERFACE signalsmith-dsp)

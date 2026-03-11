cmake_minimum_required(VERSION 3.15...3.31)
project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Required on Linux: static libs linked into shared .so need PIC
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

# Signalsmith DSP (header-only)
add_subdirectory(thirdparty/signalsmith)

# DaisySP (MIT + LGPL-2.1)
add_subdirectory(thirdparty/DaisySP)
add_subdirectory(thirdparty/DaisySP/DaisySP-LGPL)
# Suppress warnings and fix MSVC compat in thirdparty DaisySP sources
if(MSVC)
    set(MSVC_COMPAT "${CMAKE_CURRENT_SOURCE_DIR}/cmake/msvc_compat.h")
    target_compile_options(DaisySP PRIVATE /w /FI"${MSVC_COMPAT}")
    target_compile_options(DaisySP_LGPL PRIVATE /w)
else()
    target_compile_options(DaisySP PRIVATE -w)
    target_compile_options(DaisySP_LGPL PRIVATE -w)
endif()

# Madronalib DSP (header-only, MIT)
add_library(madronalib-dsp INTERFACE)
target_include_directories(madronalib-dsp INTERFACE
    thirdparty/madronalib/source/DSP
    thirdparty/madronalib/source/app
    thirdparty/madronalib/include
    thirdparty/madronalib/external/sse2neon
)

# HISSTools Library (SIMD-accelerated DSP)
set(HISSTOOLS_SOURCES
    thirdparty/HISSTools_Library/HISSTools_FFT/HISSTools_FFT.cpp
    thirdparty/HISSTools_Library/HIRT_Multichannel_Convolution/PartitionedConvolve.cpp
    thirdparty/HISSTools_Library/HIRT_Multichannel_Convolution/MonoConvolve.cpp
    thirdparty/HISSTools_Library/HIRT_Multichannel_Convolution/NToMonoConvolve.cpp
    thirdparty/HISSTools_Library/HIRT_Multichannel_Convolution/Convolver.cpp
    thirdparty/HISSTools_Library/HIRT_Multichannel_Convolution/TimeDomainConvolve.cpp
)
add_library(hisstools STATIC ${HISSTOOLS_SOURCES})
target_include_directories(hisstools PUBLIC
    thirdparty/HISSTools_Library
    thirdparty/HISSTools_Library/HISSTools_FFT
    thirdparty/HISSTools_Library/HIRT_Multichannel_Convolution
)
if(APPLE)
    target_link_libraries(hisstools PRIVATE "-framework Accelerate")
else()
    target_compile_definitions(hisstools PUBLIC NO_NATIVE_FFT)
endif()
# HISSTools checks __arm64__ (Apple-specific); GCC/Linux uses __aarch64__.
# Force-include a compat header that bridges this using compiler target defines,
# which is cross-compilation safe (unlike CMAKE_SYSTEM_PROCESSOR).
set(HISS_ARCH_COMPAT "${CMAKE_CURRENT_SOURCE_DIR}/cmake/hisstools_arch_compat.h")
if(MSVC)
    target_compile_definitions(hisstools PRIVATE NOMINMAX)
    target_compile_options(hisstools PRIVATE /w /FI"${HISS_ARCH_COMPAT}")
else()
    target_compile_options(hisstools PRIVATE -w -include "${HISS_ARCH_COMPAT}")
endif()

# GrainflowLib (header-only, MIT)
add_library(grainflow-dsp INTERFACE)
target_include_directories(grainflow-dsp INTERFACE
    thirdparty/GrainflowLib/include
    thirdparty/GrainflowLib/lib/AudioFile
)

# VA Filters (header-only, MIT-style STK-4.3 license)
add_library(vafilters-dsp INTERFACE)
target_include_directories(vafilters-dsp INTERFACE
    thirdparty/vafilters
)

# FX DSP algorithms (header-only)
add_library(fxdsp INTERFACE)
target_include_directories(fxdsp INTERFACE
    thirdparty/fxdsp
)

# DspFilters (MIT -- Vinnie Falco)
set(DSPFILTERS_SOURCES
    thirdparty/DspFilters/Biquad.cpp
    thirdparty/DspFilters/Cascade.cpp
    thirdparty/DspFilters/Design.cpp
    thirdparty/DspFilters/Filter.cpp
    thirdparty/DspFilters/Param.cpp
    thirdparty/DspFilters/PoleFilter.cpp
    thirdparty/DspFilters/RootFinder.cpp
    thirdparty/DspFilters/State.cpp
    thirdparty/DspFilters/Butterworth.cpp
    thirdparty/DspFilters/ChebyshevI.cpp
    thirdparty/DspFilters/ChebyshevII.cpp
    thirdparty/DspFilters/Elliptic.cpp
    thirdparty/DspFilters/Bessel.cpp
    thirdparty/DspFilters/Legendre.cpp
    thirdparty/DspFilters/RBJ.cpp
    thirdparty/DspFilters/Custom.cpp
)
add_library(dspfilters STATIC ${DSPFILTERS_SOURCES})
target_include_directories(dspfilters PUBLIC
    thirdparty/DspFilters/include
)
if(MSVC)
    target_compile_options(dspfilters PRIVATE /w)
else()
    target_compile_options(dspfilters PRIVATE -w)
endif()

# STK (subset -- no RtAudio/RtMidi/Socket/File I/O)
set(STK_SOURCES
    thirdparty/stk/src/Stk.cpp
    thirdparty/stk/src/SineWave.cpp
    thirdparty/stk/src/ADSR.cpp
    thirdparty/stk/src/Asymp.cpp
    thirdparty/stk/src/Envelope.cpp
    thirdparty/stk/src/Noise.cpp
    thirdparty/stk/src/Blit.cpp
    thirdparty/stk/src/BlitSaw.cpp
    thirdparty/stk/src/BlitSquare.cpp
    thirdparty/stk/src/Modulate.cpp
    thirdparty/stk/src/BiQuad.cpp
    thirdparty/stk/src/OnePole.cpp
    thirdparty/stk/src/OneZero.cpp
    thirdparty/stk/src/TwoPole.cpp
    thirdparty/stk/src/TwoZero.cpp
    thirdparty/stk/src/PoleZero.cpp
    thirdparty/stk/src/Resonate.cpp
    thirdparty/stk/src/FormSwep.cpp
    thirdparty/stk/src/Delay.cpp
    thirdparty/stk/src/DelayA.cpp
    thirdparty/stk/src/DelayL.cpp
    thirdparty/stk/src/TapDelay.cpp
    thirdparty/stk/src/FreeVerb.cpp
    thirdparty/stk/src/JCRev.cpp
    thirdparty/stk/src/NRev.cpp
    thirdparty/stk/src/PRCRev.cpp
    thirdparty/stk/src/Echo.cpp
    thirdparty/stk/src/Chorus.cpp
    thirdparty/stk/src/PitShift.cpp
    thirdparty/stk/src/LentPitShift.cpp
    thirdparty/stk/src/Clarinet.cpp
    thirdparty/stk/src/Flute.cpp
    thirdparty/stk/src/Brass.cpp
    thirdparty/stk/src/Bowed.cpp
    thirdparty/stk/src/Plucked.cpp
    thirdparty/stk/src/Sitar.cpp
    thirdparty/stk/src/StifKarp.cpp
    thirdparty/stk/src/Saxofony.cpp
    thirdparty/stk/src/Recorder.cpp
    thirdparty/stk/src/BlowBotl.cpp
    thirdparty/stk/src/BlowHole.cpp
    thirdparty/stk/src/Whistle.cpp
    thirdparty/stk/src/Guitar.cpp
    thirdparty/stk/src/Twang.cpp
    thirdparty/stk/src/Fir.cpp
    thirdparty/stk/src/Iir.cpp
    thirdparty/stk/src/Sphere.cpp
    thirdparty/stk/src/FileRead.cpp
    thirdparty/stk/src/FileWvIn.cpp
)
add_library(stk_subset STATIC ${STK_SOURCES})
target_include_directories(stk_subset PUBLIC thirdparty/stk/include)
target_compile_definitions(stk_subset PRIVATE __LITTLE_ENDIAN__)
if(MSVC)
    target_compile_definitions(stk_subset PRIVATE _USE_MATH_DEFINES)
    target_compile_options(stk_subset PRIVATE /w)
else()
    target_compile_options(stk_subset PRIVATE -w)
endif()

nanobind_add_module(_core
    src/nanodsp/_core.cpp
    src/nanodsp/_core_signalsmith.cpp
    src/nanodsp/_core_daisysp.cpp
    src/nanodsp/_core_stk.cpp
    src/nanodsp/_core_madronalib.cpp
    src/nanodsp/_core_hisstools.cpp
    src/nanodsp/_core_choc.cpp
    src/nanodsp/_core_grainflow.cpp
    src/nanodsp/_core_vafilters.cpp
    src/nanodsp/_core_bloscillators.cpp
    src/nanodsp/_core_fxdsp.cpp
    src/nanodsp/_core_iirdesign.cpp
)
target_link_libraries(_core PRIVATE signalsmith-dsp DaisySP DaisySP_LGPL stk_subset madronalib-dsp hisstools grainflow-dsp vafilters-dsp fxdsp dspfilters)
target_include_directories(_core SYSTEM PRIVATE thirdparty/choc)

# Mark all thirdparty include dirs as SYSTEM to suppress warnings from their headers
get_target_property(_sigdsp_inc signalsmith-dsp INTERFACE_INCLUDE_DIRECTORIES)
if(_sigdsp_inc)
    set_target_properties(signalsmith-dsp PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${_sigdsp_inc}")
endif()
get_target_property(_daisy_inc DaisySP INTERFACE_INCLUDE_DIRECTORIES)
if(_daisy_inc)
    set_target_properties(DaisySP PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${_daisy_inc}")
endif()
get_target_property(_daisy_lgpl_inc DaisySP_LGPL INTERFACE_INCLUDE_DIRECTORIES)
if(_daisy_lgpl_inc)
    set_target_properties(DaisySP_LGPL PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${_daisy_lgpl_inc}")
endif()
get_target_property(_stk_inc stk_subset INTERFACE_INCLUDE_DIRECTORIES)
if(_stk_inc)
    set_target_properties(stk_subset PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${_stk_inc}")
endif()
get_target_property(_madro_inc madronalib-dsp INTERFACE_INCLUDE_DIRECTORIES)
if(_madro_inc)
    set_target_properties(madronalib-dsp PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${_madro_inc}")
endif()
get_target_property(_hiss_inc hisstools INTERFACE_INCLUDE_DIRECTORIES)
if(_hiss_inc)
    set_target_properties(hisstools PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${_hiss_inc}")
endif()
get_target_property(_gf_inc grainflow-dsp INTERFACE_INCLUDE_DIRECTORIES)
if(_gf_inc)
    set_target_properties(grainflow-dsp PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${_gf_inc}")
endif()
get_target_property(_dspf_inc dspfilters INTERFACE_INCLUDE_DIRECTORIES)
if(_dspf_inc)
    set_target_properties(dspfilters PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${_dspf_inc}")
endif()

if(MSVC)
    target_compile_options(_core PRIVATE /W4)
    target_compile_definitions(_core PRIVATE NOMINMAX _USE_MATH_DEFINES)
else()
    target_compile_options(_core PRIVATE -Wall -Wextra -Wno-unknown-pragmas -Wno-sign-compare)
    # GCC's -Wmaybe-uninitialized fires on inlined code from SYSTEM headers (e.g. HISSTools Statistics.hpp)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(_core PRIVATE -Wno-maybe-uninitialized)
    endif()
endif()

install(TARGETS _core DESTINATION nanodsp)

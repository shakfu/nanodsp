cmake_minimum_required(VERSION 3.14)

project(DaisySP_LGPL VERSION 0.0.1)

add_library(DaisySP_LGPL STATIC 
Source/Control/line.cpp
Source/Dynamics/balance.cpp
Source/Dynamics/compressor.cpp
Source/Effects/bitcrush.cpp
Source/Effects/fold.cpp
Source/Effects/reverbsc.cpp
Source/Filters/allpass.cpp
Source/Filters/atone.cpp
Source/Filters/biquad.cpp
Source/Filters/comb.cpp
Source/Filters/mode.cpp
Source/Filters/moogladder.cpp
Source/Filters/nlfilt.cpp
Source/Filters/tone.cpp
Source/PhysicalModeling/pluck.cpp
Source/Synthesis/blosc.cpp
Source/Utility/jitter.cpp
Source/Utility/port.cpp
)


set_target_properties(DaisySP_LGPL PROPERTIES PUBLIC
  CXX_STANDARD 14 
  CXX_STANDARD_REQUIRED
  )


target_link_libraries(DaisySP_LGPL PRIVATE DaisySP)
target_include_directories(DaisySP_LGPL PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}/Source
  PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/../Source"
  "${CMAKE_CURRENT_LIST_DIR}/../Source/Control"
  "${CMAKE_CURRENT_LIST_DIR}/../Source/Drums"
  "${CMAKE_CURRENT_LIST_DIR}/../Source/Dynamics"
  "${CMAKE_CURRENT_LIST_DIR}/../Source/Effects"
  "${CMAKE_CURRENT_LIST_DIR}/../Source/Filters"
  "${CMAKE_CURRENT_LIST_DIR}/../Source/Noise"
  "${CMAKE_CURRENT_LIST_DIR}/../Source/PhysicalModeling"
  "${CMAKE_CURRENT_LIST_DIR}/../Source/Synthesis"
  "${CMAKE_CURRENT_LIST_DIR}/../Source/Utility"
  "${CMAKE_CURRENT_LIST_DIR}/Source"
  "${CMAKE_CURRENT_LIST_DIR}/Source/Control"
  "${CMAKE_CURRENT_LIST_DIR}/Source/Dynamics"
  "${CMAKE_CURRENT_LIST_DIR}/Source/Effects"
  "${CMAKE_CURRENT_LIST_DIR}/Source/Filters"
  "${CMAKE_CURRENT_LIST_DIR}/Source/PhysicalModeling"
  "${CMAKE_CURRENT_LIST_DIR}/Source/Synthesis"
  "${CMAKE_CURRENT_LIST_DIR}/Source/Utility"
  )

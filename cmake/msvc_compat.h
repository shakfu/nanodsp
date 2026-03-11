/* MSVC compatibility for GCC/Clang-specific extensions used in thirdparty code */
#ifdef _MSC_VER
#  ifndef _USE_MATH_DEFINES
#    define _USE_MATH_DEFINES
#  endif
#  include <cmath>
#  define __attribute__(x)
#endif

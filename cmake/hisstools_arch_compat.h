/* HISSTools architecture compatibility.
   HISSTools checks __arm__ || __arm64__ for NEON, but GCC/Linux
   defines __aarch64__ instead of __arm64__ (which is Apple-specific).
   This header bridges the gap using the compiler's own target defines. */
#if defined(__aarch64__) && !defined(__arm64__)
#  define __arm64__
#endif

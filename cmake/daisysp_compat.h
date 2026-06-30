/* Compatibility shim for vendored DaisySP / DaisySP-LGPL sources.
 *
 * Several DaisySP headers and translation units use the unqualified name
 * `size_t` in the global namespace without including <cstddef> or <stddef.h>.
 * Older toolchains leaked `::size_t` transitively through other standard
 * headers, so this compiled; newer libc++/SDK versions (e.g. recent Xcode)
 * no longer do, causing "unknown type name 'size_t'" errors.
 *
 * Force-including this header (via the compiler's -include / /FI flag) makes
 * `size_t` available in the global namespace for every DaisySP translation
 * unit, without editing the vendored sources. It mirrors the existing
 * msvc_compat.h / hisstools_arch_compat.h shims used elsewhere in this build.
 */
#pragma once
#include <cstddef>
using std::size_t;

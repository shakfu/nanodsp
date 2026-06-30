/* Forwarding shim.
 *
 * The vendored signalsmith-stretch.h (v1.1.1) includes its DSP dependency as
 * `dsp/spectral.h`. Rather than edit the upstream header, this shim redirects
 * that include to the signalsmith-dsp library already vendored at
 * thirdparty/signalsmith, keeping signalsmith-stretch.h byte-identical to
 * upstream for clean future updates.
 */
#pragma once
#include <signalsmith-dsp/spectral.h>

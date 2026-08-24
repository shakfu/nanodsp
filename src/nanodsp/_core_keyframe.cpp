// Keyframe time stretching via extrema sampling.
//
// An original implementation of the algorithm described in:
//
//   M. Nielsen, "Keyframe Time Stretching via Extrema Sampling", Proc. 29th
//   Int. Conf. Digital Audio Effects (DAFx26), Cambridge, MA, USA, Sept. 2026.
//   Paper distributed under CC BY 4.0.
//
// Written from the paper's equations and pseudocode only. Equation and
// algorithm numbers in the comments below refer to that paper. The author's
// own firmware (heavylight-industries/capicola) is AGPL-3.0 and no part of it
// was consulted or reproduced; see thirdparty/VERSIONS.md for why that matters.
//
// The method reduces a uniformly sampled signal to a sparse set of its local
// extrema ("keyframes"). Because the spacing between extrema tracks the
// signal's local bandwidth, that spacing doubles as a free estimate of
// information density, and the stretch engine uses it to size each overlap-add
// splice: short where extrema crowd together (transients), long across
// sustained material. No FFT and no correlation search are involved.

#include "_core_common.h"

#include <nanobind/stl/pair.h>

namespace {

// ---------------------------------------------------------------------------
// Cubic B-spline kernel, eq (2), and its derivative, eq (3)
// ---------------------------------------------------------------------------
//
// Used for the analysis pass only. Reconstruction uses a Hermite kernel
// instead (eq 8), which is a different basis.

inline void bspline_basis(double t, double b[4]) {
    const double t2 = t * t, t3 = t2 * t, u = 1.0 - t;
    b[0] = (u * u * u) / 6.0;
    b[1] = (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0;
    b[2] = (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0;
    b[3] = t3 / 6.0;
}

// Sample with edge clamping; the 4-tap kernel reaches i-1 .. i+2.
inline float tap(const float *x, long n, long i) {
    if (i < 0) i = 0;
    if (i >= n) i = n - 1;
    return x[i];
}

// Evaluate the B-spline approximation of x at a fractional index, eq (1).
inline double bspline_at(const float *x, long n, double pos) {
    const long i = (long)std::floor(pos);
    double b[4];
    bspline_basis(pos - (double)i, b);
    return b[0] * tap(x, n, i - 1) + b[1] * tap(x, n, i)
         + b[2] * tap(x, n, i + 1) + b[3] * tap(x, n, i + 2);
}

// Bandlimited derivative at an integer index. Evaluating eq (3) at t = 0
// leaves taps (-1/2, 0, 1/2, 0). Unlike the naive difference x[n] - x[n-1],
// whose gain peaks at Nyquist, this has a zero there, so quantisation noise
// and fine texture do not manufacture extrema that are not in the signal.
inline double bspline_derivative_at(const float *x, long n, long i) {
    return -0.5 * tap(x, n, i - 1) + 0.5 * tap(x, n, i + 1);
}

// ---------------------------------------------------------------------------
// Sparse representation
// ---------------------------------------------------------------------------

struct Sparse {
    std::vector<double> idx;  // n_m, subsample position in the uniform buffer
    std::vector<float> val;   // v_m, value there
    size_t frames = 0;        // length of the uniform signal it came from
};

// Algorithm 1: extrema analysis.
Sparse analyze(const float *x, long n, double threshold) {
    Sparse s;
    s.frames = (size_t)n;
    if (n <= 0) return s;

    s.idx.push_back(0.0);  // anchor, Algorithm 1 line 2
    s.val.push_back(x[0]);
    if (n == 1) return s;

    double v_prev = x[0];
    // The last *nonzero* derivative, rather than simply the previous one.
    //
    // Algorithm 1 line 1 seeds d_prev to 0 and line 5 tests sign(d[n]) !=
    // sign(d_prev). Two things go wrong if that is taken literally. At the
    // start of the signal sign(0) differs from the first real derivative, so
    // sample 1 always reports an extremum that is not one. And an extremum
    // sitting exactly on a sample makes the central difference cancel to
    // exactly 0.0 -- routine, not a corner case: any tone whose period divides
    // the sample rate does it, and a 1 kHz sine at 48 kHz does it at every
    // peak and trough. That zero then reads as a sign change on the way in and
    // another on the way out, so the extremum is either double-counted or,
    // once a "previous sign must exist" guard is added, dropped entirely.
    //
    // Comparing against the last nonzero derivative handles both: no sign
    // exists until the signal moves, and a run of zeros is transparent. The
    // subsample refinement below then lands on the exact sample, because
    // alpha (eq 4) evaluates to 0 when d[i-1] is the zero.
    double d_prev = 0.0;
    for (long i = 1; i < n; ++i) {
        const double d = bspline_derivative_at(x, n, i);
        if (d == 0.0) continue;  // no sign yet; d_prev is left alone
        if (d_prev != 0.0 && ((d < 0.0) != (d_prev < 0.0))) {
            // Difference thresholding, section 2.2. Measured against the last
            // *saved* extremum rather than the last candidate, which gives the
            // decision a hysteresis-like deadband.
            if (std::fabs(x[i] - v_prev) > threshold) {
                const double a = std::fabs(bspline_derivative_at(x, n, i - 1));
                const double denom = a + std::fabs(d);
                const double alpha = denom > 0.0 ? a / denom : 0.5;  // eq (4)
                const double n_m = (double)(i - 1) + alpha;          // eq (5)
                const double v_m = bspline_at(x, n, n_m);            // eq (6)
                s.idx.push_back(n_m);
                s.val.push_back((float)v_m);
                v_prev = v_m;
            }
        }
        d_prev = d;
    }

    // Terminating anchor, so the final span is defined. Like the leading
    // anchor this is not an extremum, so the zero-tangent simplification below
    // does not strictly hold across the first and last spans; the error is
    // local to them and the paper accepts the same trade for block boundaries.
    if (s.idx.back() < (double)(n - 1)) {
        s.idx.push_back((double)(n - 1));
        s.val.push_back((float)bspline_at(x, n, (double)(n - 1)));
    }
    return s;
}

// ---------------------------------------------------------------------------
// Reconstruction
// ---------------------------------------------------------------------------
//
// Non-uniform cubic Hermite, eq (7)-(8). Every saved keyframe is an extremum,
// so its derivative is zero by construction; setting both tangents to zero
// collapses eq (7) to eq (11), which costs two multiplies and no divisions.
// The remaining basis pair is the smoothstep function.
inline float hermite_zero_tangent(float v0, float v1, double t) {
    const double t2 = t * t, t3 = t2 * t;
    const double h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    const double h10 = -2.0 * t3 + 3.0 * t2;
    return (float)(v0 * h00 + v1 * h10);
}

// Locate the window (keyframe pair) containing phi, updating m in place.
// Playheads move sequentially, so this is an amortised O(1) local scan.
inline double window_at(const Sparse &s, size_t &m, double phi) {
    const size_t last = s.idx.size() - 2;
    if (m > last) m = last;
    while (m < last && s.idx[m + 1] <= phi) ++m;
    while (m > 0 && s.idx[m] > phi) --m;
    const double span = s.idx[m + 1] - s.idx[m];
    if (span <= 0.0) return 0.0;
    double t = (phi - s.idx[m]) / span;
    if (t < 0.0) t = 0.0;
    if (t > 1.0) t = 1.0;
    return t;
}

// Algorithm 2: uniform resampling of the sparse buffer.
void reconstruct(const Sparse &s, float *y, long n_out) {
    if (s.idx.size() < 2) {
        const float fill = s.val.empty() ? 0.0f : s.val[0];
        for (long i = 0; i < n_out; ++i) y[i] = fill;
        return;
    }
    size_t m = 0;
    for (long i = 0; i < n_out; ++i) {
        const double t = window_at(s, m, (double)i);
        y[i] = hermite_zero_tangent(s.val[m], s.val[m + 1], t);
    }
}

// Algorithm 3: sparse-domain time stretching.
//
// Three playheads. The reference advances at the time rate and marks where
// playback *should* be; the playhead produces audio and advances at the pitch
// rate; a temporary playhead produces audio during a splice. The leash between
// reference and playhead is measured in keyframes rather than samples, so it
// contracts in time wherever the signal is dense -- which is what makes the
// splice duration adapt to transients without detecting them.
void stretch_render(const Sparse &s, float *y, long n_out, double time_rate,
                    double pitch_rate, long k, double max_splice) {
    if (s.idx.size() < 2) {
        const float fill = s.val.empty() ? 0.0f : s.val[0];
        for (long i = 0; i < n_out; ++i) y[i] = fill;
        return;
    }

    const double end = s.idx.back();
    double phi_ref = 0.0, phi_play = 0.0, phi_temp = 0.0;
    size_t m_ref = 0, m_play = 0, m_temp = 0;
    bool splicing = false;
    double t_temp = 0.0, dt_temp = 0.0;

    for (long i = 0; i < n_out; ++i) {
        window_at(s, m_ref, phi_ref < end ? phi_ref : end);
        double t_play = window_at(s, m_play, phi_play < end ? phi_play : end);

        if (!splicing) {
            const long d = (long)m_play - (long)m_ref;
            if (std::labs(d) > k) {
                phi_temp = phi_ref;
                m_temp = m_ref;
                // eq (12): the splice spans k keyframes ahead of the reference,
                // so its duration in samples is however long that takes here.
                size_t ahead = m_ref + (size_t)k;
                if (ahead >= s.idx.size()) ahead = s.idx.size() - 1;
                double span = s.idx[ahead] - s.idx[m_ref];
                // Section 3.7: long silences or otherwise sparse passages
                // produce very long splices, so the span is capped.
                if (max_splice > 0.0 && span > max_splice) span = max_splice;
                if (span > 1.0) {
                    dt_temp = 1.0 / span;
                    t_temp = 0.0;
                    splicing = true;
                }
            }
        }

        if (splicing && t_temp <= 1.0) {
            const double t_t = window_at(s, m_temp, phi_temp < end ? phi_temp : end);
            const float y_play = hermite_zero_tangent(s.val[m_play], s.val[m_play + 1], t_play);
            const float y_temp = hermite_zero_tangent(s.val[m_temp], s.val[m_temp + 1], t_t);
            y[i] = (float)(y_temp * t_temp + y_play * (1.0 - t_temp));
            phi_play += pitch_rate;
            phi_temp += pitch_rate;
            t_temp += dt_temp * pitch_rate;
            phi_ref += time_rate;
            continue;
        }
        if (splicing) {  // t_temp > 1: the fresh playhead has fully faded in
            phi_play = phi_temp;
            m_play = m_temp;
            splicing = false;
            t_play = window_at(s, m_play, phi_play < end ? phi_play : end);
        }

        y[i] = hermite_zero_tangent(s.val[m_play], s.val[m_play + 1], t_play);
        phi_play += pitch_rate;
        phi_ref += time_rate;
    }
}

}  // namespace

void bind_keyframe(nb::module_ &m) {
    nb::module_ kf = m.def_submodule(
        "keyframe", "Keyframe time stretching via extrema sampling (DAFx26).");

    kf.def(
        "analyze",
        [](ArrayF input, double threshold) {
            const long n = (long)input.shape(0);
            const float *x = input.data();
            Sparse s;
            { nb::gil_scoped_release rel;
              s = analyze(x, n, threshold);
            }
            const size_t count = s.idx.size();
            auto *idx = new float[count];
            auto *val = new float[count];
            for (size_t i = 0; i < count; ++i) {
                idx[i] = (float)s.idx[i];
                val[i] = s.val[i];
            }
            return std::make_pair(make_f1(idx, count), make_f1(val, count));
        },
        "input"_a, "threshold"_a = 0.001,
        "Reduce a signal to its local extrema. Returns (indices, values), where "
        "indices are subsample positions in the input. Lower thresholds keep "
        "more detail; the paper reports 0.001 (-60 dB) as faithful.");

    kf.def(
        "sparsify",
        [](ArrayF input, double threshold) {
            const long n = (long)input.shape(0);
            const float *x = input.data();
            auto *out = new float[(size_t)n];
            { nb::gil_scoped_release rel;
              const Sparse s = analyze(x, n, threshold);
              reconstruct(s, out, n);
            }
            return make_f1(out, (size_t)n);
        },
        "input"_a, "threshold"_a = 0.001,
        "Analysis followed by reconstruction at the original length. The "
        "round trip is lossy -- this exposes the cost of the representation "
        "on its own, with no time or pitch change applied.");

    kf.def(
        "stretch",
        [](ArrayF input, double time_rate, double pitch_rate, long k,
           double threshold, double max_splice, long n_out) {
            const long n = (long)input.shape(0);
            const float *x = input.data();
            util_check_count((int)n_out, "n_out");
            auto *out = new float[(size_t)n_out];
            { nb::gil_scoped_release rel;
              const Sparse s = analyze(x, n, threshold);
              stretch_render(s, out, n_out, time_rate, pitch_rate, k, max_splice);
            }
            return make_f1(out, (size_t)n_out);
        },
        "input"_a, "time_rate"_a, "pitch_rate"_a = 1.0, "k"_a = 16,
        "threshold"_a = 0.001, "max_splice"_a = 0.0, "n_out"_a = 0,
        "Render n_out samples at the given time and pitch rates. time_rate is "
        "the reciprocal of the stretch factor; pitch_rate is the playback "
        "speed ratio. k is the splice threshold in keyframes.");
}

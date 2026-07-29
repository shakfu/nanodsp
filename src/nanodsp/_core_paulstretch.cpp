#include "_core_common.h"
#include <signalsmith-dsp/fft.h>
#include <random>

// ============================================================================
// PaulStretch -- extreme time-stretching via phase-randomized spectral
// resynthesis.
//
// Algorithm by Nasca Octavian Paul (public domain). This is an original
// implementation built on the signalsmith RealFFT; it does NOT vendor the
// GPLv3 paulxstretch application sources. It reproduces the core PaulStretch
// technique plus optional onset preservation and spectral effects.
//
// For each output frame:
//   1. read a windowed input segment (input advances by displace/stretch)
//   2. forward real FFT, extract per-bin magnitude
//   3. (optional) reshape the magnitude spectrum: pitch shift, harmonics,
//      spectral spread (blur), band-pass filtering
//   4. assign each bin a random phase (the source of the smeared character);
//      on a detected onset the original phases are kept to preserve transients
//   5. inverse FFT, window again, overlap-add into the output at a fixed hop
//
// Output is energy-normalized by the overlap-add of the squared window, which
// is the correct model for incoherent (phase-randomized) overlap.
// ============================================================================

using RFFT = signalsmith::fft::RealFFT<float>;
using cfloat = std::complex<float>;

namespace {

constexpr float kTwoPi = 6.283185307179586f;

class PaulStretch {
public:
    PaulStretch(size_t window_size, float sample_rate)
        : sampleRate(sample_rate), rng(42u) {
        setWindowSize(window_size);
    }

    size_t windowSizeValue() const { return windowSize; }
    float sampleRateValue() const { return sampleRate; }

    void setWindowSize(size_t n) {
        if (n < 16) n = 16;
        if (n % 2) n += 1;  // require even size
        windowSize = n;
        bins = n / 2;        // signalsmith RealFFT packs N/2 complex bins
        rfft.setSize(n);

        // Nasca PaulStretch window: w(x) = (1 - (2x-1)^2)^1.25, x in [0,1).
        window.resize(n);
        for (size_t i = 0; i < n; ++i) {
            double x = static_cast<double>(i) / static_cast<double>(n);
            double v = 1.0 - (2.0 * x - 1.0) * (2.0 * x - 1.0);
            if (v < 0.0) v = 0.0;
            window[i] = static_cast<float>(std::pow(v, 1.25));
        }

        winFrame.assign(n, 0.0f);
        timeOut.assign(n, 0.0f);
        spec.assign(bins, cfloat(0.0f, 0.0f));
        // Unpacked spectra span DC..Nyquist == bins + 1 points.
        mag.assign(bins + 1, 0.0f);
        magScratch.assign(bins + 1, 0.0f);
        phase.assign(bins + 1, 0.0f);
        warpBuf.assign(bins + 1, 0.0f);
        envBuf.assign(bins + 1, 0.0f);
        reset();
    }

    void reset() {
        energyAvg = 0.0f;
        haveAvg = false;
    }

    void setSeed(unsigned s) { rng.seed(s); }

    // --- tunable parameters (set from Python before process) ---
    float onsetSensitivity = 0.0f;   // 0 disables transient preservation; (0,1]
    float pitchSemitones = 0.0f;     // spectral pitch/octave shift in semitones
    int harmonics = 0;               // number of added harmonic copies (0 = off)
    float spread = 0.0f;             // spectral blur radius in bins
    float spreadOctaves = 0.0f;      // constant-Q spread width; 0 = off
    float tonalVsNoise = 0.0f;       // [-1,1]; >0 keeps tonal, <0 keeps noise
    float tonalNoiseOctaves = 0.2f;  // envelope-estimator width in octaves
    float lowpassHz = 0.0f;          // <=0 disables; zero bins above this freq
    float highpassHz = 0.0f;         // <=0 disables; zero bins below this freq

    // Stretch `in` (length inLen) by `stretch` and return the result.
    std::vector<float> process(const float *in, size_t inLen, double stretch) {
        if (stretch < 1e-6) stretch = 1e-6;
        const size_t N = windowSize;
        const size_t displace = N / 2;                       // output hop
        double readIncr = static_cast<double>(displace) / stretch;  // input hop
        if (readIncr < 1e-9) readIncr = 1e-9;

        // Number of analysis frames: step the read head until it passes the end.
        size_t numFrames = 0;
        for (double pos = 0.0; static_cast<size_t>(pos) < inLen; pos += readIncr)
            ++numFrames;
        if (numFrames == 0) numFrames = 1;

        const size_t outLen = (numFrames - 1) * displace + N;
        std::vector<float> out(outLen, 0.0f);
        std::vector<float> env(outLen, 0.0f);  // overlap-add of window^2

        std::uniform_real_distribution<float> uni(0.0f, 1.0f);
        const float invN = 1.0f / static_cast<float>(N);  // RealFFT round-trip is N

        double pos = 0.0;
        for (size_t f = 0; f < numFrames; ++f) {
            const size_t istart = static_cast<size_t>(pos);

            // Windowed analysis frame (zero-pad past the end of the input).
            for (size_t i = 0; i < N; ++i) {
                float s = (istart + i < inLen) ? in[istart + i] : 0.0f;
                winFrame[i] = s * window[i];
            }

            rfft.fft(winFrame.data(), spec.data());

            // Unpack packed bins into magnitude/phase over DC..Nyquist.
            // spec[0] holds {DC, Nyquist} as two real values.
            mag[0] = std::fabs(spec[0].real());
            phase[0] = spec[0].real() < 0.0f ? float(M_PI) : 0.0f;
            mag[bins] = std::fabs(spec[0].imag());
            phase[bins] = spec[0].imag() < 0.0f ? float(M_PI) : 0.0f;
            for (size_t k = 1; k < bins; ++k) {
                mag[k] = std::abs(spec[k]);
                phase[k] = std::arg(spec[k]);
            }

            applySpectralEffects();

            const bool onset = detectOnset();

            // Rebuild the packed spectrum from (possibly reshaped) magnitudes.
            // Random phase produces the smear; preserved phase keeps transients.
            // DC and Nyquist must stay real: a random *sign* is their "phase".
            {
                float dcSign, nyqSign;
                if (onset) {
                    dcSign = std::cos(phase[0]);
                    nyqSign = std::cos(phase[bins]);
                } else {
                    dcSign = uni(rng) < 0.5f ? -1.0f : 1.0f;
                    nyqSign = uni(rng) < 0.5f ? -1.0f : 1.0f;
                }
                spec[0] = cfloat(mag[0] * dcSign, mag[bins] * nyqSign);
            }
            for (size_t k = 1; k < bins; ++k) {
                float ph = onset ? phase[k] : uni(rng) * kTwoPi;
                spec[k] = cfloat(mag[k] * std::cos(ph), mag[k] * std::sin(ph));
            }

            rfft.ifft(spec.data(), timeOut.data());

            // Synthesis window + overlap-add, tracking the window^2 envelope.
            const size_t opos = f * displace;
            for (size_t i = 0; i < N; ++i) {
                float w = window[i];
                out[opos + i] += timeOut[i] * invN * w;
                env[opos + i] += w * w;
            }

            pos += readIncr;
        }

        // Normalize by the RMS of the overlapping windows. Phase-randomized
        // frames overlap incoherently, so their amplitudes add in quadrature;
        // dividing by sqrt(sum of window^2) equalizes the level across the
        // whole signal, including the tapered edges (dividing by env directly
        // would blow up where only one tapering window covers a sample).
        for (size_t i = 0; i < outLen; ++i) {
            if (env[i] > 1e-8f) out[i] /= std::sqrt(env[i]);
        }
        return out;
    }

private:
    // Linear interpolation into the magnitude array at fractional bin `x`.
    float magAt(const std::vector<float> &m, double x) const {
        if (x < 0.0 || x > static_cast<double>(bins)) return 0.0f;
        size_t i0 = static_cast<size_t>(x);
        if (i0 >= bins) return m[bins];
        float frac = static_cast<float>(x - static_cast<double>(i0));
        return m[i0] * (1.0f - frac) + m[i0 + 1] * frac;
    }

    // Constant-Q smoothing of the magnitude spectrum.
    //
    // A plain box blur over bins smears low partials far more (in musical
    // terms) than high ones, because bin spacing is linear in frequency. This
    // instead warps the spectrum onto a log-frequency axis spanning
    // [20 Hz, Nyquist], smooths there, and warps back -- so each partial is
    // spread by a constant fraction of its own frequency.
    //
    // The smoother is a one-pole run forward then backward (zero phase, no
    // group delay in the frequency direction), repeated `kPasses` times for a
    // steeper skirt. `octaves` is the target standard deviation of the
    // resulting kernel, so the smoothing width is set in musical terms and is
    // independent of FFT size and sample rate. `dst` must not alias `src`.
    void logSmooth(const std::vector<float> &src, std::vector<float> &dst,
                   float octaves) const {
        const size_t nf = bins;             // bin nf sits at Nyquist
        const size_t M = nf + 1;
        const double maxFreq = 0.5 * static_cast<double>(sampleRate);
        const double minFreq = 20.0;
        if (maxFreq <= minFreq || nf < 2) {
            dst.assign(src.begin(), src.end());
            return;
        }
        const double logMin = std::log(minFreq);
        const double logRange = std::log(maxFreq) - logMin;

        // --- warp linear -> log: log-axis point i samples the linear bin
        //     holding frequency exp(logMin + (i/nf) * logRange) ---
        for (size_t i = 0; i < M; ++i) {
            double f = std::exp(logMin +
                                (static_cast<double>(i) / nf) * logRange);
            warpBuf[i] = magAt(src, f / maxFreq * nf);
        }

        // --- smooth on the log axis ---
        //
        // Convert the requested width in octaves to a one-pole coefficient.
        // A single one-pole with coefficient a has impulse-response variance
        // a/(1-a)^2 samples^2; running it forward and backward doubles that,
        // and kPasses rounds multiply it again, so the cascade reaches
        //     sigma^2 = 2 * kPasses * a / (1-a)^2.
        // Solving that quadratic for a gives the closed form below, which
        // stays well conditioned for both very narrow and very wide settings.
        constexpr int kPasses = 2;
        const double octPerSample =
            (logRange / std::log(2.0)) / static_cast<double>(nf);
        const double sigma = static_cast<double>(octaves) / octPerSample;
        const double v = sigma * sigma / (2.0 * kPasses);
        double a = 0.0;
        if (v > 1e-12)
            a = (2.0 * v + 1.0 - std::sqrt(4.0 * v + 1.0)) / (2.0 * v);
        if (!(a > 0.0)) a = 0.0;
        if (a > 0.999999) a = 0.999999;
        const float af = static_cast<float>(a);
        const float bf = 1.0f - af;

        for (int p = 0; p < kPasses; ++p) {
            warpBuf[0] = 0.0f;
            for (size_t i = 1; i < M; ++i)
                warpBuf[i] = warpBuf[i - 1] * af + warpBuf[i] * bf;
            warpBuf[M - 1] = 0.0f;
            for (size_t i = M - 1; i-- > 1;)
                warpBuf[i] = warpBuf[i + 1] * af + warpBuf[i] * bf;
        }

        // --- warp log -> linear; bins below minFreq have no log-axis
        //     counterpart and read as zero ---
        dst[0] = 0.0f;
        for (size_t i = 1; i < M; ++i) {
            double f = static_cast<double>(i) / nf * maxFreq;
            double x = (std::log(f) - logMin) / logRange * nf;
            dst[i] = (x > 0.0) ? magAt(warpBuf, x) : 0.0f;
        }
    }

    // Split the spectrum into tonal peaks and the noise floor beneath them,
    // then keep one and discard the other.
    //
    // The constant-Q smoothed spectrum is a local envelope: whatever stands
    // above its own neighbourhood is tonal, the rest is the noise floor. The
    // spectrum is split into those two parts once, using a fixed detection
    // threshold, and `tonalVsNoise` then blends the input toward one of them:
    // +1 keeps only the peaks (a stretched choir holds its pitches and loses
    // its hiss), -1 keeps only the floor (the pitches drop out and the breath
    // remains), 0 leaves the spectrum untouched.
    //
    // Note this is not the mapping PaulXStretch uses. Scaling the threshold
    // with the parameter, as it does, is degenerate at both extremes: the
    // envelope is a smoothed copy of the same spectrum, so once the threshold
    // exceeds a peak's own ratio to its neighbourhood the peak is destroyed
    // along with the noise, and the setting collapses to silence. Blending
    // between the two components instead keeps the whole range usable and
    // monotone, and makes both endpoints mean what they say.
    void applyTonalVsNoise() {
        const size_t M = bins + 1;
        // How far above the local envelope a bin must sit to count as tonal.
        constexpr float kThreshold = 2.0f;

        float bw = tonalNoiseOctaves;
        if (bw <= 0.0f) bw = 0.2f;
        logSmooth(mag, envBuf, bw);

        float t = tonalVsNoise;
        if (t > 1.0f) t = 1.0f;
        if (t < -1.0f) t = -1.0f;

        const float mix = std::fabs(t);
        const float keep = 1.0f - mix;
        for (size_t k = 0; k < M; ++k) {
            float m = mag[k];
            float tonal = m - (envBuf[k] + 1e-12f) * kThreshold;
            if (tonal < 0.0f) tonal = 0.0f;
            // The two parts sum back to the original magnitude.
            float target = (t >= 0.0f) ? tonal : (m - tonal);
            mag[k] = m * keep + target * mix;
        }
    }

    void applySpectralEffects() {
        const size_t M = bins + 1;

        // --- tonal / noise separation: applied to the raw analysis spectrum,
        //     before any reshaping smears the peaks it needs to detect ---
        if (tonalVsNoise != 0.0f) applyTonalVsNoise();

        // --- pitch / octave shift: resample the magnitude spectrum ---
        if (pitchSemitones != 0.0f) {
            double ratio = std::pow(2.0, pitchSemitones / 12.0);
            for (size_t k = 0; k < M; ++k)
                magScratch[k] = magAt(mag, static_cast<double>(k) / ratio);
            mag.swap(magScratch);
        }

        // --- harmonics: add integer-multiple copies with geometric decay ---
        if (harmonics > 0) {
            for (size_t k = 0; k < M; ++k) magScratch[k] = mag[k];
            float decay = 0.6f;
            float amp = decay;
            for (int h = 2; h <= harmonics + 1; ++h) {
                for (size_t k = 0; k < M; ++k)
                    magScratch[k] += amp * magAt(mag, static_cast<double>(k) / h);
                amp *= decay;
            }
            mag.swap(magScratch);
        }

        // --- spectral spread: box blur of the magnitude over `spread` bins ---
        if (spread > 0.0f) {
            int r = static_cast<int>(spread + 0.5f);
            if (r > 0) {
                float invCount = 1.0f / static_cast<float>(2 * r + 1);
                for (int k = 0; k < static_cast<int>(M); ++k) {
                    float acc = 0.0f;
                    for (int j = -r; j <= r; ++j) {
                        int idx = k + j;
                        if (idx < 0) idx = 0;
                        if (idx >= static_cast<int>(M)) idx = static_cast<int>(M) - 1;
                        acc += mag[idx];
                    }
                    magScratch[k] = acc * invCount;
                }
                mag.swap(magScratch);
            }
        }

        // --- constant-Q spread: smear each partial across a fixed fraction of
        //     its own frequency rather than a fixed number of bins ---
        if (spreadOctaves > 0.0f) {
            logSmooth(mag, magScratch, spreadOctaves);
            mag.swap(magScratch);
        }

        // --- spectral band filtering: zero bins outside [highpass, lowpass] ---
        if (lowpassHz > 0.0f || highpassHz > 0.0f) {
            float binHz = sampleRate / static_cast<float>(windowSize);
            for (size_t k = 0; k < M; ++k) {
                float fHz = static_cast<float>(k) * binHz;
                if (highpassHz > 0.0f && fHz < highpassHz) mag[k] = 0.0f;
                if (lowpassHz > 0.0f && fHz > lowpassHz) mag[k] = 0.0f;
            }
        }
    }

    // Energy-based onset detector with an exponential moving average baseline.
    bool detectOnset() {
        if (onsetSensitivity <= 0.0f) return false;
        float energy = 0.0f;
        for (size_t k = 0; k <= bins; ++k) energy += mag[k] * mag[k];

        bool onset = false;
        if (haveAvg) {
            // sensitivity in (0,1] maps to a ratio threshold in [1.5, 5.5].
            float s = onsetSensitivity > 1.0f ? 1.0f : onsetSensitivity;
            float thresh = 1.5f + (1.0f - s) * 4.0f;
            onset = energy > thresh * (energyAvg + 1e-12f);
        }
        float a = 0.2f;
        energyAvg = haveAvg ? (a * energy + (1.0f - a) * energyAvg) : energy;
        haveAvg = true;
        return onset;
    }

    float sampleRate;
    size_t windowSize = 0;
    size_t bins = 0;
    RFFT rfft{2};
    std::mt19937 rng;

    std::vector<float> window;
    std::vector<float> winFrame;
    std::vector<float> timeOut;
    std::vector<cfloat> spec;
    std::vector<float> mag;
    std::vector<float> magScratch;
    std::vector<float> phase;
    mutable std::vector<float> warpBuf;  // log-axis scratch for logSmooth
    std::vector<float> envBuf;           // spectral envelope for tonal/noise

    float energyAvg = 0.0f;
    bool haveAvg = false;
};

}  // namespace

void bind_paulstretch(nb::module_ &m) {
    auto sub = m.def_submodule(
        "paulstretch",
        "PaulStretch extreme time-stretching (phase-randomized spectral "
        "resynthesis) with onset preservation and spectral effects.");

    nb::class_<PaulStretch>(sub, "PaulStretch",
        "Extreme time-stretch processor. Construct with a window size (in "
        "samples) and sample rate, set optional parameters, then call "
        "process(input, stretch).")
        .def(nb::init<size_t, float>(), "window_size"_a, "sample_rate"_a)
        .def("reset", &PaulStretch::reset,
             "Reset the onset-detector running state.")
        .def("set_seed", &PaulStretch::setSeed, "seed"_a,
             "Seed the phase-randomization RNG for reproducible output.")
        .def_prop_ro("window_size", &PaulStretch::windowSizeValue)
        .def_prop_ro("sample_rate", &PaulStretch::sampleRateValue)
        .def_rw("onset_sensitivity", &PaulStretch::onsetSensitivity,
                "Transient preservation in (0,1]; 0 disables it.")
        .def_rw("pitch_semitones", &PaulStretch::pitchSemitones,
                "Spectral pitch shift in semitones (+12 = up one octave).")
        .def_rw("harmonics", &PaulStretch::harmonics,
                "Number of added harmonic copies (0 = off).")
        .def_rw("spread", &PaulStretch::spread,
                "Spectral blur radius in bins (linear-frequency box blur).")
        .def_rw("spread_octaves", &PaulStretch::spreadOctaves,
                "Constant-Q spectral spread width in octaves; 0 disables it. "
                "Smears each partial across a fixed fraction of its own "
                "frequency instead of a fixed number of bins.")
        .def_rw("tonal_vs_noise", &PaulStretch::tonalVsNoise,
                "Tonal/noise separation in [-1,1]; 0 disables it. Positive "
                "keeps tonal peaks, negative keeps the noise floor.")
        .def_rw("tonal_noise_octaves", &PaulStretch::tonalNoiseOctaves,
                "Width in octaves of the spectral envelope estimator used by "
                "tonal_vs_noise.")
        .def_rw("lowpass_hz", &PaulStretch::lowpassHz,
                "Spectral low-pass cutoff in Hz (<=0 disables).")
        .def_rw("highpass_hz", &PaulStretch::highpassHz,
                "Spectral high-pass cutoff in Hz (<=0 disables).")
        .def("process",
             [](PaulStretch &self, ArrayF input, double stretch) {
                 size_t n = input.shape(0);
                 const float *in = input.data();
                 std::vector<float> result;
                 {
                     nb::gil_scoped_release rel;
                     result = self.process(in, n, stretch);
                 }
                 size_t outN = result.size();
                 auto *out = new float[outN];
                 std::copy(result.begin(), result.end(), out);
                 return make_f1(out, outN);
             },
             "input"_a, "stretch"_a,
             "Time-stretch a mono float32 array by `stretch` (>1 = longer). "
             "Returns a new float32 array.");
}

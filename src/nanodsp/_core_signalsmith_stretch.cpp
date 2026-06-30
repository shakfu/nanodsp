#include "_core_common.h"
#include <signalsmith-stretch.h>

// ============================================================================
// Signalsmith Stretch -- high-quality time-stretching and pitch-shifting.
//
// Wraps the MIT-licensed signalsmith-stretch library (Geraint Luff /
// Signalsmith Audio), vendored at release 1.1.1 on top of the already-vendored
// signalsmith-dsp. Unlike PaulStretch (which smears phase for extreme ratios),
// this is a phase-vocoder-derived stretcher tuned to stay musical and
// transient-aware at modest ratios, with independent pitch-shifting.
//
// The library is real-time/block-based: the time-stretch ratio is set by the
// relative number of input vs output samples handed to process(). This wrapper
// drives the canonical offline whole-buffer pattern (seek -> process -> flush
// -> pre-roll fold-back) so a fixed-length input becomes an exact-length,
// latency-trimmed output. Stretch and pitch are decoupled: `stretch` changes
// duration, `transpose_semitones` changes pitch, either independently.
//
// All channels are processed together in a single pass (the library keeps a
// stereo image coherent), so the binding takes and returns planar
// [channels, frames] float32 arrays rather than looping per channel.
// ============================================================================

using Stretch = signalsmith::stretch::SignalsmithStretch<float>;

namespace {

class StretchProcessor {
public:
    StretchProcessor(int channels, float sample_rate, bool cheaper, long seed)
        : channels_(channels), sampleRate_(sample_rate), cheaper_(cheaper),
          seed_(seed) {
        if (channels < 1)
            throw std::invalid_argument("channels must be >= 1");
        if (sample_rate <= 0.0f)
            throw std::invalid_argument("sample_rate must be positive");
    }

    int channelsValue() const { return channels_; }
    float sampleRateValue() const { return sampleRate_; }
    bool cheaperValue() const { return cheaper_; }

    void setSeed(long s) { seed_ = s; }

    // --- tunable parameters (set from Python before process) ---
    float transposeSemitones = 0.0f;  // pitch shift in semitones (independent of stretch)
    float tonalityHz = 0.0f;          // <=0 disables; above this, shift acts like the unprocessed signal

    // Offline time-stretch + pitch-shift. `input` is planar [C][N]; the result
    // is written planar into a flat [C * outFrames] vector (row-major), with
    // outFrames set to round(N * stretch).
    std::vector<float> process(const float *const *input, int N, double stretch,
                               int &outFrames) {
        if (stretch < 1e-6) stretch = 1e-6;
        const int C = channels_;

        // Fresh instance per call so a given (seed, params, input) is fully
        // reproducible -- the library only seeds its RNG at construction, and
        // its phase-randomisation (engaged past ~2x) would otherwise depend on
        // prior state.
        Stretch stretcher(seed_);
        if (cheaper_)
            stretcher.presetCheaper(C, sampleRate_);
        else
            stretcher.presetDefault(C, sampleRate_);
        float tonality = tonalityHz > 0.0f ? tonalityHz / sampleRate_ : 0.0f;
        stretcher.setTransposeSemitones(transposeSemitones, tonality);

        const int inLat = stretcher.inputLatency();
        const int outLat = stretcher.outputLatency();
        const int outputLength = static_cast<int>(std::lround(static_cast<double>(N) * stretch));
        const int padIn = N + inLat;            // trailing silence flushes the tail
        const int padOut = outputLength + outLat;  // pre-roll lives in [0, outLat)

        // Planar padded input; zero-pad the trailing inLat samples.
        std::vector<std::vector<float>> inBuf(C, std::vector<float>(padIn, 0.0f));
        std::vector<const float *> inPtr(C), inPtrOffset(C);
        for (int c = 0; c < C; ++c) {
            std::copy(input[c], input[c] + N, inBuf[c].begin());
            inPtr[c] = inBuf[c].data();
            inPtrOffset[c] = inBuf[c].data() + inLat;
        }

        std::vector<std::vector<float>> outBuf(C, std::vector<float>(padOut, 0.0f));
        std::vector<float *> outPtr(C), outPtrTail(C);
        for (int c = 0; c < C; ++c) {
            outPtr[c] = outBuf[c].data();
            outPtrTail[c] = outBuf[c].data() + outputLength;
        }

        // Prime with the first inLat input samples, stretch the body, then
        // flush the remaining outLat samples of pre-rolled output.
        stretcher.seek(inPtr, inLat, 1.0 / stretch);
        stretcher.process(inPtrOffset, N, outPtr, outputLength);
        stretcher.flush(outPtrTail, outLat);

        // Exact-length trim: fold the reflected, negated pre-roll back over the
        // start so the usable output begins cleanly at offset outLat.
        const int fold = std::min(outLat, outputLength);
        for (int c = 0; c < C; ++c)
            for (int i = 0; i < fold; ++i)
                outBuf[c][outLat + i] -= outBuf[c][outLat - 1 - i];

        outFrames = outputLength;
        std::vector<float> out(static_cast<size_t>(C) * outputLength);
        for (int c = 0; c < C; ++c)
            std::copy(outBuf[c].begin() + outLat,
                      outBuf[c].begin() + outLat + outputLength,
                      out.begin() + static_cast<size_t>(c) * outputLength);
        return out;
    }

private:
    int channels_;
    float sampleRate_;
    bool cheaper_;
    long seed_;
};

}  // namespace

void bind_signalsmith_stretch(nb::module_ &m) {
    auto sub = m.def_submodule(
        "signalsmith_stretch",
        "Signalsmith high-quality time-stretching and pitch-shifting "
        "(phase-vocoder-derived, transient-aware).");

    nb::class_<StretchProcessor>(sub, "SignalsmithStretch",
        "Time-stretch / pitch-shift processor. Construct with channel count "
        "and sample rate, set optional parameters, then call "
        "process(input, stretch).")
        .def(nb::init<int, float, bool, long>(), "channels"_a, "sample_rate"_a,
             "cheaper"_a = false, "seed"_a = 0)
        .def("set_seed", &StretchProcessor::setSeed, "seed"_a,
             "Seed the internal RNG for reproducible output.")
        .def_prop_ro("channels", &StretchProcessor::channelsValue)
        .def_prop_ro("sample_rate", &StretchProcessor::sampleRateValue)
        .def_prop_ro("cheaper", &StretchProcessor::cheaperValue)
        .def_rw("transpose_semitones", &StretchProcessor::transposeSemitones,
                "Pitch shift in semitones (independent of time-stretch; "
                "+12 = up one octave). Fractional values are allowed.")
        .def_rw("tonality_hz", &StretchProcessor::tonalityHz,
                "Tonality limit in Hz (<=0 disables). Above this frequency the "
                "pitch shift is rolled back toward the original, preserving "
                "high-frequency timbre/air.")
        .def("process",
             [](StretchProcessor &self, Array2F input, double stretch) {
                 int C = static_cast<int>(input.shape(0));
                 int N = static_cast<int>(input.shape(1));
                 if (C != self.channelsValue())
                     throw std::invalid_argument(
                         "input channel count does not match the configured "
                         "channels");
                 const float *base = input.data();
                 std::vector<const float *> chans(C);
                 for (int c = 0; c < C; ++c)
                     chans[c] = base + static_cast<size_t>(c) * N;

                 int outFrames = 0;
                 std::vector<float> result;
                 {
                     nb::gil_scoped_release rel;
                     result = self.process(chans.data(), N, stretch, outFrames);
                 }
                 auto *out = new float[result.size()];
                 std::copy(result.begin(), result.end(), out);
                 return make_f2(out, static_cast<size_t>(C),
                                static_cast<size_t>(outFrames));
             },
             "input"_a, "stretch"_a,
             "Time-stretch a planar [channels, frames] float32 array by "
             "`stretch` (>1 = longer) while pitch-shifting by "
             "`transpose_semitones`. Returns a new [channels, out_frames] "
             "float32 array with out_frames ~= frames * stretch.");
}

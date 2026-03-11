#include "_core_common.h"

// GrainflowLib headers
#include "gfGrainCollection.h"
#include "gfGenericBufferReader.h"
#include "gfPanner.h"
#include "gfRecord.h"
#include "gfSyn.h"

#include <nanobind/stl/tuple.h>

// ---------- Type aliases ----------
using SigType = float;
static constexpr size_t kGfBlock = 64;
using GfBuf = Grainflow::gf_buffer<SigType>;
using GfBufReader = Grainflow::gf_buffer_reader<SigType>;
using GfCollection = Grainflow::gf_grain_collection<GfBuf, kGfBlock, SigType>;
using GfIoConfig = Grainflow::gf_io_config<SigType>;

// ---------- GfBufferWrapper ----------
struct GfBufferWrapper {
    std::unique_ptr<GfBuf> buf_;

    GfBufferWrapper(int frames, int channels, int samplerate) {
        buf_ = std::make_unique<GfBuf>(frames, channels, samplerate);
    }

    void set_data(Array2F data) {
        int channels = data.shape(0);
        int frames = data.shape(1);
        buf_->resize(frames, channels, buf_->data_->getSampleRate());
        const float *src = data.data();
        for (int ch = 0; ch < channels; ++ch) {
            for (int f = 0; f < frames; ++f) {
                buf_->data_->samples[ch][f] = src[ch * frames + f];
            }
        }
    }

    NpF2 get_data() {
        int channels = buf_->data_->getNumChannels();
        int frames = buf_->data_->getNumSamplesPerChannel();
        float *out = new float[(size_t)channels * frames];
        for (int ch = 0; ch < channels; ++ch) {
            for (int f = 0; f < frames; ++f) {
                out[ch * frames + f] = buf_->data_->samples[ch][f];
            }
        }
        return make_f2(out, channels, frames);
    }

    int channels() const { return buf_->data_->getNumChannels(); }
    int frames() const { return buf_->data_->getNumSamplesPerChannel(); }
    int samplerate() const { return buf_->data_->getSampleRate(); }
    GfBuf *ptr() { return buf_.get(); }
};

// ---------- GrainCollectionWrapper ----------
struct GrainCollectionWrapper {
    std::unique_ptr<GfCollection> collection_;
    // We own all GfBuf objects we pass to grains (since ~gf_grain_collection leaks)
    std::vector<std::unique_ptr<GfBuf>> owned_buffers_;
    int num_grains_;

    // Persistent io backing storage (reused across process calls)
    std::vector<float> output_storage_;
    std::vector<float *> output_ptrs_;

    GrainCollectionWrapper(int num_grains, int samplerate) : num_grains_(num_grains) {
        auto reader = GfBufReader::get_gf_buffer_reader();
        collection_ = std::make_unique<GfCollection>(reader, num_grains);
        collection_->samplerate = samplerate;
    }

    void set_buffer(GfBufferWrapper &buf, int buf_type, int target) {
        // Create a new GfBuf, copy data from the wrapper
        int ch = buf.channels();
        int fr = buf.frames();
        int sr = buf.samplerate();
        auto new_buf = std::make_unique<GfBuf>(fr, ch, sr);
        for (int c = 0; c < ch; ++c) {
            for (int f = 0; f < fr; ++f) {
                new_buf->data_->samples[c][f] = buf.buf_->data_->samples[c][f];
            }
        }
        GfBuf *raw = new_buf.get();
        owned_buffers_.push_back(std::move(new_buf));
        collection_->set_buffer(static_cast<Grainflow::gf_buffers>(buf_type), raw, target);
    }

    void set_buffer_str(GfBufferWrapper &buf, const std::string &type_str, int target) {
        int ch = buf.channels();
        int fr = buf.frames();
        int sr = buf.samplerate();
        auto new_buf = std::make_unique<GfBuf>(fr, ch, sr);
        for (int c = 0; c < ch; ++c) {
            for (int f = 0; f < fr; ++f) {
                new_buf->data_->samples[c][f] = buf.buf_->data_->samples[c][f];
            }
        }
        GfBuf *raw = new_buf.get();
        owned_buffers_.push_back(std::move(new_buf));
        collection_->set_buffer(type_str, raw, target);
    }

    void param_set(int target, int param_name, int param_type, float value) {
        collection_->param_set(
            target,
            static_cast<Grainflow::gf_param_name>(param_name),
            static_cast<Grainflow::gf_param_type>(param_type),
            value);
    }

    int param_set_str(int target, const std::string &name, float value) {
        return static_cast<int>(collection_->param_set(target, name, value));
    }

    float param_get(int target, int param_name) {
        return collection_->param_get(target, static_cast<Grainflow::gf_param_name>(param_name));
    }

    float param_get_typed(int target, int param_name, int param_type) {
        return collection_->param_get(
            target,
            static_cast<Grainflow::gf_param_name>(param_name),
            static_cast<Grainflow::gf_param_type>(param_type));
    }

    void set_active_grains(int n) { collection_->set_active_grains(n); }
    int active_grains() const { return collection_->active_grains(); }
    int grains() const { return collection_->grains(); }

    void set_auto_overlap(bool v) { collection_->set_auto_overlap(v); }
    bool get_auto_overlap() { return collection_->get_auto_overlap(); }

    void stream_set(int mode, int nstreams) {
        collection_->stream_set(static_cast<Grainflow::gf_stream_set_type>(mode), nstreams);
    }
    void stream_set_manual(int grain, int stream_id) {
        collection_->stream_set(grain, stream_id);
    }
    int stream_get(int grain) { return collection_->stream_get(grain); }
    int streams() const { return collection_->streams(); }

    nb::tuple process(Array2F clock, Array2F traversal, Array2F fm, Array2F am, int samplerate) {
        int block_size = clock.shape(1);
        if (block_size % kGfBlock != 0)
            throw std::invalid_argument("block_size must be a multiple of 64");

        int clock_chans = clock.shape(0);
        int trav_chans = traversal.shape(0);
        int fm_chans = fm.shape(0);
        int am_chans = am.shape(0);
        int ng = collection_->grains();

        // Pitfall: grain_clock[0] == grain_clock[1] check requires >= 2 clock channels
        // Duplicate channel 0 if only 1 clock channel provided
        std::vector<float> clock_dup;
        float *clock_data = const_cast<float *>(clock.data());
        int actual_clock_chans = clock_chans;
        if (clock_chans < 2) {
            actual_clock_chans = 2;
            clock_dup.resize(2 * block_size);
            std::copy(clock_data, clock_data + block_size, clock_dup.data());
            std::copy(clock_data, clock_data + block_size, clock_dup.data() + block_size);
            clock_data = clock_dup.data();
        }

        // Allocate output arrays: 8 outputs x num_grains x block_size
        size_t total_out = (size_t)8 * ng * block_size;
        output_storage_.resize(total_out);
        std::fill(output_storage_.begin(), output_storage_.end(), 0.0f);

        // Build output pointer-of-pointers (8 arrays of ng pointers)
        output_ptrs_.resize(8 * ng);
        for (int out_idx = 0; out_idx < 8; ++out_idx) {
            for (int g = 0; g < ng; ++g) {
                output_ptrs_[out_idx * ng + g] =
                    output_storage_.data() + ((size_t)out_idx * ng + g) * block_size;
            }
        }

        // Build input pointer arrays
        std::vector<float *> clock_ptrs(actual_clock_chans);
        for (int i = 0; i < actual_clock_chans; ++i)
            clock_ptrs[i] = clock_data + (size_t)i * block_size;

        std::vector<float *> trav_ptrs(trav_chans);
        for (int i = 0; i < trav_chans; ++i)
            trav_ptrs[i] = const_cast<float *>(traversal.data()) + (size_t)i * block_size;

        std::vector<float *> fm_ptrs(fm_chans);
        for (int i = 0; i < fm_chans; ++i)
            fm_ptrs[i] = const_cast<float *>(fm.data()) + (size_t)i * block_size;

        std::vector<float *> am_ptrs(am_chans);
        for (int i = 0; i < am_chans; ++i)
            am_ptrs[i] = const_cast<float *>(am.data()) + (size_t)i * block_size;

        GfIoConfig config;
        config.grain_output = &output_ptrs_[0 * ng];
        config.grain_state = &output_ptrs_[1 * ng];
        config.grain_progress = &output_ptrs_[2 * ng];
        config.grain_playhead = &output_ptrs_[3 * ng];
        config.grain_amp = &output_ptrs_[4 * ng];
        config.grain_envelope = &output_ptrs_[5 * ng];
        config.grain_buffer_channel = &output_ptrs_[6 * ng];
        config.grain_stream_channel = &output_ptrs_[7 * ng];

        config.grain_clock = clock_ptrs.data();
        config.traversal_phasor = trav_ptrs.data();
        config.fm = fm_ptrs.data();
        config.am = am_ptrs.data();

        config.grain_clock_chans = actual_clock_chans;
        config.traversal_phasor_chans = trav_chans;
        config.fm_chans = fm_chans;
        config.am_chans = am_chans;

        config.block_size = block_size;
        config.samplerate = samplerate;
        config.livemode = false;

        { nb::gil_scoped_release rel;
          collection_->process(config);
        }

        // Copy outputs to numpy arrays
        auto copy_output = [&](int out_idx) -> NpF2 {
            float *data = new float[(size_t)ng * block_size];
            for (int g = 0; g < ng; ++g) {
                std::copy(
                    output_ptrs_[out_idx * ng + g],
                    output_ptrs_[out_idx * ng + g] + block_size,
                    data + (size_t)g * block_size);
            }
            return make_f2(data, ng, block_size);
        };

        return nb::make_tuple(
            copy_output(0),  // grain_output
            copy_output(1),  // grain_state
            copy_output(2),  // grain_progress
            copy_output(3),  // grain_playhead
            copy_output(4),  // grain_amp
            copy_output(5),  // grain_envelope
            copy_output(6),  // grain_buffer_channel
            copy_output(7)   // grain_stream_channel
        );
    }
};

// ---------- PannerWrapper ----------
struct PannerWrapper {
    int pan_mode_;
    // One instance per mode (only one will be used)
    std::unique_ptr<Grainflow::gf_panner<kGfBlock, Grainflow::gf_pan_mode::bipolar, SigType>> panner_bipolar_;
    std::unique_ptr<Grainflow::gf_panner<kGfBlock, Grainflow::gf_pan_mode::unipolar, SigType>> panner_unipolar_;
    std::unique_ptr<Grainflow::gf_panner<kGfBlock, Grainflow::gf_pan_mode::stereo, SigType>> panner_stereo_;

    PannerWrapper(int in_channels, int out_channels, int pan_mode) : pan_mode_(pan_mode) {
        switch (pan_mode) {
            case 0:
                panner_bipolar_ = std::make_unique<
                    Grainflow::gf_panner<kGfBlock, Grainflow::gf_pan_mode::bipolar, SigType>>(
                    in_channels, out_channels);
                break;
            case 1:
                panner_unipolar_ = std::make_unique<
                    Grainflow::gf_panner<kGfBlock, Grainflow::gf_pan_mode::unipolar, SigType>>(
                    in_channels, out_channels);
                break;
            case 2:
            default:
                panner_stereo_ = std::make_unique<
                    Grainflow::gf_panner<kGfBlock, Grainflow::gf_pan_mode::stereo, SigType>>(
                    in_channels, out_channels);
                pan_mode_ = 2;
                break;
        }
    }

    void set_pan_position(float v) {
        switch (pan_mode_) {
            case 0: panner_bipolar_->pan_position.store(v); break;
            case 1: panner_unipolar_->pan_position.store(v); break;
            case 2: panner_stereo_->pan_position.store(v); break;
        }
    }
    void set_pan_spread(float v) {
        switch (pan_mode_) {
            case 0: panner_bipolar_->pan_spread.store(v); break;
            case 1: panner_unipolar_->pan_spread.store(v); break;
            case 2: panner_stereo_->pan_spread.store(v); break;
        }
    }
    void set_pan_quantization(float v) {
        switch (pan_mode_) {
            case 0: panner_bipolar_->pan_quantization.store(v); break;
            case 1: panner_unipolar_->pan_quantization.store(v); break;
            case 2: panner_stereo_->pan_quantization.store(v); break;
        }
    }

    float get_pan_position() {
        switch (pan_mode_) {
            case 0: return panner_bipolar_->pan_position.load();
            case 1: return panner_unipolar_->pan_position.load();
            case 2: return panner_stereo_->pan_position.load();
        }
        return 0;
    }
    float get_pan_spread() {
        switch (pan_mode_) {
            case 0: return panner_bipolar_->pan_spread.load();
            case 1: return panner_unipolar_->pan_spread.load();
            case 2: return panner_stereo_->pan_spread.load();
        }
        return 0;
    }

    NpF2 process(Array2F grains, Array2F grain_states, int out_channels) {
        int in_channels = grains.shape(0);
        int block_size = grains.shape(1);
        if (block_size % kGfBlock != 0)
            throw std::invalid_argument("block_size must be a multiple of 64");

        // Build grain input pointers
        std::vector<float *> grain_ptrs(in_channels);
        std::vector<float *> state_ptrs(in_channels);
        for (int i = 0; i < in_channels; ++i) {
            grain_ptrs[i] = const_cast<float *>(grains.data()) + (size_t)i * block_size;
            state_ptrs[i] = const_cast<float *>(grain_states.data()) + (size_t)i * block_size;
        }

        // Allocate output
        float *out = new float[(size_t)out_channels * block_size];
        std::fill(out, out + (size_t)out_channels * block_size, 0.0f);
        std::vector<float *> out_ptrs(out_channels);
        for (int i = 0; i < out_channels; ++i)
            out_ptrs[i] = out + (size_t)i * block_size;

        { nb::gil_scoped_release rel;
          switch (pan_mode_) {
              case 0:
                  panner_bipolar_->process(grain_ptrs.data(), state_ptrs.data(), out_ptrs.data(), block_size);
                  break;
              case 1:
                  panner_unipolar_->process(grain_ptrs.data(), state_ptrs.data(), out_ptrs.data(), block_size);
                  break;
              case 2:
                  panner_stereo_->process(grain_ptrs.data(), state_ptrs.data(), out_ptrs.data(), block_size);
                  break;
          }
        }

        nb::capsule del(out, [](void *p) noexcept { delete[] static_cast<float *>(p); });
        size_t shape[2] = {(size_t)out_channels, (size_t)block_size};
        return NpF2(out, 2, shape, del);
    }
};

// ---------- RecorderWrapper ----------
struct RecorderWrapper {
    using GfRec = Grainflow::gfRecorder<GfBuf, kGfBlock, SigType>;
    std::unique_ptr<GfRec> recorder_;
    std::unique_ptr<GfBuf> target_buf_;
    int target_channels_ = 0;
    int target_frames_ = 0;

    RecorderWrapper(int samplerate) {
        auto reader = GfBufReader::get_gf_buffer_reader();
        recorder_ = std::make_unique<GfRec>(reader);
        recorder_->samplerate = samplerate;
    }

    void set_target(int frames, int channels, int samplerate) {
        target_buf_ = std::make_unique<GfBuf>(frames, channels, samplerate);
        target_channels_ = channels;
        target_frames_ = frames;
        recorder_->set_n_filter_channels(channels);
    }

    void set_n_filters(int n) { recorder_->set_n_filters(n); }
    void set_filter_params(int idx, float freq, float q, float mix) {
        recorder_->set_filter_params(idx, freq, q, mix);
    }

    NpF1 process(Array2F input, float time_override) {
        if (!target_buf_)
            throw std::runtime_error("set_target() must be called before process()");

        int channels = input.shape(0);
        int frames = input.shape(1);
        if (frames % kGfBlock != 0)
            throw std::invalid_argument("frames must be a multiple of 64");

        std::vector<float *> input_ptrs(channels);
        for (int i = 0; i < channels; ++i)
            input_ptrs[i] = const_cast<float *>(input.data()) + (size_t)i * frames;

        float *head_out = new float[frames];
        std::fill(head_out, head_out + frames, 0.0f);

        { nb::gil_scoped_release rel;
          recorder_->process(input_ptrs.data(), time_override, target_buf_.get(),
                             frames, channels, head_out);
        }

        return make_f1(head_out, frames);
    }

    NpF2 get_buffer_data() {
        if (!target_buf_)
            throw std::runtime_error("set_target() must be called first");
        int ch = target_buf_->data_->getNumChannels();
        int fr = target_buf_->data_->getNumSamplesPerChannel();
        float *out = new float[(size_t)ch * fr];
        for (int c = 0; c < ch; ++c)
            for (int f = 0; f < fr; ++f)
                out[c * fr + f] = target_buf_->data_->samples[c][f];
        return make_f2(out, ch, fr);
    }

    // Properties
    void set_overdub(float v) { recorder_->overdub = v; }
    float get_overdub() { return recorder_->overdub; }
    void set_freeze(bool v) { recorder_->freeze = v; }
    bool get_freeze() { return recorder_->freeze; }
    void set_sync(bool v) { recorder_->sync = v; }
    bool get_sync() { return recorder_->sync; }
    void set_state(bool v) { recorder_->state = v; }
    bool get_state() { return recorder_->state; }
    void set_rec_range(float lo, float hi) {
        recorder_->recRange[0].store(lo);
        recorder_->recRange[1].store(hi);
    }
    nb::tuple get_rec_range() {
        return nb::make_tuple(recorder_->recRange[0].load(), recorder_->recRange[1].load());
    }
};

// ---------- Phasor binding ----------
using GfPhasor = Grainflow::phasor<SigType, kGfBlock>;

// ---------- bind_grainflow ----------
void bind_grainflow(nb::module_ &m) {
    auto gf = m.def_submodule("grainflow", "GrainflowLib granular synthesis");

    // --- GfBuffer ---
    nb::class_<GfBufferWrapper>(gf, "GfBuffer")
        .def(nb::init<int, int, int>(),
             nb::arg("frames"), nb::arg("channels"), nb::arg("samplerate"))
        .def("set_data", &GfBufferWrapper::set_data, nb::arg("data"),
             "Copy [channels, frames] float32 numpy array into buffer")
        .def("get_data", &GfBufferWrapper::get_data,
             "Return buffer contents as [channels, frames] float32 numpy array")
        .def_prop_ro("channels", &GfBufferWrapper::channels)
        .def_prop_ro("frames", &GfBufferWrapper::frames)
        .def_prop_ro("samplerate", &GfBufferWrapper::samplerate);

    // --- GrainCollection ---
    nb::class_<GrainCollectionWrapper>(gf, "GrainCollection")
        .def(nb::init<int, int>(), nb::arg("num_grains"), nb::arg("samplerate"))
        .def("set_buffer", &GrainCollectionWrapper::set_buffer,
             nb::arg("buf"), nb::arg("buf_type"), nb::arg("target") = 0,
             "Set buffer for grain collection (buf_type: BUF_* constant, target: 0=all)")
        .def("set_buffer_str", &GrainCollectionWrapper::set_buffer_str,
             nb::arg("buf"), nb::arg("type_str"), nb::arg("target") = 0,
             "Set buffer using string type name")
        .def("param_set", &GrainCollectionWrapper::param_set,
             nb::arg("target"), nb::arg("param_name"), nb::arg("param_type"), nb::arg("value"),
             "Set parameter by enum constants")
        .def("param_set_str", &GrainCollectionWrapper::param_set_str,
             nb::arg("target"), nb::arg("name"), nb::arg("value"),
             "Set parameter by reflection string (e.g. 'delay', 'rateRandom')")
        .def("param_get", &GrainCollectionWrapper::param_get,
             nb::arg("target"), nb::arg("param_name"),
             "Get parameter value")
        .def("param_get_typed", &GrainCollectionWrapper::param_get_typed,
             nb::arg("target"), nb::arg("param_name"), nb::arg("param_type"),
             "Get parameter value by type")
        .def("set_active_grains", &GrainCollectionWrapper::set_active_grains, nb::arg("n"))
        .def_prop_ro("active_grains", &GrainCollectionWrapper::active_grains)
        .def_prop_ro("grains", &GrainCollectionWrapper::grains)
        .def("set_auto_overlap", &GrainCollectionWrapper::set_auto_overlap, nb::arg("v"))
        .def("get_auto_overlap", &GrainCollectionWrapper::get_auto_overlap)
        .def("stream_set", &GrainCollectionWrapper::stream_set,
             nb::arg("mode"), nb::arg("nstreams"),
             "Set stream assignment mode")
        .def("stream_set_manual", &GrainCollectionWrapper::stream_set_manual,
             nb::arg("grain"), nb::arg("stream_id"),
             "Manually assign grain to stream")
        .def("stream_get", &GrainCollectionWrapper::stream_get, nb::arg("grain"))
        .def_prop_ro("streams", &GrainCollectionWrapper::streams)
        .def("process", &GrainCollectionWrapper::process,
             nb::arg("clock"), nb::arg("traversal"), nb::arg("fm"), nb::arg("am"),
             nb::arg("samplerate"),
             "Process grains. Returns (output, state, progress, playhead, amp, envelope, "
             "buffer_channel, stream_channel) each [num_grains, block_size].");

    // --- Panner ---
    nb::class_<PannerWrapper>(gf, "Panner")
        .def(nb::init<int, int, int>(),
             nb::arg("in_channels"), nb::arg("out_channels") = 2, nb::arg("pan_mode") = 2,
             "Create panner (pan_mode: 0=bipolar, 1=unipolar, 2=stereo)")
        .def("set_pan_position", &PannerWrapper::set_pan_position, nb::arg("v"))
        .def("set_pan_spread", &PannerWrapper::set_pan_spread, nb::arg("v"))
        .def("set_pan_quantization", &PannerWrapper::set_pan_quantization, nb::arg("v"))
        .def_prop_ro("pan_position", &PannerWrapper::get_pan_position)
        .def_prop_ro("pan_spread", &PannerWrapper::get_pan_spread)
        .def("process", &PannerWrapper::process,
             nb::arg("grains"), nb::arg("grain_states"), nb::arg("out_channels") = 2,
             "Pan grains to output channels. Returns [out_channels, block_size].");

    // --- Recorder ---
    nb::class_<RecorderWrapper>(gf, "Recorder")
        .def(nb::init<int>(), nb::arg("samplerate"))
        .def("set_target", &RecorderWrapper::set_target,
             nb::arg("frames"), nb::arg("channels"), nb::arg("samplerate"),
             "Allocate recording target buffer")
        .def("set_n_filters", &RecorderWrapper::set_n_filters, nb::arg("n"))
        .def("set_filter_params", &RecorderWrapper::set_filter_params,
             nb::arg("idx"), nb::arg("freq"), nb::arg("q"), nb::arg("mix"))
        .def("process", &RecorderWrapper::process,
             nb::arg("input"), nb::arg("time_override") = 0.0f,
             "Record input into target buffer. Returns head position array.")
        .def("get_buffer_data", &RecorderWrapper::get_buffer_data,
             "Read recorded audio back as [channels, frames] numpy array")
        .def_prop_rw("overdub", &RecorderWrapper::get_overdub, &RecorderWrapper::set_overdub)
        .def_prop_rw("freeze", &RecorderWrapper::get_freeze, &RecorderWrapper::set_freeze)
        .def_prop_rw("sync", &RecorderWrapper::get_sync, &RecorderWrapper::set_sync)
        .def_prop_rw("state", &RecorderWrapper::get_state, &RecorderWrapper::set_state)
        .def("set_rec_range", &RecorderWrapper::set_rec_range,
             nb::arg("lo"), nb::arg("hi"))
        .def("get_rec_range", &RecorderWrapper::get_rec_range);

    // --- Phasor ---
    nb::class_<GfPhasor>(gf, "Phasor")
        .def(nb::init<SigType, int>(), nb::arg("rate"), nb::arg("samplerate"))
        .def("set_rate", [](GfPhasor &self, float rate, int samplerate) {
            self.set_rate(rate, samplerate);
        }, nb::arg("rate"), nb::arg("samplerate"))
        .def("perform", [](GfPhasor &self, int frames) -> NpF1 {
            if (frames % kGfBlock != 0)
                throw std::invalid_argument("frames must be a multiple of 64");
            float *buf = new float[frames];
            std::fill(buf, buf + frames, 0.0f);
            { nb::gil_scoped_release rel;
              self.perform(buf, frames);
            }
            return make_f1(buf, frames);
        }, nb::arg("frames") = 64,
        "Generate phasor ramp [0, 1). frames must be multiple of 64.");

    // --- Enum constants ---
    // gf_param_name
    gf.attr("PARAM_ERR") = 0;
    gf.attr("PARAM_DELAY") = (int)Grainflow::gf_param_name::delay;
    gf.attr("PARAM_RATE") = (int)Grainflow::gf_param_name::rate;
    gf.attr("PARAM_GLISSON") = (int)Grainflow::gf_param_name::glisson;
    gf.attr("PARAM_GLISSON_ROWS") = (int)Grainflow::gf_param_name::glisson_rows;
    gf.attr("PARAM_GLISSON_POSITION") = (int)Grainflow::gf_param_name::glisson_position;
    gf.attr("PARAM_WINDOW") = (int)Grainflow::gf_param_name::window;
    gf.attr("PARAM_AMPLITUDE") = (int)Grainflow::gf_param_name::amplitude;
    gf.attr("PARAM_SPACE") = (int)Grainflow::gf_param_name::space;
    gf.attr("PARAM_ENVELOPE_POSITION") = (int)Grainflow::gf_param_name::envelope_position;
    gf.attr("PARAM_N_ENVELOPES") = (int)Grainflow::gf_param_name::n_envelopes;
    gf.attr("PARAM_DIRECTION") = (int)Grainflow::gf_param_name::direction;
    gf.attr("PARAM_START_POINT") = (int)Grainflow::gf_param_name::start_point;
    gf.attr("PARAM_STOP_POINT") = (int)Grainflow::gf_param_name::stop_point;
    gf.attr("PARAM_RATE_QUANTIZE_SEMI") = (int)Grainflow::gf_param_name::rate_quantize_semi;
    gf.attr("PARAM_LOOP_MODE") = (int)Grainflow::gf_param_name::loop_mode;
    gf.attr("PARAM_CHANNEL") = (int)Grainflow::gf_param_name::channel;
    gf.attr("PARAM_DENSITY") = (int)Grainflow::gf_param_name::density;
    gf.attr("PARAM_VIBRATO_RATE") = (int)Grainflow::gf_param_name::vibrato_rate;
    gf.attr("PARAM_VIBRATO_DEPTH") = (int)Grainflow::gf_param_name::vibrato_depth;
    gf.attr("PARAM_TRANSPOSE") = (int)Grainflow::gf_param_name::transpose;
    gf.attr("PARAM_GLISSON_ST") = (int)Grainflow::gf_param_name::glisson_st;
    gf.attr("PARAM_STREAM") = (int)Grainflow::gf_param_name::stream;

    // gf_param_type
    gf.attr("PTYPE_ERR") = 0;
    gf.attr("PTYPE_BASE") = (int)Grainflow::gf_param_type::base;
    gf.attr("PTYPE_RANDOM") = (int)Grainflow::gf_param_type::random;
    gf.attr("PTYPE_OFFSET") = (int)Grainflow::gf_param_type::offset;
    gf.attr("PTYPE_MODE") = (int)Grainflow::gf_param_type::mode;
    gf.attr("PTYPE_VALUE") = (int)Grainflow::gf_param_type::value;

    // gf_stream_set_type
    gf.attr("STREAM_AUTOMATIC") = (int)Grainflow::gf_stream_set_type::automatic_streams;
    gf.attr("STREAM_PER") = (int)Grainflow::gf_stream_set_type::per_streams;
    gf.attr("STREAM_RANDOM") = (int)Grainflow::gf_stream_set_type::random_streams;
    gf.attr("STREAM_MANUAL") = (int)Grainflow::gf_stream_set_type::manual_streams;

    // gf_buffers
    gf.attr("BUF_BUFFER") = (int)Grainflow::gf_buffers::buffer;
    gf.attr("BUF_ENVELOPE") = (int)Grainflow::gf_buffers::envelope;
    gf.attr("BUF_RATE") = (int)Grainflow::gf_buffers::rate_buffer;
    gf.attr("BUF_DELAY") = (int)Grainflow::gf_buffers::delay_buffer;
    gf.attr("BUF_WINDOW") = (int)Grainflow::gf_buffers::window_buffer;
    gf.attr("BUF_GLISSON") = (int)Grainflow::gf_buffers::glisson_buffer;

    // gf_buffer_mode
    gf.attr("BUFMODE_NORMAL") = (int)Grainflow::gf_buffer_mode::normal;
    gf.attr("BUFMODE_SEQUENCE") = (int)Grainflow::gf_buffer_mode::buffer_sequence;
    gf.attr("BUFMODE_RANDOM") = (int)Grainflow::gf_buffer_mode::buffer_random;

    // gf_pan_mode
    gf.attr("PAN_BIPOLAR") = (int)Grainflow::gf_pan_mode::bipolar;
    gf.attr("PAN_UNIPOLAR") = (int)Grainflow::gf_pan_mode::unipolar;
    gf.attr("PAN_STEREO") = (int)Grainflow::gf_pan_mode::stereo;
}

#include "_core_common.h"

#include <filesystem>
#include <fstream>

// choc FLAC codec (header-only)
#include "choc/audio/choc_AudioFileFormat_FLAC.h"

void bind_choc(nb::module_ &m) {
    auto choc_m = m.def_submodule("choc", "CHOC audio utilities (FLAC codec)");

    choc_m.def("read_flac", [](const std::string &path) -> nb::tuple {
        choc::audio::FLACAudioFileFormat<true> format;
        std::unique_ptr<choc::audio::AudioFileReader> reader;
        { nb::gil_scoped_release rel;
          reader = format.createReader(std::filesystem::path(path));
        }
        if (!reader)
            throw std::runtime_error("Failed to open FLAC file: " + path);

        auto &props = reader->getProperties();
        uint32_t numChannels = props.numChannels;
        uint32_t numFrames = static_cast<uint32_t>(props.numFrames);
        double sampleRate = props.sampleRate;

        if (numFrames == 0) {
            float *out = new float[1]; // dummy allocation for capsule
            return nb::make_tuple(make_f2(out, numChannels, 0), sampleRate);
        }

        float *out = new float[(size_t)numChannels * numFrames];
        { nb::gil_scoped_release rel;
          auto buffer = reader->readEntireStream<float>();
          for (uint32_t ch = 0; ch < numChannels; ++ch) {
              auto iter = buffer.getIterator(ch);
              float *dst = out + (size_t)ch * numFrames;
              for (uint32_t f = 0; f < numFrames; ++f) {
                  dst[f] = *iter;
                  ++iter;
              }
          }
        }

        return nb::make_tuple(make_f2(out, numChannels, numFrames), sampleRate);
    }, nb::arg("path"),
    "Read a FLAC file. Returns (data, sample_rate) where data is [channels, frames] float32.");

    choc_m.def("write_flac", [](const std::string &path, Array2F data,
                                 double sample_rate, int bit_depth) {
        uint32_t numChannels = data.shape(0);
        uint32_t numFrames = data.shape(1);

        choc::audio::AudioFileProperties props;
        props.sampleRate = sample_rate;
        props.numChannels = numChannels;
        props.numFrames = numFrames;

        switch (bit_depth) {
            case 16: props.bitDepth = choc::audio::BitDepth::int16; break;
            case 24: props.bitDepth = choc::audio::BitDepth::int24; break;
            default:
                throw std::invalid_argument(
                    "Unsupported bit_depth for FLAC: " + std::to_string(bit_depth) +
                    " (use 16 or 24)");
        }

        // Build channel pointer array from 2D contiguous numpy (rows = channels)
        std::vector<float *> channelPtrs(numChannels);
        for (uint32_t ch = 0; ch < numChannels; ++ch)
            channelPtrs[ch] = data.data() + (size_t)ch * numFrames;

        { nb::gil_scoped_release rel;
          choc::audio::FLACAudioFileFormat<true> format;
          auto writer = format.createWriter(std::filesystem::path(path), props);
          if (!writer)
              throw std::runtime_error("Failed to create FLAC writer for: " + path);

          if (numFrames == 0) {
              writer->flush();
              return;
          }

          auto view = choc::buffer::createChannelArrayView(
              channelPtrs.data(), numChannels, numFrames);

          if (!writer->appendFrames(view))
              throw std::runtime_error("Failed to write FLAC frames to: " + path);

          if (!writer->flush())
              throw std::runtime_error("Failed to flush FLAC file: " + path);
        }
    }, nb::arg("path"), nb::arg("data"), nb::arg("sample_rate"),
       nb::arg("bit_depth") = 16,
    "Write audio data to a FLAC file. data is [channels, frames] float32.");
}

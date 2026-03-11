#pragma once

#include <atomic>
#include "gfIBufferReader.h"
#include "gfParam.h"
#include <AudioFile.h>

namespace Grainflow{
    template<typename SigType>
    class gf_buffer{
        public:
        std::atomic<bool> latch_;
        std::unique_ptr<AudioFile<SigType>> data_;

        private:


        void clear(){
            for (auto& c : data_->samples){
                std::fill(c.begin(), c.end(), 0);
            }
        }

        void replace(std::string& file_path){
            data_->load(file_path);
        }

        public:
		gf_buffer(){
			data_ = std::make_unique<AudioFile<SigType>>();
		}
        gf_buffer(int frames, int channels, int samplerate){
            data_ = std::make_unique<AudioFile<SigType>>();
            data_->setAudioBufferSize(channels, frames);
            data_->setSampleRate(samplerate);
        }
        gf_buffer(std::string& file_path){
            data_ = std::make_unique<AudioFile<SigType>>();
			data_->load(file_path);
        }
		void resize(int frames, int channels, int samplerate = 0){
            data_->setAudioBufferSize(channels, frames);
            data_->setSampleRate(samplerate);
        }

    };
    template<typename SigType>
    struct buffer_lock{ 
        private:
        SigType dummy_;
        gf_buffer<SigType>* buffer_ {nullptr};
        bool valid_ {false};
        public:
        buffer_lock(gf_buffer<SigType>* buffer){
            buffer_ = buffer;
            if (buffer_->latch_.load()){
                return;
            }
            buffer_->latch_.store(true);
            valid_ = true;
        }
        ~buffer_lock(){
			 buffer_->latch_.store(false);
		};
        bool valid(){
            return valid_;
        }

        bool clear(){
            buffer_->clear();
        }

        std::vector<std::vector<SigType>>& get_samples(){
             return buffer_->data_->samples;
        }

        void get_info(gf_buffer_info* info){
            info->buffer_frames = frame_count();
            info->one_over_buffer_frames= 1/info->buffer_frames;
            info->n_channels = channel_count();
            info->samplerate = samplerate();
            info->one_over_samplerate = 1/info->samplerate;
        }

        void replace(std::string audio_file_path){
            buffer_->replace(audio_file_path);
        }

        void resize(int frames, int channels, int samplerate = 0){
            buffer_->resize(frames, channels, samplerate);
        }


        int frame_count(){
            return buffer_->data_->getNumSamplesPerChannel();
        }

        int channel_count(){
            return buffer_->data_->getNumChannels();
        }

        int samplerate(){
            return buffer_->data_->getSampleRate();
        }

        inline SigType& lookup(int frame, int channel){
            return buffer_->data_->samples[channel][frame];
        }

    };
    template<typename SigType>
    class gf_buffer_reader{
        public:
		static bool update_buffer_info(gf_buffer<SigType>* buffer, const gf_io_config<SigType>& io_config,
		                               gf_buffer_info* buffer_info)
		{
			if (buffer == nullptr){
                return false;
            }
			buffer_lock<SigType> buffer_lock(buffer);
            if (!buffer_lock.valid()){
                return false;
            }

            buffer_lock.get_info(buffer_info);
            buffer_info->sample_rate_adjustment = buffer_info->samplerate / io_config.samplerate;
            
			return true;
		}

		static bool sample_param_buffer(gf_buffer<SigType>* buffer, gf_param* param, const int grain_id)
		{
			
			if (param->mode == gf_buffer_mode::normal || buffer == nullptr)
			{
				return false;
			}
			buffer_lock<SigType> param_buf(buffer);
			if (!param_buf.valid())
				return false;
			size_t frame = 0;
			size_t frames = param_buf.frame_count();
			if (frames <= 0) return false;
			if (param->mode == gf_buffer_mode::buffer_sequence)
			{
				frame = grain_id % frames;
			}
			else if (param->mode == gf_buffer_mode::buffer_random)
			{
				frame = (rand() % frames);
			}
			param->value = param_buf.lookup(frame, 0) + param->random * (rand() % 10000) * 0.0001 + param->offset *
				grain_id;
			return true;
		}

		static void sample_buffer(gf_buffer<SigType>* buffer, const int channel, SigType* __restrict samples,
		                          const SigType* positions, 
		                          const int size, const float lower_bound, const float upper_bound)
		{
			buffer_lock<SigType> sample_lock(buffer);
            if (!sample_lock.valid()){
                return;
            }
	
			
			const int max_frame = static_cast<int>(sample_lock.frame_count())-1;
			const int lower_frame = max_frame * lower_bound;
			const int upper_frame = max_frame * upper_bound;
			if (upper_frame == lower_frame) return;
			int channels = static_cast<int>(sample_lock.channel_count());
			channels = std::max(channels, 1);
			const auto chan = channel % channels;
			for (int i = 0; i < size; i++)
			{
				const auto position = positions[i];
				const auto first_frame = static_cast<int>(position);
				const auto tween = position - first_frame;
				const bool frame_overflow = first_frame >= upper_frame;
				const int second_frame = (first_frame + 1) * !frame_overflow + lower_frame * frame_overflow;
				samples[i] = sample_lock.lookup(first_frame, chan) * (1 - tween) + sample_lock.lookup(second_frame, chan)* tween;
			}

		}

		static void read_buffer(gf_buffer<SigType>* buffer, int channel, SigType* __restrict samples, int start_sample,
			const int size)
		{
			buffer_lock<SigType> sample_lock(buffer);
            if (!sample_lock.valid()){
                return;
            }
            auto& buffer_samples = sample_lock.get_samples();

            const int frames = static_cast<int>(sample_lock.frame_count());
			int channels = static_cast<int>(sample_lock.channel_count());
			if (channels <= 0) return;
			auto write_channel = channel % channels;
			auto is_segmented = (start_sample + size) >= frames;

			if (!is_segmented)
			{
				for (int i = 0; i < size; i++)
				{
					samples[i] = buffer_samples[channel][(start_sample + i)];
				}
			}
			else
			{
				for (int i = 0; i < size; i++)
				{
					samples[i] = buffer_samples[channel][(((start_sample + i) % frames))];
				}
			}
			
		}

		static void write_buffer(gf_buffer<SigType>* buffer, const int channel, const SigType* samples,
			const int start_position, const int size)
		{
			buffer_lock<SigType> sample_lock(buffer);
            if (!sample_lock.valid()){
                return;
            }

            auto& buffer_samples = sample_lock.get_samples();
            const int frames = static_cast<int>(sample_lock.frame_count());
			int channels = static_cast<int>(sample_lock.channel_count());
			if (channels <= 0 || frames <= 0) return;
			auto write_channel = channel % channels;
			auto is_segmented = (start_position + size) >= frames;

			if (!is_segmented)
			{
				for (int i = 0; i < size; i++)
				{
					buffer_samples[channel][(start_position + i)] = samples[i];
				}
				return;
			}
			auto first_chunk = (start_position + size) - frames;
			for (int i = 0; i < size; i++)
			{
				buffer_samples[channel][(((start_position + i) % frames))] = samples[i];
			}            
		}


		static void sample_envelope(gf_buffer<SigType>* buffer, const bool use_default, const int n_envelopes,
		                            const float env2d_pos, SigType* __restrict samples,
		                            const SigType* __restrict grain_clock, const int size)
		{
            if (use_default)
			{
				for (int i = 0; i < size; i++)
				{
					const auto frame = static_cast<int>(
						std::fmax(((std::fmin((grain_clock[i] * 1024.0), 1023.0))), 0.0));
					samples[i] = Grainflow::gf_envelopes::hanning_envelope[frame];
				}
				return;
			}

			buffer_lock<SigType> sample_lock(buffer);
            if (!sample_lock.valid()){
                return;
            }
            auto& buffer_samples = sample_lock.get_samples();
			int frames = sample_lock.frame_count();

            if (n_envelopes <= 1)
			{
				for (int i = 0; i < size; i++)
				{
					if (!sample_lock.valid()) return;
					const auto frame = static_cast<int>(grain_clock[i] * frames);
					samples[i] = buffer_samples[0][frame];
				}
				return;
			}
			for (int i = 0; i < size; i++)
			{
				if (!sample_lock.valid()) return;
				const int size_per_envelope = frames / n_envelopes;
				const int env1 = static_cast<int>(env2d_pos * static_cast<float>(n_envelopes));
				const int env2 = env1 + 1;
				const float fade = env2d_pos * static_cast<float>(n_envelopes) - static_cast<float>(env1);
				const auto frame = static_cast<int>((grain_clock[i] * size_per_envelope));
				samples[i] = buffer_samples[0][(env1 * size_per_envelope + frame) % frames] * (1 - fade) + buffer_samples[0][(
					env2 *
					size_per_envelope + frame) % frames] * fade;
			}
			

		}
         
		static gf_i_buffer_reader<gf_buffer<SigType>, SigType> get_gf_buffer_reader()
		{
			gf_i_buffer_reader<gf_buffer<SigType>, SigType> _bufferReader;
			_bufferReader.sample_buffer = gf_buffer_reader<SigType>::sample_buffer;
			_bufferReader.sample_envelope = gf_buffer_reader<SigType>::sample_envelope;
			_bufferReader.update_buffer_info = gf_buffer_reader<SigType>::update_buffer_info;
			_bufferReader.sample_param_buffer = gf_buffer_reader<SigType>::sample_param_buffer;
			_bufferReader.write_buffer = gf_buffer_reader<SigType>::write_buffer;
			_bufferReader.read_buffer = gf_buffer_reader<SigType>::read_buffer;
			return _bufferReader;
		}
        
    };



}
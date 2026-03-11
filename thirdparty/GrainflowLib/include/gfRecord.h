#include "gfIBufferReader.h"
#include <atomic>
#include <algorithm>
#include "gfFilters.h"

namespace Grainflow
{
	template <typename T, size_t INTERNALBLOCK, typename SigType = double>
	class gfRecorder
	{
	private:
		struct filter_data
		{
		public:
			std::vector<biquad<SigType>> sample_filter;
			std::vector<biquad<SigType>> od_filter;
			biquad_params<SigType> filter_params;
			float freq{0};
			float q{0};
			float overdub;
		};

		gf_io_config<SigType> config_{};
		gf_buffer_info buffer_info_{};
		size_t write_position_ = 0;
		std::array<std::array<SigType, INTERNALBLOCK>, 4> temp_;
		int channels_ = 1;
		gf_i_buffer_reader<T, SigType> buffer_reader_{};

		std::vector<filter_data> filter_data_;
		int _n_filters{0};

	public:
		std::array<std::atomic<float>, 2> recRange{0.0, 1.0};
		SigType write_position_norm = 0.0;
		SigType write_position_ms = 0.0;
		int write_position_samps = 0;
		bool sync = false;
		bool freeze = false;
		bool state = false;
		float overdub = 0;
		size_t samplerate = 48000;

	private:
		void write_simple(SigType** __restrict input, T* __restrict buffer, const int block, const int channels)
		{
			for (int c = 0; c < channels; ++c)
			{
				const auto samps = input[c] + block * INTERNALBLOCK;
				if (overdub <= 0)
				{
					buffer_reader_.write_buffer(buffer, c, samps, write_position_, INTERNALBLOCK);
					continue;
				}

				buffer_reader_.read_buffer(buffer, c, temp_[0].data(), write_position_, INTERNALBLOCK);
				const auto old_mix = overdub;
				const auto new_mix = 1 - overdub;
				std::transform(samps, samps + INTERNALBLOCK, temp_[0].begin(), temp_[0].begin(),
				               [old_mix, new_mix](auto a, auto b)
				               {
					               return a * new_mix + b * old_mix;
				               });
				buffer_reader_.write_buffer(buffer, c, temp_[0].data(), write_position_, INTERNALBLOCK);
			}
		}

		void write_with_filters(SigType** __restrict input, T* __restrict buffer, const int block, const int channels)
		{
			for (int c = 0; c < channels; ++c)
			{
				const auto input_samps = input[c] + block * INTERNALBLOCK;

				buffer_reader_.read_buffer(buffer, c, temp_[0].data(), write_position_, INTERNALBLOCK);


				auto& sample_data = temp_[0];
				auto& filter_samples = temp_[1];
				auto& filter_output = temp_[2];
				auto& residual = temp_[3];

				if (filter_data_.size() < 0 || filter_data_[0].od_filter.size() < channels)
				{
					return;
				}
				std::fill(filter_output.begin(), filter_output.end(), 0); //Clear filter output accumulator

				//Perform filters on samples in buffer
				std::copy(sample_data.begin(), sample_data.end(), residual.begin());
				//Copy buffer results into the residual array
				for (auto& filter : filter_data_)
				{
					const auto old_mix = filter.overdub;

					filter.od_filter[c].perform(residual.data(), INTERNALBLOCK, filter.filter_params,
					                            filter_samples.data()); //Apply the filter to the residual

					std::transform(filter_samples.begin(), filter_samples.end(), filter_output.begin(),
					               filter_output.begin(), [old_mix](auto a, auto b)
					               {
						               return a * old_mix + b;
					               }); //Add the filtered result with a mix to the filter output 

					std::transform(residual.begin(), residual.end(), filter_samples.begin(),
					               residual.begin(), [](auto a, auto b)
					               {
						               return a - b;
					               });
				}

				//Mix in the old remainder according to overdub 
				const auto old_mix = overdub;

				std::transform(residual.begin(), residual.end(),
				               sample_data.begin(), [old_mix](auto a)
				               {
					               return a * old_mix;
				               });

				std::copy_n(input_samps, INTERNALBLOCK, residual.begin());
				//Perform filters on input samples 
				for (auto& filter : filter_data_)
				{
					const auto new_mix = 1 - filter.overdub;

					filter.sample_filter[c].
						perform(residual.data(), INTERNALBLOCK, filter.filter_params, filter_samples.data());

					std::transform(filter_samples.begin(), filter_samples.end(), filter_output.begin(),
					               filter_output.begin(), [new_mix](auto a, auto b)
					               {
						               return a * new_mix + b;
					               });

					std::transform(residual.begin(), residual.end(), filter_samples.begin(),
					               residual.begin(), [](auto a, auto b)
					               {
						               return a - b;
					               });
				}

				//Mix in the new remainder according to overdub 
				const auto new_mix = 1 - overdub;

				std::transform(residual.begin(), residual.end(),
				               residual.begin(), [new_mix](auto a)
				               {
					               return (a) * new_mix;
				               });

				//Mix everything together
				std::transform(residual.begin(), residual.end(), sample_data.begin(),
				               sample_data.begin(), [](auto a, auto b) { return a + b; });

				std::transform(filter_output.begin(), filter_output.end(), sample_data.begin(),
				               sample_data.begin(), [](auto a, auto b) { return a + b; });

				buffer_reader_.write_buffer(buffer, c, sample_data.data(), write_position_, INTERNALBLOCK);
			}
		}

	public:
		gfRecorder(gf_i_buffer_reader<T, SigType> buffer_reader)
		{
			buffer_reader_ = buffer_reader;
			config_.livemode = true;
		}

		~gfRecorder()
		{
		}

		void set_n_filters(const int number)
		{
			if (number < 1)
			{
				_n_filters = 0;
				return;
			}
			_n_filters = number;
			filter_data_.resize(number);
			set_n_filter_channels(channels_);
		}

		void set_n_filter_channels(const int number)
		{
			channels_ = number;
			for (auto& filter : filter_data_)
			{
				filter.od_filter.resize(number);
				filter.sample_filter.resize(number);
			}
		}

		void set_filter_params(const int& idx, const float& freq, const float& q, const float& mix)
		{
			if (idx >= filter_data_.size()) return;
			filter_data_[idx].freq = freq;
			filter_data_[idx].q = q;
			filter_data_[idx].overdub = std::min(1.0f, std::max(0.0f, mix));
			biquad_params<SigType>::bandpass(filter_data_[idx].filter_params, freq, q, samplerate);
		}

		void pre_process_filters()
		{
			//Currently we just sort by decending q
			std::sort(filter_data_.begin(), filter_data_.end(), [](auto& a, auto& b)
			{
				return a.q > b.q;
			});
		}

		void clear(T* buffer)
		{
			for (auto& filter : filter_data_)
			{
				filter.sample_filter.clear();
				filter.od_filter.clear();
			}
			buffer_reader_.clear_buffer(buffer);
		}

		void get_position(SigType& position_samps, SigType& position_norm, SigType& position_ms)
		{
			position_samps = write_position_samps;
			position_norm = write_position_norm;
			position_ms = write_position_ms;
		}

		void process(SigType** __restrict input, const SigType time_override, T* buffer, int frames, int channels,
		             SigType* __restrict recorded_head_out)
		{
			const auto blocks = frames / INTERNALBLOCK;
			float recBase = recRange[0].load();
			float rangeMax = recRange[1].load();
			float recRangeSize = std::abs(rangeMax - recBase);
			float recRangeSign = rangeMax < recBase ? -1 : 1;


			if (!state)
			{
				for (int b = 0; b < blocks; ++b)
				{
					if (buffer_info_.buffer_frames == 0)
					{
						for (int i = 0; i < INTERNALBLOCK; ++i)
						{
							recorded_head_out[b * INTERNALBLOCK + i] = 0;
						}
						return;
					}
					if (!freeze)
					{
						for (int i = 0; i < INTERNALBLOCK; ++i)
						{
							recorded_head_out[b * INTERNALBLOCK + i] = static_cast<float>(write_position_) /
								buffer_info_.
								buffer_frames;
						}
					}
					else
					{
						for (int i = 0; i < INTERNALBLOCK; ++i)
						{
							recorded_head_out[b * INTERNALBLOCK + i] = write_position_norm;
						}
					}
				}
				return;
			}

			auto success = buffer_reader_.update_buffer_info(buffer, config_, &buffer_info_);
			if (!success || buffer_info_.buffer_frames <= 0)
			{
				write_position_ = 0;
				write_position_samps = 0;
				write_position_norm = 0;
				write_position_ms = 0;
				return;
			}

			if (sync)
			{
				write_position_ = buffer_info_.buffer_frames * (gf_utils::mod<SigType>(time_override, 1));
			}
			if (buffer_info_.buffer_frames == 0) return;
			int sampleRange = buffer_info_.buffer_frames * recRangeSize;
			int sampleBase = buffer_info_.buffer_frames * recBase;
			int increment = INTERNALBLOCK * recRangeSign;
			for (int b = 0; b < blocks; ++b)
			{
				if (_n_filters < 1)
				{
					write_simple(input, buffer, b, channels);
				}
				else
				{
					write_with_filters(input, buffer, b, channels);
				}

				if (!freeze)
				{
					for (int i = 0; i < INTERNALBLOCK; ++i)
					{
						recorded_head_out[b * INTERNALBLOCK + i] = static_cast<float>((write_position_ + i) %
							buffer_info_.buffer_frames) / buffer_info_.buffer_frames;
					}
					write_position_ = ((write_position_ + increment) + sampleRange) % sampleRange + sampleBase;
					write_position_samps = write_position_;
					write_position_norm = static_cast<float>((write_position_samps + INTERNALBLOCK) % buffer_info_.
						buffer_frames) / buffer_info_.buffer_frames;
					write_position_ms = (write_position_samps + INTERNALBLOCK) * 1000 / samplerate;
				}
				else
				{
					for (int i = 0; i < INTERNALBLOCK; ++i)
					{
						recorded_head_out[b * INTERNALBLOCK + i] = write_position_norm;
					}

					write_position_ = ((write_position_ + increment) + sampleRange) % sampleRange + sampleBase;
				}
			}
		}
	};
}

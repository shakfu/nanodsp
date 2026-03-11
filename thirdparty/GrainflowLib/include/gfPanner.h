#pragma once
#include <vector>
#include <random>
#include <algorithm>
#include <atomic>
#include <numeric>
#include <thread>
#include<chrono>
#include "gfEnvelopes.h"
#include "gfUtils.h"

namespace Grainflow
{
	enum class gf_pan_mode
	{
		bipolar = 0,
		unipolar = 1,
		stereo = 2
	};

	template <size_t InternalBlock, gf_pan_mode pan_mode, typename sigtype = double>
	class gf_panner
	{
	private:
		int channels_ = 0;
		std::atomic<int> output_channels_ = 0;
		float positions_[InternalBlock] = {0};
		std::vector<sigtype> last_samples_ = {0};
		std::vector<float> last_position_ = {0};
		std::atomic<bool> latch_{false};


		static void determine_pan_position(const int idx, const size_t block_size, const int channels,
		                                   const float pan_center,
		                                   const float pan_spread, const float quantization,
		                                   std::vector<float>& last_positions, const int channel,
		                                   const int output_channels,
		                                   float* __restrict out_pan_positions)
		{
			const float last_position = last_positions[channel];
			float position = 0;
			switch (pan_mode)
			{
			case gf_pan_mode::bipolar:
				position = gf_utils::deviate(pan_center, pan_spread);
				break;
			case gf_pan_mode::unipolar:
				position = gf_utils::random_range(pan_center, pan_center + pan_spread);
				break;
			case gf_pan_mode::stereo:
				position = std::clamp(gf_utils::deviate(pan_center, pan_spread * 0.5f), 0.0f, 1.0f);
			}
			const sigtype n_outputs = output_channels;
			position = static_cast<float>(std::max<float>(
				gf_utils::mod<float>(position + n_outputs * 5, n_outputs),
				0.0));
			position = gf_utils::mod<float>(static_cast<float>(gf_utils::round(position, quantization)), output_channels);

			for (int i = 0; i < block_size; i++)
			{
				const auto new_pos = static_cast<int>(i > idx);
				out_pan_positions[i] = position * new_pos + last_position * (1.0f - new_pos);
			}

			last_positions[channel] = out_pan_positions[block_size - 1];
		}

		static void perform_pan(const sigtype* __restrict input_stream, const float* __restrict positions,
		                        const size_t block_size, sigtype** __restrict output_stream, const size_t block_offset,
		                        const int output_channels)
		{
			for (int j = 0; j < block_size; j++)
			{
				const auto position = positions[j];
				const auto low = static_cast<int>(position);
				const auto high = (low + 1) % output_channels;
				const auto mix = position - low;
				output_stream[low][block_offset + j] += input_stream[j] * gf_envelopes::quarter_sine_wave[static_cast<
						int>
					((1 - mix) * 4095)];
				output_stream[high][block_offset + j] += input_stream[j] * gf_envelopes::quarter_sine_wave[static_cast<
					int>((mix) * 4095)];
			}
		}

	public:
		std::atomic<float> pan_position = 0.5;
		std::atomic<float> pan_spread = 0.25;
		std::atomic<float> pan_quantization = 0;

		gf_panner(const int in_channels, const int out_channels = 2)
		{
			set_channels(in_channels, out_channels);
		}

		void set_channels(const int channels, const int output_channels = 2)
		{
			while (latch_.load()){
				std::this_thread::sleep_for(std::chrono::milliseconds(10));
			}
			latch_.store(true);
			channels_ = channels;
			output_channels_ = output_channels;
			if (channels != last_samples_.size())
			{
				last_samples_.resize(channels);
				std::fill(last_samples_.begin(), last_samples_.end(), 0);
				last_position_.resize(channels);
				std::fill(last_position_.begin(), last_position_.end(), 0);
			}
			latch_.store(false);
		}


		std::vector<float> get_positions() const
		{
			std::vector<float> positions;
			positions.resize(last_position_.size());
			for (int i = 0; i < positions.size(); i++)
			{
				positions[i] = last_position_[i];
			}
			return positions;
		}


		void process(sigtype** __restrict grains, sigtype** __restrict grain_states, sigtype** __restrict output_stream,
		             const int block_size)
		{

			const auto position = pan_position.load();
			const auto spread = pan_spread.load();
			const auto quantization = pan_quantization.load();
			const auto blocks = block_size / InternalBlock;
			const auto output_chans = output_channels_.load();
			if (output_chans < 1) return;
			if (latch_.load()){
				return;
			}
			latch_.store(true);
			for (int ch = 0; ch < channels_; ++ch)
			{
				for (int i = 0; i < blocks; ++i)
				{
					auto this_block = i * InternalBlock;
					auto input = &grains[ch][this_block];
					auto states = &grain_states[ch][this_block];

					const sigtype absSum = std::transform_reduce(states, &states[InternalBlock], 0,
					                                             std::plus<sigtype>{},
					                                             static_cast<sigtype(*)(sigtype)>(std::fabs));
					if (absSum <= 0.0)
					{
						continue;
					}
					auto idx = gf_utils::detect_one_transition<
						sigtype>(states, InternalBlock, last_samples_.data(), ch);

					determine_pan_position(idx, InternalBlock, channels_, position, spread, quantization,
					                       last_position_, ch, output_chans, positions_);
					perform_pan(input, positions_, InternalBlock, output_stream,
					            this_block, output_chans);
								
				}
			}
			latch_.store(false);

		}
	};
}

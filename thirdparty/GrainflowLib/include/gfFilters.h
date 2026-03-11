#pragma once
#include <array>
#include "gfUtils.h"

#ifndef M_PI
	#define _USE_MATH_DEFINES
	#include <cmath>
#endif 



namespace Grainflow
{
	template <typename SigType = double>
	struct biquad_params
	{
		float a1{0};
		float a2{0};
		float b0{0};
		float b1{0};
		float b2{0};

		inline static void lerp_block(biquad_params& a, biquad_params& b, const SigType amount, const int blocksize,
		                              biquad_params* params_block)
		{
			for (int i = 0; i < blocksize; ++i)
			{
				auto& params = params_block[i];
				params.a1 = gf_utils::lerp(a.a1, b.a1, amount * i);
				params.a2 = gf_utils::lerp(a.a2, b.a2, amount * i);
				params.b0 = gf_utils::lerp(a.b0, b.b0, amount * i);
				params.b1 = gf_utils::lerp(a.b1, b.b1, amount * i);
				params.b2 = gf_utils::lerp(a.b2, b.b2, amount * i);
			}
		}

		inline static void split_block(biquad_params& a, biquad_params& b, const SigType amount, const int blocksize,
		                               int split,
		                               biquad_params* params_block)
		{
			std::fill_n(params_block, split, a);
			std::fill_n(params_block + split, blocksize - split, b);
		}

		inline static void bandpass(biquad_params& params, float center, float q, int fs)
		{
			const float omega = M_2_PI * std::max<float>(center * 10, 1.0f) / fs;
			const float sn = std::sin(omega);
			const float cs = std::cos(omega);
			float alpha = sn * 0.5 / q;
			float a0 = 1.0f / (1 + alpha);
			params.b0 = alpha * a0;
			params.b1 = 0;
			params.b2 = -alpha * a0;
			params.a1 = -2.0f * cs * a0;
			params.a2 = (1.0f - alpha) * a0;
		}

		inline static void morph(biquad_params& params, const SigType center, const SigType q,
		                         const SigType morph, const int fs)
		{
			const SigType freq = std::min(std::max(center, 1.0), fs * 0.5) * 10;
			const float omega = M_2_PI * std::max<float>(freq, 1.0f) / fs;
			const float sn = std::sin(omega);
			const float cs = std::cos(omega);
			const float alpha = sn * 0.5 / std::max(q, 0.01);
			const float morph_value = morph;
			const float bp_norm = std::min(10.0, std::max(1.0, q));
			const float a0 = (1 / (1 + alpha));
			//Bandpass
			params.a1 = -2 * cs * a0;
			params.a2 = (1 - alpha) * a0;

			float bp_b0 = (alpha * a0 * bp_norm);
			float bp_b1 = 0;
			float bp_b2 = (-alpha * a0 * bp_norm);

			float lp_b0 = ((1 - cs) / 2 * a0);
			float lp_b1 = ((1 - cs) * a0);
			float lp_b2 = ((1 - cs) / 2 * a0);

			float hp_b0 = (1 + cs) / 2 * a0;
			float hp_b1 = -(1 + cs) * a0;
			float hp_b2 = (1 + cs) / 2 * a0;


			float lp_morph = 1 - std::min(1.0f, std::abs(morph_value - 0));
			float bp_morph = 1 - std::min(1.0f, std::abs(morph_value - 1));
			float hp_morph = 1 - std::min(1.0f, std::abs(morph_value - 2));

			params.b0 = lp_b0 * lp_morph + bp_b0 * bp_morph + hp_b0 * hp_morph;
			params.b1 = lp_b1 * lp_morph + bp_b1 * bp_morph + hp_b1 * hp_morph;
			params.b2 = lp_b2 * lp_morph + bp_b2 * bp_morph + hp_b2 * hp_morph;
		}
	};


	template <typename SigType = double>
	class biquad
	{
	private:
		SigType mem[4]{0};

	public:
		biquad()
		{
		}

		~biquad()
		{
		}

		void clear()
		{
			std::fill_n(mem, 4, 0);
		}

		inline void perform(SigType* __restrict block, const int blocksize, const biquad_params<SigType>& params,
		                    SigType* __restrict out)
		{
			for (int i = 0; i < blocksize; ++i)
			{
				//This in not optimized-- basic biquad 
				out[i] = block[i] * params.b0 + mem[0] * params.b1 + mem[1] * params.b2 - (mem[2] * params.a1 + mem[3] *
					params.a2);
				mem[1] = mem[0];
				mem[3] = mem[2];
				mem[0] = block[i];
				mem[2] = out[i];
			}
		}

		inline void perform(SigType* __restrict block, const int blocksize,
		                    const biquad_params<SigType>* __restrict params_block,
		                    SigType* __restrict out)
		{
			for (int i = 0; i < blocksize; ++i)
			{
				auto& params = params_block[i];
				//This in not optimized-- basic biquad 
				out[i] = block[i] * params.b0 + mem[0] * params.b1 + mem[1] * params.b2 - (mem[2] * params.a1 + mem[3] *
					params.a2);
				mem[1] = mem[0];
				mem[3] = mem[2];
				mem[0] = block[i];
				mem[2] = out[i];
			}
		}


		inline static void get_needed_mem_size(int& mem_size)
		{
			mem_size = 4;
		}
	};
}

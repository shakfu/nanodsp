#include "bitcrush.h"
#include <math.h>

using namespace daisysp;

// nanodsp local patch: the Fold used by Process() was a file-scope `static Fold
// fold;` shared by every Bitcrush in the process, so two instances interfered
// and concurrent Process() calls raced on it. It is a member now (bitcrush.h).
// See thirdparty/VERSIONS.md.

void Bitcrush::Init(float sample_rate)
{
    bit_depth_   = 8;
    crush_rate_  = 10000;
    sample_rate_ = sample_rate;
    fold_.Init();
}

float Bitcrush::Process(float in)
{
    float bits    = pow(2, bit_depth_);
    float foldamt = sample_rate_ / crush_rate_;
    float out;

    out = in * 65536.0f;
    out += 32768;
    out *= (bits / 65536.0f);
    out = floor(out);
    // nanodsp local patch: was `out *= (65536.0f / bits) - 32768;`, which
    // parses as out * ((65536/bits) - 32768) and so applies a bit-depth
    // dependent gain of about 2^(bit_depth-1) with an inverted sign, instead
    // of undoing the two encode steps above. See thirdparty/VERSIONS.md.
    out *= (65536.0f / bits);
    out -= 32768;

    fold_.SetIncrement(foldamt);
    out = fold_.Process(out);
    out /= 65536.0;

    return out;
}

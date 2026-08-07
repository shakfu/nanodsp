/***************************************************/
/*! \class Noise
    \brief STK noise generator.

    Generic random number generation using the
    C rand() function.  The quality of the rand()
    function varies from one OS to another.

    by Perry R. Cook and Gary P. Scavone, 1995--2023.
*/
/***************************************************/

#include "Noise.h"
#include <time.h>

namespace stk {

Noise :: Noise( unsigned int seed )
{
  // Seed the random number generator
  this->setSeed( seed );
}

void Noise :: setSeed( unsigned int seed )
{
  // nanodsp local patch: upstream seeds from the wall clock when seed == 0,
  // which is the default for every Noise member inside an STK voice. That made
  // any voice containing noise render differently on every run landing in a
  // different second, and -- because srand() is process global and DaisySP
  // draws from the same rand() state -- silently randomised unrelated DaisySP
  // generators (pluck, drip, the snare drums) as a side effect of merely
  // constructing an STK instrument.
  //
  // Leaving rand() untouched for seed == 0 keeps its deterministic default
  // state, so renders are reproducible. Callers wanting run-to-run variation
  // seed explicitly via nanodsp._core.stk.set_random_seed(). See
  // thirdparty/VERSIONS.md.
  if ( seed != 0 )
    srand( seed );
}

} // stk namespace



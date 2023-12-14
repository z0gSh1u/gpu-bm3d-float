#ifndef _PARAMS_H_
#define _PARAMS_H_

#include <vector_types.h>
#include <vector_functions.h>

#ifndef uint
#define uint unsigned int
#endif

struct Params {
  /*
  RESTRICTIONS:
      k must be divisible by p
  */
  unsigned int patch_size; // width and height of a patch
  unsigned int stripe;     // Step between reference patches
  unsigned int
      max_group_size; // Maximal number of similar blocks in stack (without reference block)
  unsigned int searching_window_size; // Area where similar blocks are searched
  unsigned int
      distance_threshold_1; // Distance treshold under which two blocks are simialr for step 1
  unsigned int
      distance_threshold_2; // Distance treshold under which two blocks are simialr for step 2
  float sigma;              // Expexted noise variance
  float lambda_3d;          // Treshold in first step colaborative filtering
  float beta; // Kaiser window parameter that affects the sidelobe attenuation of the transform of
              // the window.

  // [ VNC ]
  Params(unsigned int searching_window_size = 16, unsigned int patch_size = 8,
         unsigned int max_group_size = 16, unsigned int distance_threshold_1 = 20*10 * 8 * 8,
         unsigned int distance_threshold_2 = 100 * 8 * 8, unsigned int stripe = 3, float sigma = 20,
         float lambda_3d = 2.7f, float beta = 2.0f)
      : searching_window_size(searching_window_size), patch_size(patch_size),
        max_group_size(max_group_size), distance_threshold_1(distance_threshold_1),
        distance_threshold_2(distance_threshold_2), stripe(stripe), sigma(sigma),
        lambda_3d(lambda_3d), beta(beta) {}
};

// structure to store information of matched blocks
struct Q {
  uint distance;
  uint2 position;
};

#endif

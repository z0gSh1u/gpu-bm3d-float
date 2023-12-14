#include "bm3d.h"

/*
 * Read-only variables for all cuda kernels. These variables
 * will be stored in the "constant" memory on GPU for fast read.
 */
__constant__ GlobalConstants cu_const_params;

#include "block_matching.cu_inl"
#include "aggregation.cu_inl"
#include <cufft.h>

////////////////////////////////////////////////////////////////////////////////////////
// Putting all the cuda kernels here
///////////////////////////////////////////////////////////////////////////////////////
__device__ float norm2(cuComplex &a) { return (a.x * a.x) + (a.y * a.y); }

__global__ void real2complex(float *h_data, cufftComplex *output) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int index = j * cu_const_params.image_width + i;

  if (i < cu_const_params.image_width && j < cu_const_params.image_height) {
    output[index].x = h_data[index];
    output[index].y = 0.0f;
  }
}

__global__ void complex2real(cufftComplex *data, float *output, int total_size, int trans_size) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index >= total_size) {
    return;
  }
  output[index] = data[index].x / (float)(trans_size);
}

/*
 *  normalize cufft inverse result by dividing number of elements per batch
 */
__global__ void normalize(cufftComplex *data, int size) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int index = idx2(i, j, cu_const_params.image_width);
  data[index].x = data[index].x / (float)(size);
  data[index].y = data[index].y / (float)(size);
}

/*
 * taking d_rearrange_stacks and perform thresholding. Count number of non zeros
 * Also will normalize the 1D transform result.
 */
__global__ void hard_filter(cufftComplex *d_transformed_stacks, float *d_weight) {
  int group_id = threadIdx.x + blockIdx.x * blockDim.x;
  if (group_id >= cu_const_params.total_ref_patches) {
    return;
  }
  int patch_size = cu_const_params.patch_size;
  int non_zero = 0;
  float threshold = cu_const_params.lambda_3d * cu_const_params.lambda_3d * cu_const_params.sigma *
                    cu_const_params.sigma * patch_size * patch_size *
                    cu_const_params.max_group_size;
  // printf("Threshold %f\n", threshold);
  int offset = group_id * cu_const_params.max_group_size * patch_size * patch_size;

  float x, y, val;
  for (int i = 0; i < patch_size * patch_size * cu_const_params.max_group_size; i++) {
    x = d_transformed_stacks[offset + i].x;
    y = d_transformed_stacks[offset + i].y;
    val = x * x + y * y;
    if (val < threshold) {
      // printf("below threshold\n");
      x = 0.0f;
      y = 0.0f;
    } else {
      ++non_zero;
    }
    d_transformed_stacks[offset + i].x = x;
    d_transformed_stacks[offset + i].y = y;
  }
  d_weight[group_id] = 1.0f / (float)non_zero;
}

__global__ void get_wiener_coef(cufftComplex *d_transformed_stacks, float *d_wien_coef) {
  int group_id = threadIdx.x + blockIdx.x * blockDim.x;
  if (group_id >= cu_const_params.total_ref_patches) {
    return;
  }
  const int patch_size = cu_const_params.patch_size;
  const int sigma = cu_const_params.sigma;
  const int norm_fator = patch_size * patch_size * cu_const_params.max_group_size;
  int offset = group_id * cu_const_params.max_group_size * patch_size * patch_size;

  float val;
  for (int i = 0; i < patch_size * patch_size * cu_const_params.max_group_size; i++) {
    val = norm2(d_transformed_stacks[offset + i]) / (float)norm_fator;
    d_wien_coef[offset + i] = val / (val + sigma * sigma);
  }
}

__global__ void apply_wiener_coef(cufftComplex *d_transformed_stacks, float *d_wien_coef,
                                  float *d_wien_weight) {
  int group_id = threadIdx.x + blockIdx.x * blockDim.x;
  if (group_id >= cu_const_params.total_ref_patches) {
    return;
  }
  const int patch_size = cu_const_params.patch_size;
  int offset = group_id * cu_const_params.max_group_size * patch_size * patch_size;
  float wien_acc = 0.0f;
  for (int i = 0; i < patch_size * patch_size * cu_const_params.max_group_size; i++) {
    float wien = d_wien_coef[offset + i];
    d_transformed_stacks[offset + i].x *= wien;
    d_transformed_stacks[offset + i].y *= wien;
    wien_acc += wien * wien;
  }
  d_wien_weight[group_id] = 1.0f / wien_acc;
}

/*
 *  Each thread maps to a group, d_transformed_stacks is organized as (w, h, patch in group)
 */
__global__ void fill_patch_major_from_source(Q *d_stacks, uint *d_num_patches_in_stack,
                                             float *input_data,
                                             cufftComplex *d_transformed_stacks) {
  int group_id = threadIdx.x + blockIdx.x * blockDim.x;
  if (group_id >= cu_const_params.total_ref_patches) {
    return;
  }
  int width = cu_const_params.image_width;
  int patch_size = cu_const_params.patch_size;

  // start patch num
  int start = group_id * cu_const_params.max_group_size;
  int offset = start * patch_size * patch_size;

  for (int z = 0; z < d_num_patches_in_stack[group_id]; z++) {
    // fill in the actual data
    uint patch_x = d_stacks[z + start].position.x;
    uint patch_y = d_stacks[z + start].position.y;
    for (int k = 0; k < patch_size * patch_size; k++) {
      int index = idx2(patch_x + (k % patch_size), patch_y + (k / patch_size), width);
      int output_index = idx2(k, z, patch_size * patch_size);
      d_transformed_stacks[output_index + offset].x = (float)(input_data[index]);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////
// Class member functions
///////////////////////////////////////////////////////////////////////////////////////

/*
 * Initialize params struct
 */
Bm3d::Bm3d() {
  h_width = 0;
  h_height = 0;
  h_channels = 0;
  d_noisy_image = NULL;
  d_denoised_image = NULL;

  d_stacks = NULL;
  d_num_patches_in_stack = NULL;
  d_weight = NULL;
  d_wien_coef = NULL;
  d_kaiser_window = NULL;
}

Bm3d::~Bm3d() {}

/*
 * Set first step params
 */
void Bm3d::set_fst_step_param() {}

/*
 * Set second step params
 */
void Bm3d::set_2nd_step_param() {}

/*
 * Set device params and allocate device memories
 */
void Bm3d::set_device_param() {
  total_patches =
      (h_width - h_fst_step_params.patch_size + 1) * (h_height - h_fst_step_params.patch_size + 1);
  total_ref_patches = ((h_width - h_fst_step_params.patch_size) / h_fst_step_params.stripe + 1) *
                      ((h_height - h_fst_step_params.patch_size) / h_fst_step_params.stripe + 1);
  // copy original image to cuda
  const uint size = h_width * h_height;
  cudaMalloc(&d_noisy_image, sizeof(float) * h_channels * size);

  cudaMalloc(&d_stacks, sizeof(Q) * total_ref_patches * h_fst_step_params.max_group_size);
  cudaMalloc(&d_num_patches_in_stack, sizeof(uint) * total_ref_patches);
  cudaMalloc(&d_transformed_stacks, sizeof(cufftComplex) * h_fst_step_params.patch_size *
                                        h_fst_step_params.patch_size *
                                        h_fst_step_params.max_group_size * total_ref_patches);

  cudaMalloc(&d_numerator, sizeof(float) * size);
  cudaMalloc(&d_denominator, sizeof(float) * size);
  cudaMalloc(&d_weight, sizeof(float) * total_ref_patches);
  cudaMalloc(&d_wien_coef, sizeof(float) * h_fst_step_params.patch_size *
                               h_fst_step_params.patch_size * h_fst_step_params.max_group_size *
                               total_ref_patches);
  cudaMalloc(&d_wien_weight, sizeof(float) * total_ref_patches);

  cudaMalloc(&d_denoised_image, sizeof(float) * size);

  // Only use the generic params for now
  GlobalConstants params;
  params.image_width = h_width;
  params.image_height = h_height;
  params.image_channels = h_channels;

  params.patch_size = h_fst_step_params.patch_size;
  params.searching_window_size = h_fst_step_params.searching_window_size;
  params.stripe = h_fst_step_params.stripe;
  params.max_group_size = h_fst_step_params.max_group_size;
  params.distance_threshold_1 = h_fst_step_params.distance_threshold_1;
  params.distance_threshold_2 = h_fst_step_params.distance_threshold_2;
  params.sigma = h_fst_step_params.sigma;
  params.lambda_3d = h_fst_step_params.lambda_3d;
  params.beta = h_fst_step_params.beta;
  params.total_ref_patches = total_ref_patches;

  cudaMemcpyToSymbol(cu_const_params, &params, sizeof(GlobalConstants));
  int dim3D[3] = {h_fst_step_params.patch_size, h_fst_step_params.patch_size,
                  h_fst_step_params.max_group_size};
  int size_3d = h_fst_step_params.patch_size * h_fst_step_params.patch_size *
                h_fst_step_params.max_group_size;
  if (cufftPlanMany(&plan3D, 3, dim3D, NULL, 1, size_3d, NULL, 1, size_3d, CUFFT_C2C,
                    total_ref_patches) != CUFFT_SUCCESS) {
    fprintf(stderr, "CUFFT Plan error: Plan failed");
    return;
  }
}

/*
 * Initialize image stats and allocate memory
 */
void Bm3d::copy_image_to_device(float *src_image) {
  // set width and height
  cudaMemcpy(d_noisy_image, src_image, sizeof(float) * h_channels * h_height * h_width,
             cudaMemcpyHostToDevice);
}

void Bm3d::free_device_params() {
  if (d_noisy_image) {
    cudaFree(d_noisy_image);
  }
}

void Bm3d::clean_up_buffer() {
  // clean up buffer
  cudaMemset(d_numerator, 0, sizeof(float) * h_width * h_height);
  cudaMemset(d_denominator, 0, sizeof(float) * h_width * h_height);
  // cudaMemset(d_stacks, 0, sizeof(Q) * total_ref_patches * h_fst_step_params.max_group_size);
  // cudaMemset(d_num_patches_in_stack, 0, sizeof(uint) * total_ref_patches);
  cudaMemset(d_transformed_stacks, 0,
             sizeof(cufftComplex) * h_fst_step_params.patch_size * h_fst_step_params.patch_size *
                 h_fst_step_params.max_group_size * total_ref_patches);

  cudaMemset(d_weight, 0, sizeof(float) * total_ref_patches);
  cudaMemset(d_wien_coef, 0,
             sizeof(float) * h_fst_step_params.patch_size * h_fst_step_params.patch_size *
                 h_fst_step_params.max_group_size * total_ref_patches);
  cudaMemset(d_wien_weight, 0, sizeof(float) * total_ref_patches);

  cudaMemset(d_denoised_image, 0, sizeof(float) * h_width * h_height);
}

void Bm3d::set_up_realtime(int width, int height, int channels) {
  h_width = width;
  h_height = height;
  h_channels = channels;
  set_device_param();
}

/*
 * need to call set_up_realtime first
 */
void Bm3d::realtime_denoise(float *src_image, float *dst_image) {
  copy_image_to_device(src_image);
  clean_up_buffer();
  denoise_fst_step();
  cudaMemset(d_transformed_stacks, 0,
             sizeof(cufftComplex) * h_fst_step_params.patch_size * h_fst_step_params.patch_size *
                 h_fst_step_params.max_group_size * total_ref_patches);
  denoise_2nd_step();
  cudaMemcpy(dst_image, d_denoised_image, sizeof(float) * h_width * h_height,
             cudaMemcpyDeviceToHost);
}

/*
 * Take an image and run the algorithm to denoise.
 */
void Bm3d::denoise(float *src_image, float *dst_image, int width, int height, int sigma,
                   int channels, int step, int verbose = 1) {
  Stopwatch init_time;
  Stopwatch first_step;
  Stopwatch sed_step;

  h_width = width;
  h_height = height;
  h_channels = channels;

  init_time.start();
  set_device_param();
  init_time.stop();

  copy_image_to_device(src_image);

  first_step.start();
  denoise_fst_step();
  first_step.stop();

  sed_step.start();
  if (step == 2) {
    denoise_2nd_step();
  }
  sed_step.stop();

  // copy image from device to host
  printf("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n");
  printf("Init takes %f\n", init_time.getSeconds());
  printf("First step takes %f\n", first_step.getSeconds());
  printf("Second step takes %f\n", sed_step.getSeconds());

  const uint num_pixels = h_width * h_height;
  cudaMemcpy(dst_image, d_denoised_image, sizeof(float) * num_pixels, cudaMemcpyDeviceToHost);
}

/*
 * Perform the first step denoise
 */
void Bm3d::denoise_fst_step() {
  // Block matching, each thread maps to a ref patch
  do_block_matching(d_noisy_image, h_fst_step_params.distance_threshold_1);

  // gather patches
  arrange_block(d_noisy_image);

  // perform 3D dct transform;

  if (cufftExecC2C(plan3D, d_transformed_stacks, d_transformed_stacks, CUFFT_FORWARD) !=
      CUFFT_SUCCESS) {
    fprintf(stderr, "CUFFT error: 3D Forward failed");
    return;
  }

  // hard thresholding and normalize
  hard_threshold();

  // perform inverse 3D dct transform;
  if (cufftExecC2C(plan3D, d_transformed_stacks, d_transformed_stacks, CUFFT_INVERSE) !=
      CUFFT_SUCCESS) {
    fprintf(stderr, "CUFFT error: 3D inverse failed");
    return;
  }
  // Need to normalize 3D inverse result by dividing patch_size * patch_size
  // aggregate to single image by writing into buffer
  do_aggregation(d_weight);
}

/*
 * Perform the second step denoise
 */
void Bm3d::denoise_2nd_step() {
  // Block matching estimate image, each thread maps to a ref patch
  do_block_matching(d_denoised_image, h_fst_step_params.distance_threshold_2);
  // gather patches for estimate image
  arrange_block(d_denoised_image);
  // perform 3d transform for estimate groups
  if (cufftExecC2C(plan3D, d_transformed_stacks, d_transformed_stacks, CUFFT_FORWARD) !=
      CUFFT_SUCCESS) {
    fprintf(stderr, "CUFFT error: 3D Forward failed");
    return;
  }
  // calculate Wiener coefficient for each estimate group
  cal_wiener_coef();
  // gather noisy image patches according to estimate block matching result
  arrange_block(d_noisy_image);
  // perform 3d transform on original image
  if (cufftExecC2C(plan3D, d_transformed_stacks, d_transformed_stacks, CUFFT_FORWARD) !=
      CUFFT_SUCCESS) {
    fprintf(stderr, "CUFFT error: 3D Forward failed");
    return;
  }
  // apply wiener coefficient to each group of transformed noisy data
  apply_wien_filter();
  // inverse 3d transform
  if (cufftExecC2C(plan3D, d_transformed_stacks, d_transformed_stacks, CUFFT_INVERSE) !=
      CUFFT_SUCCESS) {
    fprintf(stderr, "CUFFT error: 3D Forward failed");
    return;
  }
  // aggregate to single image by writing into buffer
  do_aggregation(d_wien_weight);
  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    fprintf(stderr, "Cuda error: %s\n", cudaGetErrorString(code));
    return;
  }
}

/*
 *  arrange_block - according to the stacked patch indices, fetching data from the transformed
 *                  data array for 2D DCT. Input is an array of uint2, every N uint2
 *                  is a group. This kernel will put each group into an continuous array
 *                  of cufftComplex num with x component to be the value, y component to be 0.f
 */
void Bm3d::arrange_block(float *input_data) {
  // input: Q* each struct is a patch with top left index
  // output: d_transformed_stacks, each patch got patch*patch size continuous chunk
  // each group will be assigned a thread
  Stopwatch arrange;
  arrange.start();
  int thread_per_block = 512;
  int num_blocks = (total_ref_patches + thread_per_block - 1) / thread_per_block;
  fill_patch_major_from_source<<<num_blocks, thread_per_block>>>(d_stacks, d_num_patches_in_stack,
                                                                 input_data, d_transformed_stacks);
  cudaDeviceSynchronize();
  arrange.stop();
  // printf("Arrange block takes %f\n", arrange.getSeconds());
}

/*
 * do_block_matching - launch kernel to run block matching
 */
void Bm3d::do_block_matching(float *input_image, const uint distance_threshold) {
  // determine how many threads we need to spawn
  Stopwatch bm_time;
  bm_time.start();
  const int total_num_threads = total_ref_patches;
  const int threads_per_block = 512;
  const int num_blocks = (total_num_threads + threads_per_block - 1) / threads_per_block;
  block_matching<<<num_blocks, threads_per_block>>>(d_stacks, d_num_patches_in_stack, input_image,
                                                    distance_threshold);
  cudaDeviceSynchronize();
  bm_time.stop();
  // printf("Block Matching: %f\n", bm_time.getSeconds());
}

void Bm3d::hard_threshold() {
  Stopwatch hard_threshold;
  hard_threshold.start();
  int thread_per_block = 512;
  int num_blocks = (total_ref_patches + thread_per_block - 1) / thread_per_block;
  hard_filter<<<num_blocks, thread_per_block>>>(d_transformed_stacks, d_weight);
  cudaDeviceSynchronize();
  hard_threshold.stop();
  // printf("Hard threshold takes %.5f\n", hard_threshold.getSeconds());
}

void Bm3d::cal_wiener_coef() {
  Stopwatch wiener_coef;
  wiener_coef.start();
  int thread_per_block = 512;
  int num_blocks = (total_ref_patches + thread_per_block - 1) / thread_per_block;
  get_wiener_coef<<<num_blocks, thread_per_block>>>(d_transformed_stacks, d_wien_coef);
  cudaDeviceSynchronize();
  wiener_coef.stop();
  // printf("Get wiener takes %.5f\n", wiener_coef.getSeconds());
}

void Bm3d::apply_wien_filter() {
  Stopwatch apply_wiener;
  apply_wiener.start();
  int thread_per_block = 512;
  int num_blocks = (total_ref_patches + thread_per_block - 1) / thread_per_block;
  apply_wiener_coef<<<num_blocks, thread_per_block>>>(d_transformed_stacks, d_wien_coef,
                                                      d_wien_weight);
  cudaDeviceSynchronize();
  apply_wiener.stop();
  // printf("Apply wiener takes %.5f\n", apply_wiener.getSeconds());
}

void Bm3d::do_aggregation(float *weight) {
  Stopwatch ag_time;
  ag_time.start();
  const uint num_threads_per_block = 512;
  // step 1: do aggregation, one thread per stack
  uint num_blocks = (total_ref_patches + num_threads_per_block - 1) / num_threads_per_block;
  aggregate<<<num_blocks, num_threads_per_block>>>(
      d_stacks, d_num_patches_in_stack, weight, d_transformed_stacks, d_numerator, d_denominator);
  // step 2: reduction. calculate how many pixels
  const uint num_pixels = h_width * h_height;
  num_blocks = (num_pixels + num_threads_per_block - 1) / num_threads_per_block;
  reduction<<<num_blocks, num_threads_per_block>>>(d_numerator, d_denominator, d_denoised_image,
                                                   num_pixels);
  cudaDeviceSynchronize();
  ag_time.stop();
  // printf("Aggregation: %f\n", ag_time.getSeconds());
}
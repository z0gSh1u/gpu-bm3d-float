#include <stdlib.h>
#include <stdio.h>
#include <string>

#include "bm3d.h"
#include "Cimg.h"
#include "getopt.h"

#include <vector_types.h>
#include <vector_functions.h>

using namespace cimg_library;
using namespace std;

void usage(const char *progname) {
  printf("Usage: %s [options] InputFile OutputFile\n", progname);
  printf("Program Options:\n");
  printf("  -s  <INT>       Noisy level (sigma)\n");
  printf("  -h  <INT>       Height of image\n");
  printf("  -w  <INT>       Width of image\n");
  printf("  -t  <INT>       Step of denoise, 1: first, 2: both\n");
  printf("  -v              Print addtional infomation\n");
  printf("  -?              Help message\n");
}

int main(int argc, char **argv) {
  int opt, channels = 1, step = 2, verbose = 0, sigma = 0, height, width;
  string input_file, output_file;

  while ((opt = getopt(argc, argv, "s:t:h:w:v?")) != EOF) {
    switch (opt) {
    case 's':
      sigma = atoi(optarg);
      break;
    case 't':
      step = atoi(optarg);
      break;
    case 'h':
      height = atoi(optarg);
      break;
    case 'w':
      width = atoi(optarg);
      break;
    case 'v':
      verbose = 1;
      break;
    case '?':
    default:
      usage(argv[0]);
      return 1;
    }
  }

  if (optind + 2 > argc) {
    fprintf(stderr, "Error: missing file name\n");
    usage(argv[0]);
    return 1;
  }

  input_file = argv[optind];
  output_file = argv[optind + 1];
  if (verbose) {
    printf("Sigma: %d\n", sigma);
    if (channels == 1) {
      printf("Image: Grayscale\n");
    }
    printf("Steps: %d\n", step);
  }

  // Allocate images
  CImg<float> inputImage;
  inputImage.load_raw(input_file.c_str(), width, height);

  // Check for invalid input
  if (!inputImage.data()) {
    fprintf(stderr, "Error: Could not open file\n");
    return 1;
  }

  printf("Width: %d, Height: %d\n", inputImage.width(), inputImage.height());

  // time it
  Stopwatch timer;
  timer.start();
  // Launch BM3D
  CImg<float> outputImage(inputImage.width(), inputImage.height(), 1, channels, 0);
  Bm3d bm3d;
  bm3d.set_up_realtime(inputImage.width(), inputImage.height(), channels);
  bm3d.realtime_denoise(inputImage.data(), outputImage.data());
  cudaDeviceSynchronize();
  timer.stop();
  printf("Time elapsed [sec]: %f \n", timer.getSeconds());

  outputImage = outputImage.get_channel(0);
  outputImage.save(output_file.c_str());

  printf("BM3D done.\n");
  return 0;
}

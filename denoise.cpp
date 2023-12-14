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
  printf("  -s  --sigma <INT>          Noisy level\n");
  printf("  -c  --color                Color Image\n");
  printf(
      "  -t  --step  <INT>          Perform which step of denoise, 1: first step, 2: both step\n");
  printf("  -v  --verbose              Print addtional infomation\n");
  printf("  -?  --help                 This message\n");
}

int main(int argc, char **argv) {
  printf("this is float ver\n");

  int opt;
  int channels = 1;
  int step = 2;
  int verbose = 0;
  int sigma = 0;
  string input_file, output_file;

  while ((opt = getopt(argc, argv, "s:ct:v?")) != EOF) {
    switch (opt) {
    case 's':
      sigma = atoi(optarg);
      break;
    case 'c':
      // color image
      channels = 3;
      break;
    case 't':
      step = atoi(optarg);
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
    fprintf(stderr, "Error: missing File name\n");
    usage(argv[0]);
    return 1;
  }

  input_file = argv[optind];
  output_file = argv[optind + 1];
  if (verbose) {
    printf("Sigma: %d\n", sigma);
    if (channels == 1) {
      printf("Image: Grayscale\n");
    } else {
      printf("Image: Color\n");
    }
    printf("Steps: %d\n", step);
  }

  // Allocate images
  CImg<float> image;
  image.load_raw(input_file.c_str(), 512, 512);

  CImg<float> image2(image.width(), image.height(), 1, channels, 0);

  // Check for invalid input
  if (!image.data()) {
    fprintf(stderr, "Error: Could not open file\n");
    return 1;
  }

  printf("Width: %d, Height: %d\n", image.width(), image.height());

  // Launch BM3D
  Bm3d bm3d;
  bm3d.set_up_realtime(image.width(), image.height(), channels);

  // time it
  Stopwatch timer;
  timer.start();
  for (int i = 0; i < 100; i++) {
    bm3d.realtime_denoise(image.data(), image2.data());
    cudaDeviceSynchronize();
  }
  timer.stop();
  printf("100 circle, timer [sec]: %f \n", timer.getSeconds());

  image2 = image2.get_channel(0);
  image2.save(output_file.c_str());

  printf("exec done\n");
  return 0;
}

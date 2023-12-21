# bm3d-float-cuda

This is the implementation of BM3D based on [https://github.com/JeffOwOSun/gpu-bm3d](https://github.com/JeffOwOSun/gpu-bm3d) with supports for float32 images, and CMake cross-platform compilation (Windows and Linux both OK).

![demo](./demo.png)

```sh
$ bm3d -s 20  # noise level (sigma)
       -h 256 # height
       -w 256 # width
       -t 2   # 1 for one-step, 2 for both-step
       -v     # verbose
       lena_noisy.raw    # input float32 raw image
       lena_denoised.raw # output float32 raw image
```

Note that in this implementation, the Discrete Cosine Transform (DCT) is substituted by Discrete Fourier Transform (DCT/FFT), which is different from the official implementation. Also, you might need to finetune those parameters in [params.h](params.h).


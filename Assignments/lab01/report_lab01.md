# Lab Report — Lab 01: Efficient Separable Gaussian Convolution

*Fill in each section below. Be specific — name the actual methods/parameters you used, since
your report is graded alongside your code.*

## Overview

_1 sentence: What was the goal of this lab?_

The goal of this lab was to blur a grayscale image using Gaussian convolution and make the process more efficient by taking advantage of the Gaussian kernel’s separability.

## Implementation

_1-2 sentences: Briefly describe the steps you performed to implment efficient convolution._

I used the gauss_kernel() function to create a normalized kernel with the chosen sigma value. In my_conv_method() I started by padding the image to deal with edge effects. Then I applied the kernel in both directions using cv2.filter2D. After that I cropped the result back, to the image size and converted the final output to uint8 format.


![Results](<Screenshot 2026-10-05 134030-1.png>)
## Results

_1-2 sentences: How well did your method perform vs the baseline in the pytests? ._

The output kept the image size and showed the blur that was expected. The fine details became less clear when sigma was 10. My version took 0.1221 seconds. The convolve2d baseline took 6.0797 seconds. So my version ran at 0.02 times the time of the baseline.


![pytests](<Screenshot 2026-10-05 134351.png>)

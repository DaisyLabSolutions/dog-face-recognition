#ifndef HEADER_TOOLS_H_
#define HEADER_TOOLS_H_

#include "net.h"
#include <bits/stdc++.h>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <vector>

namespace facedogrecognition {
void NonMaxSuppression(float iouThresh, std::vector<std::vector<float>> &dets);
inline float Sigmoid(float x);
void DecodeResult(const ncnn::Mat &data, int stride,
                  std::vector<std::vector<int>> anchors, float scoreThresh,
                  std::vector<std::vector<float>> &dets, int frameWidth,
                  int frameHeight, int width, int height, int numclass_);

} // namespace facedogrecognition

#endif

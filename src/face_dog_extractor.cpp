#include "header/face_dog_extractor.h"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

namespace facedogrecognition {

FaceDogExtractor::FaceDogExtractor(const char *param_buffer,
                                   const unsigned char *weight_buffer,

                                   int input_width, int input_height,
                                   bool use_gpu)
    : NCNNModel(param_buffer, weight_buffer, use_gpu),
      ImageModel(input_width, input_height) {}

FaceDogExtractor::FaceDogExtractor(const std::string &param_file,
                                   const std::string &weight_file,
                                   int input_width, int input_height,
                                   bool use_gpu)
    : NCNNModel(param_file, weight_file, use_gpu),
      ImageModel(input_width, input_height) {}

void FaceDogExtractor::Preprocess(const cv::Mat &image, ncnn::Mat &in) {
  cv::Mat img=image.clone();
  int img_w = img.cols;
  int img_h = img.rows;

  in = ncnn::Mat::from_pixels_resize(img.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h,
                              InputWidth(), InputHeight());
  const float mean_vals[3] = {0.485f * 255.f, 0.456f * 255.f, 0.406f * 255.f};
  const float norm_vals[3] = {1 / 0.229f / 255.f, 1 / 0.224f / 255.f,
                              1 / 0.225f / 255.f};

  in.substract_mean_normalize(mean_vals, norm_vals);
}

int FaceDogExtractor::Predict(const cv::Mat &image,
                              facedogrecognition::types::Feature &feature) {

  float *prob = feature.feature;
  ncnn::Mat input;
  ncnn::Mat out;
  ncnn::Mat in;

  Preprocess(image, in);

  // Inference
  int result = Infer(in, out, "input", "output");
  if (result != 0) {
    return result;
  }

  for (int j = 0; j < 512; j++) {
    feature.feature[j] = out[j];
  }
  cv::Mat out_m(512, 1, CV_32FC1, prob);
  cv::normalize(out_m, feature.feature_norm);

  return 0;
}
} // namespace facedogrecognition
#ifndef HEADER_FACE_DOG_DETECTOR_H_
#define HEADER_FACE_DOG_DETECTOR_H_

#include "header/image_model.h"
#include "header/ncnn_model.h"
#include "header/tools.h"
#include "header/types.h"
#include "opencv2/opencv.hpp"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

namespace facedogrecognition {

class FaceDogDetector : public NCNNModel, public ImageModel {
public:
  FaceDogDetector(const char *param_buffer, const unsigned char *weight_buffer,
                  float score_threshold = 0.7, float iou_threshold = 0.5,
                  int input_width = 320, int input_height = 320,
                  bool use_gpu = false);

  FaceDogDetector(const std::string &param_file, const std::string &weight_file,
                  float score_threshold = 0.7, float iou_threshold = 0.5,
                  int input_width = 320, int input_height = 320,
                  bool use_gpu = false);

  int Predict(const cv::Mat &image,
              std::vector<facedogrecognition::types::Det> &facedogdets);

private:
  void Preprocess(const cv::Mat &image, ncnn::Mat &net_input) override;
  std::vector<std::vector<std::vector<int>>> anchors_{
      {{10, 13}, {16, 30}, {33, 23}},
      {{30, 61}, {62, 45}, {59, 119}},
      {{116, 90}, {156, 198}, {373, 326}}};

  float score_threshold_;
  float iou_threshold_;
  const int numclass_ = 1;
  int width_img_;
  int height_img_;
  std::vector<std::string> output_names_{"output", "397", "417"};
};
} // namespace facedogrecognition

#endif
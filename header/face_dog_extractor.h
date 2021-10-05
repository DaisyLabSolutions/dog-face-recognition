#ifndef HEADER_FACE_DOG_EXTRACTOR_H_
#define HEADER_FACE_DOG_EXTRACTOR_H_

#include "header/image_model.h"
#include "header/ncnn_model.h"
#include "types.h"
#include <benchmark.h>
#include <datareader.h>
#include <net.h>
#include <opencv2/core/core.hpp>
#include <platform.h>

namespace facedogrecognition {

class FaceDogExtractor : public NCNNModel, public ImageModel {
public:
  FaceDogExtractor(const char *param_buffer, const unsigned char *weight_buffer,
                int input_width = 112, int input_height = 112,
                bool use_gpu = false);

  FaceDogExtractor(const std::string &param_file, const std::string &weight_file,
                int input_width = 112, int input_height = 112,
                bool use_gpu = false);

  int Predict(const cv::Mat &image, facedogrecognition::types::Feature &feature);

private:
  /// Preprocess image data to obtain net input.
  void Preprocess(const cv::Mat &image, ncnn::Mat &net_input) override;
};
} // namespace facerecognition
#endif
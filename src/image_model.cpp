#include "header/image_model.h"

#include <string>
#include <vector>

namespace facedogrecognition {

ImageModel::ImageModel(int input_width, int input_height) {
  input_height_ = input_height;
  input_width_ = input_width;
}

int ImageModel::InputWidth() { return input_width_; }

int ImageModel::InputHeight() { return input_height_; }

} // namespace facedogrecognition
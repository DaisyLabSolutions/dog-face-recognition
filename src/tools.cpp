#include "header/tools.h"

namespace facedogrecognition {

void NonMaxSuppression(float iouThresh, std::vector<std::vector<float>> &dets) {
  int length = dets.size();
  int index = length - 1;

  sort(dets.begin(), dets.end());
  std::vector<float> areas(length);
  for (int i = 0; i < length; i++) {
    areas[i] = (dets[i][4] - dets[i][2]) * (dets[i][3] - dets[i][1]);
  }

  while (index > 0) {
    int i = 0;
    while (i < index) {
      float left = std::max(dets[index][1], dets[i][1]);
      float top = std::max(dets[index][2], dets[i][2]);
      float right = std::min(dets[index][3], dets[i][3]);
      float bottom = std::min(dets[index][4], dets[i][4]);
      float overlap =
          std::max(0.0f, right - left) * std::max(0.0f, bottom - top);
      if (overlap / (areas[index] + areas[i] - overlap) > iouThresh) {
        areas.erase(areas.begin() + i);
        dets.erase(dets.begin() + i);
        index--;
      } else {
        i++;
      }
    }
    index--;
  }
}

inline float Sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }

void DecodeResult(const ncnn::Mat &data, int stride,
                  std::vector<std::vector<int>> anchors, float scoreThresh,
                  std::vector<std::vector<float>> &dets, int frameWidth,
                  int frameHeight, int width, int height, int numclass_) {
  for (int c = 0; c < data.c; c++) {
    const float *ptr = data.channel(c);
    for (int y = 0; y < data.h; y++) {
      float score = Sigmoid(ptr[4]);
      if (score > scoreThresh) {
        std::vector<float> det(6);
        det[1] = (Sigmoid(ptr[0]) * 2 - 0.5 + y % (int)(width / stride)) *
                 stride * frameWidth / width; // center_x
        det[2] = (Sigmoid(ptr[1]) * 2 - 0.5 + (int)(y / (width / stride))) *
                 stride * frameHeight / height; // center_y
        det[3] = pow((Sigmoid(ptr[2]) * 2), 2) * anchors[c][0] * frameWidth /
                 width; // w
        det[4] = pow((Sigmoid(ptr[3]) * 2), 2) * anchors[c][1] * frameHeight /
                 height; // h

        det[1] = det[1] - det[3] / 2 > 0 ? det[1] - det[3] / 2 : 0.0; // left
        det[2] = det[2] - det[4] / 2 > 0 ? det[2] - det[4] / 2 : 0.0; // top
        det[3] = det[1] + det[3] < width ? det[1] + det[3] : width;   // right
        det[4] = det[2] + det[4] < height ? det[2] + det[4] : height; // bottom

        for (int i = 5; i < numclass_ + 5; i++) {
          float conf = Sigmoid(ptr[i]);
          if (conf * score > det[0]) {
            det[0] = conf * score; // score
            det[5] = i - 5;        // class_id
          }
        }
        dets.push_back(det);
      }
      ptr += data.w;
    }
  }
}
} // namespace facedogrecognition
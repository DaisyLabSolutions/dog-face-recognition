#include "header/face_dog_detector.h"

namespace facedogrecognition {

FaceDogDetector::FaceDogDetector(const char *param_buffer,
                                 const unsigned char *weight_buffer,
                                 float score_threshold, float iou_threshold,
                                 int input_width, int input_height,
                                 bool use_gpu)
    : NCNNModel(param_buffer, weight_buffer, use_gpu),
      ImageModel(input_width, input_height) {
  score_threshold_ = score_threshold;
  iou_threshold_ = iou_threshold;
}

FaceDogDetector::FaceDogDetector(const std::string &param_file,
                                 const std::string &weight_file,
                                 float score_threshold, float iou_threshold,
                                 int input_width, int input_height,
                                 bool use_gpu)
    : NCNNModel(param_file, weight_file, use_gpu),
      ImageModel(input_width, input_height) {
  score_threshold_ = score_threshold;
  iou_threshold_ = iou_threshold;
}

void FaceDogDetector::Preprocess(const cv::Mat &image, ncnn::Mat &net_input) {
  cv::Mat img;
  int width = InputWidth();
  int height = InputHeight();
  width_img_ = image.cols;
  height_img_ = image.rows;

  cv::resize(image, img, cv::Size(width, height));
  cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
  net_input = ncnn::Mat::from_pixels(img.data, ncnn::Mat::PIXEL_RGB, img.cols,
                                     img.rows);
  float norm[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
  float mean[3] = {0, 0, 0};
  net_input.substract_mean_normalize(mean, norm);
}

int FaceDogDetector::Predict(
    const cv::Mat &image,
    std::vector<facedogrecognition::types::Det> &facedogdets) {

  ncnn::Mat in;
  std::vector<std::vector<float>> dets;
  int width = InputWidth();
  int height = InputHeight();

  Preprocess(image, in);
  std::map<std::string, ncnn::Mat> out;
  int res = Infer(in, out, "images", output_names_);
  if (res != 0) {
    return res;
  }
  ncnn::Mat output0 = (ncnn::Mat)out[output_names_[0]];
  ncnn::Mat output1 = (ncnn::Mat)out[output_names_[1]];
  ncnn::Mat output2 = (ncnn::Mat)out[output_names_[2]];
  DecodeResult(output0, 8, anchors_[0], 0.6, dets, width_img_, height_img_,
               width, height, numclass_);
  DecodeResult(output1, 16, anchors_[1], 0.6, dets, width_img_, height_img_,
               width, height, numclass_);
  DecodeResult(output2, 32, anchors_[2], 0.6, dets, width_img_, height_img_,
               width, height, numclass_);
  NonMaxSuppression(0.5, dets);

  for (int i = 0; i < dets.size(); i++) {
    facedogrecognition::types::Det det;
    float left = dets[i][1];
    float top = dets[i][2];
    float right = dets[i][3];
    float bottom = dets[i][4];
    float score = dets[i][0];
    int classID = dets[i][5];
    det.box = cv::Rect(left, top, (right - left), (bottom - top));
    det.img_cropped = image(det.box);
    facedogdets.push_back(det);
  }
  return 1;
}
} // namespace facedogrecognition

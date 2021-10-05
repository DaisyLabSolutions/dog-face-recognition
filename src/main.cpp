#include "header/face_dog_detector.h"
#include "header/face_dog_extractor.h"
#include "opencv2/opencv.hpp"
#include <time.h>

using namespace facedogrecognition;
using namespace types;

FaceDogDetector *fdd =
    new FaceDogDetector("../weights/facedog.param", "../weights/facedog.bin",
                        0.7, 0.5, 640, 640, true);

FaceDogExtractor *face_dog_extractor =
    new FaceDogExtractor("../weights/facedog_res18_1.param",
                         "../weights/facedog_res18_1.bin", 112, 112, true);

cv::Mat img1 = cv::imread("../images/testset1/1.jpg");
cv::Mat img2 = cv::imread("../images/testset1/2.jpg");
cv::Mat img3 = cv::imread("../images/testset1/3.jpg");
cv::Mat img4 = cv::imread("../images/testset1/4.jpg");
float distance(cv::Mat f1, cv::Mat f2) { return cv::norm(f1, f2); }

int main() {
  std::vector<facedogrecognition::types::Det> facedogdets1, facedogdets2,
      facedogdets3, facedogdets4;
  facedogrecognition::types::Feature feature1, feature2, feature3, feature4;

  fdd->Predict(img1, facedogdets1);
  fdd->Predict(img2, facedogdets2);
  fdd->Predict(img3, facedogdets3);
  fdd->Predict(img4, facedogdets4);


  face_dog_extractor->Predict(facedogdets1[0].img_cropped, feature1);
  face_dog_extractor->Predict(facedogdets2[0].img_cropped, feature2);
  face_dog_extractor->Predict(facedogdets3[0].img_cropped, feature3);
  face_dog_extractor->Predict(facedogdets4[0].img_cropped, feature4);

  std::cout << "distance similar : "
            << distance(feature1.feature_norm, feature2.feature_norm) << "\n";
  std::cout << "distance similar : "
            << distance(feature3.feature_norm, feature2.feature_norm) << "\n";
  std::cout << "distance similar : "
            << distance(feature1.feature_norm, feature3.feature_norm) << "\n";

  std::cout << "distance unsimilar : "
            << distance(feature4.feature_norm, feature1.feature_norm) << "\n";
  std::cout << "distance unsimilar : "
            << distance(feature4.feature_norm, feature2.feature_norm) << "\n";
  std::cout << "distance unsimilar : "
            << distance(feature4.feature_norm, feature3.feature_norm) << "\n";

  return 0;
}
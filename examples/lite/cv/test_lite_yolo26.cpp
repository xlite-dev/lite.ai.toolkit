//
// Created by lite.ai.toolkit on 2026/8/26.
//

#include "lite/lite.h"

#include <iomanip>

int main(int argc, char *argv[])
{
#ifdef ENABLE_ONNXRUNTIME
  const std::string onnx_path = argc > 1 ? argv[1] : "../../../examples/hub/onnx/cv/yolo26n.onnx";
  const std::string image_path = argc > 2 ? argv[2] : "../../../examples/lite/resources/test_lite_detection_1.jpg";
  const std::string output_path = argc > 3 ? argv[3] : "../../../examples/logs/test_lite_yolo26.jpg";

  cv::Mat image = cv::imread(image_path);
  if (image.empty())
  {
    std::cerr << "Failed to read image: " << image_path << std::endl;
    return 1;
  }

  lite::onnxruntime::cv::detection::YOLO26 detector(onnx_path);
  std::vector<lite::types::Boxf> boxes;
  detector.detect(image, boxes);

  std::cout << std::fixed << std::setprecision(6);
  for (const auto &box: boxes)
    std::cout << box.x1 << " " << box.y1 << " " << box.x2 << " " << box.y2
              << " " << box.score << " " << box.label << std::endl;

  lite::utils::draw_boxes_inplace(image, boxes);
  if (!cv::imwrite(output_path, image))
  {
    std::cerr << "Failed to write result: " << output_path << std::endl;
    return 1;
  }

  std::cout << "YOLO26 detected boxes: " << boxes.size() << std::endl;
  std::cout << "Saved result to: " << output_path << std::endl;
#else
  std::cerr << "ONNX Runtime is not enabled." << std::endl;
  return 1;
#endif
  return 0;
}

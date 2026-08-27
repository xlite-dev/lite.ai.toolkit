//
// Created by zhuohaoyang on 2026/8/26.
//

#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../examples/hub/onnx/cv/yolov26n.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_detection_1.jpg";
  std::string save_img_path = "../../../examples/logs/test_lite_yolov26_1.jpg";

  // 1. Test Default Engine ONNXRuntime
  lite::cv::detection::YoloV26 *yolov26 =
      new lite::cv::detection::YoloV26(onnx_path); // default

  std::vector<lite::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  yolov26->detect(img_bgr, detected_boxes);

  lite::utils::draw_boxes_inplace(img_bgr, detected_boxes);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "Default Version Detected Boxes Num: " << detected_boxes.size() << std::endl;

  delete yolov26;
}

static void test_onnxruntime()
{
#ifdef ENABLE_ONNXRUNTIME
  std::string onnx_path = "../../../examples/hub/onnx/cv/yolov26n.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_detection_1.jpg";
  std::string save_img_path = "../../../examples/logs/test_lite_yolov26_2.jpg";

  // 2. Test Specific Engine ONNXRuntime
  lite::onnxruntime::cv::detection::YoloV26 *yolov26 =
      new lite::onnxruntime::cv::detection::YoloV26(onnx_path);

  std::vector<lite::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  yolov26->detect(img_bgr, detected_boxes);

  lite::utils::draw_boxes_inplace(img_bgr, detected_boxes);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "ONNXRuntime Version Detected Boxes Num: " << detected_boxes.size() << std::endl;

  delete yolov26;
#endif
}

static void test_lite()
{
  test_default();
  test_onnxruntime();
}

int main(__unused int argc, __unused char *argv[])
{
  test_lite();
  return 0;
}

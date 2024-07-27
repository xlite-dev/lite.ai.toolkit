//
// Created by DefTruth on 2022/5/8.
//

#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../examples/hub/onnx/cv/yolov5face-blazeface-640x640.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_face_detector.jpg";
  std::string save_img_path = "../../../examples/logs/test_lite_yolov5_blazeface.jpg";

  lite::cv::face::detect::YOLOv5BlazeFace *yolov5_blazeface =
      new lite::cv::face::detect::YOLOv5BlazeFace(onnx_path);

  std::vector<lite::types::BoxfWithLandmarks> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  yolov5_blazeface->detect(img_bgr, detected_boxes);

  lite::utils::draw_boxes_with_landmarks_inplace(img_bgr, detected_boxes);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "Default Version Done! Detected Face Num: " << detected_boxes.size() << std::endl;

  delete yolov5_blazeface;
}

static void test_onnxruntime()
{
#ifdef ENABLE_ONNXRUNTIME
  std::string onnx_path = "../../../examples/hub/onnx/cv/yolov5face-blazeface-640x640.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_face_detector_2.jpg";
  std::string save_img_path = "../../../examples/logs/test_lite_yolov5_blazeface_onnx_2.jpg";

  lite::onnxruntime::cv::face::detect::YOLOv5BlazeFace *yolov5_blazeface =
      new lite::onnxruntime::cv::face::detect::YOLOv5BlazeFace(onnx_path);

  std::vector<lite::types::BoxfWithLandmarks> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  yolov5_blazeface->detect(img_bgr, detected_boxes);

  lite::utils::draw_boxes_with_landmarks_inplace(img_bgr, detected_boxes);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "ONNXRuntime Version Done! Detected Face Num: " << detected_boxes.size() << std::endl;

  delete yolov5_blazeface;
#endif
}

static void test_mnn()
{
#ifdef ENABLE_MNN
  std::string mnn_path = "../../../examples/hub/mnn/cv/yolov5face-blazeface-640x640.mnn";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_face_detector_2.jpg";
  std::string save_img_path = "../../../examples/logs/test_lite_yolov5_blazeface_mnn_2.jpg";

  lite::mnn::cv::face::detect::YOLOv5BlazeFace *yolov5_blazeface =
      new lite::mnn::cv::face::detect::YOLOv5BlazeFace(mnn_path);

  std::vector<lite::types::BoxfWithLandmarks> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  yolov5_blazeface->detect(img_bgr, detected_boxes);

  lite::utils::draw_boxes_with_landmarks_inplace(img_bgr, detected_boxes);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "MNN Version Done! Detected Face Num: " << detected_boxes.size() << std::endl;

  delete yolov5_blazeface;
#endif
}

static void test_ncnn()
{
#ifdef ENABLE_NCNN
#endif
}

static void test_tnn()
{
#ifdef ENABLE_TNN
#endif
}

static void test_tensorrt()
{
#ifdef ENABLE_ONNXRUNTIME
    std::string onnx_path = "../../../examples/hub/trt/yolov5face-fp32.engine";
    std::string test_img_path = "../../../examples/lite/resources/test_lite_face_detector.jpg";
    std::string save_img_path = "../../../examples/logs/test_lite_yolov5_blazeface_trt_3.jpg";

    lite::trt::cv::face::detection::YOLOV5Face *yolov5_blazeface = new lite::trt::cv::face::detection::YOLOV5Face(onnx_path);

    std::vector<lite::types::BoxfWithLandmarks> detected_boxes;
    cv::Mat img_bgr = cv::imread(test_img_path);
    yolov5_blazeface->detect(img_bgr, detected_boxes);

    lite::utils::draw_boxes_with_landmarks_inplace(img_bgr, detected_boxes);

    cv::imwrite(save_img_path, img_bgr);

    std::cout << "TensorRT Version Done! Detected Face Num: " << detected_boxes.size() << std::endl;

    delete yolov5_blazeface;
#endif
}



static void test_lite()
{
  test_default();
  test_onnxruntime();
  test_mnn();
  test_ncnn();
  test_tnn();
  test_tensorrt();
}

int main(__unused int argc, __unused char *argv[])
{
  test_lite();
  return 0;
}
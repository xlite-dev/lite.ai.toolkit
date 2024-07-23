//
// Created by DefTruth on 2021/7/20.
//

#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../examples/hub/onnx/cv/yolox_s.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_yolox_1.jpg";
  std::string save_img_path = "../../../examples/logs/test_lite_yolox_1.jpg";

  // 1. Test Default Engine ONNXRuntime
  lite::cv::detection::YoloX *yolox = new lite::cv::detection::YoloX(onnx_path); // default

  std::vector<lite::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  yolox->detect(img_bgr, detected_boxes);

  lite::utils::draw_boxes_inplace(img_bgr, detected_boxes);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "Default Version Detected Boxes Num: " << detected_boxes.size() << std::endl;

  delete yolox;

}

static void test_onnxruntime()
{
#ifdef ENABLE_ONNXRUNTIME
  std::string onnx_path = "../../../examples/hub/onnx/cv/yolox_s.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_yolox_2.jpg";
  std::string save_img_path = "../../../examples/logs/test_lite_yolox_2.jpg";

  // 2. Test Specific Engine ONNXRuntime
  lite::onnxruntime::cv::detection::YoloX *yolox =
      new lite::onnxruntime::cv::detection::YoloX(onnx_path);

  std::vector<lite::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  yolox->detect(img_bgr, detected_boxes);

  lite::utils::draw_boxes_inplace(img_bgr, detected_boxes);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "ONNXRuntime Version Detected Boxes Num: " << detected_boxes.size() << std::endl;

  delete yolox;
#endif
}

static void test_mnn()
{
#ifdef ENABLE_MNN
  std::string mnn_path = "../../../examples/hub/mnn/cv/yolox_s.mnn";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_yolox_2.jpg";
  std::string save_img_path = "../../../examples/logs/test_lite_yolox_mnn_2.jpg";

  // 3. Test Specific Engine MNN
  lite::mnn::cv::detection::YoloX *yolox =
      new lite::mnn::cv::detection::YoloX(mnn_path);

  std::vector<lite::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  yolox->detect(img_bgr, detected_boxes);

  lite::utils::draw_boxes_inplace(img_bgr, detected_boxes);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "MNN Version Detected Boxes Num: " << detected_boxes.size() << std::endl;

  delete yolox;
#endif
}

static void test_ncnn()
{
#ifdef ENABLE_NCNN
  std::string param_path = "../../../examples/hub/ncnn/cv/yolox_s.opt.param";
  std::string bin_path = "../../../examples/hub/ncnn/cv/yolox_s.opt.bin";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_yolox_2.jpg";
  std::string save_img_path = "../../../examples/logs/test_lite_yolox_ncnn_2.jpg";

  // 4. Test Specific Engine NCNN
  lite::ncnn::cv::detection::YoloX *yolox =
      new lite::ncnn::cv::detection::YoloX(param_path, bin_path);

  std::vector<lite::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  yolox->detect(img_bgr, detected_boxes);

  lite::utils::draw_boxes_inplace(img_bgr, detected_boxes);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "NCNN Version Detected Boxes Num: " << detected_boxes.size() << std::endl;

  delete yolox;
#endif
}

static void test_tnn()
{
#ifdef ENABLE_TNN
  std::string proto_path = "../../../examples/hub/tnn/cv/yolox_s.opt.tnnproto";
  std::string model_path = "../../../examples/hub/tnn/cv/yolox_s.opt.tnnmodel";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_yolox_2.jpg";
  std::string save_img_path = "../../../examples/logs/test_lite_yolox_tnn_2.jpg";

  // 5. Test Specific Engine TNN
  lite::tnn::cv::detection::YoloX *yolox =
      new lite::tnn::cv::detection::YoloX(proto_path, model_path);

  std::vector<lite::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  yolox->detect(img_bgr, detected_boxes);

  lite::utils::draw_boxes_inplace(img_bgr, detected_boxes);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "TNN Version Detected Boxes Num: " << detected_boxes.size() << std::endl;

  delete yolox;
#endif
}

static void test_tensorrt()
{
#ifdef ENABLE_TENSORRT

    std::string engine_path = "../../..//examples/hub/trt/yolox_s_fp32.engine";
    std::string test_img_path = "../../..//examples/lite/resources/test_lite_yolox_2.jpg";
    std::string save_img_path = "../../..//examples/logs/test_lite_yolox_trt_4.jpg";

    // 2. Test Specific Engine TensorRT
    lite::trt::cv::detection::YoloX  *yolox =
            new lite::trt::cv::detection::YoloX (engine_path);

    std::vector<lite::types::Boxf> detected_boxes;
    cv::Mat img_bgr = cv::imread(test_img_path);
    yolox->detect(img_bgr, detected_boxes);

    lite::utils::draw_boxes_inplace(img_bgr, detected_boxes);

    cv::imwrite(save_img_path, img_bgr);

    delete yolox;
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

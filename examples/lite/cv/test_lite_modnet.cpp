//
// Created by DefTruth on 2022/3/30.
//

#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../examples/hub/onnx/cv/modnet_photographic_portrait_matting-512x512.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_matting_input.jpg";
  std::string test_bgr_path = "../../../examples/lite/resources/test_lite_matting_bgr.jpg";
  std::string save_fgr_path = "../../../examples/logs/test_lite_modnet_fgr.jpg";
  std::string save_pha_path = "../../../examples/logs/test_lite_modnet_pha.jpg";
  std::string save_merge_path = "../../../examples/logs/test_lite_modnet_merge.jpg";
  std::string save_swap_path = "../../../examples/logs/test_lite_modnet_swap.jpg";

  lite::cv::matting::MODNet *modnet =
      new lite::cv::matting::MODNet(onnx_path, 16); // 16 threads

  lite::types::MattingContent content;
  cv::Mat img_bgr = cv::imread(test_img_path);
  cv::Mat bgr_mat = cv::imread(test_bgr_path);

  // 1. image matting.
  modnet->detect(img_bgr, content, true, true);

  if (content.flag)
  {
    if (!content.fgr_mat.empty()) cv::imwrite(save_fgr_path, content.fgr_mat);
    if (!content.pha_mat.empty()) cv::imwrite(save_pha_path, content.pha_mat * 255.);
    if (!content.merge_mat.empty()) cv::imwrite(save_merge_path, content.merge_mat);
    // swap background
    cv::Mat out_mat;
    if (!content.fgr_mat.empty())
      lite::utils::swap_background(content.fgr_mat, content.pha_mat, bgr_mat, out_mat, true);
    else
      lite::utils::swap_background(img_bgr, content.pha_mat, bgr_mat, out_mat, false);

    if (!out_mat.empty())
    {
      cv::imwrite(save_swap_path, out_mat);
      std::cout << "Saved Swap Image Done!" << std::endl;
    }

    std::cout << "Default Version MGMatting Done!" << std::endl;
  }

  delete modnet;
}

static void test_onnxruntime()
{
#ifdef ENABLE_ONNXRUNTIME
  std::string onnx_path = "../../../examples/hub/onnx/cv/modnet_photographic_portrait_matting-512x512.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_matting_input.jpg";
  std::string test_bgr_path = "../../../examples/lite/resources/test_lite_matting_bgr.jpg";
  std::string save_fgr_path = "../../../examples/logs/test_lite_modnet_fgr_onnx.jpg";
  std::string save_pha_path = "../../../examples/logs/test_lite_modnet_pha_onnx.jpg";
  std::string save_merge_path = "../../../examples/logs/test_lite_modnet_merge_onnx.jpg";
  std::string save_swap_path = "../../../examples/logs/test_lite_modnet_swap_onnx.jpg";

  lite::onnxruntime::cv::matting::MODNet *modnet =
      new lite::onnxruntime::cv::matting::MODNet(onnx_path, 16); // 16 threads

  lite::types::MattingContent content;
  cv::Mat img_bgr = cv::imread(test_img_path);
  cv::Mat bgr_mat = cv::imread(test_bgr_path);

  // 1. image matting.
  modnet->detect(img_bgr, content, true, true);

  if (content.flag)
  {
    if (!content.fgr_mat.empty()) cv::imwrite(save_fgr_path, content.fgr_mat);
    if (!content.pha_mat.empty()) cv::imwrite(save_pha_path, content.pha_mat * 255.);
    if (!content.merge_mat.empty()) cv::imwrite(save_merge_path, content.merge_mat);
    // swap background
    cv::Mat out_mat;
    if (!content.fgr_mat.empty())
      lite::utils::swap_background(content.fgr_mat, content.pha_mat, bgr_mat, out_mat, true);
    else
      lite::utils::swap_background(img_bgr, content.pha_mat, bgr_mat, out_mat, false);

    if (!out_mat.empty())
    {
      cv::imwrite(save_swap_path, out_mat);
      std::cout << "Saved Swap Image Done!" << std::endl;
    }

    std::cout << "Default Version MGMatting Done!" << std::endl;
  }

  delete modnet;
#endif
}



static void test_tensorrt()
{
#ifdef ENABLE_TENSORRT
    std::string engine_path = "../../../examples/hub/trt/modnet_fp16.engine";
    std::string test_img_path = "../../../examples/lite/resources/test_lite_matting_input.jpg";
    std::string test_bgr_path = "../../../examples/lite/resources/test_lite_matting_bgr.jpg";
    std::string save_fgr_path = "../../../examples/logs/test_lite_modnet_fgr_trt.jpg";
    std::string save_pha_path = "../../../examples/logs/test_lite_modnet_pha_trt.jpg";
    std::string save_merge_path = "../../../examples/logs/test_lite_modnet_merge_trt.jpg";
    std::string save_swap_path = "../../../examples/logs/test_lite_modnet_swap_trt.jpg";


    lite::trt::cv::matting::MODNet  *modnet = new lite::trt::cv::matting::MODNet (engine_path);

    lite::types::MattingContent content;
    cv::Mat img_bgr = cv::imread(test_img_path);
    cv::Mat bgr_mat = cv::imread(test_bgr_path);

    // 1. image matting.
    modnet->detect(img_bgr, content, true, true);

    if (content.flag)
    {
        if (!content.fgr_mat.empty()) cv::imwrite(save_fgr_path, content.fgr_mat);
        if (!content.pha_mat.empty()) cv::imwrite(save_pha_path, content.pha_mat * 255.);
        if (!content.merge_mat.empty()) cv::imwrite(save_merge_path, content.merge_mat);
        // swap background
        cv::Mat out_mat;
        if (!content.fgr_mat.empty())
            lite::utils::swap_background(content.fgr_mat, content.pha_mat, bgr_mat, out_mat, true);
        else
            lite::utils::swap_background(img_bgr, content.pha_mat, bgr_mat, out_mat, false);

        if (!out_mat.empty())
        {
            cv::imwrite(save_swap_path, out_mat);
            std::cout << "Saved Swap Image Done!" << std::endl;
        }

        std::cout << "Default Version MGMatting Done!" << std::endl;
    }

    delete modnet;
#endif
}


static void test_mnn()
{
#ifdef ENABLE_MNN
  std::string mnn_path = "../../../examples/hub/mnn/cv/modnet_photographic_portrait_matting-512x512.mnn";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_matting_input.jpg";
  std::string test_bgr_path = "../../../examples/lite/resources/test_lite_matting_bgr.jpg";
  std::string save_fgr_path = "../../../examples/logs/test_lite_modnet_fgr_mnn.jpg";
  std::string save_pha_path = "../../../examples/logs/test_lite_modnet_pha_mnn.jpg";
  std::string save_merge_path = "../../../examples/logs/test_lite_modnet_merge_mnn.jpg";
  std::string save_swap_path = "../../../examples/logs/test_lite_modnet_swap_mnn.jpg";

  lite::mnn::cv::matting::MODNet *modnet =
      new lite::mnn::cv::matting::MODNet(mnn_path, 16); // 16 threads

  lite::types::MattingContent content;
  cv::Mat img_bgr = cv::imread(test_img_path);
  cv::Mat bgr_mat = cv::imread(test_bgr_path);

  // 1. image matting.
  modnet->detect(img_bgr, content, true, true);

  if (content.flag)
  {
    if (!content.fgr_mat.empty()) cv::imwrite(save_fgr_path, content.fgr_mat);
    if (!content.pha_mat.empty()) cv::imwrite(save_pha_path, content.pha_mat * 255.);
    if (!content.merge_mat.empty()) cv::imwrite(save_merge_path, content.merge_mat);
    // swap background
    cv::Mat out_mat;
    if (!content.fgr_mat.empty())
      lite::utils::swap_background(content.fgr_mat, content.pha_mat, bgr_mat, out_mat, true);
    else
      lite::utils::swap_background(img_bgr, content.pha_mat, bgr_mat, out_mat, false);

    if (!out_mat.empty())
    {
      cv::imwrite(save_swap_path, out_mat);
      std::cout << "Saved Swap Image Done!" << std::endl;
    }
    std::cout << "MNN Version MGMatting Done!" << std::endl;
  }

  delete modnet;
#endif
}

static void test_ncnn()
{
#ifdef ENABLE_NCNN
  std::string proto_path = "../../../examples/hub/ncnn/cv/modnet_photographic_portrait_matting-512x512.opt.param";
  std::string bin_path = "../../../examples/hub/ncnn/cv/modnet_photographic_portrait_matting-512x512.opt.bin";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_matting_input.jpg";
  std::string test_bgr_path = "../../../examples/lite/resources/test_lite_matting_bgr.jpg";
  std::string save_fgr_path = "../../../examples/logs/test_lite_modnet_fgr_ncnn.jpg";
  std::string save_pha_path = "../../../examples/logs/test_lite_modnet_pha_ncnn.jpg";
  std::string save_merge_path = "../../../examples/logs/test_lite_modnet_merge_ncnn.jpg";
  std::string save_swap_path = "../../../examples/logs/test_lite_modnet_swap_ncnn.jpg";

  lite::ncnn::cv::matting::MODNet *modnet =
      new lite::ncnn::cv::matting::MODNet(proto_path, bin_path, 16, 512, 512); // 16 threads

  lite::types::MattingContent content;
  cv::Mat img_bgr = cv::imread(test_img_path);
  cv::Mat bgr_mat = cv::imread(test_bgr_path);

  // 1. image matting.
  modnet->detect(img_bgr, content, true, true);

  if (content.flag)
  {
    if (!content.fgr_mat.empty()) cv::imwrite(save_fgr_path, content.fgr_mat);
    if (!content.pha_mat.empty()) cv::imwrite(save_pha_path, content.pha_mat * 255.);
    if (!content.merge_mat.empty()) cv::imwrite(save_merge_path, content.merge_mat);
    // swap background
    cv::Mat out_mat;
    if (!content.fgr_mat.empty())
      lite::utils::swap_background(content.fgr_mat, content.pha_mat, bgr_mat, out_mat, true);
    else
      lite::utils::swap_background(img_bgr, content.pha_mat, bgr_mat, out_mat, false);

    if (!out_mat.empty())
    {
      cv::imwrite(save_swap_path, out_mat);
      std::cout << "Saved Swap Image Done!" << std::endl;
    }
    std::cout << "NCNN Version MGMatting Done!" << std::endl;
  }

  delete modnet;
#endif
}

static void test_tnn()
{
#ifdef ENABLE_TNN
  std::string proto_path = "../../../examples/hub/tnn/cv/modnet_photographic_portrait_matting-512x512.tnnproto";
  std::string model_path = "../../../examples/hub/tnn/cv/modnet_photographic_portrait_matting-512x512.tnnmodel";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_matting_input.jpg";
  std::string test_bgr_path = "../../../examples/lite/resources/test_lite_matting_bgr.jpg";
  std::string save_fgr_path = "../../../examples/logs/test_lite_modnet_fgr_tnn.jpg";
  std::string save_pha_path = "../../../examples/logs/test_lite_modnet_pha_tnn.jpg";
  std::string save_merge_path = "../../../examples/logs/test_lite_modnet_merge_tnn.jpg";
  std::string save_swap_path = "../../../examples/logs/test_lite_modnet_swap_tnn.jpg";

  lite::tnn::cv::matting::MODNet *modnet =
      new lite::tnn::cv::matting::MODNet(proto_path, model_path, 16); // 16 threads

  lite::types::MattingContent content;
  cv::Mat img_bgr = cv::imread(test_img_path);
  cv::Mat bgr_mat = cv::imread(test_bgr_path);

  // 1. image matting.
  modnet->detect(img_bgr, content, true, true);

  if (content.flag)
  {
    if (!content.fgr_mat.empty()) cv::imwrite(save_fgr_path, content.fgr_mat);
    if (!content.pha_mat.empty()) cv::imwrite(save_pha_path, content.pha_mat * 255.);
    if (!content.merge_mat.empty()) cv::imwrite(save_merge_path, content.merge_mat);
    // swap background
    cv::Mat out_mat;
    if (!content.fgr_mat.empty())
      lite::utils::swap_background(content.fgr_mat, content.pha_mat, bgr_mat, out_mat, true);
    else
      lite::utils::swap_background(img_bgr, content.pha_mat, bgr_mat, out_mat, false);

    if (!out_mat.empty())
    {
      cv::imwrite(save_swap_path, out_mat);
      std::cout << "Saved Swap Image Done!" << std::endl;
    }
    std::cout << "TNN Version MGMatting Done!" << std::endl;
  }

  delete modnet;
#endif
}

static void test_lite()
{
//  test_default();
//  test_onnxruntime();
//  test_mnn();
//  test_ncnn();
//  test_tnn();
    test_tensorrt();
}

int main(__unused int argc, __unused char *argv[])
{
  test_lite();
  return 0;
}

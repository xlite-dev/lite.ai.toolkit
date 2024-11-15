//
// Created by wangzijian on 11/7/24.
//
#include "lite/lite.h"

#include "lite/trt/cv/trt_face_restoration_mt.h"

static void test_default()
{
#ifdef ENABLE_ONNXRUNTIME
    std::string onnx_path = "/home/lite.ai.toolkit/examples/hub/onnx/cv/gfpgan_1.4.onnx";
    std::string test_img_path = "/home/lite.ai.toolkit/trt_result.jpg";
    std::string save_img_path = "/home/lite.ai.toolkit/trt_result_final.jpg";

    // 1. Test Default Engine ONNXRuntime
    lite::cv::face::restoration::GFPGAN *face_restoration = new  lite::cv::face::restoration::GFPGAN(onnx_path);

    std::vector<cv::Point2f> face_landmark_5 = {
            cv::Point2f(569.092041f, 398.845886f),
            cv::Point2f(701.891724f, 399.156677f),
            cv::Point2f(634.767212f, 482.927216f),
            cv::Point2f(584.270996f, 543.294617f),
            cv::Point2f(684.877991f, 543.067078f)
    };
    cv::Mat img_bgr = cv::imread(test_img_path);

    face_restoration->detect(img_bgr,face_landmark_5,save_img_path);


    std::cout<<"face restoration detect done!"<<std::endl;

    delete face_restoration;
#endif
}




static void test_tensorrt()
{
#ifdef ENABLE_TENSORRT
    std::string engine_path = "/home/lite.ai.toolkit/examples/hub/trt/gfpgan_1.4_fp32.engine";
    std::string test_img_path = "/home/lite.ai.toolkit/trt_result.jpg";
    std::string save_img_path = "/home/lite.ai.toolkit/trt_facerestoration_mt_test111.jpg";

    // 1. Test Default Engine TensorRT
//    lite::trt::cv::face::restoration::TRTGFPGAN *face_restoration_trt = new  lite::trt::cv::face::restoration::TRTGFPGAN(engine_path);

    const int num_threads = 4;  // 使用4个线程
    auto face_restoration_trt = std::make_unique<trt_face_restoration_mt>(engine_path,4);

//    trt_face_restoration_mt *face_restoration_trt = new trt_face_restoration_mt(engine_path);


    // 2. 准备测试数据 - 这里假设我们要处理4张相同的图片作为示例
    std::vector<std::string> test_img_paths = {
            "/home/lite.ai.toolkit/trt_result.jpg",
            "/home/lite.ai.toolkit/trt_result_2.jpg",
            "/home/lite.ai.toolkit/trt_result_3.jpg",
            "/home/lite.ai.toolkit/trt_result_4.jpg"
    };

    std::vector<std::string> save_img_paths = {
            "/home/lite.ai.toolkit/trt_facerestoration_mt_thread1.jpg",
            "/home/lite.ai.toolkit/trt_facerestoration_mt_thread2.jpg",
            "/home/lite.ai.toolkit/trt_facerestoration_mt_thread3.jpg",
            "/home/lite.ai.toolkit/trt_facerestoration_mt_thread4.jpg"
    };

    std::vector<cv::Point2f> face_landmark_5 = {
            cv::Point2f(569.092041f, 398.845886f),
            cv::Point2f(701.891724f, 399.156677f),
            cv::Point2f(634.767212f, 482.927216f),
            cv::Point2f(584.270996f, 543.294617f),
            cv::Point2f(684.877991f, 543.067078f)
    };
//    cv::Mat img_bgr = cv::imread(test_img_path);
//
//    face_restoration_trt->detect_async(img_bgr,face_landmark_5,save_img_path);
//
//
//    std::cout<<"face restoration detect done!"<<std::endl;
//
//    delete face_restoration_trt;
    auto start_time = std::chrono::high_resolution_clock::now();

    for (size_t i=0; i < test_img_paths.size();++i){
        cv::Mat img_bgr = cv::imread(test_img_paths[i]);
        if (img_bgr.empty()) {
            std::cerr << "Failed to read image: " << test_img_paths[i] << std::endl;
            continue;
        }
        // 异步提交任务
        face_restoration_trt->detect_async(img_bgr, face_landmark_5, save_img_paths[i]);
        std::cout << "Submitted task " << i + 1 << " for processing" << std::endl;
    }

    // 6. 等待所有任务完成
    std::cout << "Waiting for all tasks to complete..." << std::endl;
    face_restoration_trt->wait_for_completion();

    // 7. 计算和输出总耗时
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "All tasks completed!" << std::endl;
    std::cout << "Total processing time: " << duration.count() << "ms" << std::endl;
    std::cout << "Average time per image: " << duration.count() / test_img_paths.size() << "ms" << std::endl;


#endif
}

int main(__unused int argc, __unused char *argv[])
{
//    test_default();
    test_tensorrt();
    return 0;
}
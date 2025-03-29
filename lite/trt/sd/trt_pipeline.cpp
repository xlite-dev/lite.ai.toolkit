#include "trt_pipeline.h"
using trtsd::TRTPipeline;

TRTPipeline::TRTPipeline(const std::string &_clip_engine_path, const std::string &_unet_engine_path,
                         const std::string &_vae_engine_path,bool is_low_vram):
         clip_engine_path(_clip_engine_path),
          unet_engine_path(_unet_engine_path),
          vae_engine_path(_vae_engine_path) {

    // 初始时不分配资源，将成员变量置为 nullptr
    // 如果不是低显存模式 那么分配资源
    if (!is_low_vram)
    {
        clip = std::make_unique<TRTClip>(clip_engine_path);
        unet = std::make_unique<TRTUNet>(unet_engine_path);
        vae = std::make_unique<TRTVae>(vae_engine_path);
        vaeencoder =  std::make_unique<TRTVaeEncoder>("/home/lite.ai.toolkit/examples/hub/trt/vae_encoder.engine");
    }


}

void TRTPipeline::inference(std::string prompt, std::string negative_prompt, std::string image_save_path, std::string scheduler_config_path,bool is_low_vram) {

    // 如果 clip 为空则初始化
    if (!clip) {
        clip = std::make_unique<TRTClip>(clip_engine_path);
    }
    std::vector<std::string> total_prompt = {std::move(prompt), std::move(negative_prompt)};
    std::vector<std::vector<float>> clip_output;
    clip->inference(total_prompt, clip_output);
    if (is_low_vram) {
        clip.reset(); // 使用后立即释放
    }
    // 如果 unet 为空则初始化
    if (!unet) {
        unet = std::make_unique<TRTUNet>(unet_engine_path);
    }
    std::vector<float> unet_output;
    unet->inference(clip_output, unet_output, scheduler_config_path);
    if (is_low_vram)
    {
        unet.reset(); // 使用后立即释放
    }

    // 如果 vae 为空则初始化
    if (!vae) {
        vae = std::make_unique<TRTVae>(vae_engine_path);
    }
    vae->inference(unet_output, std::move(image_save_path));
    if (is_low_vram)
    {
        vae.reset(); // 使用后立即释放
    }
}

#include "opencv2/opencv.hpp"

cv::Mat openImageAsPIL(const std::string& filename) {
    // 读取图像
    cv::Mat image = cv::imread(filename, cv::IMREAD_UNCHANGED);
    return image;
}

/**
 * 预处理图像函数 (简化版)
 * @param image 输入OpenCV图像
 * @return 预处理后的图像数据，格式为NCHW的一维向量
 */
#include <opencv2/opencv.hpp>
#include <vector>

std::vector<float> preprocess_image(const cv::Mat& image) {
    // 1. 调整尺寸为32的倍数
    int w = image.cols - (image.cols % 32);
    int h = image.rows - (image.rows % 32);

    // 调整图像大小
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(w, h));

    // 2. 转换为RGB (如果输入是BGR)
    cv::Mat rgb_image;
    cv::cvtColor(resized, rgb_image, cv::COLOR_BGR2RGB);

    // 3. 转换为浮点型并归一化到[0,1]
    cv::Mat float_image;
    rgb_image.convertTo(float_image, CV_32FC3, 1.0/255.0);

    // 4. 准备输出向量 (NCHW格式)
    std::vector<float> result(3 * h * w);

    // 5. 逐像素填充并应用 2*x-1 变换
    int idx = 0;
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                // 安全地访问像素值
                float pixel_value = float_image.at<cv::Vec3f>(y, x)[c];

                // 归一化到[-1,1]
                float normalized = 2.0f * pixel_value - 1.0f;

                // 存入结果向量
                result[idx++] = normalized;

                // 调试: 打印一些像素值 (仅少量样本)
                if (x < 3 && y < 3 && c == 0) {
                    std::cout << "位置(" << x << "," << y << ") 通道" << c
                              << " 原值:" << pixel_value
                              << " 归一化值:" << normalized << std::endl;
                }
            }
        }
    }

    return result;
}

// 将预处理结果写入文本文件
void save_vector_to_txt(const std::vector<float>& vec, const std::string& filename) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "无法创建文件: " << filename << std::endl;
        return;
    }
    // 设置输出精度
    outfile << std::fixed << std::setprecision(6);
    // 每行写入一个元素
    for (const auto &value: vec) {
        outfile << value << std::endl;
    }
    outfile.close();
    std::cout << "已将数据写入文件: " << filename << " (共 " << vec.size() << " 个元素)" << std::endl;
}


void TRTPipeline::inference(std::string prompt, std::string negative_prompt,std::string init_image_path, std::string image_save_path, std::string scheduler_config_path,bool is_low_vram) {


    vaeencoder =  std::make_unique<TRTVaeEncoder>("/home/lite.ai.toolkit/examples/hub/trt/vae_encoder.engine");

    cv::Mat temp = openImageAsPIL(init_image_path);

    if (temp.rows != 512 || temp.cols != 512){
        std::cout<<"[I] Resizing input image!"<<std::endl;
        cv::resize(temp,temp,cv::Size (512,512));
    }

    std::vector preprocess_result = preprocess_image(temp);

//    save_vector_to_txt(preprocess_result,"/home/lite.ai.toolkit/preprocess_result.txt");

    std::vector<float> output_latents;
    vaeencoder->inference(preprocess_result,output_latents);



    // 如果 clip 为空则初始化
    if (!clip) {
        clip = std::make_unique<TRTClip>(clip_engine_path);
    }

    prompt = "A fantasy landscape, trending on artstation";
    std::vector<std::string> total_prompt = {std::move(prompt), std::move(negative_prompt)};
    std::vector<std::vector<float>> clip_output;
    clip->inference(total_prompt, clip_output);
    if (is_low_vram) {
        clip.reset(); // 使用后立即释放
    }
    // 如果 unet 为空则初始化
    if (!unet) {
        unet = std::make_unique<TRTUNet>(unet_engine_path);
    }
    std::vector<float> unet_output;
//    unet->inference(clip_output, unet_output, scheduler_config_path);
    unet->inference(clip_output,output_latents,unet_output, scheduler_config_path);
    if (is_low_vram)
    {
        unet.reset(); // 使用后立即释放
    }

    // 如果 vae 为空则初始化
    if (!vae) {
        vae = std::make_unique<TRTVae>(vae_engine_path);
    }
    vae->inference(unet_output, std::move(image_save_path));
    if (is_low_vram)
    {
        vae.reset(); // 使用后立即释放
    }
}
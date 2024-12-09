
<div id="lite.ai.toolkit-Introduction"></div>  

<!--
![logo-v3](https://github.com/DefTruth/lite.ai.toolkit/assets/31974251/f99f5300-ece6-4572-8c4b-56b90e6e4d74)
![lite-ai-toolkit](https://github.com/user-attachments/assets/dc567d38-3fc4-4c9c-84de-3bfdf524aeab)
-->


![lite-ai-toolkit](https://github.com/user-attachments/assets/11568474-57e3-4ef7-96c0-d2ce7028bb5f)

<div align='center'>
  <img src=https://img.shields.io/badge/Linux-pass-brightgreen.svg >
  <img src=https://img.shields.io/badge/Device-GPU/CPU-yellow.svg >
  <img src=https://img.shields.io/badge/ONNXRuntime-1.17.1-turquoise.svg >
  <img src=https://img.shields.io/badge/MNN-2.8.2-hotpink.svg >
  <img src=https://img.shields.io/badge/TensorRT-10-turquoise.svg >
  <img src=https://img.shields.io/github/stars/DefTruth/lite.ai.toolkit.svg?style=social >
</div>   

🛠**Lite.Ai.ToolKit**: A lite C++ toolkit of 100+ Awesome AI models, such as [Object Detection](#lite.ai.toolkit-object-detection), [Face Detection](#lite.ai.toolkit-face-detection), [Face Recognition](#lite.ai.toolkit-face-recognition), [Segmentation](#lite.ai.toolkit-segmentation), [Matting](#lite.ai.toolkit-matting), etc. See [Model Zoo](#lite.ai.toolkit-Model-Zoo) and [ONNX Hub](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md), [MNN Hub](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.mnn.md), [TNN Hub](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.tnn.md), [NCNN Hub](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.ncnn.md). Welcome to 🌟👆🏻star this repo to support me, many thanks ~ 🎉🎉

<div align='center'>
  <img src='https://github.com/DefTruth/lite.ai.toolkit/assets/31974251/5b28aed1-e207-4256-b3ea-3b52f9e68aed' height="80px" width="80px">
  <img src='https://github.com/DefTruth/lite.ai.toolkit/assets/31974251/28274741-8745-4665-abff-3a384b75f7fa' height="80px" width="80px">
  <img src='https://github.com/DefTruth/lite.ai.toolkit/assets/31974251/c802858c-6899-4246-8839-5721c43faffe' height="80px" width="80px">
  <img src='https://github.com/DefTruth/lite.ai.toolkit/assets/31974251/20a18d56-297c-4c72-8153-76d4380fc9ec' height="80px" width="80px">
  <img src='https://github.com/DefTruth/lite.ai.toolkit/assets/31974251/f4dd5263-8514-4bb0-a0dd-dbe532481aff' height="80px" width="80px">
  <img src='https://github.com/DefTruth/lite.ai.toolkit/assets/31974251/b6a431d2-225b-416b-8a1e-cf9617d79a63' height="80px" width="80px">
  <img src='https://github.com/DefTruth/lite.ai.toolkit/assets/31974251/84d3ed6a-b711-4c0a-8e92-a2da05a0d04e' height="80px" width="80px">
  <img src='https://github.com/DefTruth/lite.ai.toolkit/assets/31974251/157b9e11-fc92-445b-ae0d-0d859c8663ee' height="80px" width="80px">
  <img src='https://github.com/DefTruth/lite.ai.toolkit/assets/31974251/ef0eeabe-6dbe-4837-9aad-b806a8398697' height="80px" width="80px">  
</div>  

## News 👇👇
Most of my time now is focused on **LLM/VLM** Inference. Please check 📖[Awesome-LLM-Inference](https://github.com/DefTruth/Awesome-LLM-Inference)  ![](https://img.shields.io/github/stars/DefTruth/Awesome-LLM-Inference.svg?style=social) and 📖[CUDA-Learn-Notes](https://github.com/DefTruth/CUDA-Learn-Notes)  ![](https://img.shields.io/github/stars/DefTruth/CUDA-Learn-Notes.svg?style=social) for more details. Now, [lite.ai.toolkit](https://github.com/DefTruth/lite.ai.toolkit) ![](https://img.shields.io/github/stars/DefTruth/lite.ai.toolkit.svg?style=social) is mainly maintained by 🎉[@wangzijian1010](https://github.com/wangzijian1010).

## Citations 🎉🎉
```BibTeX
@misc{lite.ai.toolkit@2021,
  title={lite.ai.toolkit: A lite C++ toolkit of 100+ Awesome AI models.},
  url={https://github.com/DefTruth/lite.ai.toolkit},
  note={Open-source software available at https://github.com/DefTruth/lite.ai.toolkit},
  author={DefTruth, wangzijian1010 etc},
  year={2021}
}
```

## Features 👏👋

* **Simply and User friendly.** Simply and Consistent syntax like **lite::cv::Type::Class**, see [examples](#lite.ai.toolkit-Examples-for-Lite.AI.ToolKit).
* **Minimum Dependencies.** Only **OpenCV** and **ONNXRuntime** are required by default, see [build](#lite.ai.toolkit-Build-Lite.AI.ToolKit).
* **Many Models Supported.** **[300+](#lite.ai.toolkit-Supported-Models-Matrix)** C++ implementations and **[500+](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md)** weights 👉 **[Supported-Matrix](#lite.ai.toolkit-Supported-Models-Matrix)**.

## Build 👇👇
Download prebuilt lite.ai.toolkit library from [tag/v0.2.0](https://github.com/DefTruth/lite.ai.toolkit/releases/tag/v0.2.0), or just build it from source:
```shell
git clone --depth=1 https://github.com/DefTruth/lite.ai.toolkit.git  # latest
cd lite.ai.toolkit && sh ./build.sh # >= 0.2.0, support Linux only, tested on Ubuntu 20.04.6 LTS
```

## Quick Start 🌟🌟
<div id="lite.ai.toolkit-Quick-Start"></div>

#### Example0: Object Detection using [YOLOv5](https://github.com/ultralytics/yolov5). Download model from Model-Zoo[<sup>2</sup>](#lite.ai.toolkit-2).
```c++
#include "lite/lite.h"

int main(int argc, char *argv[]) {
  std::string onnx_path = "yolov5s.onnx";
  std::string test_img_path = "test_yolov5.jpg";
  std::string save_img_path = "test_results.jpg";

  auto *yolov5 = new lite::cv::detection::YoloV5(onnx_path); 
  std::vector<lite::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  yolov5->detect(img_bgr, detected_boxes);
  
  lite::utils::draw_boxes_inplace(img_bgr, detected_boxes);
  cv::imwrite(save_img_path, img_bgr);  
  delete yolov5;
  return 0;
}
```
You can download the prebuilt lite.ai.tooklit library and test resources from [tag/v0.2.0](https://github.com/DefTruth/lite.ai.toolkit/releases/tag/v0.2.0).
```bash
export LITE_AI_TAG_URL=https://github.com/DefTruth/lite.ai.toolkit/releases/download/v0.2.0
wget ${LITE_AI_TAG_URL}/lite-ort1.17.1+ocv4.9.0+ffmpeg4.2.2-linux-x86_64.tgz
wget ${LITE_AI_TAG_URL}/yolov5s.onnx && wget ${LITE_AI_TAG_URL}/test_yolov5.jpg
```
#### 🎉🎉[TensorRT](https://github.com/NVIDIA/TensorRT): Boost inference performance with NVIDIA GPU via TensorRT.
Run `bash ./build.sh tensorrt` to build lite.ai.toolkit with TensorRT support, and then test yolov5 with the codes below. NOTE: lite.ai.toolkit need TensorRT 10.x (or later) and CUDA 12.x (or later). Please check [build.sh](./build.sh), [tensorrt-linux-x86_64-install.zh.md](./docs/tensorrt/tensorrt-linux-x86_64.zh.md), [test_lite_yolov5.cpp](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_yolov5.cpp) and [NVIDIA/TensorRT](https://github.com/NVIDIA/TensorRT) for more details.
```c++
// trtexec --onnx=yolov5s.onnx --saveEngine=yolov5s.engine
auto *yolov5 = new lite::trt::cv::detection::YOLOV5(engine_path);
```

## Quick Setup 👀

To quickly setup `lite.ai.toolkit`, you can follow the `CMakeLists.txt` listed as belows. 👇👀

```cmake
set(lite.ai.toolkit_DIR YOUR-PATH-TO-LITE-INSTALL)
find_package(lite.ai.toolkit REQUIRED PATHS ${lite.ai.toolkit_DIR})
add_executable(lite_yolov5 test_lite_yolov5.cpp)
target_link_libraries(lite_yolov5 ${lite.ai.toolkit_LIBS})
```

## Mixed with MNN or ONNXRuntime 👇👇
The goal of lite.ai.toolkit is not to abstract on top of MNN and ONNXRuntime. So, you can use lite.ai.toolkit mixed with MNN(`-DENABLE_MNN=ON, default OFF`) or ONNXRuntime(`-DENABLE_ONNXRUNTIME=ON, default ON`). The lite.ai.toolkit installation package contains complete MNN and ONNXRuntime. The workflow may looks like:
```C++
#include "lite/lite.h"
// 0. use yolov5 from lite.ai.toolkit to detect objs.
auto *yolov5 = new lite::cv::detection::YoloV5(onnx_path);
// 1. use OnnxRuntime or MNN to implement your own classfier.
interpreter = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(mnn_path));
// or: session = new Ort::Session(ort_env, onnx_path, session_options);
classfier = interpreter->createSession(schedule_config);
// 2. then, classify the detected objs use your own classfier ...
```
The included headers of MNN and ONNXRuntime can be found at [mnn_config.h](./lite/mnn/core/mnn_config.h) and [ort_config.h](./lite/ort/core/ort_config.h). 

<details>
<summary> 🔑️ Check the detailed Quick Start！Click here! </summary>    

### Download resources

You can download the prebuilt lite.ai.tooklit library and test resources from [tag/v0.2.0](https://github.com/DefTruth/lite.ai.toolkit/releases/tag/v0.2.0).
```bash
export LITE_AI_TAG_URL=https://github.com/DefTruth/lite.ai.toolkit/releases/download/v0.2.0
wget ${LITE_AI_TAG_URL}/lite-ort1.17.1+ocv4.9.0+ffmpeg4.2.2-linux-x86_64.tgz
wget ${LITE_AI_TAG_URL}/yolov5s.onnx && wget ${LITE_AI_TAG_URL}/test_yolov5.jpg
tar -zxvf lite-ort1.17.1+ocv4.9.0+ffmpeg4.2.2-linux-x86_64.tgz
```
### Write test code

write YOLOv5 example codes and name it `test_lite_yolov5.cpp`:
```c++
#include "lite/lite.h"

int main(int argc, char *argv[]) {
  std::string onnx_path = "yolov5s.onnx";
  std::string test_img_path = "test_yolov5.jpg";
  std::string save_img_path = "test_results.jpg";

  auto *yolov5 = new lite::cv::detection::YoloV5(onnx_path); 
  std::vector<lite::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  yolov5->detect(img_bgr, detected_boxes);
  
  lite::utils::draw_boxes_inplace(img_bgr, detected_boxes);
  cv::imwrite(save_img_path, img_bgr);  
  delete yolov5;
  return 0;
}
```

### Setup CMakeLists.txt 
```cmake
cmake_minimum_required(VERSION 3.10)
project(lite_yolov5)
set(CMAKE_CXX_STANDARD 17)

set(lite.ai.toolkit_DIR YOUR-PATH-TO-LITE-INSTALL)
find_package(lite.ai.toolkit REQUIRED PATHS ${lite.ai.toolkit_DIR})
if (lite.ai.toolkit_Found)
    message(STATUS "lite.ai.toolkit_INCLUDE_DIRS: ${lite.ai.toolkit_INCLUDE_DIRS}")
    message(STATUS "        lite.ai.toolkit_LIBS: ${lite.ai.toolkit_LIBS}")
    message(STATUS "   lite.ai.toolkit_LIBS_DIRS: ${lite.ai.toolkit_LIBS_DIRS}")
endif()
add_executable(lite_yolov5 test_lite_yolov5.cpp)
target_link_libraries(lite_yolov5 ${lite.ai.toolkit_LIBS})
```
### Build example

```bash
mkdir build && cd build && cmake .. && make -j1
```
Then, export the lib paths to `LD_LIBRARY_PATH` which listed by `lite.ai.toolkit_LIBS_DIRS`. 
```bash
export LD_LIBRARY_PATH=YOUR-PATH-TO-LITE-INSTALL/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=YOUR-PATH-TO-LITE-INSTALL/third_party/opencv/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=YOUR-PATH-TO-LITE-INSTALL/third_party/onnxruntime/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=YOUR-PATH-TO-LITE-INSTALL/third_party/MNN/lib:$LD_LIBRARY_PATH # if -DENABLE_MNN=ON
```

### Run binary:
```bash
cp ../yolov5s.onnx ../test_yolov.jpg .
./lite_yolov5
```
The output logs:
```bash
LITEORT_DEBUG LogId: ../examples/hub/onnx/cv/yolov5s.onnx
=============== Input-Dims ==============
Name: images
Dims: 1
Dims: 3
Dims: 640
Dims: 640
=============== Output-Dims ==============
Output: 0 Name: pred Dim: 0 :1
Output: 0 Name: pred Dim: 1 :25200
Output: 0 Name: pred Dim: 2 :85
Output: 1 Name: output2 Dim: 0 :1
......
Output: 3 Name: output4 Dim: 1 :3
Output: 3 Name: output4 Dim: 2 :20
Output: 3 Name: output4 Dim: 3 :20
Output: 3 Name: output4 Dim: 4 :85
========================================
detected num_anchors: 25200
generate_bboxes num: 48
```
</details>
  
<div id="lite.ai.toolkit-Supported-Models-Matrix"></div>

<!--
<details>
<summary> 🔑️ Supported Models Matrix！Click here! </summary>    
-->

## Supported Models Matrix

* / = not supported now.
* ✅ = known work and official supported now.
* ✔️ = known work, but unofficial supported now.
* ❔ = in my plan, but not coming soon, maybe a few months later.

### NVIDIA GPU Inference: TensorRT

|Class|Class|Class|Class|Class| System | Engine |  
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|      
|✅[YOLOv5](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_yolov5.cpp)|✅[YOLOv6](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_yolov6.cpp)|✅[YOLOv8](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_yolov8.cpp)|✅[YOLOv8Face](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_yolov8face.cpp)|✅[YOLOv5Face](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_yolo5face.cpp)|  Linux | TensorRT |  
|✅[YOLOX](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_yolox.cpp)|✅[YOLOv5BlazeFace](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_yolov5_blazeface.cpp) |✅[StableDiffusion](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/sd/test_lite_sd_pipeline.cpp)| / |  / |  Linux | TensorRT |


### CPU Inference: ONNXRuntime, MNN, NCNN and TNN
|                                                       Class                                                       | Size  |       Type       |                                                          Demo                                                          | ONNXRuntime | MNN | NCNN | TNN | Linux | MacOS | Windows | Android |
|:-----------------------------------------------------------------------------------------------------------------:|:-----:|:----------------:|:----------------------------------------------------------------------------------------------------------------------:|:-----------:|:---:|:----:|:---:|:-----:|:-----:|:-------:|:-------:|
|                                  [YoloV5](https://github.com/ultralytics/yolov5)                                  |  28M  |   *detection*    |          [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_yolov5.cpp)           |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|         [YoloV3](https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/yolov3)          | 236M  |   *detection*    |          [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_yolov3.cpp)           |      ✅      |  /  |  /   |  /  |   ✅   |  ✔️   |   ✔️    |    /    |
|     [TinyYoloV3](https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/tiny-yolov3)     |  33M  |   *detection*    |        [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_tiny_yolov3.cpp)        |      ✅      |  /  |  /   |  /  |   ✅   |  ✔️   |   ✔️    |    /    |
|                              [YoloV4](https://github.com/argusswift/YOLOv4-pytorch)                               | 176M  |   *detection*    |          [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_yolov4.cpp)           |      ✅      |  /  |  /   |  /  |   ✅   |  ✔️   |   ✔️    |    /    |
|            [SSD](https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/ssd)             |  76M  |   *detection*    |            [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_ssd.cpp)            |      ✅      |  /  |  /   |  /  |   ✅   |  ✔️   |   ✔️    |    /    |
| [SSDMobileNetV1](https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/ssd-mobilenetv1) |  27M  |   *detection*    |      [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_ssd_mobilenetv1.cpp)      |      ✅      |  /  |  /   |  /  |   ✅   |  ✔️   |   ✔️    |    /    |
|                              [YoloX](https://github.com/Megvii-BaseDetection/YOLOX)                               | 3.5M  |   *detection*    |           [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_yolox.cpp)           |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                        [TinyYoloV4VOC](https://github.com/bubbliiiing/yolov4-tiny-pytorch)                        |  22M  |   *detection*    |      [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_tiny_yolov4_voc.cpp)      |      ✅      |  /  |  /   |  /  |   ✅   |  ✔️   |   ✔️    |    /    |
|                       [TinyYoloV4COCO](https://github.com/bubbliiiing/yolov4-tiny-pytorch)                        |  22M  |   *detection*    |     [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_tiny_yolov4_coco.cpp)      |      ✅      |  /  |  /   |  /  |   ✅   |  ✔️   |   ✔️    |    /    |
|                                   [YoloR](https://github.com/WongKinYiu/yolor)                                    |  39M  |   *detection*    |           [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_yolor.cpp)           |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                            [ScaledYoloV4](https://github.com/WongKinYiu/ScaledYOLOv4)                             | 270M  |   *detection*    |       [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_scaled_yolov4.cpp)       |      ✅      |  /  |  /   |  /  |   ✅   |  ✔️   |   ✔️    |    /    |
|                    [EfficientDet](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch)                    |  15M  |   *detection*    |       [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_efficientdet.cpp)        |      ✅      |  /  |  /   |  /  |   ✅   |  ✔️   |   ✔️    |    /    |
|                   [EfficientDetD7](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch)                   | 220M  |   *detection*    |      [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_efficientdet_d7.cpp)      |      ✅      |  /  |  /   |  /  |   ✅   |  ✔️   |   ✔️    |    /    |
|                   [EfficientDetD8](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch)                   | 322M  |   *detection*    |      [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_efficientdet_d8.cpp)      |      ✅      |  /  |  /   |  /  |   ✅   |  ✔️   |   ✔️    |    /    |
|                                     [YOLOP](https://github.com/hustvl/YOLOP)                                      |  30M  |   *detection*    |           [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_yolop.cpp)           |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                                  [NanoDet](https://github.com/RangiLyu/nanodet)                                   | 1.1M  |   *detection*    |          [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_nanodet.cpp)          |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                                [NanoDetPlus](https://github.com/RangiLyu/nanodet)                                 | 4.5M  |   *detection*    |       [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_nanodet_plus.cpp)        |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                               [NanoDetEffi...](https://github.com/RangiLyu/nanodet)                               |  12M  |   *detection*    | [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_nanodet_efficientnet_lite.cpp) |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                          [YoloX_V_0_1_1](https://github.com/Megvii-BaseDetection/YOLOX)                           | 3.5M  |   *detection*    |       [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_yolox_v0.1.1.cpp)        |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                               [YoloV5_V_6_0](https://github.com/ultralytics/yolov5)                               | 7.5M  |   *detection*    |        [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_yolov5_v6.0.cpp)        |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|         [GlintArcFace](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch)          |  92M  |     *faceid*     |       [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_glint_arcface.cpp)       |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|         [GlintCosFace](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch)          |  92M  |     *faceid*     |       [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_glint_cosface.cpp)       |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    /    |
|          [GlintPartialFC](https://github.com/deepinsight/insightface/tree/master/recognition/partial_fc)          | 170M  |     *faceid*     |     [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_glint_partial_fc.cpp)      |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    /    |
|                              [FaceNet](https://github.com/timesler/facenet-pytorch)                               |  89M  |     *faceid*     |          [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_facenet.cpp)          |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    /    |
|                         [FocalArcFace](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch)                          | 166M  |     *faceid*     |       [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_focal_arcface.cpp)       |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    /    |
|                       [FocalAsiaArcFace](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch)                        | 166M  |     *faceid*     |    [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_focal_asia_arcface.cpp)     |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    /    |
|                [TencentCurricularFace](https://github.com/Tencent/TFace/tree/master/tasks/distfc)                 | 249M  |     *faceid*     |  [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_tencent_curricular_face.cpp)  |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    /    |
|                    [TencentCifpFace](https://github.com/Tencent/TFace/tree/master/tasks/cifp)                     | 130M  |     *faceid*     |     [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_tencent_cifp_face.cpp)     |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    /    |
|                        [CenterLossFace](https://github.com/louis-she/center-loss.pytorch)                         | 280M  |     *faceid*     |     [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_center_loss_face.cpp)      |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    /    |
|                           [SphereFace](https://github.com/clcarwin/sphereface_pytorch)                            |  80M  |     *faceid*     |        [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_sphere_face.cpp)        |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    /    |
|                              [PoseRobustFace](https://github.com/penincillin/DREAM)                               |  92M  |     *faceid*     |     [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_pose_robust_face.cpp)      |      ✅      |  /  |  /   |  /  |   ✅   |  ✔️   |   ✔️    |    /    |
|                            [NaivePoseRobustFace](https://github.com/penincillin/DREAM)                            |  43M  |     *faceid*     |  [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_naive_pose_robust_face.cpp)   |      ✅      |  /  |  /   |  /  |   ✅   |  ✔️   |   ✔️    |    /    |
|                        [MobileFaceNet](https://github.com/Xiaoccer/MobileFaceNet_Pytorch)                         | 3.8M  |     *faceid*     |      [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_mobile_facenet.cpp)       |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                        [CavaGhostArcFace](https://github.com/cavalleria/cavaface.pytorch)                         |  15M  |     *faceid*     |    [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_cava_ghost_arcface.cpp)     |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                        [CavaCombinedFace](https://github.com/cavalleria/cavaface.pytorch)                         | 250M  |     *faceid*     |    [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_cava_combined_face.cpp)     |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    /    |
|                    [MobileSEFocalFace](https://github.com/grib0ed0v/face_recognition.pytorch)                     | 4.5M  |     *faceid*     |    [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_mobilese_focal_face.cpp)    |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                       [RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting)                        |  14M  |    *matting*     |            [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_rvm.cpp)            |      ✅      |  ✅  |  /   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                               [MGMatting](https://github.com/yucornetto/MGMatting)                                | 113M  |    *matting*     |        [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_mg_matting.cpp)         |      ✅      |  ✅  |  /   |  ✅  |   ✅   |  ✔️   |   ✔️    |    /    |
|                                    [MODNet](https://github.com/ZHKKKe/MODNet)                                     |  24M  |    *matting*     |          [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_modnet.cpp)           |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    /    |
|                                   [MODNetDyn](https://github.com/ZHKKKe/MODNet)                                   |  24M  |    *matting*     |        [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_modnet_dyn.cpp)         |      ✅      |  /  |  /   |  /  |   ✅   |  ✔️   |   ✔️    |    /    |
|                      [BackgroundMattingV2](https://github.com/PeterL1n/BackgroundMattingV2)                       |  20M  |    *matting*     |    [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_backgroundmattingv2.cpp)    |      ✅      |  ✅  |  /   |  ✅  |   ✅   |  ✔️   |   ✔️    |    /    |
|                     [BackgroundMattingV2Dyn](https://github.com/PeterL1n/BackgroundMattingV2)                     |  20M  |    *matting*     |  [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_backgroundmattingv2_dyn.cpp)  |      ✅      |  /  |  /   |  /  |   ✅   |  ✔️   |   ✔️    |    /    |
|                [UltraFace](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB)                 | 1.1M  |  *face::detect*  |         [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_ultraface.cpp)         |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                            [RetinaFace](https://github.com/biubug6/Pytorch_Retinaface)                            | 1.6M  |  *face::detect*  |        [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_retinaface.cpp)         |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                             [FaceBoxes](https://github.com/zisianw/FaceBoxes.PyTorch)                             | 3.8M  |  *face::detect*  |         [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_faceboxes.cpp)         |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                             [FaceBoxesV2](https://github.com/jhb86253817/FaceBoxesV2)                             | 3.8M  |  *face::detect*  |        [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_faceboxesv2.cpp)        |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                 [SCRFD](https://github.com/deepinsight/insightface/blob/master/detection/scrfd/)                  | 2.5M  |  *face::detect*  |           [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_scrfd.cpp)           |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |  
|                              [YOLO5Face](https://github.com/deepcam-cn/yolov5-face)                               | 4.8M  |  *face::detect*  |         [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_yolo5face.cpp)         |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |  
|                            [PFLD](https://github.com/Hsintao/pfld_106_face_landmarks)                             | 1.0M  |  *face::align*   |           [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_pfld.cpp)            |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                               [PFLD98](https://github.com/polarisZhao/PFLD-pytorch)                               | 4.8M  |  *face::align*   |          [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_pfld98.cpp)           |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                         [MobileNetV268](https://github.com/cunjian/pytorch_face_landmark)                         | 9.4M  |  *face::align*   |      [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_mobilenetv2_68.cpp)       |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                        [MobileNetV2SE68](https://github.com/cunjian/pytorch_face_landmark)                        |  11M  |  *face::align*   |     [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_mobilenetv2_se_68.cpp)     |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                            [PFLD68](https://github.com/cunjian/pytorch_face_landmark)                             | 2.8M  |  *face::align*   |          [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_pfld68.cpp)           |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                         [FaceLandmark1000](https://github.com/Single430/FaceLandmark1000)                         | 2.0M  |  *face::align*   |    [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_face_landmarks_1000.cpp)    |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                                 [PIPNet98](https://github.com/jhb86253817/PIPNet)                                 | 44.0M |  *face::align*   |         [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_pipnet98.cpp)          |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                                 [PIPNet68](https://github.com/jhb86253817/PIPNet)                                 | 44.0M |  *face::align*   |         [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_pipnet68.cpp)          |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                                 [PIPNet29](https://github.com/jhb86253817/PIPNet)                                 | 44.0M |  *face::align*   |         [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_pipnet29.cpp)          |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                                 [PIPNet19](https://github.com/jhb86253817/PIPNet)                                 | 44.0M |  *face::align*   |         [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_pipnet19.cpp)          |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                           [FSANet](https://github.com/omasaht/headpose-fsanet-pytorch)                            | 1.2M  |   *face::pose*   |          [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_fsanet.cpp)           |      ✅      |  ✅  |  /   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|            [AgeGoogleNet](https://github.com/onnx/models/tree/master/vision/body_analysis/age_gender)             |  23M  |   *face::attr*   |       [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_age_googlenet.cpp)       |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|           [GenderGoogleNet](https://github.com/onnx/models/tree/master/vision/body_analysis/age_gender)           |  23M  |   *face::attr*   |     [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_gender_googlenet.cpp)      |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|         [EmotionFerPlus](https://github.com/onnx/models/blob/master/vision/body_analysis/emotion_ferplus)         |  33M  |   *face::attr*   |      [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_emotion_ferplus.cpp)      |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|              [VGG16Age](https://github.com/onnx/models/tree/master/vision/body_analysis/age_gender)               | 514M  |   *face::attr*   |         [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_vgg16_age.cpp)         |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    /    |
|             [VGG16Gender](https://github.com/onnx/models/tree/master/vision/body_analysis/age_gender)             | 512M  |   *face::attr*   |       [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_vgg16_gender.cpp)        |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    /    |
|                               [SSRNet](https://github.com/oukohou/SSR_Net_Pytorch)                                | 190K  |   *face::attr*   |          [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_ssrnet.cpp)           |      ✅      |  ✅  |  /   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                  [EfficientEmotion7](https://github.com/HSE-asavchenko/face-emotion-recognition)                  |  15M  |   *face::attr*   |    [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_efficient_emotion7.cpp)     |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                  [EfficientEmotion8](https://github.com/HSE-asavchenko/face-emotion-recognition)                  |  15M  |   *face::attr*   |    [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_efficient_emotion8.cpp)     |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                   [MobileEmotion7](https://github.com/HSE-asavchenko/face-emotion-recognition)                    |  13M  |   *face::attr*   |      [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_mobile_emotion7.cpp)      |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                   [ReXNetEmotion7](https://github.com/HSE-asavchenko/face-emotion-recognition)                    |  30M  |   *face::attr*   |      [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_rexnet_emotion7.cpp)      |      ✅      |  ✅  |  /   |  ✅  |   ✅   |  ✔️   |   ✔️    |    /    |
|     [EfficientNetLite4](https://github.com/onnx/models/blob/master/vision/classification/efficientnet-lite4)      |  49M  | *classification* |    [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_efficientnet_lite4.cpp)     |      ✅      |  ✅  |  /   |  ✅  |   ✅   |  ✔️   |   ✔️    |    /    |
|            [ShuffleNetV2](https://github.com/onnx/models/blob/master/vision/classification/shufflenet)            | 8.7M  | *classification* |       [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_shufflenetv2.cpp)        |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                          [DenseNet121](https://pytorch.org/hub/pytorch_vision_densenet/)                          | 30.7M | *classification* |         [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_densenet.cpp)          |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    /    |
|                           [GhostNet](https://pytorch.org/hub/pytorch_vision_ghostnet/)                            |  20M  | *classification* |         [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_ghostnet.cpp)          |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                            [HdrDNet](https://pytorch.org/hub/pytorch_vision_hardnet//)                            |  13M  | *classification* |          [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_hardnet.cpp)          |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                             [IBNNet](https://pytorch.org/hub/pytorch_vision_ibnnet/)                              |  97M  | *classification* |          [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_ibnnet.cpp)           |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    /    |
|                        [MobileNetV2](https://pytorch.org/hub/pytorch_vision_mobilenet_v2/)                        |  13M  | *classification* |        [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_mobilenetv2.cpp)        |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                             [ResNet](https://pytorch.org/hub/pytorch_vision_resnet/)                              |  44M  | *classification* |          [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_resnet.cpp)           |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    /    |
|                            [ResNeXt](https://pytorch.org/hub/pytorch_vision_resnext/)                             |  95M  | *classification* |          [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_resnext.cpp)          |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    /    |
|                 [DeepLabV3ResNet101](https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/)                 | 232M  |  *segmentation*  |    [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_deeplabv3_resnet101.cpp)    |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    /    |
|                       [FCNResNet101](https://pytorch.org/hub/pytorch_vision_fcn_resnet101/)                       | 207M  |  *segmentation*  |       [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_fcn_resnet101.cpp)       |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    /    |
|      [FastStyleTransfer](https://github.com/onnx/models/blob/master/vision/style_transfer/fast_neural_style)      | 6.4M  |     *style*      |    [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_fast_style_transfer.cpp)    |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                              [Colorizer](https://github.com/richzhang/colorization)                               | 123M  |  *colorization*  |         [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_colorizer.cpp)         |      ✅      |  ✅  |  /   |  ✅  |   ✅   |  ✔️   |   ✔️    |    /    |
|                             [SubPixelCNN](https://github.com/niazwazir/SUB_PIXEL_CNN)                             | 234K  |   *resolution*   |       [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_subpixel_cnn.cpp)        |      ✅      |  ✅  |  /   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                             [SubPixelCNN](https://github.com/niazwazir/SUB_PIXEL_CNN)                             | 234K  |   *resolution*   |       [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_subpixel_cnn.cpp)        |      ✅      |  ✅  |  /   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                           [InsectDet](https://github.com/quarrying/quarrying-insect-id)                           |  27M  |   *detection*    |         [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_insectdet.cpp)         |      ✅      |  ✅  |  /   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                           [InsectID](https://github.com/quarrying/quarrying-insect-id)                            |  22M  | *classification* |         [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_insectid.cpp)          |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |   ✔️    |    ❔    |
|                            [PlantID](https://github.com/quarrying/quarrying-plant-id)                             |  30M  | *classification* |          [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_plantid.cpp)          |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |   ✔️    |    ❔    |
|                           [YOLOv5BlazeFace](https://github.com/deepcam-cn/yolov5-face)                            | 3.4M  |  *face::detect*  |     [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_yolov5_blazeface.cpp)      |      ✅      |  ✅  |  /   |  /  |   ✅   |  ✔️   |   ✔️    |    ❔    |  
|                      [YoloV5_V_6_1](https://github.com/ultralytics/yolov5/releases/tag/v6.1)                      | 7.5M  |   *detection*    |        [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_yolov5_v6.1.cpp)        |      ✅      |  ✅  |  /   |  /  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                             [HeadSeg](https://github.com/minivision-ai/photo2cartoon)                             |  31M  |  *segmentation*  |         [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_head_seg.cpp)          |      ✅      |  ✅  |  /   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                       [FemalePhoto2Cartoon](https://github.com/minivision-ai/photo2cartoon)                       |  15M  |     *style*      |   [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_female_photo2cartoon.cpp)    |      ✅      |  ✅  |  /   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                    [FastPortraitSeg](https://github.com/YexingWan/Fast-Portrait-Segmentation)                     | 400k  |  *segmentation*  |     [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_fast_portrait_seg.cpp)     |      ✅      |  ✅  |  /   |  /  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                     [PortraitSegSINet](https://github.com/clovaai/ext_portrait_segmentation)                      | 380k  |  *segmentation*  |    [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_portrait_seg_sinet.cpp)     |      ✅      |  ✅  |  /   |  /  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                  [PortraitSegExtremeC3Net](https://github.com/clovaai/ext_portrait_segmentation)                  | 180k  |  *segmentation*  | [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_portrait_seg_extremec3net.cpp) |      ✅      |  ✅  |  /   |  /  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                                 [FaceHairSeg](https://github.com/kampta/face-seg)                                 |  18M  |  *segmentation*  |       [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_face_hair_seg.cpp)       |      ✅      |  ✅  |  /   |  /  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                       [HairSeg](https://github.com/akirasosa/mobile-semantic-segmentation)                        |  18M  |  *segmentation*  |         [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_hair_seg.cpp)          |      ✅      |  ✅  |  /   |  /  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                [MobileHumanMatting](https://github.com/lizhengwei1992/mobile_phone_human_matting)                 |  3M   |    *matting*     |   [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_mobile_human_matting.cpp)    |      ✅      |  ✅  |  /   |  /  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                 [MobileHairSeg](https://github.com/wonbeomjang/mobile-hair-segmentation-pytorch)                  |  14M  |  *segmentation*  |      [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_mobile_hair_seg.cpp)      |      ✅      |  ✅  |  /   |  /  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                                    [YOLOv6](https://github.com/meituan/YOLOv6)                                    |  17M  |   *detection*    |          [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_yolov6.cpp)           |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                     [FaceParsingBiSeNet](https://github.com/zllrunning/face-parsing.PyTorch)                      |  50M  |  *segmentation*  |   [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_face_parsing_bisenet.cpp)    |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                    [FaceParsingBiSeNetDyn](https://github.com/zllrunning/face-parsing.PyTorch)                    |  50M  |  *segmentation*  | [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_face_parsing_bisenet_dyn.cpp)  |      ✅      |  /  |  /   |  /  |   /   |  ✔️   |   ✔️    |    ❔    |

<!--
</details>
-->

<div id="lite.ai.toolkit-Model-Zoo"></div>

<details>
<summary> 🔑️ Model Zoo！Click here! </summary>    

## Model Zoo.

<div id="lite.ai.toolkit-2"></div>

**Lite.Ai.ToolKit** contains almost **[100+](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md)** AI models with **[500+](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md)** frozen pretrained  files now. Most of the files are converted by myself. You can use it through **lite::cv::Type::Class** syntax, such as **[lite::cv::detection::YoloV5](#lite.ai.toolkit-object-detection)**. More details can be found at [Examples for Lite.Ai.ToolKit](#lite.ai.toolkit-Examples-for-Lite.AI.ToolKit). Note, for Google Drive, I can not upload all the *.onnx files because of the storage limitation (15G).

| File |                                Baidu Drive                                |                                             Google Drive                                             |                                                          Docker Hub                                                          |                                               Hub (Docs)                                               |  
|:----:|:-------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------:|
| ONNX | [Baidu Drive](https://pan.baidu.com/s/1elUGcx7CZkkjEoYhTMwTRQ) code: 8gin | [Google Drive](https://drive.google.com/drive/folders/1p6uBcxGeyS1exc-T61vL8YRhwjYL4iD2?usp=sharing) | [ONNX Docker v0.1.22.01.08 (28G), v0.1.22.02.02 (400M)](https://hub.docker.com/r/qyjdefdocker/lite.ai.toolkit-onnx-hub/tags) | [ONNX Hub](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md) |    
| MNN  | [Baidu Drive](https://pan.baidu.com/s/1KyO-bCYUv6qPq2M8BH_Okg) code: 9v63 |                                                  ❔                                                   |  [MNN Docker v0.1.22.01.08 (11G), v0.1.22.02.02 (213M)](https://hub.docker.com/r/qyjdefdocker/lite.ai.toolkit-mnn-hub/tags)  |  [MNN Hub](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.mnn.md)  |  
| NCNN | [Baidu Drive](https://pan.baidu.com/s/1hlnqyNsFbMseGFWscgVhgQ) code: sc7f |                                                  ❔                                                   | [NCNN Docker v0.1.22.01.08 (9G), v0.1.22.02.02 (197M)](https://hub.docker.com/r/qyjdefdocker/lite.ai.toolkit-ncnn-hub/tags)  | [NCNN Hub](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.ncnn.md) |
| TNN  | [Baidu Drive](https://pan.baidu.com/s/1lvM2YKyUbEc5HKVtqITpcw) code: 6o6k |                                                  ❔                                                   |  [TNN Docker v0.1.22.01.08 (11G), v0.1.22.02.02 (217M)](https://hub.docker.com/r/qyjdefdocker/lite.ai.toolkit-tnn-hub/tags)  |  [TNN Hub](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.tnn.md)  |

```shell
  docker pull qyjdefdocker/lite.ai.toolkit-onnx-hub:v0.1.22.01.08  # (28G)
  docker pull qyjdefdocker/lite.ai.toolkit-mnn-hub:v0.1.22.01.08   # (11G)
  docker pull qyjdefdocker/lite.ai.toolkit-ncnn-hub:v0.1.22.01.08  # (9G)
  docker pull qyjdefdocker/lite.ai.toolkit-tnn-hub:v0.1.22.01.08   # (11G)
  docker pull qyjdefdocker/lite.ai.toolkit-onnx-hub:v0.1.22.02.02  # (400M) + YOLO5Face
  docker pull qyjdefdocker/lite.ai.toolkit-mnn-hub:v0.1.22.02.02   # (213M) + YOLO5Face
  docker pull qyjdefdocker/lite.ai.toolkit-ncnn-hub:v0.1.22.02.02  # (197M) + YOLO5Face
  docker pull qyjdefdocker/lite.ai.toolkit-tnn-hub:v0.1.22.02.02   # (217M) + YOLO5Face
```

### 🔑️ How to download Model Zoo from Docker Hub?

* Firstly, pull the image from docker hub.  
  ```shell
  docker pull qyjdefdocker/lite.ai.toolkit-mnn-hub:v0.1.22.01.08 # (11G)
  docker pull qyjdefdocker/lite.ai.toolkit-ncnn-hub:v0.1.22.01.08 # (9G)
  docker pull qyjdefdocker/lite.ai.toolkit-tnn-hub:v0.1.22.01.08 # (11G)
  docker pull qyjdefdocker/lite.ai.toolkit-onnx-hub:v0.1.22.01.08 # (28G)
  ```
* Secondly, run the container with local `share` dir using `docker run -idt xxx`. A minimum example will show you as follows.  
  * make a `share` dir in your local device.
  ```shell
  mkdir share # any name is ok.
  ```
  * write `run_mnn_docker_hub.sh` script like:
  ```shell
  #!/bin/bash  
  PORT1=6072
  PORT2=6084
  SERVICE_DIR=/Users/xxx/Desktop/your-path-to/share
  CONRAINER_DIR=/home/hub/share
  CONRAINER_NAME=mnn_docker_hub_d
  
  docker run -idt -p ${PORT2}:${PORT1} -v ${SERVICE_DIR}:${CONRAINER_DIR} --shm-size=16gb --name ${CONRAINER_NAME} qyjdefdocker/lite.ai.toolkit-mnn-hub:v0.1.22.01.08

  ```
* Finally, copy the model weights from `/home/hub/mnn/cv` to your local `share` dir. 
  ```shell
  # activate mnn docker.
  sh ./run_mnn_docker_hub.sh
  docker exec -it mnn_docker_hub_d /bin/bash
  # copy the models to the share dir.
  cd /home/hub 
  cp -rf mnn/cv share/
  ```


### Model Hubs
The pretrained and converted ONNX files provide by lite.ai.toolkit are listed as follows. Also, see [Model Zoo](#lite.ai.toolkit-Model-Zoo) and [ONNX Hub](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md), [MNN Hub](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.mnn.md), [TNN Hub](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.tnn.md), [NCNN Hub](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.ncnn.md) for more details.

</details>


<div id="lite.ai.toolkit-Examples-for-Lite.AI.ToolKit"></div>

<details>
<summary> 🔑️ More Examples！Click here! </summary>    

## 🔑️ More Examples.

More examples can be found at [examples](https://github.com/DefTruth/lite.ai.toolkit/tree/main/examples/lite/cv). 

<div id="lite.ai.toolkit-object-detection"></div>

#### Example0: Object Detection using [YOLOv5](https://github.com/ultralytics/yolov5). Download model from Model-Zoo[<sup>2</sup>](#lite.ai.toolkit-2).
```c++
#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../examples/hub/onnx/cv/yolov5s.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_yolov5_1.jpg";
  std::string save_img_path = "../../../examples/logs/test_lite_yolov5_1.jpg";

  auto *yolov5 = new lite::cv::detection::YoloV5(onnx_path); 
  std::vector<lite::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  yolov5->detect(img_bgr, detected_boxes);
  
  lite::utils::draw_boxes_inplace(img_bgr, detected_boxes);
  cv::imwrite(save_img_path, img_bgr);  
  
  delete yolov5;
}
```

The output is:
<div align='center'>
  <img src='https://github.com/DefTruth/lite.ai.toolkit/assets/31974251/44dbf4ac-0f38-41b6-930b-55b032b3c2ee' height="256px">
  <img src='https://github.com/DefTruth/lite.ai.toolkit/assets/31974251/23aca3df-76a6-45c0-a48b-7968b4d4b9d8' height="256px">
</div>

Or you can use Newest 🔥🔥 ! YOLO series's detector [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) or [YoloR](https://github.com/WongKinYiu/yolor). They got the similar results.

More classes for general object detection (80 classes, COCO).
```c++
auto *detector = new lite::cv::detection::YoloX(onnx_path);  // Newest YOLO detector !!! 2021-07
auto *detector = new lite::cv::detection::YoloV4(onnx_path); 
auto *detector = new lite::cv::detection::YoloV3(onnx_path); 
auto *detector = new lite::cv::detection::TinyYoloV3(onnx_path); 
auto *detector = new lite::cv::detection::SSD(onnx_path); 
auto *detector = new lite::cv::detection::YoloV5(onnx_path); 
auto *detector = new lite::cv::detection::YoloR(onnx_path);  // Newest YOLO detector !!! 2021-05
auto *detector = new lite::cv::detection::TinyYoloV4VOC(onnx_path); 
auto *detector = new lite::cv::detection::TinyYoloV4COCO(onnx_path); 
auto *detector = new lite::cv::detection::ScaledYoloV4(onnx_path); 
auto *detector = new lite::cv::detection::EfficientDet(onnx_path); 
auto *detector = new lite::cv::detection::EfficientDetD7(onnx_path); 
auto *detector = new lite::cv::detection::EfficientDetD8(onnx_path); 
auto *detector = new lite::cv::detection::YOLOP(onnx_path);
auto *detector = new lite::cv::detection::NanoDet(onnx_path); // Super fast and tiny!
auto *detector = new lite::cv::detection::NanoDetPlus(onnx_path); // Super fast and tiny! 2021/12/25
auto *detector = new lite::cv::detection::NanoDetEfficientNetLite(onnx_path); // Super fast and tiny!
auto *detector = new lite::cv::detection::YoloV5_V_6_0(onnx_path); 
auto *detector = new lite::cv::detection::YoloV5_V_6_1(onnx_path); 
auto *detector = new lite::cv::detection::YoloX_V_0_1_1(onnx_path);  // Newest YOLO detector !!! 2021-07
auto *detector = new lite::cv::detection::YOLOv6(onnx_path);  // Newest 2022 YOLO detector !!!
```


****

<div id="lite.ai.toolkit-matting"></div>  

#### Example1: Video Matting using [RobustVideoMatting2021🔥🔥🔥](https://github.com/PeterL1n/RobustVideoMatting). Download model from Model-Zoo[<sup>2</sup>](#lite.ai.toolkit-2).

```c++
#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../examples/hub/onnx/cv/rvm_mobilenetv3_fp32.onnx";
  std::string video_path = "../../../examples/lite/resources/test_lite_rvm_0.mp4";
  std::string output_path = "../../../examples/logs/test_lite_rvm_0.mp4";
  std::string background_path = "../../../examples/lite/resources/test_lite_matting_bgr.jpg";
  
  auto *rvm = new lite::cv::matting::RobustVideoMatting(onnx_path, 16); // 16 threads
  std::vector<lite::types::MattingContent> contents;
  
  // 1. video matting.
  cv::Mat background = cv::imread(background_path);
  rvm->detect_video(video_path, output_path, contents, false, 0.4f,
                    20, true, true, background);
  
  delete rvm;
}
```
The output is:

<div align='center'>
  <img src='https://github.com/DefTruth/lite.ai.toolkit/assets/31974251/a6da4814-6643-4dfc-89ce-57f140c999fc' height="150px" width="150px">
  <img src='https://github.com/DefTruth/lite.ai.toolkit/assets/31974251/9e4f060e-3de8-44c4-a20f-74a0ff3943bb' height="150px" width="150px">  
  <img src='https://github.com/DefTruth/lite.ai.toolkit/assets/31974251/42bb2991-333a-4524-b874-6ab6156b3425' height="150px" width="150px">
  <img src='https://github.com/DefTruth/lite.ai.toolkit/assets/31974251/f8d65d8c-2a3d-4634-9169-3bc36452d997' height="150px" width="150px">
  <br>
  <img src='https://github.com/DefTruth/lite.ai.toolkit/assets/31974251/c1411bb7-5537-4d6e-81f7-c902c2256a72' height="150px" width="150px">
  <img src='https://github.com/DefTruth/lite.ai.toolkit/assets/31974251/6344f307-15e3-4593-9866-50f5ee777f43' height="150px" width="150px">  
  <img src='https://github.com/DefTruth/lite.ai.toolkit/assets/31974251/4d824828-7727-48df-8aae-64e15ca1c03b' height="150px" width="150px">
  <img src='https://github.com/DefTruth/lite.ai.toolkit/assets/31974251/e8102fd6-e869-4a42-a19f-dd6d180dda92' height="150px" width="150px">
</div>

More classes for matting (image matting, video matting, trimap/mask-free, trimap/mask-based)
```c++
auto *matting = new lite::cv::matting::RobustVideoMatting:(onnx_path);  //  WACV 2022.
auto *matting = new lite::cv::matting::MGMatting(onnx_path); // CVPR 2021
auto *matting = new lite::cv::matting::MODNet(onnx_path); // AAAI 2022
auto *matting = new lite::cv::matting::MODNetDyn(onnx_path); // AAAI 2022 Dynamic Shape Inference.
auto *matting = new lite::cv::matting::BackgroundMattingV2(onnx_path); // CVPR 2020 
auto *matting = new lite::cv::matting::BackgroundMattingV2Dyn(onnx_path); // CVPR 2020 Dynamic Shape Inference.
auto *matting = new lite::cv::matting::MobileHumanMatting(onnx_path); // 3Mb only !!!
```


****

<div id="lite.ai.toolkit-face-alignment"></div>

#### Example2: 1000 Facial Landmarks Detection using [FaceLandmarks1000](https://github.com/Single430/FaceLandmark1000). Download model from Model-Zoo[<sup>2</sup>](#lite.ai.toolkit-2).
```c++
#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../examples/hub/onnx/cv/FaceLandmark1000.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_face_landmarks_0.png";
  std::string save_img_path = "../../../examples/logs/test_lite_face_landmarks_1000.jpg";
    
  auto *face_landmarks_1000 = new lite::cv::face::align::FaceLandmark1000(onnx_path);

  lite::types::Landmarks landmarks;
  cv::Mat img_bgr = cv::imread(test_img_path);
  face_landmarks_1000->detect(img_bgr, landmarks);
  lite::utils::draw_landmarks_inplace(img_bgr, landmarks);
  cv::imwrite(save_img_path, img_bgr);
  
  delete face_landmarks_1000;
}
```
The output is:
<div align='center'>
  <img src='https://github.com/DefTruth/lite.ai.toolkit/assets/31974251/318691ec-7226-4d55-990b-a320635d8910' height="224px" width="224px">
  <img src='https://github.com/DefTruth/lite.ai.toolkit/assets/31974251/d64ae30e-a0b9-4ac9-bf4f-9d6f80c2c05a' height="224px" width="224px">
  <img src='https://github.com/DefTruth/lite.ai.toolkit/assets/31974251/c802858c-6899-4246-8839-5721c43faffe' height="224px" width="224px">
</div>    

More classes for face alignment (68 points, 98 points, 106 points, 1000 points)
```c++
auto *align = new lite::cv::face::align::PFLD(onnx_path);  // 106 landmarks, 1.0Mb only!
auto *align = new lite::cv::face::align::PFLD98(onnx_path);  // 98 landmarks, 4.8Mb only!
auto *align = new lite::cv::face::align::PFLD68(onnx_path);  // 68 landmarks, 2.8Mb only!
auto *align = new lite::cv::face::align::MobileNetV268(onnx_path);  // 68 landmarks, 9.4Mb only!
auto *align = new lite::cv::face::align::MobileNetV2SE68(onnx_path);  // 68 landmarks, 11Mb only!
auto *align = new lite::cv::face::align::FaceLandmark1000(onnx_path);  // 1000 landmarks, 2.0Mb only!
auto *align = new lite::cv::face::align::PIPNet98(onnx_path);  // 98 landmarks, CVPR2021!
auto *align = new lite::cv::face::align::PIPNet68(onnx_path);  // 68 landmarks, CVPR2021!
auto *align = new lite::cv::face::align::PIPNet29(onnx_path);  // 29 landmarks, CVPR2021!
auto *align = new lite::cv::face::align::PIPNet19(onnx_path);  // 19 landmarks, CVPR2021!
```


****

<div id="lite.ai.toolkit-colorization"></div>

#### Example3: Colorization using [colorization](https://github.com/richzhang/colorization). Download model from Model-Zoo[<sup>2</sup>](#lite.ai.toolkit-2).
```c++
#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../examples/hub/onnx/cv/eccv16-colorizer.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_colorizer_1.jpg";
  std::string save_img_path = "../../../examples/logs/test_lite_eccv16_colorizer_1.jpg";
  
  auto *colorizer = new lite::cv::colorization::Colorizer(onnx_path);
  
  cv::Mat img_bgr = cv::imread(test_img_path);
  lite::types::ColorizeContent colorize_content;
  colorizer->detect(img_bgr, colorize_content);
  
  if (colorize_content.flag) cv::imwrite(save_img_path, colorize_content.mat);
  delete colorizer;
}
```
The output is:

<div align='center'>
  <img src='https://github.com/DefTruth/lite.ai.toolkit/assets/31974251/022dd4ab-1048-4d51-8e84-f839464d013e' height="224px" width="224px">
  <img src='https://github.com/DefTruth/lite.ai.toolkit/assets/31974251/8eceb121-6da6-41d9-8dbf-949034f27247' height="224px" width="224px">
  <img src='https://github.com/DefTruth/lite.ai.toolkit/assets/31974251/54a9b811-d21b-4120-8381-df0f858dba8b' height="224px" width="224px">  
  <br> 
  <img src='https://github.com/DefTruth/lite.ai.toolkit/assets/31974251/95f265a8-ca85-4df1-b2f0-04e1dd3d8fff' height="224px" width="224px">
  <img src='https://github.com/DefTruth/lite.ai.toolkit/assets/31974251/34bd3bcb-377a-47a0-b8bf-a44603f9b275' height="224px" width="224px">
  <img src='https://github.com/DefTruth/lite.ai.toolkit/assets/31974251/e1eff9e9-9f3c-4558-8826-d05c0c254e29' height="224px" width="224px">
</div>    

More classes for colorization (gray to rgb)
```c++
auto *colorizer = new lite::cv::colorization::Colorizer(onnx_path);
```

****

<div id="lite.ai.toolkit-face-recognition"></div>  

#### Example4: Face Recognition using [ArcFace](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch). Download model from Model-Zoo[<sup>2</sup>](#lite.ai.toolkit-2).

```c++
#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../examples/hub/onnx/cv/ms1mv3_arcface_r100.onnx";
  std::string test_img_path0 = "../../../examples/lite/resources/test_lite_faceid_0.png";
  std::string test_img_path1 = "../../../examples/lite/resources/test_lite_faceid_1.png";
  std::string test_img_path2 = "../../../examples/lite/resources/test_lite_faceid_2.png";

  auto *glint_arcface = new lite::cv::faceid::GlintArcFace(onnx_path);

  lite::types::FaceContent face_content0, face_content1, face_content2;
  cv::Mat img_bgr0 = cv::imread(test_img_path0);
  cv::Mat img_bgr1 = cv::imread(test_img_path1);
  cv::Mat img_bgr2 = cv::imread(test_img_path2);
  glint_arcface->detect(img_bgr0, face_content0);
  glint_arcface->detect(img_bgr1, face_content1);
  glint_arcface->detect(img_bgr2, face_content2);

  if (face_content0.flag && face_content1.flag && face_content2.flag)
  {
    float sim01 = lite::utils::math::cosine_similarity<float>(
        face_content0.embedding, face_content1.embedding);
    float sim02 = lite::utils::math::cosine_similarity<float>(
        face_content0.embedding, face_content2.embedding);
    std::cout << "Detected Sim01: " << sim  << " Sim02: " << sim02 << std::endl;
  }

  delete glint_arcface;
}
```

The output is:
<div align='center'>
  <img src='https://github.com/DefTruth/lite.ai.toolkit/assets/31974251/8311a1e0-1945-4a70-a361-c15a6e55baab' height="224px" width="224px">
  <img src='https://github.com/DefTruth/lite.ai.toolkit/assets/31974251/c1619f3f-cb12-4607-9e72-4a9f9224ef09' height="224px" width="224px">
  <img src='https://github.com/DefTruth/lite.ai.toolkit/assets/31974251/21859087-8458-4be6-b1ab-f20c1546e310' height="224px" width="224px">
</div>  

> Detected Sim01: 0.721159  Sim02: -0.0626267

More classes for face recognition (face id vector extract)
```c++
auto *recognition = new lite::cv::faceid::GlintCosFace(onnx_path);  // DeepGlint(insightface)
auto *recognition = new lite::cv::faceid::GlintArcFace(onnx_path);  // DeepGlint(insightface)
auto *recognition = new lite::cv::faceid::GlintPartialFC(onnx_path); // DeepGlint(insightface)
auto *recognition = new lite::cv::faceid::FaceNet(onnx_path);
auto *recognition = new lite::cv::faceid::FocalArcFace(onnx_path);
auto *recognition = new lite::cv::faceid::FocalAsiaArcFace(onnx_path);
auto *recognition = new lite::cv::faceid::TencentCurricularFace(onnx_path); // Tencent(TFace)
auto *recognition = new lite::cv::faceid::TencentCifpFace(onnx_path); // Tencent(TFace)
auto *recognition = new lite::cv::faceid::CenterLossFace(onnx_path);
auto *recognition = new lite::cv::faceid::SphereFace(onnx_path);
auto *recognition = new lite::cv::faceid::PoseRobustFace(onnx_path);
auto *recognition = new lite::cv::faceid::NaivePoseRobustFace(onnx_path);
auto *recognition = new lite::cv::faceid::MobileFaceNet(onnx_path); // 3.8Mb only !
auto *recognition = new lite::cv::faceid::CavaGhostArcFace(onnx_path);
auto *recognition = new lite::cv::faceid::CavaCombinedFace(onnx_path);
auto *recognition = new lite::cv::faceid::MobileSEFocalFace(onnx_path); // 4.5Mb only !
```

****

<div id="lite.ai.toolkit-face-detection"></div>

#### Example5: Face Detection using [SCRFD 2021](https://github.com/deepinsight/insightface/blob/master/detection/scrfd/). Download model from Model-Zoo[<sup>2</sup>](#lite.ai.toolkit-2).
```c++
#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../examples/hub/onnx/cv/scrfd_2.5g_bnkps_shape640x640.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_face_detector.jpg";
  std::string save_img_path = "../../../examples/logs/test_lite_scrfd.jpg";
  
  auto *scrfd = new lite::cv::face::detect::SCRFD(onnx_path);
  
  std::vector<lite::types::BoxfWithLandmarks> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  scrfd->detect(img_bgr, detected_boxes);
  
  lite::utils::draw_boxes_with_landmarks_inplace(img_bgr, detected_boxes);
  cv::imwrite(save_img_path, img_bgr);
  
  delete scrfd;
}
```
The output is:
<div align='center'>
  <img src='https://github.com/DefTruth/lite.ai.toolkit/assets/31974251/b913c502-93fc-4a29-8114-9a3450c512f0' height="224px" width="224px">
  <img src='https://github.com/DefTruth/lite.ai.toolkit/assets/31974251/28274741-8745-4665-abff-3a384b75f7fa' height="224px" width="224px">
  <img src='https://github.com/DefTruth/lite.ai.toolkit/assets/31974251/a0bc4d9f-df88-4757-bcfb-214f2c1d4991' height="224px" width="224px">
</div>  

More classes for face detection (super fast face detection)
```c++
auto *detector = new lite::face::detect::UltraFace(onnx_path);  // 1.1Mb only !
auto *detector = new lite::face::detect::FaceBoxes(onnx_path);  // 3.8Mb only ! 
auto *detector = new lite::face::detect::FaceBoxesv2(onnx_path);  // 4.0Mb only ! 
auto *detector = new lite::face::detect::RetinaFace(onnx_path);  // 1.6Mb only ! CVPR2020
auto *detector = new lite::face::detect::SCRFD(onnx_path);  // 2.5Mb only ! CVPR2021, Super fast and accurate!!
auto *detector = new lite::face::detect::YOLO5Face(onnx_path);  // 2021, Super fast and accurate!!
auto *detector = new lite::face::detect::YOLOv5BlazeFace(onnx_path);  // 2021, Super fast and accurate!!
```

****

<div id="lite.ai.toolkit-segmentation"></div>   

#### Example6: Object Segmentation using [DeepLabV3ResNet101](https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/). Download model from Model-Zoo[<sup>2</sup>](#lite.ai.toolkit-2).
```c++
#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../examples/hub/onnx/cv/deeplabv3_resnet101_coco.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_deeplabv3_resnet101.png";
  std::string save_img_path = "../../../examples/logs/test_lite_deeplabv3_resnet101.jpg";

  auto *deeplabv3_resnet101 = new lite::cv::segmentation::DeepLabV3ResNet101(onnx_path, 16); // 16 threads

  lite::types::SegmentContent content;
  cv::Mat img_bgr = cv::imread(test_img_path);
  deeplabv3_resnet101->detect(img_bgr, content);

  if (content.flag)
  {
    cv::Mat out_img;
    cv::addWeighted(img_bgr, 0.2, content.color_mat, 0.8, 0., out_img);
    cv::imwrite(save_img_path, out_img);
    if (!content.names_map.empty())
    {
      for (auto it = content.names_map.begin(); it != content.names_map.end(); ++it)
      {
        std::cout << it->first << " Name: " << it->second << std::endl;
      }
    }
  }
  delete deeplabv3_resnet101;
}
```

The output is:
<div align='center'>
  <img src='https://github.com/DefTruth/lite.ai.toolkit/assets/31974251/cf143f25-a233-40f1-a4b9-7ad52f691799' height="256px">
  <img src='https://github.com/DefTruth/lite.ai.toolkit/assets/31974251/f4dd5263-8514-4bb0-a0dd-dbe532481aff' height="256px">
</div> 

More classes for object segmentation (general objects segmentation)
```c++
auto *segment = new lite::cv::segmentation::FCNResNet101(onnx_path);
auto *segment = new lite::cv::segmentation::DeepLabV3ResNet101(onnx_path);
```

****

<div id="lite.ai.toolkit-face-attributes-analysis"></div>  

#### Example7: Age Estimation using [SSRNet](https://github.com/oukohou/SSR_Net_Pytorch) . Download model from Model-Zoo[<sup>2</sup>](#lite.ai.toolkit-2).
```c++
#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../examples/hub/onnx/cv/ssrnet.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_ssrnet.jpg";
  std::string save_img_path = "../../../examples/logs/test_lite_ssrnet.jpg";

  auto *ssrnet = new lite::cv::face::attr::SSRNet(onnx_path);

  lite::types::Age age;
  cv::Mat img_bgr = cv::imread(test_img_path);
  ssrnet->detect(img_bgr, age);
  lite::utils::draw_age_inplace(img_bgr, age);
  cv::imwrite(save_img_path, img_bgr);

  delete ssrnet;
}
```
The output is:
<div align='center'>
  <img src='https://github.com/DefTruth/lite.ai.toolkit/assets/31974251/6dc688d9-95be-40f3-b9b8-1a2f69e39e1d' height="224px" width="224px">
  <img src='https://github.com/DefTruth/lite.ai.toolkit/assets/31974251/77089174-f744-4603-b417-c23caeb344d7' height="224px" width="224px">
  <img src='https://github.com/DefTruth/lite.ai.toolkit/assets/31974251/8f279483-5849-4356-885c-5806337ced2a' height="224px" width="224px">
</div>    

More classes for face attributes analysis (age, gender, emotion)
```c++
auto *attribute = new lite::cv::face::attr::AgeGoogleNet(onnx_path);  
auto *attribute = new lite::cv::face::attr::GenderGoogleNet(onnx_path); 
auto *attribute = new lite::cv::face::attr::EmotionFerPlus(onnx_path);
auto *attribute = new lite::cv::face::attr::VGG16Age(onnx_path);
auto *attribute = new lite::cv::face::attr::VGG16Gender(onnx_path);
auto *attribute = new lite::cv::face::attr::EfficientEmotion7(onnx_path); // 7 emotions, 15Mb only!
auto *attribute = new lite::cv::face::attr::EfficientEmotion8(onnx_path); // 8 emotions, 15Mb only!
auto *attribute = new lite::cv::face::attr::MobileEmotion7(onnx_path); // 7 emotions, 13Mb only!
auto *attribute = new lite::cv::face::attr::ReXNetEmotion7(onnx_path); // 7 emotions
auto *attribute = new lite::cv::face::attr::SSRNet(onnx_path); // age estimation, 190kb only!!!
```

****

<div id="lite.ai.toolkit-image-classification"></div>  

#### Example8: 1000 Classes Classification using [DenseNet](https://pytorch.org/hub/pytorch_vision_densenet/). Download model from Model-Zoo[<sup>2</sup>](#lite.ai.toolkit-2).
```c++
#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../examples/hub/onnx/cv/densenet121.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_densenet.jpg";

  auto *densenet = new lite::cv::classification::DenseNet(onnx_path);

  lite::types::ImageNetContent content;
  cv::Mat img_bgr = cv::imread(test_img_path);
  densenet->detect(img_bgr, content);
  if (content.flag)
  {
    const unsigned int top_k = content.scores.size();
    if (top_k > 0)
    {
      for (unsigned int i = 0; i < top_k; ++i)
        std::cout << i + 1
                  << ": " << content.labels.at(i)
                  << ": " << content.texts.at(i)
                  << ": " << content.scores.at(i)
                  << std::endl;
    }
  }
  delete densenet;
}
```

The output is:
<div align='center'>
  <img src='https://github.com/DefTruth/lite.ai.toolkit/assets/31974251/58e3b905-367d-486a-a3b6-062cef87d726' height="224px" width="350px">
  <img src='https://github.com/DefTruth/lite.ai.toolkit/assets/31974251/cf81d298-5903-4d3c-ad06-14882911c006' height="224px" width="350px">
</div>  

More classes for image classification (1000 classes)
```c++
auto *classifier = new lite::cv::classification::EfficientNetLite4(onnx_path);  
auto *classifier = new lite::cv::classification::ShuffleNetV2(onnx_path); // 8.7Mb only!
auto *classifier = new lite::cv::classification::GhostNet(onnx_path);
auto *classifier = new lite::cv::classification::HdrDNet(onnx_path);
auto *classifier = new lite::cv::classification::IBNNet(onnx_path);
auto *classifier = new lite::cv::classification::MobileNetV2(onnx_path); // 13Mb only!
auto *classifier = new lite::cv::classification::ResNet(onnx_path); 
auto *classifier = new lite::cv::classification::ResNeXt(onnx_path);
```

****

<div id="lite.ai.toolkit-head-pose-estimation"></div>  

#### Example9: Head Pose Estimation using [FSANet](https://github.com/omasaht/headpose-fsanet-pytorch). Download model from Model-Zoo[<sup>2</sup>](#lite.ai.toolkit-2).

```c++
#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../examples/hub/onnx/cv/fsanet-var.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_fsanet.jpg";
  std::string save_img_path = "../../../examples/logs/test_lite_fsanet.jpg";

  auto *fsanet = new lite::cv::face::pose::FSANet(onnx_path);
  cv::Mat img_bgr = cv::imread(test_img_path);
  lite::types::EulerAngles euler_angles;
  fsanet->detect(img_bgr, euler_angles);
  
  if (euler_angles.flag)
  {
    lite::utils::draw_axis_inplace(img_bgr, euler_angles);
    cv::imwrite(save_img_path, img_bgr);
    std::cout << "yaw:" << euler_angles.yaw << " pitch:" << euler_angles.pitch << " row:" << euler_angles.roll << std::endl;
  }
  delete fsanet;
}
```

The output is:
<div align='center'>
  <img src='https://github.com/DefTruth/lite.ai.toolkit/assets/31974251/20a18d56-297c-4c72-8153-76d4380fc9ec)' height="224px" width="224px">
  <img src='https://github.com/DefTruth/lite.ai.toolkit/assets/31974251/6630a13b-af81-4606-8a81-37fb416f0a64' height="224px" width="224px">
  <img src='https://github.com/DefTruth/lite.ai.toolkit/assets/31974251/fb873266-2cfd-4b08-8ffb-639aee1ca2b6' height="224px" width="224px">
</div>  

More classes for head pose estimation (euler angle, yaw, pitch, roll)
```c++
auto *pose = new lite::cv::face::pose::FSANet(onnx_path); // 1.2Mb only!
```

****

<div id="lite.ai.toolkit-style-transfer"></div>

#### Example10: Style Transfer using [FastStyleTransfer](https://github.com/onnx/models/tree/master/vision/style_transfer/fast_neural_style). Download model from Model-Zoo[<sup>2</sup>](#lite.ai.toolkit-2).
```c++
#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../examples/hub/onnx/cv/style-candy-8.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_fast_style_transfer.jpg";
  std::string save_img_path = "../../../examples/logs/test_lite_fast_style_transfer_candy.jpg";
  
  auto *fast_style_transfer = new lite::cv::style::FastStyleTransfer(onnx_path);
 
  lite::types::StyleContent style_content;
  cv::Mat img_bgr = cv::imread(test_img_path);
  fast_style_transfer->detect(img_bgr, style_content);

  if (style_content.flag) cv::imwrite(save_img_path, style_content.mat);
  delete fast_style_transfer;
}
```
The output is:

<div align='center'>
  <img src='https://github.com/DefTruth/lite.ai.toolkit/assets/31974251/c42af6ea-0b3a-4816-902a-9958fdef5653' height="224px">
  <img src='https://github.com/DefTruth/lite.ai.toolkit/assets/31974251/bbaa6e7e-50c0-4993-b6e9-aee681e61fdb' height="224px">
  <img src='https://github.com/DefTruth/lite.ai.toolkit/assets/31974251/95106e7e-c6bc-433d-b20c-95b579e85a06' height="224px">  
  <br> 
  <img src='https://github.com/DefTruth/lite.ai.toolkit/assets/31974251/13a17444-27a4-4153-a6ee-5fff0a7fc667' height="224px">
  <img src='https://github.com/DefTruth/lite.ai.toolkit/assets/31974251/6e2c1d8b-f4a2-4433-b31b-b60f381344c1' height="224px">
  <img src='https://github.com/DefTruth/lite.ai.toolkit/assets/31974251/9f3f706a-50b7-43e4-8631-13ffa9b12fb5' height="224px">
</div>

More classes for style transfer (neural style transfer, others)
```c++
auto *transfer = new lite::cv::style::FastStyleTransfer(onnx_path); // 6.4Mb only
```

**** 

#### Example11: Human Head Segmentation using [HeadSeg](https://github.com/minivision-ai/photo2cartoon). Download model from Model-Zoo[<sup>2</sup>](#lite.ai.toolkit-2).
```c++
#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../examples/hub/onnx/cv/minivision_head_seg.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_head_seg.png";
  std::string save_img_path = "../../../examples/logs/test_lite_head_seg.jpg";

  auto *head_seg = new lite::cv::segmentation::HeadSeg(onnx_path, 4); // 4 threads

  lite::types::HeadSegContent content;
  cv::Mat img_bgr = cv::imread(test_img_path);
  head_seg->detect(img_bgr, content);
  if (content.flag) cv::imwrite(save_img_path, content.mask * 255.f);

  delete head_seg;
}
```  
The output is:  

<div align='center'>
  <img src='https://github.com/DefTruth/lite.ai.toolkit/assets/31974251/5684e1d9-b3b1-45af-ac38-d9201490d46e' height="180px" width="180px">
  <img src='https://github.com/DefTruth/lite.ai.toolkit/assets/31974251/b6a431d2-225b-416b-8a1e-cf9617d79a63' height="180px" width="180px">
  <img src='https://github.com/DefTruth/lite.ai.toolkit/assets/31974251/ff9740a5-a70e-400c-8301-fc19c92c6248' height="180px" width="180px">
  <img src='https://github.com/DefTruth/lite.ai.toolkit/assets/31974251/62747811-3856-4f40-9057-9ec4db687b31' height="180px" width="180px">
</div> 

More classes for human segmentation (head, portrait, hair, others)
```c++
auto *segment = new lite::cv::segmentation::HeadSeg(onnx_path); // 31Mb
auto *segment = new lite::cv::segmentation::FastPortraitSeg(onnx_path); // <= 400Kb !!! 
auto *segment = new lite::cv::segmentation::PortraitSegSINet(onnx_path); // <= 380Kb !!!
auto *segment = new lite::cv::segmentation::PortraitSegExtremeC3Net(onnx_path); // <= 180Kb !!! Extreme Tiny !!!
auto *segment = new lite::cv::segmentation::FaceHairSeg(onnx_path); // 18M
auto *segment = new lite::cv::segmentation::HairSeg(onnx_path); // 18M
auto *segment = new lite::cv::segmentation::MobileHairSeg(onnx_path); // 14M
```

**** 

#### Example12: Photo transfer to Cartoon [Photo2Cartoon](https://github.com/minivision-ai/photo2cartoon). Download model from Model-Zoo[<sup>2</sup>](#lite.ai.toolkit-2).
```c++
#include "lite/lite.h"

static void test_default()
{
  std::string head_seg_onnx_path = "../../../examples/hub/onnx/cv/minivision_head_seg.onnx";
  std::string cartoon_onnx_path = "../../../examples/hub/onnx/cv/minivision_female_photo2cartoon.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_female_photo2cartoon.jpg";
  std::string save_mask_path = "../../../examples/logs/test_lite_female_photo2cartoon_seg.jpg";
  std::string save_cartoon_path = "../../../examples/logs/test_lite_female_photo2cartoon_cartoon.jpg";

  auto *head_seg = new lite::cv::segmentation::HeadSeg(head_seg_onnx_path, 4); // 4 threads
  auto *female_photo2cartoon = new lite::cv::style::FemalePhoto2Cartoon(cartoon_onnx_path, 4); // 4 threads

  lite::types::HeadSegContent head_seg_content;
  cv::Mat img_bgr = cv::imread(test_img_path);
  head_seg->detect(img_bgr, head_seg_content);

  if (head_seg_content.flag && !head_seg_content.mask.empty())
  {
    cv::imwrite(save_mask_path, head_seg_content.mask * 255.f);
    // Female Photo2Cartoon Style Transfer
    lite::types::FemalePhoto2CartoonContent female_cartoon_content;
    female_photo2cartoon->detect(img_bgr, head_seg_content.mask, female_cartoon_content);
    
    if (female_cartoon_content.flag && !female_cartoon_content.cartoon.empty())
      cv::imwrite(save_cartoon_path, female_cartoon_content.cartoon);
  }

  delete head_seg;
  delete female_photo2cartoon;
}
```  
The output is:

<div align='center'>
  <img src='https://github.com/DefTruth/lite.ai.toolkit/assets/31974251/5684e1d9-b3b1-45af-ac38-d9201490d46e' height="180px" width="180px">
  <img src='https://github.com/DefTruth/lite.ai.toolkit/assets/31974251/1970b922-6027-44b3-9211-9f057e2ce003' height="180px" width="180px">
  <img src='https://github.com/DefTruth/lite.ai.toolkit/assets/31974251/73494f60-9efd-48cb-a993-5a5837badb12' height="180px" width="180px">
  <img src='https://github.com/DefTruth/lite.ai.toolkit/assets/31974251/e10e9624-3176-4137-988b-c73be6103fed' height="180px" width="180px">
</div> 

More classes for photo style transfer.
```c++
auto *transfer = new lite::cv::style::FemalePhoto2Cartoon(onnx_path);
```

**** 

#### Example13: Face Parsing using [FaceParsing](https://github.com/zllrunning/face-parsing.PyTorch). Download model from Model-Zoo[<sup>2</sup>](#lite.ai.toolkit-2).
```c++
#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../examples/hub/onnx/cv/face_parsing_512x512.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_face_parsing.png";
  std::string save_img_path = "../../../examples/logs/test_lite_face_parsing_bisenet.jpg";

  auto *face_parsing_bisenet = new lite::cv::segmentation::FaceParsingBiSeNet(onnx_path, 8); // 8 threads

  lite::types::FaceParsingContent content;
  cv::Mat img_bgr = cv::imread(test_img_path);
  face_parsing_bisenet->detect(img_bgr, content);

  if (content.flag && !content.merge.empty())
    cv::imwrite(save_img_path, content.merge);
  
  delete face_parsing_bisenet;
}
```  
The output is:

<div align='center'>
  <img src='https://github.com/DefTruth/lite.ai.toolkit/assets/31974251/32532cbc-ef90-4afb-9fa9-0a1f52b18654' height="180px" width="180px">
  <img src='https://github.com/DefTruth/lite.ai.toolkit/assets/31974251/157b9e11-fc92-445b-ae0d-0d859c8663ee' height="180px" width="180px">
  <img src='https://github.com/DefTruth/lite.ai.toolkit/assets/31974251/e7945202-e4dc-4e70-b931-019afdc5a95b' height="180px" width="180px">
  <img src='https://github.com/DefTruth/lite.ai.toolkit/assets/31974251/7dbba712-078a-4cd6-b968-d6f565e10a3e' height="180px" width="180px">
</div> 

More classes for face parsing (hair, eyes, nose, mouth, others)
```c++
auto *segment = new lite::cv::segmentation::FaceParsingBiSeNet(onnx_path); // 50Mb
auto *segment = new lite::cv::segmentation::FaceParsingBiSeNetDyn(onnx_path); // Dynamic Shape Inference.
```
 </details>

## ©️License
GNU General Public License v3.0

## 🎉Contribute
Please consider ⭐ this repo if you like it, as it is the simplest way to support me.

<div align='center'>
<a href="https://star-history.com/#DefTruth/lite.ai.toolkit&Date">
  <picture align='center'>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=DefTruth/lite.ai.toolkit&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=DefTruth/lite.ai.toolkit&type=Date" />
    <img width=450 height=300 alt="Star History Chart" src="https://api.star-history.com/svg?repos=DefTruth/lite.ai.toolkit&type=Date" />
  </picture>
</a>  
</div>


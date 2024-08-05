//
// Created by DefTruth on 2021/3/18.
//

#ifndef LITE_AI_ORT_ORT_CORE_H
#define LITE_AI_ORT_ORT_CORE_H

#include "ort_config.h"
#include "ort_handler.h"
#include "ort_types.h"

namespace ortcv
{
  class LITE_EXPORTS FSANet;                     // [0] * reference: https://github.com/omasaht/headpose-fsanet-pytorch
  class LITE_EXPORTS PFLD;                       // [1] * reference: https://github.com/Hsintao/pfld_106_face_landmarks
  class LITE_EXPORTS UltraFace;                  // [2] * reference: https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB
  class LITE_EXPORTS AgeGoogleNet;               // [3] * reference: https://github.com/onnx/models/tree/master/vision/body_analysis/age_gender
  class LITE_EXPORTS GenderGoogleNet;            // [4] * reference: https://github.com/onnx/models/tree/master/vision/body_analysis/age_gender
  class LITE_EXPORTS EmotionFerPlus;             // [5] * reference: https://github.com/onnx/models/blob/master/vision/body_analysis/emotion_ferplus
  class LITE_EXPORTS VGG16Age;                   // [6] * reference: https://github.com/onnx/models/tree/master/vision/body_analysis/age_gender
  class LITE_EXPORTS VGG16Gender;                // [7] * reference: https://github.com/onnx/models/tree/master/vision/body_analysis/age_gender
  class LITE_EXPORTS SSRNet;                     // [8] * reference: https://github.com/oukohou/SSR_Net_Pytorch
  class LITE_EXPORTS FastStyleTransfer;          // [9] * reference: https://github.com/onnx/models/blob/master/vision/style_transfer/fast_neural_style
  class LITE_EXPORTS GlintArcFace;               // [10] * reference: https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch
  class LITE_EXPORTS Colorizer;                  // [11] * reference: https://github.com/richzhang/colorization
  class LITE_EXPORTS SubPixelCNN;                // [12] * reference: https://github.com/niazwazir/SUB_PIXEL_CNN
  class LITE_EXPORTS YoloV4;                     // [13] * reference: https://github.com/argusswift/YOLOv4-pytorch
  class LITE_EXPORTS YoloV5;                     // [14] * reference: https://github.com/ultralytics/yolov5
  class LITE_EXPORTS YoloV3;                     // [15] * reference: https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/yolov3
  class LITE_EXPORTS EfficientNetLite4;           // [16] * reference: https://github.com/onnx/models/blob/master/vision/classification/efficientnet-lite4
  class LITE_EXPORTS ShuffleNetV2;                // [17] * reference: https://github.com/onnx/models/blob/master/vision/classification/shufflenet
  class LITE_EXPORTS TinyYoloV3;                 // [18] * reference: https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/tiny-yolov3
  class LITE_EXPORTS SSD;                        // [19] * reference: https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/ssd
  class LITE_EXPORTS SSDMobileNetV1;             // [20] * reference: https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/ssd-mobilenetv1
  class LITE_EXPORTS DeepLabV3ResNet101;         // [21] * reference: https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/
  class LITE_EXPORTS DenseNet;                   // [22] * reference: https://pytorch.org/hub/pytorch_vision_densenet/
  class LITE_EXPORTS FCNResNet101;               // [23] * reference: https://pytorch.org/hub/pytorch_vision_fcn_resnet101/
  class LITE_EXPORTS GhostNet;                   // [24] * reference：https://pytorch.org/hub/pytorch_vision_ghostnet/
  class LITE_EXPORTS HdrDNet;                    // [25] * reference: https://pytorch.org/hub/pytorch_vision_hardnet/
  class LITE_EXPORTS IBNNet;                     // [26] * reference: https://pytorch.org/hub/pytorch_vision_ibnnet/
  class LITE_EXPORTS MobileNetV2;                // [27] * reference: https://pytorch.org/hub/pytorch_vision_mobilenet_v2/
  class LITE_EXPORTS ResNet;                     // [28] * reference: https://pytorch.org/hub/pytorch_vision_resnet/
  class LITE_EXPORTS ResNeXt;                    // [29] * reference: https://pytorch.org/hub/pytorch_vision_resnext/
  class LITE_EXPORTS GlintCosFace;               // [30] * reference: https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch
  class LITE_EXPORTS GlintPartialFC;             // [31] * reference: https://github.com/deepinsight/insightface/tree/master/recognition/partial_fc
  class LITE_EXPORTS FaceNet;                    // [32] * reference: https://github.com/timesler/facenet-pytorch
  class LITE_EXPORTS FocalArcFace;               // [33] * reference: https://github.com/ZhaoJ9014/face.evoLVe.PyTorch
  class LITE_EXPORTS FocalAsiaArcFace;           // [34] * reference: https://github.com/ZhaoJ9014/face.evoLVe.PyTorch
  class LITE_EXPORTS TencentCurricularFace;      // [35] * reference: https://github.com/Tencent/TFace/tree/master/tasks/distfc
  class LITE_EXPORTS TencentCifpFace;            // [36] * reference: https://github.com/Tencent/TFace/tree/master/tasks/cifp
  class LITE_EXPORTS CenterLossFace;             // [37] * reference: https://github.com/louis-she/center-loss.pytorch
  class LITE_EXPORTS SphereFace;                 // [38] * reference: https://github.com/clcarwin/sphereface_pytorch
  class LITE_EXPORTS PoseRobustFace;             // [39] * reference: https://github.com/penincillin/DREAM
  class LITE_EXPORTS NaivePoseRobustFace;        // [40] * reference: https://github.com/penincillin/DREAM
  class LITE_EXPORTS MobileFaceNet;              // [41] * reference: https://github.com/Xiaoccer/MobileFaceNet_Pytorch
  class LITE_EXPORTS CavaGhostArcFace;           // [42] * reference: https://github.com/cavalleria/cavaface.pytorch
  class LITE_EXPORTS CavaCombinedFace;           // [43] * reference: https://github.com/cavalleria/cavaface.pytorch
  class LITE_EXPORTS MobileSEFocalFace;          // [44] * reference: https://github.com/grib0ed0v/face_recognition.pytorch
  class LITE_EXPORTS EfficientEmotion7;           // [45] * reference: https://github.com/HSE-asavchenko/face-emotion-recognition
  class LITE_EXPORTS EfficientEmotion8;           // [46] * reference: https://github.com/HSE-asavchenko/face-emotion-recognition
  class LITE_EXPORTS MobileEmotion7;             // [47] * reference: https://github.com/HSE-asavchenko/face-emotion-recognition
  class LITE_EXPORTS ReXNetEmotion7;             // [48] * reference: https://github.com/HSE-asavchenko/face-emotion-recognition
  class LITE_EXPORTS PFLD98;                     // [49] * reference: https://github.com/polarisZhao/PFLD-pytorch
  class LITE_EXPORTS MobileNetV268;              // [50] * reference: https://github.com/cunjian/pytorch_face_landmark
  class LITE_EXPORTS MobileNetV2SE68;            // [51] * reference: https://github.com/cunjian/pytorch_face_landmark
  class LITE_EXPORTS PFLD68;                     // [52] * reference: https://github.com/cunjian/pytorch_face_landmark
  class LITE_EXPORTS FaceLandmark1000;           // [53] * reference: https://github.com/Single430/FaceLandmark1000
  class LITE_EXPORTS RetinaFace;                 // [54] * reference: https://github.com/biubug6/Pytorch_Retinaface
  class LITE_EXPORTS FaceBoxes;                  // [55] * reference: https://github.com/zisianw/FaceBoxes.PyTorch
  class LITE_EXPORTS YoloX;                      // [56] * reference: https://github.com/Megvii-BaseDetection/YOLOX
  class LITE_EXPORTS TinyYoloV4VOC;              // [57] * reference: https://github.com/bubbliiiing/yolov4-tiny-pytorch
  class LITE_EXPORTS TinyYoloV4COCO;             // [58] * reference: https://github.com/bubbliiiing/yolov4-tiny-pytorch
  class LITE_EXPORTS YoloR;                      // [59] * reference: https://github.com/WongKinYiu/yolor
  class LITE_EXPORTS ScaledYoloV4;               // [60] * reference: https://github.com/WongKinYiu/ScaledYOLOv4
  class LITE_EXPORTS EfficientDet;                // [61] * reference: https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch
  class LITE_EXPORTS EfficientDetD7;              // [62] * reference: https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch
  class LITE_EXPORTS EfficientDetD8;              // [63] * reference: https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch
  class LITE_EXPORTS YOLOP;                      // [64] * reference: https://github.com/hustvl/YOLOP
  class LITE_EXPORTS RobustVideoMatting;         // [65] * reference: https://github.com/PeterL1n/RobustVideoMatting
  class LITE_EXPORTS NanoDet;                    // [66] * reference: https://github.com/RangiLyu/nanodet
  class LITE_EXPORTS NanoDetEfficientNetLite;     // [67] * reference: https://github.com/RangiLyu/nanodet
  class LITE_EXPORTS YoloX_V_0_1_1;              // [68] * reference: https://github.com/Megvii-BaseDetection/YOLOX
  class LITE_EXPORTS YoloV5_V_6_0;               // [69] * reference: https://github.com/ultralytics/yolov5
  class LITE_EXPORTS MGMatting;                  // [70] * reference: https://github.com/yucornetto/MGMatting
  class LITE_EXPORTS NanoDetPlus;                // [71] * reference: https://github.com/RangiLyu/nanodet
  class LITE_EXPORTS SCRFD;                      // [72] * reference: https://github.com/deepinsight/insightface/tree/master/detection/scrfd
  class LITE_EXPORTS YOLO5Face;                  // [73] * reference: https://github.com/deepcam-cn/yolov5-face
  class LITE_EXPORTS FaceBoxesV2;                // [74] * reference: https://github.com/jhb86253817/FaceBoxesV2
  class LITE_EXPORTS PIPNet19;                   // [75] * reference: https://github.com/jhb86253817/PIPNet
  class LITE_EXPORTS PIPNet29;                   // [76] * reference: https://github.com/jhb86253817/PIPNet
  class LITE_EXPORTS PIPNet68;                   // [77] * reference: https://github.com/jhb86253817/PIPNet
  class LITE_EXPORTS PIPNet98;                   // [78] * reference: https://github.com/jhb86253817/PIPNet
  class LITE_EXPORTS InsectDet;                  // [79] * reference: https://github.com/quarrying/quarrying-insect-id
  class LITE_EXPORTS InsectID;                   // [80] * reference: https://github.com/quarrying/quarrying-insect-id
  class LITE_EXPORTS PlantID;                    // [81] * reference: https://github.com/quarrying/quarrying-plant-id
  class LITE_EXPORTS MODNet;                     // [82] * reference: https://github.com/ZHKKKe/MODNet
  class LITE_EXPORTS MODNetDyn;                  // [83] * reference: https://github.com/ZHKKKe/MODNet (With Dynamic Input Shape)
  class LITE_EXPORTS BackgroundMattingV2;        // [84] * reference: https://github.com/PeterL1n/BackgroundMattingV2
  class LITE_EXPORTS BackgroundMattingV2Dyn;     // [85] * reference: https://github.com/PeterL1n/BackgroundMattingV2 (With Dynamic Input Shape)
  class LITE_EXPORTS YOLOv5BlazeFace;            // [86] * reference: https://github.com/deepcam-cn/yolov5-face
  class LITE_EXPORTS YoloV5_V_6_1;               // [87] * reference: https://github.com/ultralytics/yolov5/releases/tag/v6.1
  class LITE_EXPORTS HeadSeg;                    // [88] * reference: https://github.com/minivision-ai/photo2cartoon
  class LITE_EXPORTS FemalePhoto2Cartoon;        // [89] * reference: https://github.com/minivision-ai/photo2cartoon
  class LITE_EXPORTS FastPortraitSeg;            // [90] * reference: https://github.com/YexingWan/Fast-Portrait-Segmentation
  class LITE_EXPORTS PortraitSegExtremeC3Net;    // [91] * reference: https://github.com/clovaai/ext_portrait_segmentation
  class LITE_EXPORTS PortraitSegSINet;           // [92] * reference: https://github.com/clovaai/ext_portrait_segmentation
  class LITE_EXPORTS FaceHairSeg;                // [93] * reference: https://github.com/kampta/face-seg
  class LITE_EXPORTS HairSeg;                    // [94] * reference: https://github.com/akirasosa/mobile-semantic-segmentation
  class LITE_EXPORTS MobileHumanMatting;         // [95] * reference: https://github.com/lizhengwei1992/mobile_phone_human_matting
  class LITE_EXPORTS MobileHairSeg;              // [96] * reference: https://github.com/wonbeomjang/mobile-hair-segmentation-pytorch
  class LITE_EXPORTS RealSR;                     // [97] * reference: https://github.com/jixiaozhong/RealSR
  class LITE_EXPORTS BSRGAN;                     // [98] * reference: https://github.com/cszn/BSRGAN
  class LITE_EXPORTS SCUNet;                     // [99] * reference: https://github.com/cszn/SCUNet
  class LITE_EXPORTS MobileHumanPose3D;          // [100] * reference: https://github.com/ibaiGorordo/ONNX-Mobile-Human-Pose-3D
  class LITE_EXPORTS FaceParsingBiSeNet;         // [101] * reference: https://github.com/zllrunning/face-parsing.PyTorch
  class LITE_EXPORTS FaceParsingBiSeNetDyn;      // [102] * reference: https://github.com/zllrunning/face-parsing.PyTorch
  class LITE_EXPORTS YOLOv6;                     // [103] * reference: https://github.com/meituan/YOLOv6
  class LITE_EXPORTS FaceMesh;                   // [104] * reference: https://github.com/google/mediapipe/tree/master/mediapipe/graphs/face_mesh
  class LITE_EXPORTS IrisLandmarks;              // [105] * reference: https://github.com/google/mediapipe/tree/master/mediapipe/graphs/iris_tracking
  class LITE_EXPORTS YoloFaceV8;                 // [106] * reference: https://github.com/derronqi/yolov8-face
}

namespace ortnlp
{
  class LITE_EXPORTS TextCNN; // todo
  class LITE_EXPORTS ChineseBert; // todo
  class LITE_EXPORTS ChineseOCRLite; // todo
}

namespace ortsd
{
    class LITE_EXPORTS Clip; // [1] * reference: https://github.com/openai/CLIP
}

namespace ortcv
{
  using core::BasicOrtHandler;
  using core::BasicMultiOrtHandler;
}
namespace ortnlp
{
  using core::BasicOrtHandler;
  using core::BasicMultiOrtHandler;
}
namespace ortasr
{
  using core::BasicOrtHandler;
  using core::BasicMultiOrtHandler;
}

namespace ortsd
{
    using core::BasicOrtHandler;
    using core::BasicMultiOrtHandler;
}

#endif //LITE_AI_ORT_ORT_CORE_H

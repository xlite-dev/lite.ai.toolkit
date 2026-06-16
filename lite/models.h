//
// Created by DefTruth on 2021/8/8.
//

#ifndef LITE_AI_MODELS_H
#define LITE_AI_MODELS_H

#include "config.h"

// ENABLE_ONNXRUNTIME
#ifdef ENABLE_ONNXRUNTIME

#include "lite/ort/core/ort_core.h"
#include "lite/ort/core/ort_utils.h"
#include "lite/ort/cv/age_googlenet.h"
#include "lite/ort/cv/glint_arcface.h"
#include "lite/ort/cv/colorizer.h"
#include "lite/ort/cv/deeplabv3_resnet101.h"
#include "lite/ort/cv/densenet.h"
#include "lite/ort/cv/efficientnet_lite4.h"
#include "lite/ort/cv/emotion_ferplus.h"
#include "lite/ort/cv/fast_style_transfer.h"
#include "lite/ort/cv/fcn_resnet101.h"
#include "lite/ort/cv/fsanet.h"
#include "lite/ort/cv/gender_googlenet.h"
#include "lite/ort/cv/ghostnet.h"
#include "lite/ort/cv/hardnet.h"
#include "lite/ort/cv/ibnnet.h"
#include "lite/ort/cv/mobilenetv2.h"
#include "lite/ort/cv/pfld.h"
#include "lite/ort/cv/resnet.h"
#include "lite/ort/cv/resnext.h"
#include "lite/ort/cv/shufflenetv2.h"
#include "lite/ort/cv/ssd.h"
#include "lite/ort/cv/ssd_mobilenetv1.h"
#include "lite/ort/cv/ssrnet.h"
#include "lite/ort/cv/subpixel_cnn.h"
#include "lite/ort/cv/tiny_yolov3.h"
#include "lite/ort/cv/ultraface.h"
#include "lite/ort/cv/vgg16_age.h"
#include "lite/ort/cv/vgg16_gender.h"
#include "lite/ort/cv/yolov3.h"
#include "lite/ort/cv/yolov4.h"
#include "lite/ort/cv/yolov5.h"
#include "lite/ort/cv/glint_cosface.h"
#include "lite/ort/cv/glint_partial_fc.h"
#include "lite/ort/cv/facenet.h"
#include "lite/ort/cv/focal_arcface.h"
#include "lite/ort/cv/focal_asia_arcface.h"
#include "lite/ort/cv/tencent_cifp_face.h"
#include "lite/ort/cv/tencent_curricular_face.h"
#include "lite/ort/cv/center_loss_face.h"
#include "lite/ort/cv/sphere_face.h"
#include "lite/ort/cv/pose_robust_face.h"
#include "lite/ort/cv/naive_pose_robust_face.h"
#include "lite/ort/cv/mobile_facenet.h"
#include "lite/ort/cv/cava_ghost_arcface.h"
#include "lite/ort/cv/cava_combined_face.h"
#include "lite/ort/cv/yolox.h"
#include "lite/ort/cv/mobilese_focal_face.h"
#include "lite/ort/cv/efficient_emotion7.h"
#include "lite/ort/cv/efficient_emotion8.h"
#include "lite/ort/cv/mobile_emotion7.h"
#include "lite/ort/cv/rexnet_emotion7.h"
#include "lite/ort/cv/pfld98.h"
#include "lite/ort/cv/pfld68.h"
#include "lite/ort/cv/mobilenetv2_68.h"
#include "lite/ort/cv/mobilenetv2_se_68.h"
#include "lite/ort/cv/face_landmarks_1000.h"
#include "lite/ort/cv/retinaface.h"
#include "lite/ort/cv/faceboxes.h"
#include "lite/ort/cv/tiny_yolov4_voc.h"
#include "lite/ort/cv/tiny_yolov4_coco.h"
#include "lite/ort/cv/yolor.h"
#include "lite/ort/cv/scaled_yolov4.h"
#include "lite/ort/cv/efficientdet.h"
#include "lite/ort/cv/efficientdet_d7.h"
#include "lite/ort/cv/efficientdet_d8.h"
#include "lite/ort/cv/yolop.h"
#include "lite/ort/cv/rvm.h"
#include "lite/ort/cv/nanodet.h"
#include "lite/ort/cv/nanodet_efficientnet_lite.h"
#include "lite/ort/cv/yolox_v0.1.1.h"
#include "lite/ort/cv/yolov5_v6.0.h"
#include "lite/ort/cv/mg_matting.h"
#include "lite/ort/cv/nanodet_plus.h"
#include "lite/ort/cv/scrfd.h"
#include "lite/ort/cv/yolo5face.h"
#include "lite/ort/cv/faceboxesv2.h"
#include "lite/ort/cv/pipnet98.h"
#include "lite/ort/cv/pipnet68.h"
#include "lite/ort/cv/pipnet29.h"
#include "lite/ort/cv/pipnet19.h"
#include "lite/ort/cv/insectdet.h"
#include "lite/ort/cv/insectid.h"
#include "lite/ort/cv/plantid.h"
#include "lite/ort/cv/modnet.h"
#include "lite/ort/cv/modnet_dyn.h"
#include "lite/ort/cv/backgroundmattingv2.h"
#include "lite/ort/cv/backgroundmattingv2_dyn.h"
#include "lite/ort/cv/yolov5_blazeface.h"
#include "lite/ort/cv/yolov5_v6.1.h"
#include "lite/ort/cv/head_seg.h"
#include "lite/ort/cv/female_photo2cartoon.h"
#include "lite/ort/cv/fast_portrait_seg.h"
#include "lite/ort/cv/portrait_seg_sinet.h"
#include "lite/ort/cv/portrait_seg_extremec3net.h"
#include "lite/ort/cv/hair_seg.h"
#include "lite/ort/cv/face_hair_seg.h"
#include "lite/ort/cv/mobile_human_matting.h"
#include "lite/ort/cv/mobile_hair_seg.h"
#include "lite/ort/cv/yolov6.h"
#include "lite/ort/cv/face_parsing_bisenet.h"
#include "lite/ort/cv/face_parsing_bisenet_dyn.h"
#include "lite/ort/cv/yolofacev8.h"
#include "lite/ort/cv/light_enhance.h"
#include "lite/ort/cv/docunwarp.h"
#include "lite/ort/cv/real_esr_gan.h"
#include "lite/ort/cv/face_68landmarks.h"
#include "lite/ort/cv/face_recognizer.h"
#include "lite/ort/cv/face_swap.h"
#include "lite/ort/cv/face_restoration.h"
#include "lite/ort/cv/face_fusion_pipeline.h"
#include "lite/ort/sd/clip.h"
#include "lite/ort/sd/unet.h"
#include "lite/ort/sd/vae.h"
#include "lite/ort/sd/pipeline.h"
#endif


// ENABLE_TRT
#ifdef ENABLE_TENSORRT

#include "lite/trt/core/trt_utils.h"
#include "lite/trt/core/trt_core.h"
#include "lite/trt/cv/trt_yolofacev8.h"
#include "lite/trt/cv/trt_yolov5.h"
#include "lite/trt/cv/trt_yolov11.h"
#include "lite/trt/cv/trt_yolox.h"
#include "lite/trt/cv/trt_yolov8.h"
#include "lite/trt/cv/trt_yolov6.h"
#include "lite/trt/cv/trt_modnet.h"
#include "lite/trt/cv/trt_yolov5_blazeface.h"
#include "lite/trt/cv/trt_lightenhance.h"
#include "lite/trt/cv/trt_realesrgan.h"
#include "lite/trt/cv/trt_face_68landmarks.h"
#include "lite/trt/cv/trt_face_recognizer.h"
#include "lite/trt/cv/trt_face_swap.h"
#include "lite/trt/cv/trt_face_restoration.h"
#include "lite/trt/cv/trt_facefusion_pipeline.h"
#include "lite/trt/sd/trt_clip.h"
#include "lite/trt/sd/trt_vae.h"
#include "lite/trt/sd/trt_unet.h"
#include "lite/trt/sd/trt_pipeline.h"
#endif

// ONNXRuntime version
namespace lite
{
#ifdef ENABLE_ONNXRUNTIME
  namespace onnxruntime
  {
    namespace cv
    {
      typedef ortcv::FSANet _ONNXFSANet;
      typedef ortcv::PFLD _ONNXPFLD;
      typedef ortcv::UltraFace _ONNXUltraFace;
      typedef ortcv::AgeGoogleNet _ONNXAgeGoogleNet;
      typedef ortcv::GenderGoogleNet _ONNXGenderGoogleNet;
      typedef ortcv::EmotionFerPlus _ONNXEmotionFerPlus;
      typedef ortcv::VGG16Age _ONNXVGG16Age;
      typedef ortcv::VGG16Gender _ONNXVGG16Gender;
      typedef ortcv::SSRNet _ONNXSSRNet;
      typedef ortcv::FastStyleTransfer _ONNXFastStyleTransfer;
      typedef ortcv::GlintArcFace _ONNXGlintArcFace;
      typedef ortcv::Colorizer _ONNXColorizer;
      typedef ortcv::SubPixelCNN _ONNXSubPixelCNN;
      typedef ortcv::YoloV4 _ONNXYoloV4;
      typedef ortcv::YoloV3 _ONNXYoloV3;
      typedef ortcv::YoloV5 _ONNXYoloV5;
      typedef ortcv::EfficientNetLite4 _ONNXEfficientNetLite4;
      typedef ortcv::ShuffleNetV2 _ONNXShuffleNetV2;
      typedef ortcv::TinyYoloV3 _ONNXTinyYoloV3;
      typedef ortcv::SSD _ONNXSSD;
      typedef ortcv::SSDMobileNetV1 _ONNXSSDMobileNetV1;
      typedef ortcv::DeepLabV3ResNet101 _ONNXDeepLabV3ResNet101;
      typedef ortcv::DenseNet _ONNXDenseNet;
      typedef ortcv::FCNResNet101 _ONNXFCNResNet101;
      typedef ortcv::GhostNet _ONNXGhostNet;
      typedef ortcv::HdrDNet _ONNXHdrDNet;
      typedef ortcv::IBNNet _ONNXIBNNet;
      typedef ortcv::MobileNetV2 _ONNXMobileNetV2;
      typedef ortcv::ResNet _ONNXResNet;
      typedef ortcv::ResNeXt _ONNXResNeXt;
      typedef ortcv::GlintCosFace _ONNXGlintCosFace;
      typedef ortcv::GlintPartialFC _ONNXGlintPartialFC;
      typedef ortcv::FaceNet _ONNXFaceNet;
      typedef ortcv::FocalArcFace _ONNXFocalArcFace;
      typedef ortcv::FocalAsiaArcFace _ONNXFocalAsiaArcFace;
      typedef ortcv::TencentCifpFace _ONNXTencentCifpFace;
      typedef ortcv::TencentCurricularFace _ONNXTencentCurricularFace;
      typedef ortcv::CenterLossFace _ONNXCenterLossFace;
      typedef ortcv::SphereFace _ONNXSphereFace;
      typedef ortcv::PoseRobustFace _ONNXPoseRobustFace;
      typedef ortcv::NaivePoseRobustFace _ONNXNaivePoseRobustFace;
      typedef ortcv::MobileFaceNet _ONNXMobileFaceNet;
      typedef ortcv::CavaGhostArcFace _ONNXCavaGhostArcFace;
      typedef ortcv::CavaCombinedFace _ONNXCavaCombinedFace;
      typedef ortcv::YoloX _ONNXYoloX;
      typedef ortcv::MobileSEFocalFace _ONNXMobileSEFocalFace;
      typedef ortcv::EfficientEmotion7 _ONNXEfficientEmotion7;
      typedef ortcv::EfficientEmotion8 _ONNXEfficientEmotion8;
      typedef ortcv::MobileEmotion7 _ONNXMobileEmotion7;
      typedef ortcv::ReXNetEmotion7 _ONNXReXNetEmotion7;
      typedef ortcv::PFLD98 _ONNXPFLD98;
      typedef ortcv::PFLD68 _ONNXPFLD68;
      typedef ortcv::MobileNetV268 _ONNXMobileNetV268;
      typedef ortcv::MobileNetV2SE68 _ONNXMobileNetV2SE68;
      typedef ortcv::FaceLandmark1000 _ONNXFaceLandmark1000;
      typedef ortcv::RetinaFace _ONNXRetinaFace;
      typedef ortcv::FaceBoxes _ONNXFaceBoxes;
      typedef ortcv::TinyYoloV4VOC _ONNXTinyYoloV4VOC;
      typedef ortcv::TinyYoloV4COCO _ONNXTinyYoloV4COCO;
      typedef ortcv::YoloR _ONNXYoloR;
      typedef ortcv::ScaledYoloV4 _ONNXScaledYoloV4;
      typedef ortcv::EfficientDet _ONNXEfficientDet;
      typedef ortcv::EfficientDetD7 _ONNXEfficientDetD7;
      typedef ortcv::EfficientDetD8 _ONNXEfficientDetD8;
      typedef ortcv::YOLOP _ONNXYOLOP;
      typedef ortcv::RobustVideoMatting _ONNXRobustVideoMatting;
      typedef ortcv::NanoDet _ONNXNanoDet;
      typedef ortcv::NanoDetEfficientNetLite _ONNXNanoDetEfficientNetLite;
      typedef ortcv::YoloX_V_0_1_1 _ONNXYoloX_V_0_1_1;
      typedef ortcv::YoloV5_V_6_0 _ONNXYoloV5_V_6_0;
      typedef ortcv::MGMatting _ONNXMGMatting;
      typedef ortcv::NanoDetPlus _ONNXNanoDetPlus;
      typedef ortcv::SCRFD _ONNXSCRFD;
      typedef ortcv::YOLO5Face _ONNXYOLO5Face;
      typedef ortcv::FaceBoxesV2 _ONNXFaceBoxesV2;
      typedef ortcv::PIPNet98 _ONNXPIPNet98;
      typedef ortcv::PIPNet68 _ONNXPIPNet68;
      typedef ortcv::PIPNet29 _ONNXPIPNet29;
      typedef ortcv::PIPNet19 _ONNXPIPNet19;
      typedef ortcv::InsectDet _ONNXInsectDet;
      typedef ortcv::InsectID _ONNXInsectID;
      typedef ortcv::PlantID _ONNXPlantID;
      typedef ortcv::MODNet _ONNXMODNet;
      typedef ortcv::MODNetDyn _ONNXMODNetDyn;
      typedef ortcv::BackgroundMattingV2 _ONNXBackgroundMattingV2;
      typedef ortcv::BackgroundMattingV2Dyn _ONNXBackgroundMattingV2Dyn;
      typedef ortcv::YOLOv5BlazeFace _ONNXYOLOv5BlazeFace;
      typedef ortcv::YoloV5_V_6_1 _ONNXYoloV5_V_6_1;
      typedef ortcv::HeadSeg _ONNXHeadSeg;
      typedef ortcv::FemalePhoto2Cartoon _ONNXFemalePhoto2Cartoon;
      typedef ortcv::FastPortraitSeg _ONNXFastPortraitSeg;
      typedef ortcv::PortraitSegSINet _ONNXPortraitSegSINet;
      typedef ortcv::PortraitSegExtremeC3Net _ONNXPortraitSegExtremeC3Net;
      typedef ortcv::HairSeg _ONNXHairSeg;
      typedef ortcv::FaceHairSeg _ONNXFaceHairSeg;
      typedef ortcv::MobileHumanMatting _ONNXMobileHumanMatting;
      typedef ortcv::MobileHairSeg _ONNXMobileHairSeg;
      typedef ortcv::YOLOv6 _ONNXYOLOv6;
      typedef ortcv::FaceParsingBiSeNet _ONNXFaceParsingBiSeNet;
      typedef ortcv::FaceParsingBiSeNetDyn _ONNXFaceParsingBiSeNetDyn;
      typedef ortcv::YoloFaceV8 _ONNXYOLOFaceNet;
      typedef ortcv::LightEnhance _ONNXLightEnhance;
      typedef ortcv::DocUnWarp _ONNXDocUnWarp;
      typedef ortcv::RealESRGAN _ONNXRealESRGAN;
      typedef ortcv::Face_68Landmarks _ONNXFace_68Landmarks;
      typedef ortcv::Face_Recognizer _ONNXFace_Recognizer;
      typedef ortcv::Face_Swap _ONNXFace_Swap;
      typedef ortcv::Face_Restoration _ONNXFace_Restoration;
      typedef ortcv::Face_Fusion_Pipeline _ONNXFace_Fusion_Pipeline;

      // 1. classification
      namespace classification
      {
        typedef _ONNXEfficientNetLite4 EfficientNetLite4;
        typedef _ONNXShuffleNetV2 ShuffleNetV2;
        typedef _ONNXDenseNet DenseNet;
        typedef _ONNXGhostNet GhostNet;
        typedef _ONNXHdrDNet HdrDNet;
        typedef _ONNXIBNNet IBNNet;
        typedef _ONNXMobileNetV2 MobileNetV2;
        typedef _ONNXResNet ResNet;
        typedef _ONNXResNeXt ResNeXt;
        typedef _ONNXInsectID InsectID;
        typedef _ONNXPlantID PlantID;
      }

      // 2. general object detection
      namespace detection
      {
        typedef _ONNXYoloV3 YoloV3;
        typedef _ONNXYoloV4 YoloV4;
        typedef _ONNXYoloV5 YoloV5;
        typedef _ONNXTinyYoloV3 TinyYoloV3;
        typedef _ONNXSSD SSD;
        typedef _ONNXSSDMobileNetV1 SSDMobileNetV1;
        typedef _ONNXYoloX YoloX;
        typedef _ONNXTinyYoloV4VOC TinyYoloV4VOC;
        typedef _ONNXTinyYoloV4COCO TinyYoloV4COCO;
        typedef _ONNXYoloR YoloR;
        typedef _ONNXScaledYoloV4 ScaledYoloV4;
        typedef _ONNXEfficientDet EfficientDet;
        typedef _ONNXEfficientDetD7 EfficientDetD7;
        typedef _ONNXEfficientDetD8 EfficientDetD8;
        typedef _ONNXYOLOP YOLOP;
        typedef _ONNXNanoDet NanoDet;
        typedef _ONNXNanoDetEfficientNetLite NanoDetEfficientNetLite;
        typedef _ONNXYoloX_V_0_1_1 YoloX_V_0_1_1;
        typedef _ONNXYoloV5_V_6_0 YoloV5_V_6_0;
        typedef _ONNXNanoDetPlus NanoDetPlus;
        typedef _ONNXInsectDet InsectDet;
        typedef _ONNXYoloV5_V_6_1 YoloV5_V_6_1;
        typedef _ONNXYOLOv6 YOLOv6;
      }
      // 3. face detection & facial attributes detection
      namespace face
      {
        namespace detect
        {
          typedef _ONNXUltraFace UltraFace;  // face detection.
          typedef _ONNXRetinaFace RetinaFace;
          typedef _ONNXFaceBoxes FaceBoxes;
          typedef _ONNXSCRFD SCRFD;
          typedef _ONNXYOLO5Face YOLO5Face;
          typedef _ONNXFaceBoxesV2 FaceBoxesV2;
          typedef _ONNXYOLOv5BlazeFace YOLOv5BlazeFace;
          typedef _ONNXYOLOFaceNet YOLOV8Face;
        }

        namespace align
        {
          typedef _ONNXPFLD PFLD; // facial landmarks detection. 106 points
          typedef _ONNXPFLD98 PFLD98; // 98 points
          typedef _ONNXPFLD68 PFLD68; // 68 points
          typedef _ONNXMobileNetV268 MobileNetV268; // 68 points
          typedef _ONNXMobileNetV2SE68 MobileNetV2SE68; // 68 points
          typedef _ONNXFaceLandmark1000 FaceLandmark1000; // 1000 points
          typedef _ONNXPIPNet98 PIPNet98; // 98 points
          typedef _ONNXPIPNet68 PIPNet68; // 68 points
          typedef _ONNXPIPNet29 PIPNet29; // 29 points
          typedef _ONNXPIPNet19 PIPNet19; // 19 points
        }

        namespace align3d
        {

        }

        namespace swap
        {
            namespace facefusion
            {
                typedef _ONNXYOLOFaceNet YOLOV8Face;
                typedef _ONNXFace_Swap InSwapper;
                typedef _ONNXFace_Restoration GFPGAN;
                typedef _ONNXFace_68Landmarks Face_68Landmarks;
                typedef _ONNXFace_Recognizer Face_Recognizer;
                typedef _ONNXFace_Fusion_Pipeline PipeLine;
            }
            typedef _ONNXFace_Swap InSwapper;
        }

        namespace restoration
        {
            typedef _ONNXFace_Restoration GFPGAN;
        }

        namespace pose
        {
          typedef _ONNXFSANet FSANet; // head pose estimation.
        }

        namespace attr
        {
          typedef _ONNXAgeGoogleNet AgeGoogleNet; // age estimation
          typedef _ONNXGenderGoogleNet GenderGoogleNet; // gender estimation
          typedef _ONNXVGG16Age VGG16Age; // age estimation
          typedef _ONNXVGG16Gender VGG16Gender; // gender estimation
          typedef _ONNXEmotionFerPlus EmotionFerPlus; // emotion detection
          typedef _ONNXSSRNet SSRNet; // age estimation
          typedef _ONNXEfficientEmotion7 EfficientEmotion7;
          typedef _ONNXEfficientEmotion8 EfficientEmotion8;
          typedef _ONNXMobileEmotion7 MobileEmotion7;
          typedef _ONNXReXNetEmotion7 ReXNetEmotion7;
        }
      }
      // 4. face recognition
      namespace faceid
      {
        typedef _ONNXGlintArcFace GlintArcFace; //
        typedef _ONNXGlintCosFace GlintCosFace; //
        typedef _ONNXGlintPartialFC GlintPartialFC;
        typedef _ONNXFaceNet FaceNet;
        typedef _ONNXFocalArcFace FocalArcFace;
        typedef _ONNXFocalAsiaArcFace FocalAsiaArcFace;
        typedef _ONNXTencentCifpFace TencentCifpFace;
        typedef _ONNXTencentCurricularFace TencentCurricularFace;
        typedef _ONNXCenterLossFace CenterLossFace;
        typedef _ONNXSphereFace SphereFace;
        typedef _ONNXPoseRobustFace PoseRobustFace;
        typedef _ONNXNaivePoseRobustFace NaivePoseRobustFace;
        typedef _ONNXMobileFaceNet MobileFaceNet;
        typedef _ONNXCavaGhostArcFace CavaGhostArcFace;
        typedef _ONNXCavaCombinedFace CavaCombinedFace;
        typedef _ONNXMobileSEFocalFace MobileSEFocalFace;
        typedef _ONNXFace_68Landmarks Face_68Landmarks;
        typedef _ONNXFace_Recognizer Face_Recognizer;
      }
      // 5. segmentation
      namespace segmentation
      {
        typedef _ONNXDeepLabV3ResNet101 DeepLabV3ResNet101;
        typedef _ONNXFCNResNet101 FCNResNet101;
        typedef _ONNXHeadSeg HeadSeg;
        typedef _ONNXFastPortraitSeg FastPortraitSeg;
        typedef _ONNXPortraitSegSINet PortraitSegSINet;
        typedef _ONNXPortraitSegExtremeC3Net PortraitSegExtremeC3Net;
        typedef _ONNXHairSeg HairSeg;
        typedef _ONNXFaceHairSeg FaceHairSeg;
        typedef _ONNXMobileHairSeg MobileHairSeg;
        typedef _ONNXFaceParsingBiSeNet FaceParsingBiSeNet;
        typedef _ONNXFaceParsingBiSeNetDyn FaceParsingBiSeNetDyn;
      }
      // 6. reid
      namespace reid
      {

      }

      // 7. ocr
      namespace ocr
      {
          typedef _ONNXDocUnWarp DocUnWarp;
      }
      // 8. neural rendering
      namespace render
      {

      }
      // 9. style transfer
      namespace style
      {
        typedef _ONNXFastStyleTransfer FastStyleTransfer;
        typedef _ONNXFemalePhoto2Cartoon FemalePhoto2Cartoon;
      }

      // 10. colorization
      namespace colorization
      {
        typedef _ONNXColorizer Colorizer;
      }
      namespace lightenhance
      {
          typedef  _ONNXLightEnhance LightEnhance;
      }
      namespace upscale
      {
          typedef  _ONNXRealESRGAN RealESRGAN;
      }
      // 11. super resolution
      namespace resolution
      {
        typedef _ONNXSubPixelCNN SubPixelCNN;
      }
      // 12. image & face & human matting
      namespace matting
      {
        typedef _ONNXRobustVideoMatting RobustVideoMatting;
        typedef _ONNXMGMatting MGMatting;
        typedef _ONNXMODNet MODNet;
        typedef _ONNXMODNetDyn MODNetDyn;
        typedef _ONNXBackgroundMattingV2 BackgroundMattingV2;
        typedef _ONNXBackgroundMattingV2Dyn BackgroundMattingV2Dyn;
        typedef _ONNXMobileHumanMatting MobileHumanMatting;
      }
    }
    namespace sd
    {
        typedef ortsd::Clip _ONNXClip;
        typedef ortsd::UNet _ONNXUNet;
        typedef ortsd::Vae _ONNXVae;
        typedef ortsd::Pipeline _ONNXPipeline;
        namespace text_encoder
        {
            typedef _ONNXClip Clip;
        }
        namespace denoise
        {
            typedef _ONNXUNet UNet;
        }
        namespace image_decoder
        {
            typedef _ONNXVae Vae;
        }
        namespace pipeline
        {
            typedef _ONNXPipeline Pipeline;
        }
    }

  }
#endif
}


// TRT version
namespace lite{
#ifdef ENABLE_TENSORRT
    namespace trt
    {
        namespace cv
        {
            typedef trtcv::TRTYoloFaceV8 _TRT_YOLOFaceNet;
            typedef trtcv::TRTYoloV5 _TRT_YOLOv5;
            typedef trtcv::TRTYoloV8 _TRT_YOLOv8;
            typedef trtcv::TRTYOLOV11 _TRT_YOLOV11;
            typedef trtcv::TRTYoloX _TRT_YoloX;
            typedef trtcv::TRTYoloV6 _TRT_YOLOv6;
            typedef trtcv::TRTYOLO5Face _TRT_YOLO5Face;
            typedef trtcv::TRTLightEnhance _TRT_LightEnhance;
            typedef trtcv::TRTRealESRGAN _TRT_RealESRGAN;
            typedef trtcv::TRTMODNet _TRT_MODNet;
            typedef trtcv::TRTFaceFusionFace68Landmarks _TRT_FaceFusionFace68Landmarks;
            typedef trtcv::TRTFaceFusionFaceRecognizer _TRTFaceFusionFaceRecognizer;
            typedef trtcv::TRTFaceFusionFaceSwap _TRTFaceFusionFaceSwap;
            typedef trtcv::TRTFaceFusionFaceRestoration _TRTFaceFusionFaceRestoration;
            typedef trtcv::TRTFaceFusionPipeLine _TRTFaceFusionPipeLine;
            namespace classification
            {

            }
            namespace matting
            {
                typedef _TRT_MODNet MODNet;
            }
            namespace detection
            {
                typedef _TRT_YOLOv5 YOLOV5;
                typedef _TRT_YOLOv8 YOLOV8;
                typedef _TRT_YoloX YoloX;
                typedef _TRT_YOLOv6 YOLOV6;
                typedef _TRT_YOLOV11 YOLOV11;
            }
            namespace face
            {
                namespace detection
                {
                    typedef _TRT_YOLOFaceNet YOLOV8Face;
                    typedef _TRT_YOLO5Face  YOLOV5Face;
                }
                namespace swap
                {
                    typedef _TRTFaceFusionFaceSwap FaceFusionFaceSwap;
                    typedef _TRTFaceFusionPipeLine FaceFusionPipeLine;
                }
                namespace restoration
                {
                    typedef _TRTFaceFusionFaceRestoration TRTGFPGAN;
                }
            }
            namespace faceid
            {
                typedef _TRT_FaceFusionFace68Landmarks FaceFusionFace68Landmarks;
                typedef _TRTFaceFusionFaceRecognizer FaceFusionFaceRecognizer;
            }
            namespace lightenhance
            {
                typedef _TRT_LightEnhance LightEnhance;
            }
            namespace upscale
            {
                typedef _TRT_RealESRGAN RealESRGAN;
            }
        }

        namespace sd
        {
            typedef trtsd::TRTUNet _TRT_UNet;
            typedef trtsd::TRTClip _TRT_Clip;
            typedef trtsd::TRTVae _TRT_Vae;
            typedef trtsd::TRTPipeline _TRT_Pipeline;
            namespace text_encoder
            {
                typedef _TRT_Clip Clip;
            }
            namespace image_decoder
            {
                typedef _TRT_Vae Vae;
            }
            namespace denoise
            {
                typedef _TRT_UNet UNet;
            }
            namespace pipeline
            {
                typedef _TRT_Pipeline PipeLine;
            }
        }
    }
#endif
}



// Default Engine ONNXRuntime
namespace lite
{
#if defined(ENABLE_ONNXRUNTIME)
  namespace cv = lite::onnxruntime::cv;
#endif

}

#endif //LITE_AI_MODELS_H

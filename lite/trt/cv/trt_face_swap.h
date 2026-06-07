//
// Created by wangzijian on 11/13/24.
//

#ifndef LITE_AI_TOOLKIT_TRT_FACE_SWAP_H
#define LITE_AI_TOOLKIT_TRT_FACE_SWAP_H
#include "lite/ort/cv/face_utils.h"
#include "lite/trt/core/trt_core.h"
#include "lite/trt/core/trt_utils.h"
#include "lite/trt/core/trt_types.h"
#include "lite/trt/kernel/face_swap_postproces_manager.h"
#include "lite/trt/kernel/paste_back_manager.h"
#include "lite/trt/kernel/device_frame.h"
#include "lite/trt/kernel/warp_affine_npp.h"
#include "lite/trt/kernel/face_restoration_preprocess_manager.h"

namespace trtcv{
    class LITE_EXPORTS TRTFaceFusionFaceSwap : BasicTRTHandler{
    public:
        explicit TRTFaceFusionFaceSwap(const std::string& _trt_model_path,unsigned int _num_threads = 1):
                BasicTRTHandler(_trt_model_path,_num_threads){
            // Constant inputs — load/build once here (used to be done every frame in preprocess:
            // a load_npy() disk read and a create_static_box_mask() rebuild).
            model_matrix_ = face_utils::load_npy(std::string(SOURCE_PATH) + "/examples/lite/resources/model_matrix.npy");
            box_mask_ = face_utils::create_static_box_mask(std::vector<float>{128.0f, 128.0f});
        };
    private:
        cv::Mat affine_martix;
        std::vector<float> model_matrix_;   // loaded once in ctor (was load_npy every frame)
        cv::Mat box_mask_;                  // cached static 128 box mask (was rebuilt every frame)
        PasteBackGPU paste_back_gpu_;   // GPU-fused paste-back, reused device buffers (same as restoration)
        WarpAffineNpp warp_npp_;            // GPU (NPP) affine warp for the 128 crop (was CPU warpAffine)
        FaceRestorePreprocessGPU preprocess_gpu_;  // fused bgr2rgb+/255+CHW straight into buffers[0]
        DeviceFrame target_dev_;            // target frame uploaded ONCE; shared by warp + paste-back
    public:
        void detect(cv::Mat &target_image,std::vector<float> source_face_embeding,std::vector<cv::Point2f> target_landmark_5,
                    cv::Mat &face_swap_image);

        // Device-pipeline variant: the swapped full frame stays GPU-resident in out_frame (paste
        // writes straight to device, no D2H) so restoration can consume it without re-uploading.
        void detect(cv::Mat &target_image,std::vector<float> source_face_embeding,std::vector<cv::Point2f> target_landmark_5,
                    DeviceFrame &out_frame);

    private:
        // shared body: infer + postprocess into the host BGR float crop `mat` + its affine.
        void swap_core(cv::Mat &target_image, std::vector<float> &source_face_embeding,
                       std::vector<cv::Point2f> &target_landmark_5, cv::Mat &mat_out);
    };
}


#endif //LITE_AI_TOOLKIT_TRT_FACE_SWAP_H

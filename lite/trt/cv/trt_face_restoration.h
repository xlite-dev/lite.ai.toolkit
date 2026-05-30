//
// Created by wangzijian on 11/14/24.
//

#ifndef LITE_AI_TOOLKIT_TRT_FACE_RESTORATION_H
#define LITE_AI_TOOLKIT_TRT_FACE_RESTORATION_H
#include "lite/trt/core/trt_core.h"
#include "lite/trt/core/trt_utils.h"
#include "lite/trt/core/trt_config.h"
#include "lite/ort/cv/face_utils.h"
#include "lite/trt/kernel/face_restoration_postprocess_manager.h"
#include "lite/trt/kernel/bgr2rgb_manager.h"
#include "lite/trt/kernel/paste_back_manager.h"

// Forward declaration for benchmark timing; library passes nullptr by default (zero overhead)
namespace lite { namespace bench { class Profiler; } }

namespace trtcv{
    class LITE_EXPORTS TRTFaceFusionFaceRestoration : BasicTRTHandler{
    public:
        explicit TRTFaceFusionFaceRestoration(const std::string& _trt_model_path,unsigned int _num_threads = 1) :
                BasicTRTHandler(_trt_model_path,_num_threads){};;
    public:
        // writes the restored frame straight to disk
        void detect(cv::Mat &face_swap_image,std::vector<cv::Point2f > &target_landmarks_5 ,const std::string &face_enchaner_path);

        // Core compute: returns the restored full frame without writing to disk; when prof is
        // non-null, records per-stage timings (preprocess / infer / postprocess / paste_back).
        cv::Mat restore(cv::Mat &face_swap_image, std::vector<cv::Point2f> &target_landmarks_5,
                        lite::bench::Profiler *prof = nullptr);

    private:
        PasteBackGPU paste_back_gpu_;  // GPU fused paste_back, reuses device buffers

    };
}

#endif //LITE_AI_TOOLKIT_TRT_FACE_RESTORATION_H

//
// Created by wangzijian on 24-7-11.
//

#ifndef LITE_AI_TOOLKIT_TRT_CORE_H
#define LITE_AI_TOOLKIT_TRT_CORE_H

#include "trt_config.h"
#include "trt_handler.h"
#include "trt_types.h"

namespace trtcv{
    class LITE_EXPORTS TRTYoloFaceV8; // [1] * reference: https://github.com/derronqi/yolov8-face
    class LITE_EXPORTS TRTYoloV5;     // [2] * reference: https://github.com/ultralytics/yolov5
    class LITE_EXPORTS TRTYoloX;      // [3] * reference: https://github.com/Megvii-BaseDetection/YOLOX
    class LITE_EXPORTS TRTYoloV8;     // [4] * reference: https://github.com/ultralytics/ultralytics/tree/main
    class LITE_EXPORTS TRTYoloV6;     // [5] * reference: https://github.com/meituan/YOLOv6
    class LITE_EXPORTS TRTYOLO5Face;     // [6] * reference: https://github.com/deepcam-cn/yolov5-face
}

namespace trtcv{
    using trtcore::BasicTRTHandler;
}

namespace trtsd{
    class LITE_EXPORTS TRTClip;
}



#endif //LITE_AI_TOOLKIT_TRT_CORE_H

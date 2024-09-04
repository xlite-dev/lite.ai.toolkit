//
// Created by DefTruth on 2021/3/29.
//

#ifndef LITE_AI_ORT_CORE_ORT_CONFIG_H
#define LITE_AI_ORT_CORE_ORT_CONFIG_H

#include "ort_defs.h"
#include "lite/lite.ai.headers.h"

#ifdef ENABLE_ONNXRUNTIME
#include "onnxruntime_cxx_api.h"
#endif

inline static std::string OrtCompatiableGetInputName(size_t index, OrtAllocator* allocator, 
                                               Ort::Session *ort_session) {
#if ORT_API_VERSION >= 14
    return std::string(ort_session->GetInputNameAllocated(index, allocator).get());
#else  
    return std::string(ort_session->GetInputName(index, allocator));
#endif
}

inline static std::string OrtCompatiableGetOutputName(size_t index, OrtAllocator* allocator, 
                                                      Ort::Session *ort_session) {
#if ORT_API_VERSION >= 14
    return std::string(ort_session->GetOutputNameAllocated(index, allocator).get());
#else  
    return std::string(ort_session->GetOutputName(index, allocator));
#endif
}

namespace core {}

#endif //LITE_AI_ORT_CORE_ORT_CONFIG_H

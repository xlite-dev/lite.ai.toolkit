set(CUDA_DIR "" CACHE PATH "If build tensorrt backend, need to define path of cuda library.")
set(TensorRT_DIR "" CACHE PATH "If build tensorrt backend, need to define path of tensorrt library.")

if(NOT CUDA_DIR)
  set(CUDA_DIR "/usr/local/cuda")
  message(STATUS "CUDA_DIR is not defined, use default dir: ${CUDA_DIR}")
else()
  message(STATUS "custom CUDA_DIR is defined as: ${CUDA_DIR}") 
endif()

if(NOT TensorRT_DIR)
  set(TensorRT_DIR "/usr/local/tensorrt")
  message(STATUS "TensorRT_DIR is not defined, use default dir: ${TensorRT_DIR}")
else()
  message(STATUS "custom TensorRT_DIR is defined as: ${TensorRT_DIR}") 
endif()

# TODO: download tensorrt need user operation if trt doesn't exist
if(NOT EXISTS ${CUDA_DIR})
    message(FATAL_ERROR "[Lite.AI.Toolkit][E] ${CUDA_DIR} is not exists! Please define -DCUDA_DIR=xxx while TensorRT Backend is enabled.")
endif()

if(NOT EXISTS ${TensorRT_DIR})
    message(FATAL_ERROR "[Lite.AI.Toolkit][E] ${TensorRT_DIR} is not exists! Please define -DTensorRT_DIR=xxx while TensorRT Backend is enabled.")
endif()

execute_process(COMMAND sh -c "nm -D libnvinfer.so | grep tensorrt_version" 
                WORKING_DIRECTORY ${TensorRT_DIR}/lib
                RESULT_VARIABLE result
                OUTPUT_VARIABLE curr_out
                ERROR_VARIABLE  curr_out)

string(STRIP ${curr_out} TensorRT_Version)
set(TensorRT_Version ${TensorRT_Version} CACHE STRING "TensorRT version" FORCE)

include_directories(${CUDA_DIR}/include)
link_directories(${CUDA_DIR}/lib64)

include_directories(${TensorRT_DIR}/include)
link_directories(${TensorRT_DIR}/lib)

# 1. glob sources files
file(GLOB TENSORRT_CORE_SRCS ${CMAKE_SOURCE_DIR}/lite/trt/core/*.cpp)
file(GLOB TENSORRT_CUDA_KERNEL_SRCS_CPP ${CMAKE_SOURCE_DIR}/lite/trt/kernel/*.cpp)
file(GLOB TENSORRT_CUDA_KERNEL_SRCS_CU ${CMAKE_SOURCE_DIR}/lite/trt/kernel/*.cu)
file(GLOB TENSORRT_CV_SRCS ${CMAKE_SOURCE_DIR}/lite/trt/cv/*.cpp)
file(GLOB TENSORRT_NLP_SRCS ${CMAKE_SOURCE_DIR}/lite/trt/nlp/*.cpp)
file(GLOB TENSORRT_ASR_SRCS ${CMAKE_SOURCE_DIR}/lite/trt/asr/*.cpp)
file(GLOB TENSORRT_SD_SRCS ${CMAKE_SOURCE_DIR}/lite/trt/sd/*.cpp)

# 2. glob headers files
file(GLOB TENSORRT_CORE_HEAD ${CMAKE_SOURCE_DIR}/lite/trt/core/*.h)
file(GLOB TENSORRT_CV_HEAD ${CMAKE_SOURCE_DIR}/lite/trt/cv/*.h)
file(GLOB TENSORRT_NLP_HEAD ${CMAKE_SOURCE_DIR}/lite/trt/nlp/*.h)
file(GLOB TENSORRT_ASR_HEAD ${CMAKE_SOURCE_DIR}/lite/trt/asr/*.h)
file(GLOB TENSORRT_SD_HEAD ${CMAKE_SOURCE_DIR}/lite/trt/sd/*.h)
file(GLOB TENSORRT_CUDA_KERNEL_HEAD_CPP ${CMAKE_SOURCE_DIR}/lite/trt/kernel/*.h)
file(GLOB TENSORRT_CUDA_KERNEL_HEAD_CU ${CMAKE_SOURCE_DIR}/lite/trt/kernel/*.cuh)



set(TRT_SRCS ${TENSORRT_CV_SRCS} ${TENSORRT_NLP_SRCS} ${TENSORRT_ASR_SRCS} ${TENSORRT_CORE_SRCS} ${TENSORRT_SD_SRCS}
        ${TENSORRT_CUDA_KERNEL_SRCS_CPP} ${TENSORRT_CUDA_KERNEL_SRCS_CU})
set_source_files_properties(${TENSORRT_CUDA_KERNEL_SRCS_CU} ${TENSORRT_CUDA_KERNEL_SRCS_CPP}
        ${TENSORRT_CUDA_KERNEL_HEAD_CPP} ${TENSORRT_CUDA_KERNEL_HEAD_CU}
        PROPERTIES LANGUAGE CUDA)

# 3. copy
message("[Lite.AI.Toolkit][I] Installing Lite.AI.ToolKit Headers for TensorRT Backend ...")
# "INSTALL" can copy all files from the list to the specified path.
# "COPY" only copies one file to a specified path
file(INSTALL ${TENSORRT_CORE_HEAD} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/lite/trt/core)
file(INSTALL ${TENSORRT_CV_HEAD} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/lite/trt/cv)
file(INSTALL ${TENSORRT_ASR_HEAD} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/lite/trt/asr)
file(INSTALL ${TENSORRT_NLP_HEAD} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/lite/trt/nlp)
file(INSTALL ${TENSORRT_SD_HEAD} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/lite/trt/sd)
file(INSTALL ${TENSORRT_CUDA_KERNEL_HEAD_CPP} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/lite/trt/kernel)
file(INSTALL ${TENSORRT_CUDA_KERNEL_HEAD_CU} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/lite/trt/kernel)

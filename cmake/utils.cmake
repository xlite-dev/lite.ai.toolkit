function(download_and_decompress url filename decompress_dir)
  if(NOT EXISTS ${filename})
    message("[Lite.AI.Toolkit][I] Downloading file from ${url} to ${filename} ...")
    file(DOWNLOAD ${url} ${CMAKE_CURRENT_BINARY_DIR}/${filename}.tmp SHOW_PROGRESS)
    if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/${filename}.tmp)
      message(FATAL_ERROR "Can not found ${filename}.tmp!")
    endif()
    file(RENAME ${CMAKE_CURRENT_BINARY_DIR}/${filename}.tmp ${CMAKE_CURRENT_BINARY_DIR}/${filename})
  endif()
  if(NOT EXISTS ${decompress_dir})
    file(MAKE_DIRECTORY ${decompress_dir})
  endif()
  message("[Lite.AI.Toolkit][I] Decompress file ${filename} ...")
  execute_process(COMMAND ${CMAKE_COMMAND} -E tar -xf ${CMAKE_CURRENT_BINARY_DIR}/${filename} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  string(REGEX REPLACE ".tgz|.zip|.tar.gz|.tar" "" strip_filename ${filename})
  file(RENAME ${CMAKE_CURRENT_BINARY_DIR}/${strip_filename} ${decompress_dir})
endfunction()

function(_create_syslink_if_not_found lib_dir src_lib dest_lib)
  if (NOT EXISTS ${lib_dir}/${dest_lib})
    if (EXISTS ${lib_dir}/${src_lib})
      message("[Lite.AI.Toolkit][I] CREATE_LINK ${lib_dir}: ${src_lib} -> ${dest_lib}")
      file(CREATE_LINK ${lib_dir}/${src_lib} ${lib_dir}/${dest_lib})
    endif()
  endif()
endfunction()

function(create_ffmpeg_syslinks_if_not_found lib_dir)
  _create_syslink_if_not_found(${lib_dir} libavcodec.so libavcodec.so.58)
  _create_syslink_if_not_found(${lib_dir} libavformat.so libavformat.so.58)
  _create_syslink_if_not_found(${lib_dir} libavutil.so libavutil.so.56)
  _create_syslink_if_not_found(${lib_dir} libswscale.so libswscale.so.5)
  _create_syslink_if_not_found(${lib_dir} libswresample.so libswresample.so.3)
endfunction()

# config lite.ai shared lib.
function(add_lite_ai_toolkit_shared_library version soversion)
    configure_file(
            "${CMAKE_SOURCE_DIR}/lite/config.h.in"
            "${CMAKE_SOURCE_DIR}/lite/config.h"
    )
    file(GLOB LITE_SRCS ${CMAKE_SOURCE_DIR}/lite/*.cpp)
    set(LITE_DEPENDENCIES ${OpenCV_LIBS})

    if (ENABLE_ONNXRUNTIME)
        include(cmake/onnxruntime.cmake)
        set(LITE_SRCS ${LITE_SRCS} ${ORT_SRCS})
        set(LITE_DEPENDENCIES ${LITE_DEPENDENCIES} onnxruntime ddim_scheduler_cpp)
        link_directories(${CMAKE_SOURCE_DIR}/lite/bin)
    endif ()

    if (ENABLE_TENSORRT)
        include(cmake/tensorrt.cmake)
        set(LITE_SRCS ${LITE_SRCS} ${TRT_SRCS})
        set(LITE_DEPENDENCIES ${LITE_DEPENDENCIES} cudart nvinfer nvonnxparser 
                                                   nvinfer_plugin ddim_scheduler_cpp)
        link_directories(${CMAKE_SOURCE_DIR}/lite/bin)
    endif ()

    if (ENABLE_MNN)
        include(cmake/MNN.cmake)
        set(LITE_SRCS ${LITE_SRCS} ${MNN_SRCS})
        set(LITE_DEPENDENCIES ${LITE_DEPENDENCIES} MNN)
    endif ()

    if (ENABLE_NCNN)
        include(cmake/ncnn.cmake)
        set(LITE_SRCS ${LITE_SRCS} ${NCNN_SRCS})
        set(LITE_DEPENDENCIES ${LITE_DEPENDENCIES} ncnn)
    endif ()

    if (ENABLE_TNN)
        include(cmake/TNN.cmake)
        set(LITE_SRCS ${LITE_SRCS} ${TNN_SRCS})
        set(LITE_DEPENDENCIES ${LITE_DEPENDENCIES} TNN)
    endif ()

    # 4. shared library
    add_library(lite.ai.toolkit SHARED ${LITE_SRCS})
    target_link_libraries(lite.ai.toolkit ${LITE_DEPENDENCIES})
    set_target_properties(lite.ai.toolkit PROPERTIES VERSION ${version} SOVERSION ${soversion})
    message("[Lite.AI.Toolkit][I] Added Shared Library: lite.ai.toolkit !")

endfunction()

function(add_lite_executable executable_name field)
    add_executable(${executable_name} ${field}/test_${executable_name}.cpp)
    target_link_libraries(${executable_name} lite.ai.toolkit)  # link lite.ai.toolkit
    message("[Lite.AI.Toolkit][I] Added Lite Executable: ${executable_name} !")
endfunction()


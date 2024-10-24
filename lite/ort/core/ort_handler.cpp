//
// Created by DefTruth on 2021/3/30.
//
#include "ort_handler.h"

#ifdef LITE_WIN32
#include "lite/utils.h"
#endif

using core::BasicOrtHandler;
using core::BasicMultiOrtHandler;

//**************************************BasicOrtHandler************************************/
BasicOrtHandler::BasicOrtHandler(
    const std::string &_onnx_path, unsigned int _num_threads) :
    log_id(_onnx_path.data()), num_threads(_num_threads)
{
#ifdef LITE_WIN32
  std::wstring _w_onnx_path(lite::utils::to_wstring(_onnx_path));
  onnx_path = _w_onnx_path.data();
#else
  onnx_path = _onnx_path.data();
#endif
  initialize_handler();
}

void BasicOrtHandler::initialize_handler()
{
  ort_env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, log_id);
  // 0. session options
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(num_threads);
  session_options.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_ALL);
  session_options.SetLogSeverityLevel(4);

  // GPU compatiable.
  // OrtCUDAProviderOptions provider_options;
  // session_options.AppendExecutionProvider_CUDA(provider_options);
#ifdef USE_CUDA
  OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0); // C API stable.
#endif
  // 1. session
  ort_session = new Ort::Session(ort_env, onnx_path, session_options);

  // Ort::AllocatorWithDefaultOptions allocator;
  Ort::Allocator allocator(*ort_session, memory_info_handler);
  // 2. input name & input dims
  input_node_names.resize(1);
  input_node_names_.resize(1);
  input_node_names_[0] = OrtCompatiableGetInputName(0, allocator, ort_session);
  input_node_names[0] = input_node_names_[0].data();

  // 3. type info.
  Ort::TypeInfo type_info = ort_session->GetInputTypeInfo(0);
  auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
  input_tensor_size = 1;
  input_node_dims = tensor_info.GetShape();
  for (unsigned int i = 0; i < input_node_dims.size(); ++i)
  {
      if (input_node_dims.at(i) == -1)
      {
          input_tensor_size = input_tensor_size;
      }else
      {
          input_tensor_size *= input_node_dims.at(i);
      }
  }
  input_values_handler.resize(input_tensor_size);
  // 4. output names & output dimms
  num_outputs = ort_session->GetOutputCount();
  output_node_names.resize(num_outputs);
  output_node_names_.resize(num_outputs);
  for (unsigned int i = 0; i < num_outputs; ++i)
  {
    output_node_names_[i] = OrtCompatiableGetOutputName(i, allocator, ort_session);
    output_node_names[i] = output_node_names_[i].data();
    Ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
    auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
    auto output_dims = output_tensor_info.GetShape();
    output_node_dims.push_back(output_dims);
  }
#if LITEORT_DEBUG
  this->print_debug_string();
#endif
}

BasicOrtHandler::~BasicOrtHandler()
{
  if (ort_session)
    delete ort_session;
  ort_session = nullptr;
}

void BasicOrtHandler::print_debug_string()
{
    std::cout << "LITEORT_DEBUG LogId: " << onnx_path << "\n";
    std::cout << "=============== Input-Dims ==============\n";
    std::cout << "Name: " << input_node_names[0] << "\n";
    for (unsigned int i = 0; i < input_node_dims.size(); ++i) {
        // 判断是否为动态维度（-1）
        if (input_node_dims.at(i) == -1)
            std::cout << "Dims: dynamic\n";
        else
            std::cout << "Dims: " << input_node_dims.at(i) << "\n";
    }

    std::cout << "=============== Output-Dims ==============\n";
    for (unsigned int i = 0; i < num_outputs; ++i) {
        for (unsigned int j = 0; j < output_node_dims.at(i).size(); ++j) {
            // 判断是否为动态维度（-1）
            if (output_node_dims.at(i).at(j) == -1)
                std::cout << "Output: " << i << " Name: " << output_node_names.at(i)
                          << " Dim: " << j << " : dynamic\n";
            else
                std::cout << "Output: " << i << " Name: " << output_node_names.at(i)
                          << " Dim: " << j << " :" << output_node_dims.at(i).at(j) << "\n";
        }
    }
    std::cout << "========================================\n";
}


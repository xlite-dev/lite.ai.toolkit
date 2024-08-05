//
// Created by wangzijian on 8/5/24.
//

#include "clip.h"
using ortsd::Clip;


Clip::Clip(const std::string &_onnx_path, unsigned int _num_threads)  :
        log_id(_onnx_path.data()), num_threads(_num_threads){

#ifdef LITE_WIN32
    std::wstring _w_onnx_path(lite::utils::to_wstring(_onnx_path));
  onnx_path = _w_onnx_path.data();
#else
    onnx_path = _onnx_path.data();
#endif
    ort_env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, log_id);
    // 0. session options
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(num_threads);
    session_options.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    session_options.SetLogSeverityLevel(4);

#ifdef USE_CUDA
    OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0); // C API stable.
#endif
    ort_session = new Ort::Session(ort_env, onnx_path, session_options);
#if LITEORT_DEBUG
    std::cout << "Load " << onnx_path << " done!" << std::endl;
#endif
}

Clip::~Clip() {

    if (ort_session)
        delete ort_session;
    ort_session = nullptr;
}


void Clip::inference(std::vector<std::string> input, std::vector<std::vector<float>> &output) {

    std::vector<std::vector<int>> output_encode;

    encode_text(input,output_encode);


    // flat out output_encode
    std::vector<int32_t> flat_output_encode;
    for (const auto& vec : output_encode) {
        flat_output_encode.insert(flat_output_encode.end(), vec.begin(), vec.end());
    }


    // make tensor
    int batch = output_encode.size();
    std::vector<int64_t> input_node_dims1 = {batch, 77};

    Ort::MemoryInfo allocator_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);


    auto inputTensor = Ort::Value::CreateTensor<int32_t>(
            allocator_info,
            flat_output_encode.data(),
            flat_output_encode.size(),
            input_node_dims1.data(),
            input_node_dims1.size()
    );

    Ort::RunOptions runOptions;

    // run inference
    std::vector<Ort::Value> ort_outputs = ort_session->Run(
            runOptions,
            input_node_names.data(),
            &inputTensor,
            1,
            output_node_names.data(),
            output_node_names.size()
    );

    const float *text_feature_ptr = ort_outputs[0].GetTensorMutableData<float>();

    for (int i = 0 ; i < batch ; ++i)
    {
        std::vector<float> temp;
        for (int j = 0 ; j < 512 ; ++j)
        {
            temp.push_back(text_feature_ptr[ i * 512 + j]);
        }
        output.push_back(temp);
        temp.clear();
    }



}


void Clip::encode_text(std::vector<std::string> input_text, std::vector<std::vector<int>> &output) {
    CLIPTokenizer tokenizer(VERSION_1_x);
    std::string str(reinterpret_cast<char*>(merges_utf8_c_str),sizeof(merges_utf8_c_str));
    tokenizer.load_from_merges(str);

    auto on_new_token_cb = [](std::string& str, std::vector<int32_t>& tokens) -> bool {
        return false;
    };

    for (int i = 0 ; i < input_text.size(); ++i)
    {
       auto temp = tokenizer.tokenize(input_text[i], on_new_token_cb);
       temp.push_back(49407);
        if (temp.size() < 77) {
            temp.resize(77, 0);
        }
       output.push_back(temp);
    }

}



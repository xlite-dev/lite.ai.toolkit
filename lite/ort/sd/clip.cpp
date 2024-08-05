//
// Created by wangzijian on 8/5/24.
//

#include "clip.h"
using ortsd::Clip;


void Clip::inference(std::vector<std::string> input, std::vector<float> &output) {
    std::vector<int> output_encode;
    encode_text(input,output_encode);

    int token_length = 77;

    std::vector<int32_t> text_features_input(token_length);

    for (int i = 0; i < output_encode.size(); ++i) {
        text_features_input[i] = static_cast<int32_t>(output_encode[i]);
    }

    std::vector<int64_t> input_node_dims1 = {1, 77};
    Ort::MemoryInfo allocator_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);


    auto inputTensor = Ort::Value::CreateTensor<int32_t>(
            allocator_info,
            text_features_input.data(),
            text_features_input.size(),
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
    int num_output = output_node_dims[0][1];
    for (int i = 0 ; i < num_output ; ++i)
    {
        output.push_back(text_feature_ptr[i]);
    }
}



void Clip::inference(std::vector<int> input, std::vector<float> &output) {

    int token_length = 77;

    std::vector<int32_t> text_features_input(token_length);

    for (int i = 0; i < input.size(); ++i) {
        text_features_input[i] = static_cast<int32_t>(input[i]);
    }

    std::vector<int64_t> input_node_dims1 = {1, 77};
    Ort::MemoryInfo allocator_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);


    auto inputTensor = Ort::Value::CreateTensor<int32_t>(
            allocator_info,
            text_features_input.data(),
            text_features_input.size(),
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
    int num_output = output_node_dims[0][1];
    for (int i = 0 ; i < num_output ; ++i)
    {
        output.push_back(text_feature_ptr[i]);
    }


}

void Clip::encode_text(std::vector<std::string> input_text, std::vector<int> &output) {
    CLIPTokenizer tokenizer(VERSION_1_x);
    std::string str(reinterpret_cast<char*>(merges_utf8_c_str),sizeof(merges_utf8_c_str));
    tokenizer.load_from_merges(str);
    auto on_new_token_cb = [](std::string& str, std::vector<int32_t>& tokens) -> bool {
        // 可以在这里进行自定义处理，返回 true 可以跳过该 token 的处理
        return false;
    };
    output = tokenizer.tokenize(input_text[0], on_new_token_cb);
    output.push_back(49407);
}



Ort::Value Clip::transform(const cv::Mat &mat) {

}
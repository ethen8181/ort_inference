#pragma once

#include <thread>
#include <future>
#include <string>
#include <iostream>
#ifdef OSX
#include "../third_party/onnxruntime-osx-x64-1.8.0/include/onnxruntime_cxx_api.h"
#endif
#ifdef LINUX
#include "../third_party/onnxruntime-linux-x64-1.8.0/include/onnxruntime_cxx_api.h"
#endif


namespace inference {

// Reference
// https://github.com/microsoft/onnxruntime/blob/master/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests.Capi/CXX_Api_Sample.cpp
class SequenceClassificationOrtInference {
private:
std::unique_ptr<Ort::Env> env_;
std::unique_ptr<Ort::Session> session_;
std::vector<const char*> input_node_names_;
std::vector<const char*> output_node_names_;
int64_t num_labels_;

public:
SequenceClassificationOrtInference(std::string model_path, int intra_op_num_threads) {

    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(intra_op_num_threads);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    /*
        As illustrated by the issue below, the Env variable must be in scope
        whenever we invoke a Session operation, hence this needs to be a class member
        instead of instantiating only at the constructor level
        https://github.com/microsoft/onnxruntime/issues/5320
     */
    env_ = std::make_unique<Ort::Env>(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "inference");
    session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), session_options);

    // get input and output node name, this assumes graph only contains
    // a single input and output node
    int i = 0;
    Ort::AllocatorWithDefaultOptions allocator;
    input_node_names_ = { session_->GetInputName(i, allocator) };
    output_node_names_ = { session_->GetOutputName(i, allocator) };

    // get the class count for this classification graph
    auto type_info = session_->GetOutputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> output_shape = tensor_info.GetShape();
    num_labels_ = output_shape[1];
};


std::vector<std::vector<float>>
batch_predict(std::vector<std::vector<int64_t>>& batch_input) {

    std::vector<std::future<std::vector<float>>> batch_future_probs(batch_input.size());
    std::vector<std::vector<float>> batch_probs(batch_input.size());

    for (int i = 0; i < batch_input.size(); i++) {
        // we need to pass this as the first argument when using a member functon
        // https://stackoverflow.com/questions/11758414/class-and-stdasync-on-class-member-in-c/11758662
        batch_future_probs[i] = std::async(
            &SequenceClassificationOrtInference::predict, this, std::ref(batch_input[i])
        );
    }
    for (int i = 0; i < batch_input.size(); i++) {
        batch_probs[i] = batch_future_probs[i].get();
    }
    return batch_probs;
};

std::vector<float>
predict(std::vector<int64_t>& input) {

    // assumes the graph is two dimensonal, with the first one being the
    // batch size of 1, and the second one being sequence length
    int64_t seq_len = input.size();    
    std::vector<int64_t> input_node_dims = {1, seq_len};

    // boilerplate variable for creating tensor
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator,
        OrtMemTypeDefault
    );

    Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(
        memory_info,
        input.data(),
        input.size(),
        input_node_dims.data(),
        input_node_dims.size()
    );

    // score model & input tensor, get back output tensor
    std::vector<Ort::Value> output_tensors = session_->Run(
        Ort::RunOptions{nullptr},
        input_node_names_.data(),
        &input_tensor,
        1,
        output_node_names_.data(),
        output_node_names_.size()
    );

    // Get pointer to output tensor float values
    float* score_arr = output_tensors.front().GetTensorMutableData<float>();
    std::vector<float> scores{score_arr, score_arr + num_labels_};
    return scores;
};

}; // class SequenceClassificationOrtInference

}; // namespace inference
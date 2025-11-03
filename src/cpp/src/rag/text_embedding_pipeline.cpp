// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/rag/text_embedding_pipeline.hpp"

#include <fstream>
#include <nlohmann/json.hpp>

#include "async_infer_request_queue.hpp"
#include "cmath"
#include "json_utils.hpp"
#include "logger.hpp"
#include "openvino/core/except.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "openvino/opsets/opset.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset3.hpp"
#include "openvino/opsets/opset8.hpp"
#include "utils.hpp"

namespace {
using namespace ov::genai;
using namespace ov;

ov::AnyMap remove_config_properties(const ov::AnyMap& properties) {
    auto properties_copy = properties;

    properties_copy.erase(max_length.name());
    properties_copy.erase(pad_to_max_length.name());
    properties_copy.erase(batch_size.name());
    properties_copy.erase(pooling_type.name());
    properties_copy.erase(normalize.name());
    properties_copy.erase(embed_instruction.name());
    properties_copy.erase(query_instruction.name());

    return properties_copy;
}

template <typename T>
bool has_token_type_ids_input(const T& inputs) {
    for (const auto& input : inputs) {
        if (input.get_any_name() == "token_type_ids") {
            return true;
        }
    }
    return false;
}

/**
 * CLS pooling slices first element from seq_length dimension
 * [batch_size, seq_length, hidden_size] -> [batch_size, seq_length[0], hidden_size]
 * [10, 5, 768] -> [10, 768]
 */
std::shared_ptr<op::Op> get_cls_pooling_op(const ov::Output<ov::Node>& last_hidden_state_node) {
    auto start = std::make_shared<op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0});
    auto stop = std::make_shared<op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
    auto step = std::make_shared<op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
    auto axis = std::make_shared<op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});

    auto slice = std::make_shared<op::v8::Slice>(last_hidden_state_node, start, stop, step, axis);

    auto squeeze_axis = std::make_shared<op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
    return std::make_shared<op::v15::Squeeze>(slice, squeeze_axis);
}

std::shared_ptr<op::Op> get_mean_pooling_op(std::shared_ptr<Model> model,
                                            const ov::Output<ov::Node>& last_hidden_state_node) {
    auto shape_of = std::make_shared<op::v3::ShapeOf>(last_hidden_state_node);

    auto attention_mask = model->input("attention_mask").get_node()->outputs()[0];

    auto unsqueze_axis = std::make_shared<op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-1});

    auto unsqueze = std::make_shared<op::v0::Unsqueeze>(attention_mask, unsqueze_axis);

    auto input_mask_expanded = std::make_shared<op::v3::Broadcast>(unsqueze, shape_of);

    auto input_mask_expanded_convert =
        std::make_shared<op::v0::Convert>(input_mask_expanded, last_hidden_state_node.get_element_type());

    auto last_hidden_node_with_applied_attention_mask =
        std::make_shared<op::v1::Multiply>(last_hidden_state_node, input_mask_expanded_convert->outputs()[0]);

    auto axis_1 = std::make_shared<op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
    auto sum_hidden_state = std::make_shared<op::v1::ReduceSum>(last_hidden_node_with_applied_attention_mask, axis_1);

    // f32 overflow possible
    // ReduceMean might help with overflow but its precision diverges from LlamaIndex
    auto sum_expanded_mask = std::make_shared<op::v1::ReduceSum>(input_mask_expanded_convert, axis_1);

    auto nearest_to_zero =
        std::make_shared<op::v0::Constant>(ov::element::f32, ov::Shape{1}, std::vector<float>{1e-12});
    auto max_expanded_mask = std::make_shared<op::v1::Maximum>(sum_expanded_mask, nearest_to_zero);

    // shape: [batch_size, hidden_state_size]
    return std::make_shared<op::v1::Divide>(sum_hidden_state, max_expanded_mask);
}

std::shared_ptr<Model> apply_postprocessing(std::shared_ptr<Model> model, const TextEmbeddingPipeline::Config& config) {
    ov::preprocess::PrePostProcessor processor(model);

    processor.output().postprocess().custom([model, &config](const ov::Output<ov::Node>& node) {
        if (config.pooling_type == TextEmbeddingPipeline::PoolingType::CLS) {
            return get_cls_pooling_op(node);
        } else if (config.pooling_type == TextEmbeddingPipeline::PoolingType::MEAN) {
            return get_mean_pooling_op(model, node);
        }

        OPENVINO_THROW("Pooling type is not supported");
    });

    if (config.normalize) {
        processor.output().postprocess().custom([](const ov::Output<ov::Node>& node) {
            auto axis_const = std::make_shared<op::v0::Constant>(ov::element::i32, ov::Shape{1}, std::vector{1});
            return std::make_shared<op::v0::NormalizeL2>(node, axis_const, 1e-12, op::EpsMode::MAX);
        });
    }

    return processor.build();
}

class ModelConfig {
public:
    explicit ModelConfig(const std::filesystem::path& models_path) {
        // config.json not found. Skip parameters initialization from file, use defaults.
        const std::filesystem::path& json_path = models_path / "config.json";
        if (!std::filesystem::exists(json_path)) {
            return;
        }

        using ov::genai::utils::read_json_param;

        std::ifstream f(json_path);
        OPENVINO_ASSERT(f.is_open(), "Failed to open '", json_path);

        nlohmann::json data = nlohmann::json::parse(f);

        read_json_param(data, "max_position_embeddings", max_position_embeddings);
    };

    std::optional<size_t> max_position_embeddings;
};
}  // namespace

namespace ov {
namespace genai {
using utils::read_anymap_param;

TextEmbeddingPipeline::Config::Config(const ov::AnyMap& properties) {
    read_anymap_param(properties, ov::genai::max_length.name(), max_length);
    read_anymap_param(properties, ov::genai::pad_to_max_length.name(), pad_to_max_length);
    read_anymap_param(properties, ov::genai::batch_size.name(), batch_size);
    read_anymap_param(properties, ov::genai::pooling_type.name(), pooling_type);
    read_anymap_param(properties, ov::genai::normalize.name(), normalize);
    read_anymap_param(properties, ov::genai::embed_instruction.name(), embed_instruction);
    read_anymap_param(properties, ov::genai::query_instruction.name(), query_instruction);
};

void TextEmbeddingPipeline::Config::validate() const {
    if (max_length.has_value()) {
        OPENVINO_ASSERT(max_length.value() > 0, "max_length should be greater than 0");
    }

    if (batch_size.has_value()) {
        OPENVINO_ASSERT(batch_size.value() > 0, "batch_size should be greater than 0");
    }
}

class TextEmbeddingPipeline::TextEmbeddingPipelineImpl {
public:
    TextEmbeddingPipelineImpl(const std::filesystem::path& models_path,
                              const std::string& device,
                              const Config& config,
                              const ov::AnyMap& properties = {})
        : m_config{config},
          m_tokenizer{models_path, properties},
          m_model_config{models_path} {
        m_config.validate();

        ov::Core core = utils::singleton_core();

        auto model = core.read_model(models_path / "openvino_model.xml", {}, properties);

        m_model_has_token_type_ids_input = has_token_type_ids_input(model->inputs());

        const bool should_reshape = m_config.batch_size.has_value() || m_config.max_length.has_value();
        if (should_reshape) {
            reshape_model(model);
        }

        if (device == "NPU") {
            OPENVINO_ASSERT(!model->is_dynamic(),
                            "NPU device does not support dynamic shapes. In order to fix model shape, set batch_size, "
                            "max_length and pad_to_max_length in the configuration.");
        }

        model = apply_postprocessing(model, m_config);

        if (m_config.max_length) {
            m_tokenization_params.insert({max_length.name(), *m_config.max_length});
        }

        if (m_config.pad_to_max_length) {
            m_tokenization_params.insert({pad_to_max_length.name(), *m_config.pad_to_max_length});
        }

        ov::CompiledModel compiled_model = core.compile_model(model, device, properties);

        utils::print_compiled_model_properties(compiled_model, "text embedding model");

        std::cout << "Number of infer requests: " << compiled_model.get_property(ov::optimal_number_of_infer_requests)
                  << std::endl;

        m_async_infer_queue =
            std::make_unique<AsyncInferRequestQueue>(compiled_model,
                                                     compiled_model.get_property(ov::optimal_number_of_infer_requests));

        m_embedding_finished_promise.set_value();
    };

    ~TextEmbeddingPipelineImpl() {
        // std::cout << "[TextEmbeddingPipelineImpl] Destructor called." << std::endl;
        if (m_worker_thread && m_worker_thread->joinable()) {
            m_worker_thread->join();
            // std::cout << "[TextEmbeddingPipelineImpl] Worker thread joined." << std::endl;
        }
    }

    EmbeddingResults embed_documents(const std::vector<std::string>& texts) {
        std::cout << "\n\n[main] Embedding " << texts.size() << " documents." << std::endl;
        start_embed_documents_async(texts);
        return wait_embed_documents();
    };

    void start_embed_documents_async(const std::vector<std::string>& texts) {
        // todo: test assert
        // OPENVINO_ASSERT(m_worker_thread == nullptr,
        //                 "Pipeline is already running. Please wait for the previous embedding to finish.");
        auto formatted_texts = format_texts(texts);
        m_async_infer_queue->reset_all_idle();
        m_worker_thread =
            std::make_unique<std::thread>(&TextEmbeddingPipelineImpl::embed_worker, this, formatted_texts);
    };

    EmbeddingResults wait_embed_documents() {
        return wait_embed();
    };

    EmbeddingResult embed_query(const std::string& text) {
        start_embed_query_async(text);
        return wait_embed_query();
    };

    void start_embed_query_async(const std::string& text) {
        // OPENVINO_ASSERT(m_worker_thread == nullptr,
        //                 "Pipeline is already running. Please wait for the previous embedding to finish.");
        std::vector<std::string> formatted_query{format_query(text)};
        m_async_infer_queue->reset_all_idle();
        m_worker_thread =
            std::make_unique<std::thread>(&TextEmbeddingPipelineImpl::embed_worker, this, formatted_query);
    };

    EmbeddingResult wait_embed_query() {
        const EmbeddingResults results = wait_embed();
        if (auto floats = std::get_if<std::vector<std::vector<float>>>(&results)) {
            return (*floats)[0];
        } else if (auto int8s = std::get_if<std::vector<std::vector<int8_t>>>(&results)) {
            return (*int8s)[0];
        } else if (auto uint8s = std::get_if<std::vector<std::vector<uint8_t>>>(&results)) {
            return (*uint8s)[0];
        }
        OPENVINO_THROW("Embedding result type is not supported");
    };

private:
    Tokenizer m_tokenizer;
    Config m_config;
    ModelConfig m_model_config;
    AnyMap m_tokenization_params;
    std::atomic<bool> m_model_has_fixed_batch = false;
    std::atomic<bool> m_model_has_token_type_ids_input = false;
    std::unique_ptr<AsyncInferRequestQueue> m_async_infer_queue;
    std::unique_ptr<std::thread> m_worker_thread = nullptr;
    EmbeddingResults m_embedding_results;
    std::mutex m_embedding_results_mutex;
    std::promise<void> m_embedding_finished_promise;
    std::future<void> m_embedding_finished_future = m_embedding_finished_promise.get_future();

    void reshape_model(std::shared_ptr<Model>& model) {
        ov::PartialShape target_shape{ov::Dimension::dynamic(), ov::Dimension::dynamic()};

        if (m_config.batch_size.has_value()) {
            target_shape[0] = ov::Dimension(*m_config.batch_size);
            m_model_has_fixed_batch = true;
        }

        if (m_config.max_length.has_value()) {
            const auto max_position_embeddings = m_model_config.max_position_embeddings;
            if (max_position_embeddings.has_value() && *m_config.max_length > *max_position_embeddings) {
                std::stringstream message;
                message << "max_length is set to " << *m_config.max_length
                        << " which is greater than models max_position_embeddings (" << *max_position_embeddings << ")."
                        << "Some models may fail with such configuration.";
                Logger::warn(message.str());
            }

            if (m_config.pad_to_max_length.has_value() && *m_config.pad_to_max_length) {
                target_shape[1] = ov::Dimension(*m_config.max_length);
            } else {
                target_shape[1] = ov::Dimension{0, static_cast<int64_t>(*m_config.max_length)};
            }
        }

        std::map<std::string, ov::PartialShape> input_name_to_shape;
        input_name_to_shape["input_ids"] = target_shape;
        input_name_to_shape["attention_mask"] = target_shape;

        if (m_model_has_token_type_ids_input) {
            input_name_to_shape["token_type_ids"] = target_shape;
        }

        model->reshape(input_name_to_shape);
    }

    void embed_worker(const std::vector<std::string>& texts) {
        std::cout << "[worker] thread started" << std::endl;
        m_embedding_finished_promise = std::promise<void>();
        m_embedding_finished_future = m_embedding_finished_promise.get_future();
        m_embedding_results = std::vector<std::vector<float>>(texts.size());
        size_t batch_size = m_config.batch_size.value_or(4);

        const size_t num_batches = std::ceil(static_cast<float>(texts.size()) / batch_size);

        for (size_t batch = 0; batch < num_batches; ++batch) {
            size_t start = batch * batch_size;
            size_t end = std::min(start + batch_size, texts.size());
            std::vector<std::string> batch_texts(texts.begin() + start, texts.begin() + end);

            if (m_model_has_fixed_batch && batch_texts.size() < batch_size) {
                batch_texts.resize(batch_size);
            }

            const auto encoded = m_tokenizer.encode(batch_texts, m_tokenization_params);

            std::cout << "[worker] waiting for idle request..." << std::endl;
            auto request = m_async_infer_queue->get();
            std::cout << "[worker] got idle request id: " << request->m_queue_id << std::endl;

            fill_inputs(encoded, request);

            request->set_callback([this, request, batch, num_batches, start, end]() {
                const Tensor last_hidden_state = request->get_tensor("last_hidden_state");
                fill_embedding_results(last_hidden_state, {start, end});

                // todo: not valid, requests finishing not in order
                // if (batch == num_batches - 1) {
                //     std::cout << "[worker] last batch processed, setting promise value" << std::endl;
                //     m_embedding_finished_promise.set_value();
                // } else {
                //     std::cout << "[worker] batch " << batch << " processed" << std::endl;
                // }
            });

            std::cout << "[worker] starting async request for batch: " << batch << std::endl;
            request->start_async();
        }

        // std::cout << "[worker] thread finished" << std::endl;
    };

    EmbeddingResults wait_embed() {
        std::cout << "[main] waiting for thread finish..." << std::endl;
        if (m_worker_thread && m_worker_thread->joinable()) {
            m_worker_thread->join();
            std::cout << "[main] thread joined" << std::endl;
        }

        // todo: not waiting for all_idle leads to fails, investigate
        // possible intersection with previous embed_documents call
        // m_async_infer_queue->wait_all_idle();
        // std::cout << "[main] all requests are idle" << std::endl;

        // m_embedding_finished_promise.get_future().wait();
        // std::cout << "[main] m_embedding_finished_promise finished" << std::endl;
        m_embedding_finished_future.wait();
        std::cout << "[main] m_embedding_finished_future finished" << std::endl;
        // m_worker_thread = nullptr;

        const auto results = std::move(m_embedding_results);
        m_embedding_results = {};
        return results;
    };

    std::vector<std::string> format_texts(const std::vector<std::string>& texts) {
        if (!m_config.embed_instruction) {
            return texts;
        }

        std::vector<std::string> formatted;
        formatted.reserve(texts.size());

        for (auto& text : texts) {
            formatted.emplace_back(*m_config.embed_instruction + text);
        }
        return formatted;
    }

    std::string format_query(const std::string& text) {
        if (!m_config.query_instruction) {
            return text;
        }

        return *m_config.query_instruction + text;
    }

    void fill_embedding_results(const Tensor& last_hidden_state, const std::pair<size_t, size_t>& batch_range) {
        const float* last_hidden_state_data = last_hidden_state.data<float>();

        const auto shape = last_hidden_state.get_shape();

        const size_t batch_size = batch_range.second - batch_range.first;
        const size_t hidden_size = shape[1];

        {
            std::lock_guard<std::mutex> lock(m_embedding_results_mutex);
            for (size_t batch_id = 0; batch_id < batch_size; batch_id++) {
                const auto batch_offset = batch_id * hidden_size;
                const float* batch_data = last_hidden_state_data + batch_offset;
                const std::vector<float> batch_result(batch_data, batch_data + hidden_size);

                std::cout << "[worker] filling embedding results for batch " << batch_id + batch_range.first
                          << ", size: " << batch_result.size() << std::endl;

                auto& embedding_results = std::get<std::vector<std::vector<float>>>(m_embedding_results);
                embedding_results[batch_id + batch_range.first] = batch_result;
            }
        }
    }

    void fill_inputs(const TokenizedInputs& encoded, std::shared_ptr<InferRequestAsyncWrapper> request) {
        request->set_tensor("input_ids", encoded.input_ids);
        request->set_tensor("attention_mask", encoded.attention_mask);

        // fill token_type_ids
        // todo: pass token_type_ids from tokenizer
        if (m_model_has_token_type_ids_input) {
            ov::Tensor token_type_ids{ov::element::i64, encoded.input_ids.get_shape()};
            std::fill_n(token_type_ids.data<int64_t>(), encoded.input_ids.get_size(), 0);
            request->set_tensor("token_type_ids", token_type_ids);
        }
    }
};

TextEmbeddingPipeline::TextEmbeddingPipeline(const std::filesystem::path& models_path,
                                             const std::string& device,
                                             const Config& config,
                                             const ov::AnyMap& properties) {
    m_impl = std::make_unique<TextEmbeddingPipelineImpl>(models_path, device, config, properties);
};

TextEmbeddingPipeline::TextEmbeddingPipeline(const std::filesystem::path& models_path,
                                             const std::string& device,
                                             const ov::AnyMap& properties) {
    const auto& plugin_properties = remove_config_properties(properties);

    m_impl = std::make_unique<TextEmbeddingPipelineImpl>(models_path, device, Config(properties), plugin_properties);
};

EmbeddingResults TextEmbeddingPipeline::embed_documents(const std::vector<std::string>& texts) {
    return m_impl->embed_documents(texts);
}

void TextEmbeddingPipeline::start_embed_documents_async(const std::vector<std::string>& texts) {
    return m_impl->start_embed_documents_async(texts);
}

EmbeddingResults TextEmbeddingPipeline::wait_embed_documents() {
    return m_impl->wait_embed_documents();
}

EmbeddingResult TextEmbeddingPipeline::embed_query(const std::string& text) {
    return m_impl->embed_query(text);
}

void TextEmbeddingPipeline::start_embed_query_async(const std::string& text) {
    return m_impl->start_embed_query_async(text);
}

EmbeddingResult TextEmbeddingPipeline::wait_embed_query() {
    return m_impl->wait_embed_query();
}

TextEmbeddingPipeline::~TextEmbeddingPipeline() = default;

}  // namespace genai
}  // namespace ov

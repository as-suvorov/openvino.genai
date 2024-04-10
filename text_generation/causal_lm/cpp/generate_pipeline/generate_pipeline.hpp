// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <openvino/openvino.hpp>
#include "sampling_parameters.hpp"
#include <experimental/filesystem>
#include "group_beam_searcher.hpp"

// using GenerationResult = ov::Tensor;
using GenerationResult = std::vector<std::vector<int64_t>>;
using namespace std;

std::pair<ov::Tensor, ov::Tensor> pad_left(ov::Tensor&& input_ids, ov::Tensor&& attention_mask, int64_t pad_token=2) {
    const size_t batch_size = input_ids.get_shape().at(0);
    const size_t sequence_length = input_ids.get_shape().at(1);
    int64_t* inputs_data = input_ids.data<int64_t>();
    int64_t* attention_mask_data = attention_mask.data<int64_t>();

    for (size_t batch = 0; batch < batch_size; batch++) {
        const size_t batch_offset = batch * sequence_length;

        // last token in the sequence is not a PAD_TOKEN, skipping
        if (inputs_data[batch_offset + sequence_length - 1] != pad_token) {
            continue;
        }

        size_t pad_tokens_number = 0;
        for (int i = sequence_length - 1; i >= 0; i--) {
            const size_t token_offset = batch_offset + i;

            if (inputs_data[token_offset] == pad_token) {
                continue;
            }

            if (pad_tokens_number == 0) {
                pad_tokens_number = sequence_length - i - 1;
            }

            std::swap(inputs_data[token_offset], inputs_data[token_offset + pad_tokens_number]);
            std::swap(attention_mask_data[token_offset], attention_mask_data[token_offset + pad_tokens_number]);
        }
    }

    return {input_ids, attention_mask};
}

void update_position_ids(ov::Tensor& position_ids, const ov::Tensor& attention_mask) {
    const size_t batch_size = attention_mask.get_shape().at(0);
    const size_t seq_length = attention_mask.get_shape().at(1);
    position_ids.set_shape({batch_size, 1});

    for (size_t batch = 0; batch < batch_size; batch++) {
        int64_t* start = attention_mask.data<int64_t>() + batch * seq_length;
        position_ids.data<int64_t>()[batch] = std::accumulate(start, start + seq_length - 1, 0);
    }
}

void initialize_position_ids(ov::Tensor& position_ids, const ov::Tensor& attention_mask) {
    const size_t batch_size = attention_mask.get_shape()[0];
    const size_t seq_length = attention_mask.get_shape()[1];

    const int64_t* attention_mask_data = attention_mask.data<int64_t>();
    int64_t* position_ids_data = position_ids.data<int64_t>();

    for (size_t batch = 0; batch < batch_size; batch++) {
        size_t sum = 0;
        for (size_t i = 0; i < seq_length; i++) {
            const size_t element_offset = batch * seq_length + i;
            position_ids_data[element_offset] = sum;
            if (attention_mask_data[element_offset] == 1) {
                sum += 1;
            }
        }
    }
}

class LLModel {
    ov::Tensor extend_attention(ov::Tensor attention_mask) {
        auto shape = attention_mask.get_shape();
        auto batch_size = shape[0];
        auto seq_len = shape[1];

        ov::Tensor new_atten_mask = ov::Tensor{attention_mask.get_element_type(), {batch_size, seq_len + 1}};
        auto old_data = attention_mask.data<int64_t>();
        auto new_data = new_atten_mask.data<int64_t>();
        for (size_t batch = 0; batch < batch_size; ++batch) {
            std::memcpy(new_data + batch * (seq_len + 1), old_data + batch * seq_len, seq_len * sizeof(int64_t));
            new_data[batch * (seq_len + 1) + seq_len] = 1;
        }
        return new_atten_mask;
    }

    ov::InferRequest m_model_runner;
    
    GenerationResult greedy_search(ov::Tensor input_ids, GenerationConfig sampling_params) {
        auto attention_mask = ov::Tensor{ov::element::i64, input_ids.get_shape()};
        std::fill_n(attention_mask.data<int64_t>(), attention_mask.get_size(), 1);
        return greedy_search(input_ids, attention_mask, sampling_params);
    }

    GenerationResult greedy_search(ov::Tensor input_ids, 
                                   ov::Tensor attention_mask, 
                                   GenerationConfig sampling_params) {
        ov::Shape prompts_shape = input_ids.get_shape();
        size_t batch_size = prompts_shape[0];
        
        GenerationResult results(batch_size);

        auto position_ids = ov::Tensor{ov::element::i64, input_ids.get_shape()};
        // std::iota(position_ids.data<int64_t>(), position_ids.data<int64_t>() + position_ids.get_size(), 0);
        initialize_position_ids(position_ids, attention_mask);

        size_t initial_seq_len = input_ids.get_shape()[1];

        m_model_runner.set_tensor("input_ids", input_ids);
        m_model_runner.set_tensor("attention_mask", attention_mask);
        m_model_runner.set_tensor("position_ids", position_ids);
    
        m_model_runner.get_tensor("beam_idx").set_shape({batch_size});
        auto beam_data = m_model_runner.get_tensor("beam_idx").data<int32_t>();
        std::iota(beam_data, beam_data + batch_size, 0);

        for (size_t i = 0; i < sampling_params.m_max_new_tokens; ++i) {
            m_model_runner.infer();
            auto logits = m_model_runner.get_tensor("logits");
            ov::Shape logits_shape = logits.get_shape();
            size_t seq_len = logits_shape[1], vocab_size = logits_shape[2];

            m_model_runner.get_tensor("input_ids").set_shape({batch_size, 1});
            m_model_runner.set_tensor("attention_mask", extend_attention(m_model_runner.get_tensor("attention_mask")));
            // m_model_runner.get_tensor("position_ids").set_shape({batch_size, 1});
            update_position_ids(position_ids, attention_mask);

            for (size_t batch = 0; batch < batch_size; ++batch) {
                const float * logits_data = logits.data<const float>() + seq_len * vocab_size * batch + (seq_len - 1) * vocab_size;
                int64_t out_token = std::max_element(logits_data, logits_data + vocab_size) - logits_data;
                results[batch].emplace_back(out_token);
                
                // todo: add exit criteria when pad or EOS is met
                m_model_runner.get_tensor("input_ids").data<int64_t>()[batch] = out_token;
                m_model_runner.get_tensor("position_ids").data<int64_t>()[batch] = int64_t(initial_seq_len + i);
            }
        }
        return results;
    }

    GenerationResult beam_search(ov::Tensor prompts, GenerationConfig sampling_params) {
        ov::Shape prompts_shape = prompts.get_shape();
        size_t batch_size = prompts_shape[0];
        // todo: implement for batch > 1
        OPENVINO_ASSERT(batch_size == 1);

        // initialize inputs
        auto attention_mask = ov::Tensor{ov::element::i64, prompts.get_shape()};
        std::fill_n(attention_mask.data<int64_t>(), attention_mask.get_size(), 1);
        auto position_ids = ov::Tensor{ov::element::i64, prompts.get_shape()};
        std::iota(position_ids.data<int64_t>(), position_ids.data<int64_t>() + position_ids.get_size(), 0);
        auto initial_seq_len = prompts.get_shape()[1];

        m_model_runner.set_tensor("input_ids", prompts);
        m_model_runner.set_tensor("attention_mask", attention_mask);
        m_model_runner.set_tensor("position_ids", position_ids);
    
        // set beam_idx for stateful model: no beam search is used and BATCH_SIZE = 1
        m_model_runner.get_tensor("beam_idx").set_shape({batch_size});
        m_model_runner.get_tensor("beam_idx").data<int32_t>()[0] = 0;

        const int64_t* prompt_data = prompts.data<const int64_t>();
        
        // todo: remove this duplicatino and use the same SamplingParameters for both greedy and beam
        Parameters parameters{std::vector<int64_t>{prompt_data, prompt_data + prompts.get_size()}};
        parameters.n_groups = sampling_params.m_num_groups;
        parameters.diversity_penalty = sampling_params.m_diversity_penalty;
        parameters.group_size = sampling_params.m_group_size;

        GroupBeamSearcher group_beam_searcher{parameters};
        std::vector<int64_t> next_tokens;
        std::vector<int32_t> next_beams;
        for (size_t length_count = 0; length_count < sampling_params.m_max_new_tokens; ++length_count) {
            m_model_runner.infer();
            std::tie(next_tokens, next_beams) = group_beam_searcher.select_next_tokens(m_model_runner.get_tensor("logits"));
            if (next_tokens.empty()) {
                break;
            }
            size_t batch_size = next_tokens.size();
            // Set pointers
            m_model_runner.set_tensor("input_ids", ov::Tensor{ov::element::i64, {batch_size, 1}, next_tokens.data()});
            m_model_runner.set_tensor("beam_idx", ov::Tensor{ov::element::i32, {batch_size}, next_beams.data()});
            // Set auxiliary inputs
            ov::Tensor attention_mask = m_model_runner.get_tensor("attention_mask");
            ov::Shape mask_shape{batch_size, attention_mask.get_shape()[1] + 1};
            attention_mask.set_shape(mask_shape);
            std::fill_n(attention_mask.data<int64_t>(), ov::shape_size(mask_shape), 1);

            m_model_runner.get_tensor("position_ids").set_shape({batch_size, 1});
            std::fill_n(m_model_runner.get_tensor("position_ids").data<int64_t>(), batch_size, mask_shape.at(1) - 1);
        }

        std::vector<Beam> beams;
        for (const std::vector<Beam>& group : finalize(std::move(group_beam_searcher))) {
            for (const Beam& beam : group) {
                beams.emplace_back(beam);
                // results.emplace_back(beam.tokens);
            }
        }

        auto compare_scores = [](Beam left, Beam right) { return (left.score > right.score); };
        std::sort(beams.begin(), beams.end(), compare_scores);
        
        GenerationResult results;
        for (auto beam = beams.begin(); beam != beams.begin() + sampling_params.m_num_return_sequences; ++beam) {
            results.emplace_back(beam->tokens);
        }
        return results;
    }

    GenerationResult multinomial_sampling(ov::Tensor prompts, GenerationConfig sampling_params) {
        // todo: implement
        GenerationResult results;
        return results;
    }

public:
    LLModel(ov::InferRequest& request) :
          m_model_runner(request) {
    }
    
    LLModel() = default;

    // more high level interface
    GenerationResult generate(ov::Tensor prompts, ov::Tensor attention_mask, GenerationConfig sampling_params) {
        if (sampling_params.is_gready_sampling()) {
            return greedy_search(prompts, attention_mask, sampling_params);
        } else if (sampling_params.is_beam_search()) {
            return beam_search(prompts, sampling_params);
        } else {  // if (sampling_params.is_multimomial()) {
            return multinomial_sampling(prompts, sampling_params);
        }
    }
};

std::pair<ov::Tensor, ov::Tensor> tokenize(ov::InferRequest& tokenizer, std::string prompt) {
    size_t batch_size = 1;
    tokenizer.set_input_tensor(ov::Tensor{ov::element::string, {batch_size}, &prompt});
    tokenizer.infer();

    vector<vector<int64_t>> input_ids_vec;
    input_ids_vec.reserve(1);
    auto res_tensor = tokenizer.get_tensor("input_ids");
    auto res_shape = res_tensor.get_shape();
    
    for (int i = 0; i < res_shape[0]; ++i) {
        int64_t* start = res_tensor.data<int64_t>() + i * res_shape[1];
        input_ids_vec.emplace_back(std::vector<int64_t>(start, start + res_shape[1]));
    }

    return {tokenizer.get_tensor("input_ids"), tokenizer.get_tensor("attention_mask")};
}

std::pair<ov::Tensor, ov::Tensor> tokenize(ov::InferRequest& tokenizer, std::vector<std::string> prompts) {
    tokenizer.set_input_tensor(ov::Tensor{ov::element::string, {prompts.size()}, prompts.data()});
    auto size_ = tokenizer.get_input_tensor().get_shape();
    tokenizer.infer();

    pad_left(tokenizer.get_tensor("input_ids"), tokenizer.get_tensor("attention_mask"));
    // fix mask filled with '2' instead of '0'
    ov::Tensor attention_mask = tokenizer.get_tensor("attention_mask");
    int64_t* attention_mask_data = attention_mask.data<int64_t>();
    std::replace(attention_mask_data, attention_mask_data + attention_mask.get_size(), 2, 0);
    
    vector<vector<int64_t>> input_ids_vec;
    vector<vector<int64_t>> atten_mask_vec;
    
    input_ids_vec.reserve(prompts.size());
    atten_mask_vec.reserve(prompts.size());
    auto res_tensor = tokenizer.get_tensor("input_ids");
    auto atten_tensor = tokenizer.get_tensor("attention_mask");
    auto res_shape = res_tensor.get_shape();
    
    for (int i = 0; i < res_shape[0]; ++i) {
        int64_t* start = res_tensor.data<int64_t>() + i * res_shape[1];
        input_ids_vec.emplace_back(std::vector<int64_t>(start, start + res_shape[1]));
        
        int64_t* atten_start = atten_tensor.data<int64_t>() + i * res_shape[1];
        atten_mask_vec.emplace_back(std::vector<int64_t>(atten_start, atten_start + res_shape[1]));
    }

    return {tokenizer.get_tensor("input_ids"), tokenizer.get_tensor("attention_mask")};
}

std::string detokenize(ov::InferRequest& detokenizer, std::vector<int64_t> tokens) {
    size_t batch_size = 1;
    detokenizer.set_input_tensor(ov::Tensor{ov::element::i64, {batch_size, tokens.size()}, tokens.data()});
    detokenizer.infer();
    return detokenizer.get_output_tensor().data<std::string>()[0];
}

std::vector<std::string> detokenize(ov::InferRequest& detokenizer, ov::Tensor tokens) {
    detokenizer.set_input_tensor(tokens);
    auto shape = tokens.get_shape();
    auto data = tokens.data<int64_t>();
    detokenizer.infer();
    auto res = detokenizer.get_output_tensor();
    
    std::vector<std::string> strings;
    for (int i = 0; i < res.get_shape()[0]; ++i) {
        strings.emplace_back(res.data<std::string>()[i]);
    }
    return strings;
}

std::vector<std::string> detokenize(ov::InferRequest& detokenizer, 
                                    std::vector<std::vector<int64_t>> lines, 
                                    int64_t pad_token_idx) {
    // todo: implement calling detokenizer in a single batch

    std::vector<std::string> strings;
    for (auto& line: lines){
        ov::Tensor tokens = ov::Tensor{ov::element::i64, {1, line.size()}, line.data()};
        detokenizer.set_input_tensor(tokens);
        detokenizer.infer();
        auto res = detokenizer.get_output_tensor();
        auto res_str = res.data<std::string>()[0];
        strings.emplace_back(res_str);
    }
    
    return strings;
}

// The following reasons require TextStreamer to keep a cache of previous tokens:
// detokenizer removes starting ' '. For example detokenize(tokenize(" a")) == "a",
// but detokenize(tokenize("prefix a")) == "prefix a"
// 1 printable token may consist of 2 token ids: detokenize(incomplete_token_idx) == "�"
struct TextStreamer {
    ov::InferRequest detokenizer;
    std::vector<int64_t> token_cache;
    size_t print_len = 0;

    void put(int64_t token) {
        token_cache.push_back(token);
        std::string text = detokenize(detokenizer, token_cache);
        if (!text.empty() && '\n' == text.back()) {
            // Flush the cache after the new line symbol
            std::cout << std::string_view{text.data() + print_len, text.size() - print_len};
            token_cache.clear();
            print_len = 0;
            return;
        }
        if (text.size() >= 3 && text.compare(text.size() - 3, 3, "�") == 0) {
            // Don't print incomplete text
            return;
        }
        std::cout << std::string_view{text.data() + print_len, text.size() - print_len} << std::flush;
        print_len = text.size();
    }

    void end() {
        std::string text = detokenize(detokenizer, token_cache);
        std::cout << std::string_view{text.data() + print_len, text.size() - print_len} << '\n';
        token_cache.clear();
        print_len = 0;
    }
};

class LLMPipeline {
    LLModel m_model_runner;
    ov::InferRequest m_tokenizer;
    ov::InferRequest m_detokenizer;
    std::string m_path;
    GenerationConfig m_sampling_parameters;

public:
    LLMPipeline(std::string& path) : m_path(path) {
        if (std::experimental::filesystem::exists(m_path + "/generation_config.json")) {
            m_sampling_parameters = GenerationConfig(m_path + "/generation_config.json");
        }
        // m_sampling_parameters = GenerationConfig(m_path + "/generation_config_beam.json");

        ov::Core core;
        // The model can be compiled for GPU as well
        auto model_request = core.compile_model(m_path + "/openvino_model.xml", "CPU").create_infer_request();
        m_model_runner = LLModel(model_request);

        // tokenizer and detokenizer work on CPU only
        core.add_extension(OPENVINO_TOKENIZERS_PATH);  // OPENVINO_TOKENIZERS_PATH is defined in CMakeLists.txt
        m_tokenizer = core.compile_model(m_path + "/openvino_tokenizer.xml", "CPU").create_infer_request();
        m_detokenizer = core.compile_model(m_path + "/openvino_detokenizer.xml", "CPU").create_infer_request();
    }

    GenerationConfig generation_config() const {
        return m_sampling_parameters;
    }

    std::string call(std::string text) {
        auto [input_ids, attention_mask] = tokenize(m_tokenizer, text);

        auto generate_results = m_model_runner.generate(input_ids, attention_mask, m_sampling_parameters);

        return detokenize(m_detokenizer, generate_results, 0)[0];
    }
    
    std::string call(std::string text, GenerationConfig sampling_parameters) {
        auto [input_ids, attention_mask] = tokenize(m_tokenizer, text);

        auto generate_results = m_model_runner.generate(input_ids, attention_mask, sampling_parameters);

        return detokenize(m_detokenizer, generate_results, 0)[0];
    }

    std::vector<std::string> call(std::vector<std::string> text, GenerationConfig sampling_parameters) {
        auto [input_ids, attention_mask] = tokenize(m_tokenizer, text);

        auto generate_results = m_model_runner.generate(input_ids, attention_mask, sampling_parameters);

        return detokenize(m_detokenizer, generate_results, 0);
    }

    std::string operator()(std::string text, GenerationConfig sampling_parameters) {
        return call(text, sampling_parameters);
    }
    
    std::vector<std::string> operator()(std::vector<std::string> text, GenerationConfig sampling_parameters) {
        return call(text, sampling_parameters);
    }
    
    std::vector<std::string> operator()(std::initializer_list<std::string> text, GenerationConfig sampling_parameters) {
        return call(text, sampling_parameters);
    }
};

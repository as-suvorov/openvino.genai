// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdlib>
#include <functional>
#include <nlohmann/json.hpp>
#include <fstream>
#include <group_beam_searcher.hpp>  // used only for StopCriteria

// forward declaration
class Sequence;

// Similar to HuggingFace GenerationConfig 
// but has parameters that are not present in the original SamplingParameters for continous batching
struct GenerationConfig {
    // Generic
    size_t max_new_tokens = 10;
    size_t max_length = 100; // max_new tokens should have priority over max_new_tokens
    bool ignore_eos = false;
    int64_t eos_token = 2; // There's no way to extract special token values from the tokenizer for now
    size_t num_return_sequences = 3;

    // Beam search specific
    size_t n_groups = 1;
    size_t group_size = 1; // beam_width
    float diversity_penalty = 1.0f; // 0.0 means no diversity
    
    StopCriteria stop_criteria = StopCriteria::heuristic;
    float length_penalty = 1.0f;
    size_t no_repeat_ngram_size = std::numeric_limits<size_t>::max();
    std::function<bool(const Sequence&)> early_finish = [](const Sequence&) {return false; };

    // Multinomial
    float temperature = 0.0f; // by default we use greedy sampling
    int top_k = -1; // maybe to assign vocab_size ?
    float top_p = 1.0f; // by default convsider all tokens
    bool do_sample;

    // special tokens
    int64_t bos_token_id = 0;
    int64_t eos_token_id = 0;
    int64_t pad_token_id = 0;

    GenerationConfig() = default;

    GenerationConfig(std::string json_path) {
        std::ifstream f(json_path);
        nlohmann::json data = nlohmann::json::parse(f);

        bos_token_id = data.value("bos_token_id", 0);
        eos_token_id = data.value("eos_token_id", 0);
        max_length = data.value("max_length", 0);
        pad_token_id = data.value("pad_token_id", 0);
        num_return_sequences = data.value("num_return_sequences", 1);
        max_new_tokens = data.value("max_new_tokens", 100);
        
        temperature = data.value("temperature", 0.0f);
        do_sample = data.value("do_sample", false);
        top_p = data.value("top_p", 0.0f);
        
        // beam_search_params
        n_groups = data.value("num_beam_groups", 1);
        diversity_penalty = data.value("diversity_penalty", 1.0f);
        int num_beams = data.value("num_beams", 1);
        group_size = num_beams / n_groups;
    }

    static GenerationConfig greedy() {
        GenerationConfig greedy_params;
        greedy_params.temperature = 0.0f;
        greedy_params.ignore_eos = true;
        return greedy_params;
    }

    static GenerationConfig beam_search() {
        GenerationConfig beam_search;
        beam_search.n_groups = 3;
        beam_search.group_size = 5;
        beam_search.max_new_tokens = 10;
        beam_search.diversity_penalty = 2.0f;
        return beam_search;
    }

    static GenerationConfig multimomial() {
        GenerationConfig multimomial;
        multimomial.temperature = 0.8f;
        multimomial.top_p = 0.8;
        multimomial.top_k = 20;
        multimomial.do_sample = 20;
        return multimomial;
    }

    bool is_gready_sampling() const {
        return !do_sample && !is_beam_search();
    }

    bool is_beam_search() const {
        return n_groups * group_size > 1;
    }

    bool is_multimomial() const {
        return do_sample;
    }
    
};

enum class SamplingAlgorithm{greedy, multinomial, baeam_search};

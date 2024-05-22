// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <fstream>
#include <limits>

#include <nlohmann/json.hpp>
#include <openvino/runtime/core.hpp>

#include "openvino/genai/generation_config.hpp"

#include "generation_config_helper.hpp"
#include "utils.hpp"

namespace {   


} // namespace


namespace ov {

GenerationConfig::GenerationConfig(std::string json_path) {
    using ov::generate_utils::read_json_param;

    std::ifstream f(json_path);
    OPENVINO_ASSERT(f.is_open(), "Failed to open '" + json_path + "' with generation config");

    nlohmann::json data = nlohmann::json::parse(f);
    
    read_json_param(data, "max_new_tokens", max_new_tokens);
    read_json_param(data, "max_length", max_length);
    // note that ignore_eos is not present in HF GenerationConfig
    read_json_param(data, "num_beam_groups", num_beam_groups);
    read_json_param(data, "num_beams", num_beams);
    read_json_param(data, "diversity_penalty", diversity_penalty);
    read_json_param(data, "length_penalty", length_penalty);
    read_json_param(data, "num_return_sequences", num_return_sequences);
    read_json_param(data, "no_repeat_ngram_size", no_repeat_ngram_size);
    read_json_param(data, "temperature", temperature);
    read_json_param(data, "top_p", top_p);
    read_json_param(data, "top_k", top_k);
    read_json_param(data, "do_sample", do_sample);
    read_json_param(data, "repetition_penalty", repetition_penalty);
    read_json_param(data, "pad_token_id", pad_token_id);
    read_json_param(data, "bos_token_id", bos_token_id);
    read_json_param(data, "eos_token_id", eos_token_id);
    read_json_param(data, "bos_token", bos_token);
    read_json_param(data, "eos_token", eos_token);

    if (data.contains("early_stopping")) {
        auto field_type = data["early_stopping"].type();
        if (field_type == nlohmann::json::value_t::string && data["early_stopping"] == "never") {
            stop_criteria = StopCriteria::never;
        } else if (field_type == nlohmann::json::value_t::boolean && data["early_stopping"] == true) {
            stop_criteria = StopCriteria::early;
        } else if (field_type == nlohmann::json::value_t::boolean && data["early_stopping"] == false) {
            stop_criteria = StopCriteria::heuristic;
        }
    }


}

GenerationConfig GenerationConfigHelper::anymap_to_generation_config(const ov::AnyMap& config_map) {
    using ov::generate_utils::read_anymap_param;
    
    GenerationConfig config = m_config;
    read_anymap_param(config_map, "max_new_tokens", config.max_new_tokens);
    read_anymap_param(config_map, "max_length", config.max_length);
    read_anymap_param(config_map, "ignore_eos", config.ignore_eos);
    read_anymap_param(config_map, "num_beam_groups", config.num_beam_groups);
    read_anymap_param(config_map, "num_beams", config.num_beams);
    read_anymap_param(config_map, "diversity_penalty", config.diversity_penalty);
    read_anymap_param(config_map, "length_penalty", config.length_penalty);
    read_anymap_param(config_map, "num_return_sequences", config.num_return_sequences);
    read_anymap_param(config_map, "no_repeat_ngram_size", config.no_repeat_ngram_size);
    read_anymap_param(config_map, "stop_criteria", config.stop_criteria);
    read_anymap_param(config_map, "temperature", config.temperature);
    read_anymap_param(config_map, "top_p", config.top_p);
    read_anymap_param(config_map, "top_k", config.top_k);
    read_anymap_param(config_map, "do_sample", config.do_sample);
    read_anymap_param(config_map, "repetition_penalty", config.repetition_penalty);
    read_anymap_param(config_map, "pad_token_id", config.pad_token_id);
    read_anymap_param(config_map, "bos_token_id", config.bos_token_id);
    read_anymap_param(config_map, "eos_token_id", config.eos_token_id);
    read_anymap_param(config_map, "bos_token", config.bos_token);
    read_anymap_param(config_map, "eos_token", config.eos_token);
  
    return config;
}

size_t GenerationConfigHelper::get_max_new_tokens(size_t prompt_length) {
    // max_new_tokens has priority over max_length, only if max_new_tokens was not specified use max_length
    if (m_config.max_new_tokens != SIZE_MAX) {
        return m_config.max_new_tokens;
    } else {
        return m_config.max_length - prompt_length;
    }
}

bool GenerationConfigHelper::is_greedy_decoding() const {
    return !m_config.do_sample && !is_beam_search();
}

bool GenerationConfigHelper::is_beam_search() const {
    return m_config.num_beams > 1;
}

bool GenerationConfigHelper::is_multinomial() const {
    return m_config.do_sample;
}

}  // namespace ov

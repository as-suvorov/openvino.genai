// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <chrono>  // Required for std::chrono::seconds
#include <iostream>
#include <random>
#include <thread>  // Required for std::this_thread::sleep_for

#include "openvino/genai/rag/text_embedding_pipeline.hpp"

ov::AnyMap npu_fallback_config{
    {"NPU_USE_NPUW", "YES"},
    {"NPUW_DEVICES", "CPU"},
    {"NPUW_ONLINE_PIPELINE", "NONE"},
};

std::string TEXT_DATASET{"The commercial PC market is propelled by premium\
computing solutions that drive user productivity and help\
service organizations protect and maintain devices.\
Corporations must empower mobile and hybrid workers\
while extracting value from artificial intelligence (AI) to\
improve business outcomes. Moreover, both public and\
private sectors must address sustainability initiatives\
pertaining to the full life cycle of computing fleets. An\
inflection point in computing architecture is needed to stay\
ahead of evolving requirements.\
Introducing Intel® Core™ Ultra Processors\
Intel® Core™ Ultra processors shape the future of\
commercial computing in four major ways:\
Power Efficiency\
The new product line features a holistic approach to powerefficiency that benefits mobile work. Substantial changes to\
the microarchitecture, manufacturing process, packaging\
technology, and power management software result in up to\
40% lower processor power consumption for modern tasks\
such as video conferencing with a virtual camera. \
Artificial Intelligence\
Intel Core Ultra processors incorporate an AI-optimized\
architecture that supports new user experiences and the\
next wave of commercial applications. The CPU, GPU, and\
the new neural processing unit (NPU) are all capable of\
executing AI tasks as directed by application developers.\
For example, elevated mobile collaboration is possible with\
support for AI assisted background blur, noise suppression,\
eye tracking, and picture framing. Intel Core Ultra\
processors are capable of up to 2.5x the AI inference\
performance per watt as compared to Intel’s previous\
mobile processor offering.2\
"};

std::vector<std::string> dataset_documents(const std::string& text_dataset, size_t chunk_size = 200) {
    std::vector<std::string> chunks;
    for (size_t i = 0; i < text_dataset.size(); i += chunk_size) {
        chunks.push_back(text_dataset.substr(i, chunk_size));
    }
    return chunks;
}

// generate random text
std::string generate_random_text(size_t length) {
    static const std::string characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ";
    std::string text;
    text.reserve(length);
    std::mt19937 generator(std::random_device{}());
    std::uniform_int_distribution<> distribution(0, characters.size() - 1);
    for (size_t i = 0; i < length; ++i) {
        text += characters[distribution(generator)];
    }
    return text;
}

// generate random documents
std::vector<std::string> generate_documents(size_t count) {
    std::vector<std::string> documents;
    for (size_t i = 0; i < count; ++i) {
        // documents.emplace_back("Document " + std::to_string(i) + ": This is a sample document for testing
        // purposes.");
        documents.emplace_back("Document " + std::to_string(i) + ": " + generate_random_text(100));
    }
    return documents;
}

void validate_embedding_results(const ov::genai::EmbeddingResults& results) {
    if (std::holds_alternative<std::vector<std::vector<float>>>(results)) {
        const auto& embeddings = std::get<std::vector<std::vector<float>>>(results);
        for (const auto& embedding : embeddings) {
            if (embedding.empty()) {
                throw std::runtime_error("Empty embedding vector found.");
            }
        }
    } else if (std::holds_alternative<std::vector<std::vector<int8_t>>>(results)) {
        const auto& embeddings = std::get<std::vector<std::vector<int8_t>>>(results);
        for (const auto& embedding : embeddings) {
            if (embedding.empty()) {
                throw std::runtime_error("Empty embedding vector found.");
            }
        }
    } else if (std::holds_alternative<std::vector<std::vector<uint8_t>>>(results)) {
        const auto& embeddings = std::get<std::vector<std::vector<uint8_t>>>(results);
        for (const auto& embedding : embeddings) {
            if (embedding.empty()) {
                throw std::runtime_error("Empty embedding vector found.");
            }
        }
    } else {
        throw std::runtime_error("Unsupported EmbeddingResults type.");
    }
}

void measure_performance(ov::genai::TextEmbeddingPipeline& pipeline,
                         const size_t no_of_runs,
                         const std::vector<std::string>& documents,
                         const std::string& models_path) {
    std::cout << "Starting separate thread embedding for " << documents.size() << " documents with " << no_of_runs
              << " runs...\n";
    const auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < no_of_runs; i++) {
        ov::genai::TextEmbeddingPipeline::Config config;
        // config.pooling_type = ov::genai::TextEmbeddingPipeline::PoolingType::MEAN;
        config.batch_size = 1;
        config.max_length = 64;
        config.pad_to_max_length = true;

        ov::AnyMap properties{{ov::hint::performance_mode.name(), ov::hint::PerformanceMode::THROUGHPUT}};
        // properties.insert(npu_fallback_config.begin(), npu_fallback_config.end());

        ov::genai::TextEmbeddingPipeline pipeline(models_path, "CPU", config);
        std::cout << "Run " << i + 1 << " of " << no_of_runs << std::endl;
        pipeline.embed_documents(documents);
        pipeline.embed_documents(documents);
        pipeline.embed_documents(documents);
        auto embedding_results = pipeline.embed_documents(documents);

        for (const auto& embedding : std::get<std::vector<std::vector<float>>>(embedding_results)) {
            // std::cout << "Embedding size: " << embedding.size() << std::endl;
            if (embedding.size() != 384) {
                std::cerr << "Warning: Embedding size is not 384, it is " << embedding.size() << std::endl;
            }
        }

        validate_embedding_results(embedding_results);
    }
    const auto end = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Total time for embedding " << documents.size() << " documents at " << no_of_runs
              << " runs: " << duration << " ms\n";
    std::cout << "Average time for embedding " << documents.size()
              << " documents: " << static_cast<float>(duration) / no_of_runs << " ms\n";
}

// todo: stalls on empty documents
int main(int argc, char* argv[]) try {
    if (argc < 3) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> '<TEXT 1>' ['<TEXT 2>' ...]");
    }
    // auto documents = std::vector<std::string>(argv + 2, argv + argc);

    // const auto documents = generate_documents(9);

    const auto documents = dataset_documents(TEXT_DATASET, 200);
    // std::vector<std::string> documents{
    //     "What is the capital of France?",
    //     "What is the capital of Germany?",
    //     "What is the capital of Italy?",
    //     "What is the capital of Spain?",
    //     "What is the capital of Portugal?",
    //     "What is the capital of Netherlands?",
    //     "What is the capital of Belgium?",
    //     "What is the capital of Switzerland?",
    // };
    std::string models_path = argv[1];

    std::string device = "CPU";  // GPU can be used as well

    ov::genai::TextEmbeddingPipeline::Config config;
    // config.pooling_type = ov::genai::TextEmbeddingPipeline::PoolingType::MEAN;
    config.batch_size = 1;
    config.max_length = 64;
    config.pad_to_max_length = true;

    ov::AnyMap properties{{ov::hint::performance_mode.name(), ov::hint::PerformanceMode::THROUGHPUT}};
    // properties.insert(npu_fallback_config.begin(), npu_fallback_config.end());

    ov::genai::TextEmbeddingPipeline pipeline(models_path, device, config);

    // Optimal number of infer requests: 8
    // Optimal number of infer requests: 8
    // Starting separate thread embedding for 200 documents with 20 runs...
    // Total time for embedding 200 documents at 20 runs: 62920 ms
    // Average time for embedding 200 documents: 3146 ms

    // warm up
    pipeline.embed_documents(documents);

    const size_t number_of_runs = 20;
    measure_performance(pipeline, number_of_runs, documents, models_path);

    ov::genai::EmbeddingResult query_embedding = pipeline.embed_query("What is the capital of France?");
} catch (const std::exception& error) {
    try {
        std::cerr << error.what() << '\n';
    } catch (const std::ios_base::failure&) {
    }
    return EXIT_FAILURE;
} catch (...) {
    try {
        std::cerr << "Non-exception object thrown\n";
    } catch (const std::ios_base::failure&) {
    }
    return EXIT_FAILURE;
}

// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <openvino/openvino.hpp>

#include "circular_buffer_queue.hpp"

namespace ov::genai {

using ReturnToQueueCallback = std::function<void(int queue_id)>;

class InferRequestAsyncWrapper {
public:
    InferRequestAsyncWrapper(CompiledModel& model, const ReturnToQueueCallback& queue_callback)
        : m_infer_request(model.create_infer_request()) {
        m_infer_request.set_callback([this, &queue_callback](const std::exception_ptr& ptr) {
            if (ptr) {
                std::rethrow_exception(ptr);
            }

            if (m_callback) {
                m_callback();
            }

            queue_callback(m_queue_id);
        });
    }

    void set_tensor(const std::string& tensor_name, const Tensor& tensor) {
        m_infer_request.set_tensor(tensor_name, tensor);
    }

    Tensor get_tensor(const std::string& tensor_name) {
        return m_infer_request.get_tensor(tensor_name);
    }

    void set_queue_id(const int id) {
        m_queue_id = id;
    }

    void set_callback(const std::function<void()>& callback) {
        m_callback = callback;
    }

    void start_async() {
        m_infer_request.start_async();
    }

    void wait() {
        m_infer_request.wait();
    }

    int m_queue_id = -1;

private:
    InferRequest m_infer_request;
    std::function<void()> m_callback;
};

class AsyncInferRequestQueue {
public:
    AsyncInferRequestQueue(CompiledModel& model, const size_t length)
        : m_queue{CircularBufferQueue<std::shared_ptr<InferRequestAsyncWrapper>>(length, [this, &model]() {
              return std::make_shared<InferRequestAsyncWrapper>(model, m_bound_on_request_finished);
          })} {
        m_all_idle_promise = std::promise<void>();
    }

    std::shared_ptr<InferRequestAsyncWrapper> get() {
        const int queue_id = m_queue.get_idle().get();
        const auto request = m_queue.get(queue_id);
        request->set_queue_id(queue_id);
        return request;
    }

    void reset_all_idle() {
        m_all_idle_promise = std::promise<void>();
        m_all_idle_future = m_all_idle_promise.get_future();
    }

    void wait_all_idle() {
        m_all_idle_future.wait();
    }

    void unlock_if_all_idle() {
        if (m_all_idle_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
            return;
        }

        if (m_queue.all_idle()) {
            m_all_idle_promise.set_value();
        }
    }

private:
    CircularBufferQueue<std::shared_ptr<InferRequestAsyncWrapper>> m_queue;
    std::promise<void> m_all_idle_promise;
    std::future<void> m_all_idle_future = m_all_idle_promise.get_future();
    ReturnToQueueCallback m_bound_on_request_finished =
        std::bind(&AsyncInferRequestQueue::on_request_finished, this, std::placeholders::_1);

    void on_request_finished(const int queue_id) {
        m_queue.return_to(queue_id);
        unlock_if_all_idle();
    }
};

}  // namespace ov::genai
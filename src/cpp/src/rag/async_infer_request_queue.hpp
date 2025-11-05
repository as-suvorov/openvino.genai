// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <condition_variable>
#include <mutex>
#include <openvino/openvino.hpp>

#include "circular_buffer_queue.hpp"

namespace ov::genai {

using ReturnToQueueCallback = std::function<void(int queue_id, std::exception_ptr exception)>;

class InferRequestAsyncWrapper {
public:
    InferRequestAsyncWrapper(CompiledModel& model, const ReturnToQueueCallback& queue_callback)
        : m_infer_request(model.create_infer_request()),
          m_queue_id{-1},
          m_callback{nullptr} {
        m_infer_request.set_callback([this, queue_callback](const std::exception_ptr& ptr) {
            std::exception_ptr exception = ptr;

            if (!ptr && m_callback) {
                // catch exceptions from user callback
                // if not caught request won't be returned to the queue and main thread will hang
                try {
                    m_callback();
                } catch (...) {
                    exception = std::current_exception();
                }
            }

            queue_callback(m_queue_id, exception);
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

    // todo: remove
    int get_queue_id() const {
        return m_queue_id;
    }

private:
    std::function<void()> m_callback;
    InferRequest m_infer_request;
    int m_queue_id;
};

class AsyncInferRequestQueue {
public:
    AsyncInferRequestQueue(CompiledModel& model, const size_t length)
        : m_queue{CircularBufferQueue<std::shared_ptr<InferRequestAsyncWrapper>>(length, [this, &model]() {
              return std::make_shared<InferRequestAsyncWrapper>(model,
                                                                std::bind(&AsyncInferRequestQueue::on_request_finished,
                                                                          this,
                                                                          std::placeholders::_1,
                                                                          std::placeholders::_2));
          })} {}

    std::shared_ptr<InferRequestAsyncWrapper> get_request() {
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            if (m_infer_exception) {
                std::rethrow_exception(m_infer_exception);
            }
        }

        const int queue_id = m_queue.get_idle().get();
        const auto request = m_queue.get(queue_id);
        request->set_queue_id(queue_id);
        return request;
    }

    void wait_all_idle() {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_cv.wait(lock, [this] {
            if (m_infer_exception) {
                std::rethrow_exception(m_infer_exception);
            }
            return m_queue.all_idle();
        });
    }

private:
    CircularBufferQueue<std::shared_ptr<InferRequestAsyncWrapper>> m_queue;
    std::mutex m_mutex;
    std::condition_variable m_cv;
    std::exception_ptr m_infer_exception = nullptr;

    void on_request_finished(const int queue_id, const std::exception_ptr& ptr) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (ptr) {
            m_infer_exception = ptr;
        }

        m_queue.return_to(queue_id);
        m_cv.notify_one();
    }
};

}  // namespace ov::genai
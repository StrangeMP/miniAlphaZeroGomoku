#include "multithreadhelpers.hpp"
#include <algorithm>
#include <functional>
#include <chrono>

namespace MultiThreadHelpers {

//========================================================================================
// BatchInferenceManager实现
//========================================================================================

BatchInferenceManager::BatchInferenceManager(AlphaGomoku::Network& net) 
    : network(net), inference_thread(&BatchInferenceManager::inferenceWorker, this) {
}

BatchInferenceManager::~BatchInferenceManager() {
    shutdown();
}

void BatchInferenceManager::shutdown() {
    shutdown_flag = true;
    queue_cv.notify_all();
    if (inference_thread.joinable()) {
        inference_thread.join();
    }
}

std::future<std::pair<Vec<float, Config::BOARD_SQUARES>, float>> 
BatchInferenceManager::submitRequest(const Board& board, 
                                   const std::optional<std::pair<int, int>>& last_move,
                                   AlphaGomoku::STONE_COLOR current_color) {
    
    auto request = std::make_unique<InferenceRequest>(board, last_move, current_color);
    auto future = request->promise.get_future();
    
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        request_queue.push(std::move(request));
    }
    
    queue_cv.notify_one();
    return future;
}

void BatchInferenceManager::inferenceWorker() {
    std::vector<std::unique_ptr<InferenceRequest>> batch;
    batch.reserve(MAX_BATCH_SIZE);
    
    while (!shutdown_flag) {
        batch.clear();
        
        // 收集批处理请求
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            
            // 等待第一个请求或关闭信号
            queue_cv.wait(lock, [this] { 
                return !request_queue.empty() || shutdown_flag; 
            });
            
            if (shutdown_flag) break;
            
            // 收集批次
            auto deadline = std::chrono::steady_clock::now() + BATCH_TIMEOUT;
            while (!request_queue.empty() && batch.size() < MAX_BATCH_SIZE) {
                batch.push_back(std::move(request_queue.front()));
                request_queue.pop();
                
                // 如果批次未满，等待更多请求直到超时
                if (batch.size() < MAX_BATCH_SIZE && !request_queue.empty()) {
                    queue_cv.wait_until(lock, deadline, [this] {
                        return !request_queue.empty() || shutdown_flag;
                    });
                }
            }
        }
        
        // 处理批次
        if (!batch.empty()) {
            processBatch(batch);
        }
    }
    
    // 处理剩余请求
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        while (!request_queue.empty()) {
            auto request = std::move(request_queue.front());
            request_queue.pop();
            batch.push_back(std::move(request));
        }
    }
    
    if (!batch.empty()) {
        processBatch(batch);
    }
}

void BatchInferenceManager::processBatch(std::vector<std::unique_ptr<InferenceRequest>>& batch) {
    // 目前简单实现：逐个处理请求
    // 后续可以优化为真正的批处理
    for (auto& request : batch) {
        try {
            auto result = network.feed(request->board_state, request->last_move, request->current_color);
            request->promise.set_value(result);
        } catch (...) {
            request->promise.set_exception(std::current_exception());
        }
    }
}

//========================================================================================
// ThreadPool实现
//========================================================================================

ThreadPool::ThreadPool(size_t num_threads) {
    for (size_t i = 0; i < num_threads; ++i) {
        workers.emplace_back([this] {
            for (;;) {
                std::function<void()> task;
                
                {
                    std::unique_lock<std::mutex> lock(this->queue_mutex);
                    this->condition.wait(lock, [this] { 
                        return this->stop_flag || !this->tasks.empty(); 
                    });
                    
                    if (this->stop_flag && this->tasks.empty()) {
                        return;
                    }
                    
                    task = std::move(this->tasks.front());
                    this->tasks.pop();
                }
                
                task();
            }
        });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        stop_flag = true;
    }
    
    condition.notify_all();
    
    for (std::thread &worker : workers) {
        worker.join();
    }
}

void ThreadPool::waitForAll() {
    // 简单实现：等待任务队列为空
    std::unique_lock<std::mutex> lock(queue_mutex);
    condition.wait(lock, [this] { return tasks.empty(); });
}

//========================================================================================
// MutexPool实现
//========================================================================================

MutexPool::MutexPool(size_t size) : pool_size(size) {
    mutexes.reserve(size);
    for (size_t i = 0; i < size; ++i) {
        mutexes.push_back(std::make_unique<std::mutex>());
    }
}

std::mutex& MutexPool::getMutex(const void* node_ptr) const {
    size_t hash = std::hash<const void*>{}(node_ptr);
    return *mutexes[hash % pool_size];
}

std::mutex& MutexPool::getMutex(size_t index) const {
    return *mutexes[index % pool_size];
}

} // namespace MultiThreadHelpers

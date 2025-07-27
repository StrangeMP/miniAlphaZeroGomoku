#include "multithreadhelpers.hpp"
#include <algorithm>
#include <functional>
#include <chrono>

namespace MultiThreadHelpers {

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

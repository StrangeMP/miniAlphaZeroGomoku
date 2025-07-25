#pragma once

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#include <functional>
#include <future>
#include <array>
#include "config.hpp"
#include "network.hpp"
#include "compute.hpp"  // 为了Matrix类型

// 前向声明和类型定义
using Board = Matrix<AlphaGomoku::STONE_COLOR, Config::BOARD_SIZE, Config::BOARD_SIZE>;

namespace MultiThreadHelpers {

// 前向声明
struct Node;
struct MCTS_Agent;

//========================================================================================
// 1. 线程安全的节点统计结构
//========================================================================================

// 原子化的节点统计信息
struct AtomicNodeStats {
    std::atomic<int> visit_count{0};
    std::atomic<double> value_sum{0.0};
    std::atomic<double> weight_sum{0.0};
    
    AtomicNodeStats() = default;
    
    // 禁止拷贝和移动，确保原子性
    AtomicNodeStats(const AtomicNodeStats&) = delete;
    AtomicNodeStats& operator=(const AtomicNodeStats&) = delete;
    AtomicNodeStats(AtomicNodeStats&&) = delete;
    AtomicNodeStats& operator=(AtomicNodeStats&&) = delete;
    
    // 原子更新
    void addVisit(double value, double weight = 1.0) {
        visit_count.fetch_add(1, std::memory_order_relaxed);
        
        // 使用compare_exchange_weak实现原子double更新
        double expected_value_sum = value_sum.load(std::memory_order_relaxed);
        while (!value_sum.compare_exchange_weak(expected_value_sum, 
                                              expected_value_sum + value, 
                                              std::memory_order_relaxed)) {
            // 重试直到成功
        }
        
        double expected_weight_sum = weight_sum.load(std::memory_order_relaxed);
        while (!weight_sum.compare_exchange_weak(expected_weight_sum, 
                                               expected_weight_sum + weight, 
                                               std::memory_order_relaxed)) {
            // 重试直到成功
        }
    }
    
    // 获取当前平均值
    double getAverageValue() const {
        int visits = visit_count.load(std::memory_order_relaxed);
        if (visits == 0) return 0.0;
        return value_sum.load(std::memory_order_relaxed) / visits;
    }
    
    // 获取访问次数
    int getVisitCount() const {
        return visit_count.load(std::memory_order_relaxed);
    }
    
    // 获取权重和
    double getWeightSum() const {
        return weight_sum.load(std::memory_order_relaxed);
    }
};

//========================================================================================
// 2. 虚拟损失机制
//========================================================================================

// 虚拟损失管理器
class VirtualLossManager {
private:
    std::atomic<int> virtual_loss_count{0};
    static constexpr double VIRTUAL_LOSS_VALUE = -1.0;  // 虚拟损失值
    
public:
    // 添加虚拟损失
    void addVirtualLoss() {
        virtual_loss_count.fetch_add(1, std::memory_order_relaxed);
    }
    
    // 移除虚拟损失
    void removeVirtualLoss() {
        virtual_loss_count.fetch_sub(1, std::memory_order_relaxed);
    }
    
    // 获取虚拟损失数量
    int getVirtualLossCount() const {
        return virtual_loss_count.load(std::memory_order_relaxed);
    }
    
    // 计算包含虚拟损失的总访问数
    int getTotalVisitsWithVirtualLoss(int real_visits) const {
        return real_visits + virtual_loss_count.load(std::memory_order_relaxed);
    }
    
    // 计算包含虚拟损失的值和
    double getTotalValueWithVirtualLoss(double real_value_sum) const {
        int vl_count = virtual_loss_count.load(std::memory_order_relaxed);
        return real_value_sum + (vl_count * VIRTUAL_LOSS_VALUE);
    }
};

//========================================================================================
// 3. 网络推理批处理请求
//========================================================================================

// 推理请求结构
struct InferenceRequest {
    Board board_state;
    std::optional<std::pair<int, int>> last_move;
    AlphaGomoku::STONE_COLOR current_color;
    std::promise<std::pair<Vec<float, Config::BOARD_SQUARES>, float>> promise;
    
    InferenceRequest(const Board& board, 
                    const std::optional<std::pair<int, int>>& move,
                    AlphaGomoku::STONE_COLOR color)
        : board_state(board), last_move(move), current_color(color) {}
};

// 网络推理批处理管理器
class BatchInferenceManager {
private:
    AlphaGomoku::Network& network;
    std::queue<std::unique_ptr<InferenceRequest>> request_queue;
    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    std::condition_variable batch_cv;
    
    std::atomic<bool> shutdown_flag{false};
    std::thread inference_thread;
    
    static constexpr size_t MAX_BATCH_SIZE = 8;  // 最大批处理大小
    static constexpr std::chrono::milliseconds BATCH_TIMEOUT{5};  // 批处理超时
    
    void inferenceWorker();
    void processBatch(std::vector<std::unique_ptr<InferenceRequest>>& batch);
    
public:
    explicit BatchInferenceManager(AlphaGomoku::Network& net);
    ~BatchInferenceManager();
    
    // 提交推理请求，返回future
    std::future<std::pair<Vec<float, Config::BOARD_SQUARES>, float>> 
    submitRequest(const Board& board, 
                  const std::optional<std::pair<int, int>>& last_move,
                  AlphaGomoku::STONE_COLOR current_color);
    
    // 关闭批处理管理器
    void shutdown();
};

//========================================================================================
// 4. 线程池管理
//========================================================================================

// 线程池
class ThreadPool {
private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    std::atomic<bool> stop_flag{false};
    
public:
    explicit ThreadPool(size_t num_threads);
    ~ThreadPool();
    
    // 提交任务
    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) 
        -> std::future<typename std::invoke_result<F, Args...>::type>;
    
    // 等待所有任务完成
    void waitForAll();
    
    // 获取线程数量
    size_t getThreadCount() const { return workers.size(); }
};

//========================================================================================
// 5. 互斥锁池 (用于节点级别的细粒度锁)
//========================================================================================

class MutexPool {
private:
    std::vector<std::unique_ptr<std::mutex>> mutexes;
    size_t pool_size;
    
public:
    explicit MutexPool(size_t size = 4096);  // 默认4096个锁
    ~MutexPool() = default;
    
    // 根据节点指针获取对应的锁
    std::mutex& getMutex(const void* node_ptr) const;
    
    // 根据索引获取锁
    std::mutex& getMutex(size_t index) const;
    
    size_t getPoolSize() const { return pool_size; }
};

//========================================================================================
// 6. 多线程配置
//========================================================================================

struct MultiThreadConfig {
    size_t num_search_threads = 4;           // 搜索线程数
    size_t max_batch_size = 8;               // 最大批处理大小
    std::chrono::milliseconds batch_timeout{5};  // 批处理超时
    bool enable_virtual_loss = true;         // 启用虚拟损失
    bool enable_batch_inference = true;      // 启用批处理推理
    size_t mutex_pool_size = 4096;          // 互斥锁池大小
    
    // 虚拟损失相关参数
    double virtual_loss_value = -1.0;        // 虚拟损失值
    
    // 从配置文件加载配置的静态方法
    static MultiThreadConfig getDefault() {
        MultiThreadConfig config;
        // 根据硬件线程数调整默认线程数
        config.num_search_threads = std::min(4u, std::thread::hardware_concurrency());
        return config;
    }
};

//========================================================================================
// 7. 线程安全的搜索统计
//========================================================================================

// 全局搜索统计信息
struct SearchStats {
    std::atomic<uint64_t> total_simulations{0};
    std::atomic<uint64_t> total_node_expansions{0};
    std::atomic<uint64_t> total_inference_requests{0};
    std::atomic<uint64_t> total_cache_hits{0};
    
    void incrementSimulations() { total_simulations.fetch_add(1, std::memory_order_relaxed); }
    void incrementNodeExpansions() { total_node_expansions.fetch_add(1, std::memory_order_relaxed); }
    void incrementInferenceRequests() { total_inference_requests.fetch_add(1, std::memory_order_relaxed); }
    void incrementCacheHits() { total_cache_hits.fetch_add(1, std::memory_order_relaxed); }
    
    // 获取统计信息
    uint64_t getSimulations() const { return total_simulations.load(std::memory_order_relaxed); }
    uint64_t getNodeExpansions() const { return total_node_expansions.load(std::memory_order_relaxed); }
    uint64_t getInferenceRequests() const { return total_inference_requests.load(std::memory_order_relaxed); }
    uint64_t getCacheHits() const { return total_cache_hits.load(std::memory_order_relaxed); }
    
    void reset() {
        total_simulations = 0;
        total_node_expansions = 0;
        total_inference_requests = 0;
        total_cache_hits = 0;
    }
};

} // namespace MultiThreadHelpers

//========================================================================================
// ThreadPool模板方法实现
//========================================================================================

namespace MultiThreadHelpers {

template<class F, class... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args) 
    -> std::future<typename std::invoke_result<F, Args...>::type> {
    
    using return_type = typename std::invoke_result<F, Args...>::type;
    
    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );
    
    std::future<return_type> res = task->get_future();
    
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        
        // 不允许在停止的线程池中添加新任务
        if (stop_flag) {
            throw std::runtime_error("enqueue on stopped ThreadPool");
        }
        
        tasks.emplace([task](){ (*task)(); });
    }
    
    condition.notify_one();
    return res;
}

} // namespace MultiThreadHelpers

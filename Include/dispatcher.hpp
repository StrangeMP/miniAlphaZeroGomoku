#pragma once
#include "blockingconcurrentqueue.h"
#include "config.hpp"
#include "network.hpp"
#include <atomic>
#include <condition_variable>
#include <cuda_runtime.h>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

// Forward declaration
namespace MCTS {
struct Node;
}

class InferenceDispatcher {
public:
  using InputRequest = Network::InputUnit_T;
  using OutputResult = Network::ResultType;

  static InferenceDispatcher &getInstance();

  void start();
  void stop();

  std::future<Network::ResultPtr> collect(const Network::BinaryInputUnit_T &binary_input,
                                          const Network::GlobalInputUnit_T &global_input);

private:
  // A struct to hold all resources for a single batch inference.
  struct Batch {
    // Pinned CPU-side buffers (raw pointers)
    Network::BinaryInputUnit_T *binary_inputs = nullptr;
    Network::GlobalInputUnit_T *global_inputs = nullptr;
    Network::PolicyOut_T *policy_outputs = nullptr;
    Network::ValueOut_T *value_outputs = nullptr;
    std::vector<std::promise<Network::ResultPtr>> promises;

    // GPU-side resources
    cudaStream_t stream = nullptr;
    cudaEvent_t event = nullptr;
    void *d_binary_input = nullptr;
    void *d_global_input = nullptr;
    void *d_policy_output = nullptr;
    void *d_value_output = nullptr;

    // State
    std::atomic<int> item_count = 0;
  };

  InferenceDispatcher();
  ~InferenceDispatcher();

  InferenceDispatcher(const InferenceDispatcher &) = delete;
  InferenceDispatcher &operator=(const InferenceDispatcher &) = delete;
  InferenceDispatcher(InferenceDispatcher &&) = delete;
  InferenceDispatcher &operator=(InferenceDispatcher &&) = delete;

  // Background threads
  void submitterLoop();
  void reaperLoop();

  // State management
  std::atomic<bool> running_{false};
  std::atomic<bool> shutting_down_{false};
  std::unique_ptr<std::thread> submitter_thread_;
  std::unique_ptr<std::thread> reaper_thread_;

  // Pool of batch resources
  static constexpr int NUM_BUFFERS = Config::NUM_BATCH_BUFFERS;
  std::vector<std::unique_ptr<Batch>> batch_pool_;

  // Lock-free queues for managing batch lifecycle
  moodycamel::ConcurrentQueue<Batch *> available_batches_;
  moodycamel::BlockingConcurrentQueue<Batch *> pending_processing_;
  moodycamel::BlockingConcurrentQueue<Batch *> in_flight_batches_;

  // The single batch currently being filled by worker threads
  std::atomic<Batch *> active_batch_{nullptr};

  // Synchronization for the collector side
  std::mutex collector_mutex_;
  std::condition_variable collector_cv_;
  std::mutex batch_swap_mutex_;
};

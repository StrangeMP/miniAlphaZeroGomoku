#include "dispatcher.hpp"
#include "config.hpp"
#include "multithread_mcts.hpp"
#include <chrono>
#include <stdexcept>

InferenceDispatcher &InferenceDispatcher::getInstance() {
  static InferenceDispatcher instance;
  return instance;
}

InferenceDispatcher::InferenceDispatcher() {
  // Create the pool of batch resources
  for (int i = 0; i < NUM_BUFFERS; ++i) {
    auto batch = std::make_unique<Batch>();

    // Allocate Pinned Host Memory
    cudaMallocHost(&batch->binary_inputs, Config::MAX_BATCH_SIZE * sizeof(Network::BinaryInputUnit_T));
    cudaMallocHost(&batch->global_inputs, Config::MAX_BATCH_SIZE * sizeof(Network::GlobalInputUnit_T));
    cudaMallocHost(&batch->mask_inputs, Config::MAX_BATCH_SIZE * sizeof(Network::MaskInputUnit_T));
    cudaMallocHost(&batch->policy_outputs, Config::MAX_BATCH_SIZE * sizeof(Network::PolicyOut_T));
    cudaMallocHost(&batch->value_outputs, Config::MAX_BATCH_SIZE * sizeof(Network::ValueOut_T));
    batch->node_pointers.resize(Config::MAX_BATCH_SIZE);

    // Allocate GPU memory and create CUDA resources once
    cudaStreamCreate(&batch->stream);
    cudaEventCreate(&batch->event);
    cudaMalloc(&batch->d_binary_input, Config::MAX_BATCH_SIZE * sizeof(Network::BinaryInputUnit_T));
    cudaMalloc(&batch->d_global_input, Config::MAX_BATCH_SIZE * sizeof(Network::GlobalInputUnit_T));
    cudaMalloc(&batch->d_mask_input, Config::MAX_BATCH_SIZE * sizeof(Network::MaskInputUnit_T));
    cudaMalloc(&batch->d_policy_output, Config::MAX_BATCH_SIZE * sizeof(Network::PolicyOut_T));
    cudaMalloc(&batch->d_value_output, Config::MAX_BATCH_SIZE * sizeof(Network::ValueOut_T));

    batch_pool_.push_back(std::move(batch));
  }
  // Fill the available queue
  for (auto &batch : batch_pool_) {
    available_batches_.enqueue(batch.get());
  }
}

InferenceDispatcher::~InferenceDispatcher() {
  stop();
  // Free all resources
  for (auto &batch : batch_pool_) {
    cudaFree(batch->d_binary_input);
    cudaFree(batch->d_global_input);
    cudaFree(batch->d_mask_input);
    cudaFree(batch->d_policy_output);
    cudaFree(batch->d_value_output);
    cudaFreeHost(batch->binary_inputs);
    cudaFreeHost(batch->global_inputs);
    cudaFreeHost(batch->mask_inputs);
    cudaFreeHost(batch->policy_outputs);
    cudaFreeHost(batch->value_outputs);
    cudaEventDestroy(batch->event);
    cudaStreamDestroy(batch->stream);
  }
}

void InferenceDispatcher::start() {
  if (running_.load())
    return;

  // Prime the active batch
  available_batches_.try_dequeue(active_batch_);

  running_.store(true);
  submitter_thread_ = std::make_unique<std::thread>(&InferenceDispatcher::submitterLoop, this);
  reaper_thread_ = std::make_unique<std::thread>(&InferenceDispatcher::reaperLoop, this);
}

void InferenceDispatcher::stop() {
  if (!running_.load() || shutting_down_.load())
    return;

  shutting_down_.store(true);
  collector_cv_.notify_all(); // Wake up any waiting collectors

  // Wait for all pending requests to be processed
  while (pending_processing_.size_approx() > 0 || in_flight_batches_.size_approx() > 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }

  // Now, safely stop the threads
  running_.store(false);
  if (submitter_thread_ && submitter_thread_->joinable())
    submitter_thread_->join();
  if (reaper_thread_ && reaper_thread_->joinable())
    reaper_thread_->join();
}

void InferenceDispatcher::collect(const Network::BinaryInputUnit_T &binary_input,
                                  const Network::GlobalInputUnit_T &global_input,
                                  const Network::MaskInputUnit_T &mask_input, MultiThreadMCTS::ThreadSafeNode *node) {
  if (!running_.load() || shutting_down_.load()) {
    throw std::runtime_error("Dispatcher is not running or shutting down.");
  }

  if (!node) {
    throw std::invalid_argument("Node pointer cannot be null");
  }

  Batch *current_batch;
  int slot;

  {
    // Lock to protect the check-then-act sequence for getting a slot
    std::lock_guard<std::mutex> lock(batch_swap_mutex_);

    current_batch = active_batch_.load();

    // If the active batch is null (because it was just swapped and a new one isn't ready)
    // we must wait.
    if (current_batch == nullptr) {
      std::unique_lock<std::mutex> collector_lock(collector_mutex_);
      collector_cv_.wait(collector_lock, [this] { return active_batch_.load() != nullptr || !running_.load(); });
      if (!running_.load()) {
        throw std::runtime_error("Dispatcher is stopped.");
      }
      current_batch = active_batch_.load();
    }

    slot = current_batch->item_count++;
    current_batch->binary_inputs[slot] = binary_input;
    current_batch->global_inputs[slot] = global_input;
    current_batch->mask_inputs[slot] = mask_input;
    current_batch->node_pointers[slot] = node;

    // If this request filled the batch, swap it out for a new one
    if (slot + 1 >= Config::MAX_BATCH_SIZE) {
      // Try to get a new batch from the available queue
      Batch *next_batch = nullptr;
      if (available_batches_.try_dequeue(next_batch)) {
        active_batch_.store(next_batch);
        collector_cv_.notify_all();
      } else {
        active_batch_.store(nullptr);
      }

      // Push the full batch to the processing queue
      pending_processing_.enqueue(current_batch);
    }
  }
}

void InferenceDispatcher::submitterLoop() {
  Batch *batch_to_process;
  auto last_submission_time = std::chrono::steady_clock::now();

  while (running_.load()) {
    bool dequeued = pending_processing_.wait_dequeue_timed(batch_to_process, std::chrono::milliseconds(5));

    if (dequeued) {
      // A full batch was pushed by a collector. Process it immediately.
      last_submission_time = std::chrono::steady_clock::now();
    } else {
      // No full batch was ready. Check if the active batch has waited long enough.
      if (shutting_down_.load()) continue;

      Batch* active_batch = active_batch_.load();
      if (active_batch != nullptr && active_batch->item_count > 0) {
        auto now = std::chrono::steady_clock::now();
        auto time_since_last_submission = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_submission_time);

        if (time_since_last_submission.count() > 5) { // 5ms timeout
          // Atomically swap the active batch with a new one
          std::lock_guard<std::mutex> lock(batch_swap_mutex_);
          active_batch = active_batch_.load(); // Re-load in case it changed
          if (active_batch != nullptr && active_batch->item_count > 0) {
            Batch* next_batch = nullptr;
            if (available_batches_.try_dequeue(next_batch)) {
              active_batch_.store(next_batch);
              collector_cv_.notify_all();
            } else {
              active_batch_.store(nullptr);
            }
            batch_to_process = active_batch;
            dequeued = true;
            last_submission_time = now;
          }
        }
      }
    }

    if (dequeued) {
      // Submit the batch for asynchronous execution on its own stream
      const int batch_size = batch_to_process->item_count;
      if (batch_size == 0) continue;

      const size_t binary_size = batch_size * sizeof(Network::BinaryInputUnit_T);
      const size_t global_size = batch_size * sizeof(Network::GlobalInputUnit_T);
      const size_t mask_size = batch_size * sizeof(Network::MaskInputUnit_T);
      const size_t policy_size = batch_size * sizeof(Network::PolicyOut_T);
      const size_t value_size = batch_size * sizeof(Network::ValueOut_T);

      cudaMemcpyAsync(batch_to_process->d_binary_input, batch_to_process->binary_inputs, binary_size,
                      cudaMemcpyHostToDevice, batch_to_process->stream);
      cudaMemcpyAsync(batch_to_process->d_global_input, batch_to_process->global_inputs, global_size,
                      cudaMemcpyHostToDevice, batch_to_process->stream);
      cudaMemcpyAsync(batch_to_process->d_mask_input, batch_to_process->mask_inputs, mask_size,
                      cudaMemcpyHostToDevice, batch_to_process->stream);

      Network::feed(batch_to_process->d_binary_input, batch_to_process->d_global_input,
                    batch_to_process->d_mask_input, batch_to_process->d_policy_output,
                    batch_to_process->d_value_output, batch_size, batch_to_process->stream);

      cudaMemcpyAsync(batch_to_process->policy_outputs, batch_to_process->d_policy_output, policy_size,
                      cudaMemcpyDeviceToHost, batch_to_process->stream);
      cudaMemcpyAsync(batch_to_process->value_outputs, batch_to_process->d_value_output, value_size,
                      cudaMemcpyDeviceToHost, batch_to_process->stream);
      cudaEventRecord(batch_to_process->event, batch_to_process->stream);

      // Move the batch to the in-flight queue
      in_flight_batches_.enqueue(batch_to_process);
    }
  }
}

void InferenceDispatcher::reaperLoop() {
  Batch *completed_batch;
  while (running_.load() || in_flight_batches_.size_approx() > 0) {
    // Wait for a batch to finish processing
    if (in_flight_batches_.wait_dequeue_timed(completed_batch, std::chrono::milliseconds(10))) {
      try {
        // Block until this specific batch's event is complete
        cudaEventSynchronize(completed_batch->event);

        // This batch is done. Set inference results on nodes.
        for (int i = 0; i < completed_batch->item_count; ++i) {
          MultiThreadMCTS::ThreadSafeNode *node = completed_batch->node_pointers[i];
          if (node) {
            const auto &policy = completed_batch->policy_outputs[i];
            const auto &value_vec = completed_batch->value_outputs[i];

            // Set the inference result on the node
            node->setInferenceResult(policy, value_vec);
          }
        }
      } catch (const std::exception &e) {
        // On exception, we should still try to set some default result for nodes
        // or mark them as failed evaluation, but for now just log the error
        for (int i = 0; i < completed_batch->item_count; ++i) {
          MultiThreadMCTS::ThreadSafeNode *node = completed_batch->node_pointers[i];
          if (node) {
            // Set default/error result - zero policy and neutral value
            Vec<float, Config::BOARD_SQUARES> policy{};
            node->setInferenceResult(policy, 0.0f);
          }
        }
      }

      // Reset and recycle the batch regardless of success or failure
      completed_batch->item_count = 0;
      available_batches_.enqueue(completed_batch);

      // If collectors are waiting for a batch, wake them up
      if (active_batch_.load() == nullptr) {
        Batch *new_active_batch;
        if (available_batches_.try_dequeue(new_active_batch)) {
          active_batch_.store(new_active_batch);
          collector_cv_.notify_all();
        }
      }
    }
  }
}

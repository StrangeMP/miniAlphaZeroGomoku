#include "dispatcher.hpp"
#include "network.hpp"
#include <print>
#include <vector>
#include <thread>
#include <chrono>
#include <random>
#include <atomic>

// A simple atomic counter for successful requests
std::atomic<int> successful_requests = 0;
std::atomic<int> failed_requests = 0;
// The function that each worker thread will execute
void worker_task(int worker_id, int num_requests) {
  std::print("Worker {} started.\n", worker_id);

  // Create a random number generator for delays to simulate variance
  std::mt19937 gen(worker_id);
  std::uniform_int_distribution<> distrib(1, 5); // ms delay

  std::vector<std::future<InferenceDispatcher::OutputResult>> futures;
  futures.reserve(num_requests);

  // --- Submission Phase ---
  for (int i = 0; i < num_requests; ++i) {
    try {
      // Create dummy input data. For this test, the content doesn't matter.
      Network::BinaryInputUnit_T binary_input{};
      Network::GlobalInputUnit_T global_input{};
      Network::MaskInputUnit_T mask_input{};
      
      // Simulate some "work" being done before submitting
      std::this_thread::sleep_for(std::chrono::milliseconds(distrib(gen)));

      // Submit the request and store the future
      futures.push_back(InferenceDispatcher::getInstance().collect(binary_input, global_input, mask_input));

    } catch (const std::exception& e) {
      std::print("Worker {} submission error: {}\n", worker_id, e.what());
      failed_requests++;
    }
  }

  // --- Retrieval Phase ---
  for (int i = 0; i < futures.size(); ++i) {
    try {
      // Wait for the result. This will block until the dispatcher has processed the batch.
      futures[i].get();
      successful_requests++;
    } catch (const std::exception& e) {
      std::print("Worker {} retrieval error: {}\n", worker_id, e.what());
      failed_requests++;
    }
  }

  std::print("Worker {} completed.\n", worker_id);
}

int main() {
  std::print("Starting InferenceDispatcher test...\n");

  // --- Configuration ---
  const int NUM_WORKERS = 32;
  const int REQUESTS_PER_WORKER = 64;
  const int TOTAL_REQUESTS = NUM_WORKERS * REQUESTS_PER_WORKER;

  // --- Start Dispatcher ---
  auto& dispatcher = InferenceDispatcher::getInstance();
  try {
    dispatcher.start();
  } catch (const std::exception& e) {
    std::print("Failed to start dispatcher: {}\n", e.what());
    return 1;
  }
  
  std::print("Dispatcher started.\n");
  std::print("Launching {} workers, each submitting {} requests for a total of {} requests.\n", 
         NUM_WORKERS, REQUESTS_PER_WORKER, TOTAL_REQUESTS);

  // --- Launch Workers ---
  auto start_time = std::chrono::high_resolution_clock::now();
  std::vector<std::thread> workers;
  for (int i = 0; i < NUM_WORKERS; ++i) {
    workers.emplace_back(worker_task, i, REQUESTS_PER_WORKER);
  }

  // --- Wait for Completion ---
  for (auto& w : workers) {
    w.join();
  }
  auto end_time = std::chrono::high_resolution_clock::now();

  // --- Stop Dispatcher ---
  std::print("All workers finished. Stopping dispatcher...\n");
  dispatcher.stop();
  std::print("Dispatcher stopped.\n");

  // --- Report Results ---
  std::chrono::duration<double> elapsed = end_time - start_time;
  std::print("\n--- Test Summary ---\n");
  std::print("Total requests submitted: {}\n", TOTAL_REQUESTS);
  std::print("Successful requests: {}\n", successful_requests.load());
  std::print("Failed requests: {}\n", failed_requests.load());
  std::print("Total execution time: {} seconds\n", elapsed.count());
  std::print("Throughput: {} requests/sec\n", successful_requests.load() / elapsed.count());

  if (failed_requests.load() > 0 || successful_requests.load() != TOTAL_REQUESTS) {
    std::print("\nTEST FAILED\n");
    return 1;
  }

  std::print("\nTEST PASSED\n");
  return 0;
}

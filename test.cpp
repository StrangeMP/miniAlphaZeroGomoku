#include "NNet.hpp"
#include "compute.hpp"
#include <chrono>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <utility>
#include <vector>

// Helper function to measure execution time
template <typename Func> auto measureExecutionTime(Func &&func, int iterations) {
  std::vector<double> results(iterations);
  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < iterations; ++i) {
    results[i] = func();
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  double sum = std::accumulate(results.begin(), results.end(), 0.0);
  return std::pair{elapsed.count(), sum / iterations};
}

constexpr int kFilters = 96;
constexpr int kBlocks = 6;
constexpr int kChannels = 7;
constexpr int kHeight = 15;
constexpr int kWidth = 15;
using fp_t = float;
using int_t = int8_t;
ResNet<fp_t, kFilters, kChannels, kHeight, kWidth, kBlocks> resnet_fp;
ResNet<int_t, kFilters, kChannels, kHeight, kWidth, kBlocks> resnet_int8;
int main() {

  Tensor<fp_t, kChannels, kHeight, kWidth> input_fp{{{{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}},
                                                     {{1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}},
                                                     {{2.0f, 2.0f, 2.0f}, {2.0f, 2.0f, 2.0f}, {2.0f, 2.0f, 2.0f}}}};

  Tensor<int_t, kChannels, kHeight, kWidth> input_int8{
      {{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}, {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}}, {{2, 2, 2}, {2, 2, 2}, {2, 2, 2}}}};

  volatile fp_t fp_result = 0;
  volatile int_t int8_result = 0;
  // Test float ResNet performance
  constexpr int test_iterations = 500; // Adjust based on expected speed
  auto [fp_time, _] = measureExecutionTime(
      [&]() {
        resnet_fp.feed(input_fp);
        fp_result += resnet_fp.output()[0][0][0];
        return fp_result;
      },
      test_iterations);

  // Test int8 ResNet performance
  auto [int8_time, __] = measureExecutionTime(
      [&]() {
        resnet_int8.feed(input_int8);
        int8_result += resnet_int8.output()[0][0][0];
        return int8_result;
      },
      test_iterations);

  // Calculate iterations per second
  double fp_iterations_per_second = test_iterations / fp_time;
  double int8_iterations_per_second = test_iterations / int8_time;

  // Print results
  std::cout << "Network Architecture: \n"
            << "Channels: " << kChannels << ", Height: " << kHeight << ", Width: " << kWidth
            << ", Filters: " << kFilters << ", Blocks: " << kBlocks << ", Board Size: " << kHeight * kWidth
            << std::endl;
  std::cout << "Performance Test Results:" << std::endl;
  std::cout << "------------------------" << std::endl;
  std::cout << "Float ResNet: " << fp_iterations_per_second << " iterations per second" << std::endl;
  std::cout << "Int8 ResNet: " << int8_iterations_per_second << " iterations per second" << std::endl;
  std::cout << "Int8 is " << (int8_iterations_per_second / fp_iterations_per_second) << "x faster than float"
            << std::endl;
  std::cout << "Redundant result to make sure compiler doesn't optimize away the computation: " << std::endl;
  std::cout << "Float result: " << _ << std::endl;
  std::cout << "Int8 result: " << __ << std::endl;

  return 0;
}
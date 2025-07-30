#pragma once
#include "compute.hpp"
#include "utils.hpp"
#include <cstddef>
#include <cuda_runtime.h>
#include <tuple>
#include <utility>

namespace MCTS {
  struct Node;
}

struct Network {
  using WEIGHT_T = float;
  static constexpr size_t BOARD_SIZE = 15;
  static constexpr size_t TENSOR_SIZE = 15;
  static constexpr size_t IN_BIN_CHANNELS = 22;
  static constexpr size_t IN_GLOBAL_CHANNELS = 39;
  static constexpr size_t PASS_IDX = BOARD_SIZE * BOARD_SIZE; // 225
  /*
--- TensorRT Inference Results ---
Policy Logits:
  - Shape: (1, 1, 362)
  - DType: float32
  - Size (total elements): 362

Value Logits:
  - Shape: (1, 1)
  - DType: float32
  - Size (total elements): 1
  */

  //
  using BinaryInputUnit_T = Tensor<WEIGHT_T, IN_BIN_CHANNELS, TENSOR_SIZE, TENSOR_SIZE>;
  using GlobalInputUnit_T = Vec<WEIGHT_T, IN_GLOBAL_CHANNELS>;
  using InputUnit_T = std::pair<BinaryInputUnit_T, GlobalInputUnit_T>;
  using PolicyOut_T = Vec<float, BOARD_SIZE * BOARD_SIZE + 1>;
  using ValueOut_T = float;
  using ResultType = std::pair<PolicyOut_T, ValueOut_T>;
  using ResultPtr = std::unique_ptr<ResultType>;

  static void feed(void *d_binary_input, void *d_global_input, void *d_policy_output,
                   void *d_value_output, int batch_size, cudaStream_t stream);
  static ResultPtr evaluate(const Utils::Board &board, Utils::STONE_COLOR player);

// private:
  static InputUnit_T prepareInput(const Matrix<Utils::STONE_COLOR, BOARD_SIZE, BOARD_SIZE> &board,
                                                   Utils::STONE_COLOR player);
};

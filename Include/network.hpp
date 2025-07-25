#pragma once
#include "compute.hpp"
#include "utils.hpp"
#include <cstddef>
#include <cuda_runtime.h>
#include <memory>
#include <tuple>
#include <utility>

struct Network {
  using WEIGHT_T = float;
  static constexpr size_t BOARD_SIZE = 15;
  static constexpr size_t TENSOR_SIZE = 19;
  static constexpr size_t IN_BIN_CHANNELS = 22;
  static constexpr size_t IN_GLOBAL_CHANNELS = 39;
  static constexpr size_t MASK_SIZE = TENSOR_SIZE * TENSOR_SIZE + 1;
  /*
--- TensorRT Inference Results ---
Policy Logits:
  - Shape: (1, 1, 362)
  - DType: float32
  - Size (total elements): 362

Value Logits:
  - Shape: (1, 3)
  - DType: float32
  - Size (total elements): 3
  */

  //
  using BinaryInputUnit_T = Tensor<WEIGHT_T, IN_BIN_CHANNELS, TENSOR_SIZE, TENSOR_SIZE>;
  using GlobalInputUnit_T = Vec<WEIGHT_T, IN_GLOBAL_CHANNELS>;
  using MaskInputUnit_T = Vec<WEIGHT_T, MASK_SIZE>;
  using InputUnit_T = std::tuple<BinaryInputUnit_T, GlobalInputUnit_T, MaskInputUnit_T>;
  using PolicyOut_T = Vec<float, TENSOR_SIZE * TENSOR_SIZE + 1>;
  using ValueOut_T = Vec<float, 3>;
  using RetType = std::pair<PolicyOut_T, ValueOut_T>;

  static void feed(void *d_binary_input, void *d_global_input, void *d_mask_input, void *d_policy_output,
                   void *d_value_output, int batch_size, cudaStream_t stream);
  static RetType evaluate(const Matrix<Utils::STONE_COLOR, BOARD_SIZE, BOARD_SIZE> &board, Utils::STONE_COLOR player);

private:
  static std::unique_ptr<InputUnit_T> prepareInput(const Matrix<Utils::STONE_COLOR, BOARD_SIZE, BOARD_SIZE> &board,
                                                   Utils::STONE_COLOR player);
};

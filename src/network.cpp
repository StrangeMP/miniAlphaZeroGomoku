#include "network.hpp"
#include "ForbiddenPointFinder.h"
#include "config.hpp"
#include "dispatcher.hpp"
#include <NvInfer.h>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <memory>
#include <print>
#include <vector>

// Logger for TensorRT
class Logger : public nvinfer1::ILogger {
  void log(Severity severity, const char *msg) noexcept override {
    // suppress info-level messages
    if (severity <= Severity::kWARNING)
      std::println("{}", msg);
  }
};

// RAII wrappers for TensorRT objects are no longer needed with modern TensorRT.
// std::unique_ptr with the default deleter is sufficient.

struct TensorRTState {
  Logger logger;
  std::unique_ptr<nvinfer1::IRuntime> runtime;
  std::unique_ptr<nvinfer1::ICudaEngine> engine;
  std::unique_ptr<nvinfer1::IExecutionContext> context;
  cudaStream_t stream;

  // Store tensor names
  std::string input_binary_name;
  std::string input_global_name;
  std::string input_mask_name;
  std::string output_policy_name;
  std::string output_value_name;

  TensorRTState() {
    runtime.reset(nvinfer1::createInferRuntime(logger));

    std::ifstream engine_file(Config::ENGINE_PATH, std::ios::binary);
    if (!engine_file) {
      throw std::runtime_error("Could not open engine file: " + std::string(Config::ENGINE_PATH));
    }

    engine_file.seekg(0, std::ios::end);
    size_t engine_size = engine_file.tellg();
    engine_file.seekg(0, std::ios::beg);
    std::vector<char> engine_data(engine_size);
    engine_file.read(engine_data.data(), engine_size);

    engine.reset(runtime->deserializeCudaEngine(engine_data.data(), engine_size));
    if (!engine) {
      throw std::runtime_error("Failed to deserialize engine");
    }

    context.reset(engine->createExecutionContext());
    if (!context) {
      throw std::runtime_error("Failed to create execution context");
    }

    if (cudaStreamCreate(&stream) != cudaSuccess) {
      throw std::runtime_error("Failed to create CUDA stream");
    }

    // Get I/O tensor names from the engine
    input_binary_name = engine->getIOTensorName(0);
    input_global_name = engine->getIOTensorName(1);
    input_mask_name = engine->getIOTensorName(2);
    output_policy_name = engine->getIOTensorName(3);
    output_value_name = engine->getIOTensorName(4);
  }

  ~TensorRTState() { cudaStreamDestroy(stream); }
};

// File-static instance, initialized once.
static TensorRTState g_trt_state;

void Network::feed(void *d_binary_input, void *d_global_input, void *d_mask_input, void *d_policy_output,
                   void *d_value_output, int batch_size, cudaStream_t stream) {

  if (batch_size == 0) {
    return;
  }

  // --- Set input shapes ---
  g_trt_state.context->setInputShape(g_trt_state.input_binary_name.c_str(),
                                     nvinfer1::Dims4(batch_size, IN_BIN_CHANNELS, TENSOR_SIZE, TENSOR_SIZE));
  g_trt_state.context->setInputShape(g_trt_state.input_global_name.c_str(),
                                     nvinfer1::Dims2(batch_size, IN_GLOBAL_CHANNELS));
  g_trt_state.context->setInputShape(g_trt_state.input_mask_name.c_str(), nvinfer1::Dims2(batch_size, MASK_SIZE));

  // --- Set tensor addresses ---
  g_trt_state.context->setTensorAddress(g_trt_state.input_binary_name.c_str(), d_binary_input);
  g_trt_state.context->setTensorAddress(g_trt_state.input_global_name.c_str(), d_global_input);
  g_trt_state.context->setTensorAddress(g_trt_state.input_mask_name.c_str(), d_mask_input);
  g_trt_state.context->setTensorAddress(g_trt_state.output_policy_name.c_str(), d_policy_output);
  g_trt_state.context->setTensorAddress(g_trt_state.output_value_name.c_str(), d_value_output);

  // --- Execute inference ---
  g_trt_state.context->enqueueV3(stream);
}

Network::RetType Network::evaluate(const Matrix<Utils::STONE_COLOR, BOARD_SIZE, BOARD_SIZE> &board,
                                   Utils::STONE_COLOR player) {
  static auto &dispatcher = InferenceDispatcher::getInstance();
  auto input_unit = prepareInput(board, player);
  auto future = dispatcher.collect(std::get<0>(*input_unit), std::get<1>(*input_unit), std::get<2>(*input_unit));
  input_unit.reset(); // Release input memory
  return future.get();
}

std::unique_ptr<Network::InputUnit_T>
Network::prepareInput(const Matrix<Utils::STONE_COLOR, BOARD_SIZE, BOARD_SIZE> &board, Utils::STONE_COLOR player) {
  static constexpr auto mask_vec_index = [](size_t r, size_t c) { return r * TENSOR_SIZE + c; };
  static constexpr auto get_input_template = []() {
    // bf
    /*
      0       onBoard
      1       己方棋子
      2       对方棋子
      3       己方黑棋禁手
      4       对方黑棋禁手
      5       胜点（如果有）
    */

    InputUnit_T input{};
    auto &[binary, global, mask] = input;
    for (size_t r = 0; r < Config::BOARD_SIZE; ++r) {
      for (size_t c = 0; c < Config::BOARD_SIZE; ++c) {
        binary[0][r][c] = 1.0f;
      }
    }

    /*
    // gf
    3       无禁/有禁0，无禁六不胜1
    4       无禁/无禁六不胜0，有禁1
    5       无禁/无禁六不胜0，有禁黑-1，有禁白1
    6       是否使用禁手特征（两种无禁恒为0）
    7~12    自己和对手的VCF（是否使用vcf，vcf的结果是什么）
    38      胜点是否是pass（仅可能用于vcn防守方）

    13  非VCN模式：和棋胜率，1.0是和棋己方胜，-1.0是和棋对方胜
        VCN模式：0
    14  非VCN模式：=对手是否已经pass过
        VCN模式：0

        */

    global[4] = 1.0f; // 4 无禁/无禁六不胜0，有禁1
    global[6] = 1.0f; // 6 是否使用禁手特征（两种无禁恒为0)

    // mask
    // for points not in the top-left 15x15 board, set mask to 0
    for (size_t r = 0; r < TENSOR_SIZE; ++r) {
      for (size_t c = 0; c < TENSOR_SIZE; ++c) {
        if (r < BOARD_SIZE && c < BOARD_SIZE) {
          mask[mask_vec_index(r, c)] = 1.0f; // valid move
        } else {
          mask[mask_vec_index(r, c)] = 0.0f; // invalid move
        }
      }
    }
    return input;
  };
  auto p_input = std::make_unique<Network::InputUnit_T>(get_input_template());
  auto &[binary, global, mask] = *p_input;
  CForbiddenPointFinder fpf(board);
  for (size_t r = 0; r < BOARD_SIZE; ++r) {
    for (size_t c = 0; c < BOARD_SIZE; ++c) {
      auto stone = board[r][c];
      if (stone == player) {
        binary[1][r][c] = 1.0f; // 自己的棋子
      } else if (stone != Utils::EMPTY) {
        binary[2][r][c] = 1.0f; // 对方的棋子
      }

      if (fpf.isForbidden(r, c)) {
        if (player == Utils::BLACK) {
          binary[3][r][c] = 1.0f;            // 己方黑棋禁手
          mask[mask_vec_index(r, c)] = 0.0f; // 禁手点不可下
        } else {
          binary[4][r][c] = 1.0f; // 对方黑棋禁手
        }
      }
    }
  }

  return p_input;
}
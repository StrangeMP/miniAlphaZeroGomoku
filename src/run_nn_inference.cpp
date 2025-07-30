#include "config.hpp"
#include "network.hpp"
#include "utils.hpp"
#include <algorithm>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <print>
#include <sstream>
#include <string>
#include <utility>

// Helper for CUDA error checking
#define CUDA_CHECK(call)                                                                                               \
  do {                                                                                                                 \
    cudaError_t err = call;                                                                                            \
    if (err != cudaSuccess) {                                                                                          \
      std::cerr << "CUDA error in " << #call << " at " << __FILE__ << ":" << __LINE__ << " : "                         \
                << cudaGetErrorString(err) << std::endl;                                                               \
      exit(EXIT_FAILURE);                                                                                              \
    }                                                                                                                  \
  } while (0)

// Function to read board and player from a text file
std::pair<Utils::Board, Utils::STONE_COLOR> readBoardFromFile(const std::string &filePath) {
  std::ifstream file(filePath);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file: " + filePath);
  }

  Utils::Board board;
  std::string line;
  Utils::STONE_COLOR player;

  // First line is the player color
  if (std::getline(file, line)) {
    std::istringstream iss(line);
    int playerColor;
    iss >> playerColor;
    player = playerColor == 1 ? Utils::BLACK : Utils::WHITE;
  } else {
    throw std::runtime_error("Input file is empty or missing player color line.");
  }

  // Subsequent lines are the board
  int row = 0;
  while (std::getline(file, line) && row < Config::BOARD_SIZE) {
    std::istringstream iss(line);
    for (int col = 0; col < Config::BOARD_SIZE; ++col) {
      int stone;
      iss >> stone;
      board[row][col] = (stone == 0 ? Utils::EMPTY : (stone == 1 ? Utils::BLACK : Utils::WHITE));
    }
    ++row;
  }

  return {board, player};
}

class StandaloneInference {
public:
  StandaloneInference() {
    // Allocate Device memory
    CUDA_CHECK(cudaMalloc(&d_binary_input, sizeof(Network::BinaryInputUnit_T)));
    CUDA_CHECK(cudaMalloc(&d_global_input, sizeof(Network::GlobalInputUnit_T)));
    CUDA_CHECK(cudaMalloc(&d_policy_output, sizeof(Network::PolicyOut_T)));
    CUDA_CHECK(cudaMalloc(&d_value_output, sizeof(Network::ValueOut_T)));

    // Create a CUDA stream
    CUDA_CHECK(cudaStreamCreate(&stream));
  }

  ~StandaloneInference() {
    // Clean up
    CUDA_CHECK(cudaFree(d_binary_input));
    CUDA_CHECK(cudaFree(d_global_input));
    CUDA_CHECK(cudaFree(d_policy_output));
    CUDA_CHECK(cudaFree(d_value_output));
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

  void run(const Network::BinaryInputUnit_T &binary_input, const Network::GlobalInputUnit_T &global_input,
           Network::PolicyOut_T &policy_output, Network::ValueOut_T &value_output) {
    // Copy inputs from host to device
    CUDA_CHECK(cudaMemcpyAsync(d_binary_input, &binary_input, sizeof(Network::BinaryInputUnit_T),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_global_input, &global_input, sizeof(Network::GlobalInputUnit_T),
                               cudaMemcpyHostToDevice, stream));

    // Run NN inference
    std::println("Running inference...");
    Network::feed(d_binary_input, d_global_input, d_policy_output, d_value_output, 1, stream);
    std::println("Inference submitted.");

    // Copy outputs from device to host
    CUDA_CHECK(
        cudaMemcpyAsync(&policy_output, d_policy_output, sizeof(Network::PolicyOut_T), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(
        cudaMemcpyAsync(&value_output, d_value_output, sizeof(Network::ValueOut_T), cudaMemcpyDeviceToHost, stream));

    // Wait for all operations on the stream to complete
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::println("Inference completed.");
  }

private:
  // Device memory
  void *d_binary_input, *d_global_input;
  void *d_policy_output, *d_value_output;

  // CUDA stream
  cudaStream_t stream;
};
int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::println("Usage: {} <input_file>", argv[0]);
    return 1;
  }

  try {
    // 1. Read board and player from file
    auto [board, player] = readBoardFromFile(argv[1]);

    // 2. Prepare network input
    auto [binary_input, global_input] = Network::prepareInput(board, player);

    // 3. Encapsulated Inference
    StandaloneInference inference;
    Network::PolicyOut_T policy_output;
    Network::ValueOut_T value_output;
    inference.run(binary_input, global_input, policy_output, value_output);

    // 4. Print results
    std::print("Policy Output:\n");
    for (size_t i = 0; i < policy_output.size(); ++i) {
      std::print("{} ", policy_output[i]);
    }
    std::println("");

    auto max_it = std::max_element(policy_output.begin(), policy_output.end());
    int max_index = std::distance(policy_output.begin(), max_it);

    if (max_index < Config::BOARD_SIZE * Config::BOARD_SIZE) {
      auto max_row = max_index / Config::BOARD_SIZE;
      auto max_col = max_index % Config::BOARD_SIZE;
      std::println("Max Element Value: {}, Position: ({}, {})", *max_it, max_row, max_col);
      board[max_row][max_col] = player;
    } else {
      std::println("Max Element Position: Pass");
    }

    // 5. Write updated board back to file
    std::ofstream outFile(argv[1]);
    if (!outFile.is_open()) {
      throw std::runtime_error("Failed to open file for writing: " + std::string(argv[1]));
    }

    Utils::STONE_COLOR next_player = (player == Utils::BLACK) ? Utils::WHITE : Utils::BLACK;
    outFile << (next_player == Utils::WHITE ? 2 : static_cast<int>(next_player)) << "\n";

    for (int r = 0; r < Config::BOARD_SIZE; ++r) {
      for (int c = 0; c < Config::BOARD_SIZE; ++c) {
        outFile << (board[r][c] == Utils::WHITE ? 2 : static_cast<int>(board[r][c]))
                << (c == Config::BOARD_SIZE - 1 ? "" : " ");
      }
      outFile << "\n";
    }
    outFile.close();
    std::println("Updated board state written to {}", argv[1]);

    // Assuming ValueOut_T is an array-like container with at least one element.
    std::println("Value Head Output: {}", value_output);

  } catch (const std::exception &e) {
    std::println("Error: {}", e.what());
    return 1;
  }

  return 0;
}

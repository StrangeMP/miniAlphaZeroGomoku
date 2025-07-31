#pragma once
#include <cstddef>
#include <chrono>
namespace Config {
inline constexpr int BOARD_SIZE = 15;
inline constexpr int BOARD_SQUARES = BOARD_SIZE * BOARD_SIZE;

inline constexpr float C_PUCT = 1.5f;

inline constexpr int MAX_BATCH_SIZE = 32;

inline const char *ENGINE_PATH = "engine/engine_fp32.trt";

inline constexpr int NUM_BATCH_BUFFERS = 3;

inline constexpr auto BATCH_TIMEOUT_MS = std::chrono::milliseconds(10);
} // namespace Config
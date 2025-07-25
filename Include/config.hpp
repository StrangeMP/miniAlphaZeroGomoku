#pragma once
namespace Config {
inline constexpr int BOARD_SIZE = 15;
inline constexpr int BOARD_SQUARES = BOARD_SIZE * BOARD_SIZE;

inline constexpr float C_PUCT = 2.0f;
inline constexpr int SIMULATION_TIMES = 100;

inline constexpr double FORWARD_TIME_COST = 0.007;
inline constexpr int TIME_FOR_SIMS = 985;

inline constexpr int MAX_BATCH_SIZE = 32;

inline const char *ENGINE_PATH = "engine/engine_fp32.trt";

inline constexpr int NUM_BATCH_BUFFERS = 3;
} // namespace Config
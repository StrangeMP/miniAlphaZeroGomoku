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
// FPU 相关配置
namespace FPU {
  inline constexpr float REDUCTION_MAX = 0.2f;           // 最大 FPU 削减量
  inline constexpr float LOSS_PROP = 0.0f;               // 向损失倾斜的比例
  inline constexpr float EXPLORATION_LOG = 0.0f;         // 对数探索系数
  inline constexpr float EXPLORATION_BASE = 225.0f;      // 15x15 = 225，适应棋盘大小
  inline constexpr float UTILITY_STDEV_SCALE = 0.1f;     // 效用标准差缩放
  inline constexpr bool USE_ADVANCED_FPU = true;         // 是否启用高级 FPU
  
  // 第一优先级：KataGo高级参数
  inline constexpr float VALUE_WEIGHT_EXPONENT = 0.5f;   // 价值权重指数，对差子节点降权
  inline constexpr float UNCERTAINTY_COEFF = 0.15f;      // 不确定性系数，适配五子棋
  inline constexpr float UTILITY_STDEV_PRIOR = 0.25f;    // 效用标准差先验
  inline constexpr float UTILITY_STDEV_PRIOR_WEIGHT = 1.0f; // 先验权重
  inline constexpr bool USE_UNCERTAINTY_WEIGHTING = true; // 启用不确定性权重
}
} // namespace Config
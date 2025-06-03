#include "network.hpp"
namespace Config {
inline constexpr int BOARD_SIZE = 15;
inline constexpr int BOARD_SQUARES = BOARD_SIZE * BOARD_SIZE;

inline constexpr float C_PUCT = 2.0f;
inline constexpr int SIMULATION_TIMES = 100;

inline constexpr double FORWARD_TIME_COST = 0.007;
inline constexpr int TIME_FOR_SIMS = 985;

inline auto EMPTY_STONE = AlphaGomoku::EMPTY;
inline auto BLACK_STONE = AlphaGomoku::BLACK;
inline auto WHITE_STONE = AlphaGomoku::WHITE;
} // namespace Config
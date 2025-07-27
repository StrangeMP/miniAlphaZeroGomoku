#include "../Include/zobrist.hpp"

namespace Zobrist {
std::array<std::array<std::array<Hash128, COLOR_N>, BOARD_SIZE>, BOARD_SIZE> zobrist_table = {};
std::array<Hash128, COLOR_N> player_table = {};
std::atomic<bool> initialized = false;
}

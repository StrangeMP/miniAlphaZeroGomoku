#include "../Include/zobrist.hpp"

namespace Zobrist {
std::array<std::array<std::array<size_t, COLOR_N>, BOARD_SIZE>, BOARD_SIZE> zobrist_table = {};
std::array<size_t, COLOR_N> player_table = {};
bool initialized = false;
}

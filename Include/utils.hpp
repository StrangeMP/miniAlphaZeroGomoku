#pragma once
#include "compute.hpp"
#include "config.hpp"
namespace Utils {
enum STONE_COLOR { EMPTY = 0, BLACK = 1, WHITE = -1 };
using Board = Matrix<Utils::STONE_COLOR, Config::BOARD_SIZE, Config::BOARD_SIZE>;
using Coord = std::pair<int, int>;
} // namespace Utils

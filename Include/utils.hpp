#pragma once
#include "compute.hpp"
#include "config.hpp"
namespace Utils {
enum STONE_COLOR { EMPTY = 0, BLACK = 1, WHITE = -1 };
using Board = Matrix<Utils::STONE_COLOR, Config::BOARD_SIZE, Config::BOARD_SIZE>;
using Coord = std::pair<int, int>;

inline Utils::Coord index_to_coordinate(int index) { return {index / Config::BOARD_SIZE, index % Config::BOARD_SIZE}; }

inline int coordinate_to_index(Utils::Coord coord) { return coord.first * Config::BOARD_SIZE + coord.second; }

inline auto legal_moves(const Utils::Board &board) {
  std::array<bool, Config::BOARD_SQUARES + 1> legal_vec;
  for (int i = 0; i < Config::BOARD_SQUARES; ++i) {
    auto [r, c] = Utils::index_to_coordinate(i);
    legal_vec[i] = (board[r][c] == Utils::EMPTY);
  }
  legal_vec[Config::BOARD_SQUARES] = true; // Allow pass move
  return legal_vec;
}
} // namespace Utils

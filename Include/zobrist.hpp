#pragma once
#include "compute.hpp"
#include "network.hpp"
#include <random>
#include <cstddef>
#include <array>

namespace Zobrist {
constexpr size_t BOARD_SIZE = AlphaGomoku::GodNet::BOARD_SIZE;
constexpr int COLOR_N = 3; // EMPTY, BLACK, WHITE

extern std::array<std::array<std::array<size_t, COLOR_N>, BOARD_SIZE>, BOARD_SIZE> zobrist_table;
extern std::array<size_t, COLOR_N> player_table;
extern bool initialized;

inline void init(size_t seed = 0x12345678) {
    if (initialized) return;
    std::mt19937_64 rng(seed);
    for (size_t i = 0; i < BOARD_SIZE; ++i) {
        for (size_t j = 0; j < BOARD_SIZE; ++j) {
            for (int c = 0; c < COLOR_N; ++c) {
                zobrist_table[i][j][c] = rng();
            }
        }
    }
    for (int c = 0; c < COLOR_N; ++c) {
        player_table[c] = rng();
    }
    initialized = true;
}

inline size_t hash(const Matrix<AlphaGomoku::STONE_COLOR, BOARD_SIZE, BOARD_SIZE>& board, AlphaGomoku::STONE_COLOR player) {
    size_t h = 0;
    for (size_t i = 0; i < BOARD_SIZE; ++i) {
        for (size_t j = 0; j < BOARD_SIZE; ++j) {
            int c = 0;
            if (board[i][j] == AlphaGomoku::BLACK) c = 1;
            else if (board[i][j] == AlphaGomoku::WHITE) c = 2;
            h ^= zobrist_table[i][j][c];
        }
    }
    int p = 0;
    if (player == AlphaGomoku::BLACK) p = 1;
    else if (player == AlphaGomoku::WHITE) p = 2;
    h ^= player_table[p];
    return h;
}

} // namespace Zobrist 
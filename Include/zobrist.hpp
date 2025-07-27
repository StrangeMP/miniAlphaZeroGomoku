#pragma once
#include "compute.hpp"
#include "network.hpp"
#include "hash128.hpp"
#include <random>
#include <cstddef>
#include <array>
#include <atomic>
#include <mutex>
#include <iostream> // Added for debugging output

namespace Zobrist {
constexpr size_t BOARD_SIZE = AlphaGomoku::GodNet::BOARD_SIZE;
constexpr int COLOR_N = 3; // EMPTY, BLACK, WHITE

extern std::array<std::array<std::array<Hash128, COLOR_N>, BOARD_SIZE>, BOARD_SIZE> zobrist_table;
extern std::array<Hash128, COLOR_N> player_table;
extern std::atomic<bool> initialized;

inline void init(size_t seed = 0x12345678) {
    if (initialized.load(std::memory_order_acquire)) return;
    static std::mutex init_mutex;
    std::lock_guard<std::mutex> lock(init_mutex);
    if (initialized.load(std::memory_order_acquire)) return;
    std::mt19937_64 rng(seed);
    for (size_t i = 0; i < BOARD_SIZE; ++i) {
        for (size_t j = 0; j < BOARD_SIZE; ++j) {
            for (int c = 0; c < COLOR_N; ++c) {
                zobrist_table[i][j][c] = Hash128(rng(), rng());
            }
        }
    }
    for (int c = 0; c < COLOR_N; ++c) {
        player_table[c] = Hash128(rng(), rng());
    }
    initialized.store(true, std::memory_order_release);
}

// 辅助函数：将AlphaGomoku::STONE_COLOR映射到zobrist表索引
inline int color_to_index(AlphaGomoku::STONE_COLOR c) {
    // 假设AlphaGomoku::BLACK=-1, WHITE=1, EMPTY=0
    if (c == AlphaGomoku::BLACK) return 0;
    if (c == AlphaGomoku::WHITE) return 1;
    return 2; // EMPTY
}

// 新版hash，返回Hash128，混入步数
inline Hash128 hash(const Matrix<AlphaGomoku::STONE_COLOR, BOARD_SIZE, BOARD_SIZE>& board, AlphaGomoku::STONE_COLOR player, int move_number) {
    Hash128 h;
    for (size_t i = 0; i < BOARD_SIZE; ++i) {
        for (size_t j = 0; j < BOARD_SIZE; ++j) {
            int c = color_to_index(board[i][j]);
            h ^= zobrist_table[i][j][c];
        }
    }
    int p = color_to_index(player);
    h ^= player_table[p];
    h = Hash128::mixInt(h, move_number);
    return h;
}

// KataGo风格的hash：在基础hash上添加随机数
inline Hash128 hash_with_random(const Matrix<AlphaGomoku::STONE_COLOR, BOARD_SIZE, BOARD_SIZE>& board, AlphaGomoku::STONE_COLOR player, int move_number, uint64_t rand1, uint64_t rand2) {
    Hash128 base_hash = hash(board, player, move_number);
    return base_hash ^ Hash128(rand1, rand2);
}

} // namespace Zobrist 
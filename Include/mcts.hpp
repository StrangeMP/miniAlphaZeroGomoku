#include "compute.hpp"
#include "config.hpp"
#include "network.hpp"
#include "qnnet.hpp"
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <tuple>
#include <utility>

using Board = Matrix<AlphaGomoku::STONE_COLOR, Config::BOARD_SIZE, Config::BOARD_SIZE>;

namespace Utils {
using Coordinate = std::pair<int, int>; // (row, column)

inline Coordinate index_to_coordinate(int index) { return {index / Config::BOARD_SIZE, index % Config::BOARD_SIZE}; }

inline int coordinate_to_index(Coordinate coord) { return coord.first * Config::BOARD_SIZE + coord.second; }

inline auto legal_moves(const Board &board) {
  std::array<bool, Config::BOARD_SQUARES> legal_vec;
  for (int i = 0; i < Config::BOARD_SQUARES; ++i) {
    auto [r, c] = index_to_coordinate(i);
    legal_vec[i] = (board[r][c] == Config::EMPTY_STONE);
  }
  return legal_vec;
}
} // namespace Utils

struct Node {
  Node *parent;
  AlphaGomoku::STONE_COLOR current_color;
  AlphaGomoku::STONE_COLOR opponent_color;
  float prior_p;
  int visit_count = 0;
  float value_sum = 0.0f;
  bool is_end_node = false;

  Board board_state;
  std::array<std::unique_ptr<Node>, Config::BOARD_SQUARES> children;

  int prior_action_idx;
  Vec<float, Config::BOARD_SQUARES> pi;
  float value = 0.0f; // from value head of the network, or game result if this is an end node

  Node(Node *parent_, float prior, AlphaGomoku::STONE_COLOR turn, const Board &current_board, int action_idx, auto &net)
      : parent(parent_), current_color(turn),
        opponent_color(turn == Config::BLACK_STONE ? Config::WHITE_STONE : Config::BLACK_STONE), prior_p(prior),
        board_state(current_board), prior_action_idx(action_idx) {

    std::optional<std::pair<WEIGHT_T, WEIGHT_T>> last_move = {};
    bool game_ended = false;
    float game_result = 0.0f;
    if (prior_action_idx != -1) { // this is not the initial board node
      auto [r, c] = Utils::index_to_coordinate(prior_action_idx);
      board_state[r][c] = opponent_color; // apply the move to the board state
      last_move = {r, c};
      std::tie(game_ended, game_result) = ended();
    }

    if (game_ended) {
      is_end_node = true;
      value = game_result;
    } else {
      auto [pi_, v_] = net.feed(board_state, last_move, current_color);
      pi = pi_;
      value = v_;
    }
  }

  float child_score(const Node &child) const {
    int visit_count_of_child = child.visit_count;
    float q = ((visit_count_of_child == 0) ? 0.0 : -(child.value_sum / static_cast<float>(visit_count_of_child)));
    float u = Config::C_PUCT * child.prior_p * std::sqrt(this->visit_count) / (1.0 + visit_count_of_child);
    return q + u;
  }

  std::pair<int, Node *> select_child() const {
    auto legal_moves_vec = Utils::legal_moves(board_state);
    Node *best_child = nullptr;
    int best_action_idx = -1;
    float max_score = -std::numeric_limits<float>::infinity();

    int current_node_total_visits = this->visit_count;

    for (int i = 0; i < Config::BOARD_SQUARES; ++i) {
      if (!legal_moves_vec[i])
        continue;

      float score = (children[i] != nullptr)
                        ? child_score(*children[i].get())
                        : Config::C_PUCT * pi[i] * std::sqrt(static_cast<float>(current_node_total_visits));

      if (score > max_score) {
        max_score = score;
        best_child = children[i].get();
        best_action_idx = i;
      }
    }

    return {best_action_idx, best_child};
  }

  void backpropagate() {
    Node *current = this;
    auto v = value;
    while (current != nullptr) {
      current->visit_count++;
      current->value_sum += v;
      v *= -1;
      current = current->parent;
    }
  }

private:
  // {has_ended, game_result_if_end}
  std::pair<bool, float> ended() const {
    const int B_SIZE = Config::BOARD_SIZE;
    const auto [R, C] = Utils::index_to_coordinate(prior_action_idx);
    static constexpr int STONES_TO_WIN = 5;
    // 方向：水平、垂直、对角线 (左上到右下)、反对角线 (右上到左下)
    const int dr[] = {0, 1, 1, 1};  // 行增量
    const int dc[] = {1, 0, 1, -1}; // 列增量

    for (int i = 0; i < 4; ++i) { // 遍历 4 个方向
      int count = 1;              // 计算刚落下的棋子

      // 检查正方向
      for (int k = 1; k < STONES_TO_WIN; ++k) {
        int nr = R + k * dr[i];
        int nc = C + k * dc[i];
        if (nr >= 0 && nr < B_SIZE && nc >= 0 && nc < B_SIZE && board_state[nr][nc] == opponent_color) {
          count++;
        } else {
          break;
        }
      }

      // 检查负方向
      for (int k = 1; k < STONES_TO_WIN; ++k) {
        int nr = R - k * dr[i];
        int nc = C - k * dc[i];
        if (nr >= 0 && nr < B_SIZE && nc >= 0 && nc < B_SIZE && board_state[nr][nc] == opponent_color) {
          count++;
        } else {
          break;
        }
      }

      if (count >= STONES_TO_WIN) {
        return {true, -1.0f}; // last_player_color 获胜
      }
    }

    // 检查是否平局 (棋盘已满)
    bool board_full = true;
    for (int r = 0; r < B_SIZE; ++r) {
      for (int c = 0; c < B_SIZE; ++c) {
        if (board_state[r][c] == Config::EMPTY_STONE) {
          board_full = false;
          break;
        }
      }
      if (!board_full) {
        break;
      }
    }

    if (board_full) {
      return {true, 0.0f}; // 平局
    }

    // 游戏未结束
    return {false, {}};
  }
};

struct MCTS_Agent {
  std::unique_ptr<Node> root;
  AlphaGomoku::Network &net;

  MCTS_Agent(const Board &initial_board, AlphaGomoku::STONE_COLOR player_color, AlphaGomoku::Network &network)
      : net(network) {
    root = std::make_unique<Node>(nullptr, 1.0f, player_color, initial_board, -1, net);
  }

  MCTS_Agent(MCTS_Agent &&other) : net(other.net), root(std::move(other.root)) {}

  void run_mcts() {
    Node *node = root.get();

    while (!node->is_end_node) { // sift down to a leaf node or an end node
      auto [best_action_idx, next_node] = node->select_child();
      if (next_node != nullptr) {
        node = next_node;
      } else {
        auto &best_child_ptr = node->children[best_action_idx];
        best_child_ptr = std::make_unique<Node>(node, node->pi[best_action_idx], node->opponent_color,
                                                node->board_state, best_action_idx, net);
        node = best_child_ptr.get();
        break;
      }
    }
    node->backpropagate();
  }

  int next_move_idx() const {
    int max_visits = -1;
    int best_move_idx = -1;
    for (int i = 0; i < Config::BOARD_SQUARES; ++i) {
      if (root->children[i] != nullptr && root->children[i]->visit_count > max_visits) {
        max_visits = root->children[i]->visit_count;
        best_move_idx = i;
      }
    }
    return best_move_idx;
  }

  void apply_move(int move_idx) {
    auto &new_root = root->children[move_idx];
    if (new_root == nullptr) {
      new_root = std::make_unique<Node>(nullptr, 1.0f, root->opponent_color, root->board_state, move_idx, net);
    }
    root = std::move(new_root);
    root->parent = nullptr; // reset parent to nullptr for the new root
  }

  AlphaGomoku::STONE_COLOR last_move_color() const { return root->current_color; }
  const Board &last_move_board() const { return root->board_state; }
}; // MCTS_Agent
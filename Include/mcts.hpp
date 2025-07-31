#pragma once

#include "ForbiddenPointFinder.h"
#include "config.hpp"
#include "network.hpp"
#include <array>
#include <atomic>
#include <cassert>
#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

namespace MCTS {

// Node evaluation states for handling thread collisions
enum class NodeState : int {
  UNEXPANDED = 0, // Leaf node, not yet selected for evaluation
  EXPANDING = 1,  // Currently being evaluated by a thread
  EXPANDED = 2    // Evaluation complete, network result available
};

// MCTS Node representing a game state
//
// PASS MOVE HANDLING:
// - Nodes can represent pass moves (prior_action_idx == Network::PASS_IDX)
// - Pass nodes inherit board state but represent opponent's turn
// - Two consecutive passes result in draw (detected in check_consecutive_passes)
// - Pass moves are stored in children[Network::PASS_IDX]
struct Node {
  static constexpr float VIRTUAL_LOSS_VALUE = -1.0f;
  Node *parent;
  Utils::STONE_COLOR current_color;
  Utils::STONE_COLOR opponent_color;
  float prior_p;
  std::atomic<int> visit_count{0};
  std::atomic<float> value_sum{0.0f};
  std::atomic<NodeState> state{NodeState::UNEXPANDED};
  std::atomic<int> backup_factor{0};
  std::array<bool, Config::BOARD_SQUARES + 1> legal_moves; // +1 for pass move
  bool is_end_node = false;

  Utils::Board board_state;
  std::array<std::unique_ptr<Node>, Config::BOARD_SQUARES + 1> children; // +1 for pass move
  mutable std::mutex node_mutex;                                         // For protecting non-atomic operations

  int prior_action_idx;
  Network::ResultPtr network_result; // Contains both policy and value from network
  float end_node_value = 0.0f;       // Only used for terminal nodes (game ended)
  bool is_pass_move = false;         // True if this node represents a pass move

  Node(Node *parent_, float prior, Utils::STONE_COLOR turn, const Utils::Board &current_board, int action_idx)
      : parent(parent_), current_color(turn), opponent_color(turn == Utils::BLACK ? Utils::WHITE : Utils::BLACK),
        prior_p(prior), board_state(current_board), prior_action_idx(action_idx) {
    bool game_ended = false;
    float game_result = 0.0f;
    if (prior_action_idx != -1) { // this is not the initial board node
      if (prior_action_idx == Network::PASS_IDX) {
        // Pass move: inherit board state, switch colors, check for consecutive passes
        is_pass_move = true;
        std::tie(game_ended, game_result) = check_consecutive_passes();
      } else {
        // Regular move: place stone and check for win
        is_pass_move = false;
        auto [r, c] = Utils::index_to_coordinate(prior_action_idx);
        board_state[r][c] = opponent_color; // apply the move to the board state
        std::tie(game_ended, game_result) = ended();
      }
    }

    if (game_ended) {
      is_end_node = true;
      end_node_value = game_result;
      state.store(NodeState::EXPANDED); // Terminal nodes are considered expanded
      // No network result needed for terminal nodes
    } else {
      // CRITICAL: Don't evaluate immediately to fix virtual loss timing
      // Network evaluation will happen AFTER virtual loss is applied
      is_end_node = false;
      end_node_value = 0.0f;
      state.store(NodeState::UNEXPANDED);
      // network_result will be set during evaluation
    }

    legal_moves = Utils::legal_moves(board_state);
    if (current_color == Utils::BLACK) {
      CForbiddenPointFinder fpf(board_state);
      for (int i = 0; i < Config::BOARD_SQUARES; ++i) {
        auto [r, c] = Utils::index_to_coordinate(i);
        if (fpf.isForbidden(r, c)) {
          legal_moves[i] = false;
        }
      }
    }
  }

  // Helper methods to access policy and value
  const Network::PolicyOut_T &policy() const {
    if (is_end_node || !network_result) {
      throw std::runtime_error("Invalid access to policy of end node or unevaluated node");
    }
    return network_result->first;
  }

  float value() const {
    if (is_end_node) {
      return end_node_value;
    }
    if (!network_result) {
      throw std::runtime_error("Invalid access to value of unevaluated node");
    }
    return network_result->second;
  }

  bool is_evaluated() const { return state.load() == NodeState::EXPANDED; }

  bool is_expanding() const { return state.load() == NodeState::EXPANDING; }

  bool is_unexpanded() const { return state.load() == NodeState::UNEXPANDED; }

  // Atomically claim this node for expansion
  // Returns true if successfully claimed, false if already claimed by another thread
  bool try_claim_for_expansion() {
    NodeState expected = NodeState::UNEXPANDED;
    return state.compare_exchange_strong(expected, NodeState::EXPANDING);
  }

  float score() const {
    if (parent == nullptr) {
      throw std::runtime_error("Calling score() on root node");
    }
    int visit_count_of_node = visit_count.load();
    float value_sum_of_node = value_sum.load();
    int parent_visit_count = parent->visit_count.load();
    float q = (visit_count_of_node == 0) ? 0.0f : value_sum_of_node / static_cast<float>(visit_count_of_node);
    float u =
        Config::C_PUCT * prior_p * std::sqrt(static_cast<float>(parent_visit_count)) / (1.0f + visit_count_of_node);
    return q + u;
  }

  std::pair<int, Node *> select_child() const {
    Node *best_child = nullptr;
    int best_action_idx = -1;
    float max_score = -std::numeric_limits<float>::infinity();

    int current_node_total_visits = this->visit_count.load();

    // Check regular moves
    for (int i = 0; i < Config::BOARD_SQUARES; ++i) {
      if (!legal_moves[i])
        continue;

      float score = (children[i] != nullptr)
                        ? children[i]->score()
                        : Config::C_PUCT * policy()[i] * std::sqrt(static_cast<float>(current_node_total_visits));

      if (score > max_score) {
        max_score = score;
        best_child = children[i].get();
        best_action_idx = i;
      }
    }

    // Check pass move (always legal)
    int pass_idx = Network::PASS_IDX;
    float pass_score = (children[pass_idx] != nullptr) ? children[pass_idx]->score()
                                                       : Config::C_PUCT * policy()[pass_idx] *
                                                             std::sqrt(static_cast<float>(current_node_total_visits));

    if (pass_score > max_score) {
      max_score = pass_score;
      best_child = children[pass_idx].get();
      best_action_idx = pass_idx;
    }

    return {best_action_idx, best_child};
  }

  void apply_virtual_loss(int n) { // we only define the apply function here, cleaning-up is done in backpropagation
    visit_count.fetch_add(n);
    value_sum.fetch_add(n * VIRTUAL_LOSS_VALUE);
  }

  void backup_after_evaluation(float backup_value) {
    Node *current = this;
    auto v = backup_value;
    auto bf = backup_factor.load();
    backup_factor.store(0);

    auto added_visit_count = bf;
    auto added_value_sum = bf * VIRTUAL_LOSS_VALUE;

    auto final_visit_count = 1 + bf;
    auto final_value_sum = final_visit_count * backup_value;

    auto delta_visit_count = final_visit_count - added_visit_count;
    auto delta_value_sum = final_value_sum - added_value_sum;

    while (current != nullptr) {
      current->visit_count.fetch_add(delta_visit_count);
      current->value_sum.fetch_add(delta_value_sum);
      v *= -1.0f;
      current = current->parent;
    }
  }

  void backup_end_node(float backup_value) {
    Node *current = this;
    auto v = backup_value;

    auto added_visit_count = 1;
    auto added_value_sum = VIRTUAL_LOSS_VALUE;

    auto final_visit_count = 1;
    auto final_value_sum = backup_value;

    auto delta_visit_count = final_visit_count - added_visit_count; // this is always 0
    auto delta_value_sum = final_value_sum - added_value_sum;

    while (current != nullptr) {
      // current->visit_count.fetch_add(delta_visit_count); // this is always 0
      current->value_sum.fetch_add(delta_value_sum);
      v *= -1.0f;
      current = current->parent;
    }
  }

  // Evaluate this node with network (MUST be called after successfully claiming)
  void evaluate_with_network() {
    if (is_end_node)
      return; // End nodes don't need network evaluation

    // This should only be called after try_claim_for_expansion() returned true
    // The node should be in EXPANDING state
    assert(state.load() == NodeState::EXPANDING);

    // Move the network result directly (no copying!)
    network_result = Network::evaluate(board_state, current_color);

    // Mark as expanded - evaluation complete
    state.store(NodeState::EXPANDED);
  }

  // Check for consecutive passes (draw condition)
  std::pair<bool, float> check_consecutive_passes() const {
    // If this is a pass move and parent also made a pass, it's a draw
    if (parent != nullptr && parent->is_pass_move) {
      return {true, 0.0f}; // Draw
    }
    return {false, 0.0f}; // Game continues
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
        if (board_state[r][c] == Utils::EMPTY) {
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

// Multithreaded MCTS Agent with worker thread pool
//
// PASS MOVE HANDLING:
// - Pass moves are represented by action index Network::PASS_IDX (225 for 15x15 board)
// - Pass moves inherit the board state but switch player colors
// - Consecutive passes by both players result in a draw (game_result = 0.0)
// - Pass moves are always considered legal and included in action selection

//
class MCTSAgent {

private:
  std::unique_ptr<Node> last_root;
  std::unique_ptr<Node> *root_ptr;
  static constexpr float VIRTUAL_LOSS_VALUE = -0.1f;
  std::vector<std::unique_ptr<std::thread>> worker_threads;
  std::atomic<bool> stop_search{false};
  std::atomic<int> simulations_completed{0};
  int num_threads;
  int target_simulations;

public:
  Node &root_node() { return *(*root_ptr); }
  const Node &root_node() const { return *(*root_ptr); }

private:
  void worker_loop() {
    while (!stop_search.load()) {
      if (simulations_completed.load() >= target_simulations) {
        break;
      }

      run_single_simulation();
      simulations_completed.fetch_add(1);
    }
  }

  void run_single_simulation() {
    Node *node = &root_node();
    float backup_value = 0.0f;
    // --- Phase 1: Descend through the EXPANDED part of the tree ---
    while (node->is_evaluated() && !node->is_end_node) {
      node->apply_virtual_loss(1);
      auto [best_action_idx, next_node] = node->select_child();
      if (next_node == nullptr) {
        std::lock_guard<std::mutex> lock(node->node_mutex);
        auto &child_ptr = node->children[best_action_idx];
        if (child_ptr == nullptr) {
          child_ptr = std::make_unique<Node>(node, node->policy()[best_action_idx], node->opponent_color,
                                             node->board_state, best_action_idx);
        }
        node = child_ptr.get();
      } else {
        node = next_node;
      }
    }
    // Handle the edge node
    node->apply_virtual_loss(1);
    if (node->is_end_node) { // terminal node
      node->backup_end_node(node->value());
      return;
    } else if (node->is_expanding()) { // expanding node, no value to backpropagate
      node->backup_factor.fetch_add(1);
      return;
    } else {
      // unexpanded node, try to claim and evaluate
      if (node->try_claim_for_expansion()) {
        node->evaluate_with_network();
        backup_value = node->value();
        node->backup_after_evaluation(backup_value);
        return;
      } else {
        // failed to claim, another thread is evaluating this node
        // this is an expanding node, no value to backpropagate
        node->backup_factor.fetch_add(1);
        return;
      }
    }
  }

public:
  MCTSAgent(const Utils::Board &initial_board, Utils::STONE_COLOR player_color, int num_threads = 8)
      : last_root(std::make_unique<Node>(nullptr, 1.0f, player_color, initial_board, -1)), root_ptr(&last_root),
        num_threads(num_threads) {
    // Root node should be evaluated immediately (no race condition for root)
    if (!root_node().is_end_node && root_node().try_claim_for_expansion()) {
      root_node().evaluate_with_network();
    }
  }

  ~MCTSAgent() {
    stop_search.store(true);
    for (auto &thread : worker_threads) {
      if (thread && thread->joinable()) {
        thread->join();
      }
    }
  }

  void run_mcts(int num_simulations) {
    target_simulations = num_simulations;
    simulations_completed.store(0);
    stop_search.store(false);

    // Start worker threads
    worker_threads.clear();
    for (int i = 0; i < num_threads; ++i) {
      worker_threads.emplace_back(std::make_unique<std::thread>(&MCTSAgent::worker_loop, this));
    }

    // Wait for all threads to complete
    for (auto &thread : worker_threads) {
      if (thread && thread->joinable()) {
        thread->join();
      }
    }
  }

  int last_move_idx() const { return root_node().prior_action_idx; }

  int next_move_idx() const {
    const auto &root = root_node();
    int max_visits = -1;
    int best_move_idx = -1;
    // Check all moves including pass move
    for (int i = 0; i <= Config::BOARD_SQUARES; ++i) {
      if (root.children[i] != nullptr) {
        int child_visits = root.children[i]->visit_count.load();
        if (child_visits > max_visits) {
          max_visits = child_visits;
          best_move_idx = i;
        }
      }
    }
    return best_move_idx;
  }

  void apply_move(int move_idx) {
    stop_search.store(true);
    for (auto &thread : worker_threads) {
      if (thread && thread->joinable()) {
        thread->join();
      }
    }
    worker_threads.clear();

    last_root = std::move(*root_ptr);
    auto &new_root = last_root->children[move_idx];
    if (new_root == nullptr) {
      new_root = std::make_unique<Node>(nullptr, 1.0f, last_root->opponent_color, last_root->board_state, move_idx);
    }
    root_ptr = &new_root;
    new_root->parent = nullptr; // reset parent to nullptr for the new root
    // Ensure new root is evaluated (no race condition for root)
    if (!new_root->is_end_node && new_root->try_claim_for_expansion()) {
      new_root->evaluate_with_network();
    }
  }

  void undo_last_move() {
    stop_search.store(true);
    for (auto &thread : worker_threads) {
      if (thread && thread->joinable()) {
        thread->join();
      }
    }
    worker_threads.clear();
    root_node().parent = last_root.get();
    root_ptr = &last_root;
  }

  // Helper method to check if a move is a pass
  bool is_pass_move(int move_idx) const { return move_idx == Network::PASS_IDX; }

  // Get move description for debugging
  std::string get_move_description(int move_idx) const {
    if (move_idx == -1)
      return "no move";
    if (move_idx == Network::PASS_IDX)
      return "pass";
    auto [r, c] = Utils::index_to_coordinate(move_idx);
    return "(" + std::to_string(r) + "," + std::to_string(c) + ")";
  }

  Utils::STONE_COLOR last_move_color() const { return root_node().current_color; }
  Utils::STONE_COLOR next_move_color() const { return root_node().opponent_color; }
  const Utils::Board &last_move_board() const { return root_node().board_state; }

  int get_simulations_completed() const { return simulations_completed.load(); }
  void set_num_threads(int new_num_threads) { num_threads = new_num_threads; }
}; // MCTSAgent

} // namespace MCTS
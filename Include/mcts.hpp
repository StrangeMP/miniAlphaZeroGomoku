#pragma once

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
  Node *parent;
  Utils::STONE_COLOR current_color;
  Utils::STONE_COLOR opponent_color;
  float prior_p;
  std::atomic<int> visit_count{0};
  std::atomic<float> value_sum{0.0f};
  std::atomic<int> virtual_loss_count{0};
  std::atomic<NodeState> state{NodeState::UNEXPANDED};
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
  }

  // Helper methods to access policy and value
  const Network::PolicyOut_T &policy() const {
    if (is_end_node || !network_result) {
      throw std::runtime_error("Invalid access to policy of end node or unevaluated node");
    }
    return network_result->first;
  }

  // Safe policy access for unexpanded nodes (returns uniform distribution)
  float get_policy_safe(int action_idx) const {
    if (is_end_node || !network_result) {
      // For end nodes or unevaluated nodes, return uniform distribution
      return 1.0f / static_cast<float>(Config::BOARD_SQUARES + 1); // +1 for pass move
    }
    return network_result->first[action_idx];
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

  float child_score(const Node &child) const {
    int visit_count_of_child = child.visit_count.load();
    int virtual_loss_of_child = child.virtual_loss_count.load();
    float value_sum_of_child = child.value_sum.load();

    // Adjust for virtual loss: virtual losses count as negative visits
    int effective_visits = visit_count_of_child + virtual_loss_of_child;
    float effective_value_sum = value_sum_of_child - static_cast<float>(virtual_loss_of_child);

    float q = (effective_visits == 0) ? 0.0f : -(effective_value_sum / static_cast<float>(effective_visits));
    float u = Config::C_PUCT * child.prior_p * std::sqrt(static_cast<float>(this->visit_count.load())) /
              (1.0f + effective_visits);
    return q + u;
  }

  std::pair<int, Node *> select_child() const {
    auto legal_moves_vec = Utils::legal_moves(board_state);
    Node *best_child = nullptr;
    int best_action_idx = -1;
    float max_score = -std::numeric_limits<float>::infinity();

    int current_node_total_visits = this->visit_count.load();

    // Check regular moves
    for (int i = 0; i < Config::BOARD_SQUARES; ++i) {
      if (!legal_moves_vec[i])
        continue;

      float score = (children[i] != nullptr) ? child_score(*children[i].get())
                                             : Config::C_PUCT * get_policy_safe(i) *
                                                   std::sqrt(static_cast<float>(current_node_total_visits));

      if (score > max_score) {
        max_score = score;
        best_child = children[i].get();
        best_action_idx = i;
      }
    }

    // Check pass move (always legal)
    int pass_idx = Network::PASS_IDX;
    float pass_score = (children[pass_idx] != nullptr) ? child_score(*children[pass_idx].get())
                                                       : Config::C_PUCT * get_policy_safe(pass_idx) *
                                                             std::sqrt(static_cast<float>(current_node_total_visits));

    if (pass_score > max_score) {
      max_score = pass_score;
      best_child = children[pass_idx].get();
      best_action_idx = pass_idx;
    }

    return {best_action_idx, best_child};
  }

  void backpropagate(float backup_value) {
    Node *current = this;
    auto v = backup_value;
    while (current != nullptr) {
      current->visit_count.fetch_add(1);
      current->value_sum.fetch_add(v);
      v *= -1.0f;
      current = current->parent;
    }
  }

  // Add virtual loss to the path from root to this node
  void add_virtual_loss() {
    Node *current = this;
    while (current != nullptr) {
      current->virtual_loss_count.fetch_add(1);
      current = current->parent;
    }
  }

  // Remove virtual loss from the path from root to this node
  void remove_virtual_loss() {
    Node *current = this;
    while (current != nullptr) {
      current->virtual_loss_count.fetch_sub(1);
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
public:
  std::unique_ptr<Node> root;

private:
  static constexpr float VIRTUAL_LOSS_VALUE = -0.1f;
  std::vector<std::unique_ptr<std::thread>> worker_threads;
  std::atomic<bool> stop_search{false};
  std::atomic<int> simulations_completed{0};
  int num_threads;
  int target_simulations;

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
    Node *node = root.get();
    std::vector<Node *> path;
    float backup_value = 0.0f;

    // --- Phase 1: Descend through the EXPANDED part of the tree ---
    while (node->is_evaluated() && !node->is_end_node) {
      path.push_back(node);
      auto [best_action_idx, next_node] = node->select_child();

      if (next_node == nullptr) {
        // We've chosen an action that leads to a node that needs to be created.
        std::lock_guard<std::mutex> lock(node->node_mutex);
        auto &child_ptr = node->children[best_action_idx];
        if (child_ptr == nullptr) { // Double-check after lock
          child_ptr = std::make_unique<Node>(node, node->policy()[best_action_idx], node->opponent_color,
                                             node->board_state, best_action_idx);
        }
        node = child_ptr.get();
      } else {
        // The child already existed, just move to it.
        node = next_node;
      }
    }

    // --- Phase 2: Handle the "edge" node found by the loop ---
    path.push_back(node);

    // CRITICAL: Apply virtual loss BEFORE any potential blocking/waiting
    for (Node *n : path) {
      n->virtual_loss_count.fetch_add(1);
    }

    if (node->is_end_node) {
      // We reached a terminal node.
      backup_value = node->value();
    } else if (node->is_expanding()) {
      // We collided with a node another thread is currently evaluating.
      backup_value = VIRTUAL_LOSS_VALUE; // Small negative value for virtual loss
    } else {                             // The node must be UNEXPANDED.
      // Try to claim and evaluate this leaf node.
      if (node->try_claim_for_expansion()) {
        node->evaluate_with_network();
        backup_value = node->value();
      } else {
        // We lost the race to claim it. Treat as a collision.
        backup_value = VIRTUAL_LOSS_VALUE; // Small negative value for virtual loss
      }
    }

    // --- Phase 3: Backpropagation ---
    node->backpropagate(backup_value);

    // Remove virtual loss from the entire path
    for (Node *n : path) {
      n->virtual_loss_count.fetch_sub(1);
    }
  }

public:
  MCTSAgent(const Utils::Board &initial_board, Utils::STONE_COLOR player_color, int num_threads = 8)
      : root(std::make_unique<Node>(nullptr, 1.0f, player_color, initial_board, -1)), num_threads(num_threads) {
    // Root node should be evaluated immediately (no race condition for root)
    if (!root->is_end_node && root->try_claim_for_expansion()) {
      root->evaluate_with_network();
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

  int last_move_idx() const { return root->prior_action_idx; }

  int next_move_idx() const {
    int max_visits = -1;
    int best_move_idx = -1;
    // Check all moves including pass move
    for (int i = 0; i <= Config::BOARD_SQUARES; ++i) {
      if (root->children[i] != nullptr) {
        int child_visits = root->children[i]->visit_count.load();
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

    auto &new_root = root->children[move_idx];
    if (new_root == nullptr) {
      new_root = std::make_unique<Node>(nullptr, 1.0f, root->opponent_color, root->board_state, move_idx);
    }
    root = std::move(new_root);
    root->parent = nullptr; // reset parent to nullptr for the new root
    // Ensure new root is evaluated (no race condition for root)
    if (!root->is_end_node && root->try_claim_for_expansion()) {
      root->evaluate_with_network();
    }
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

  Utils::STONE_COLOR last_move_color() const { return root->current_color; }
  Utils::STONE_COLOR next_move_color() const { return root->opponent_color; }
  const Utils::Board &last_move_board() const { return root->board_state; }

  int get_simulations_completed() const { return simulations_completed.load(); }
  void set_num_threads(int new_num_threads) { num_threads = new_num_threads; }
}; // MCTSAgent

} // namespace MCTS
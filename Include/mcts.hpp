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
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

namespace MCTS {

// Node evaluation states for handling thread collisions
enum class NodeState : int {
  UNEXPANDED = 0,  // Leaf node, not yet selected for evaluation
  EXPANDING = 1,   // Currently being evaluated by a thread
  EXPANDED = 2     // Evaluation complete, network result available
};

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
  std::array<std::unique_ptr<Node>, Config::BOARD_SQUARES> children;
  mutable std::mutex node_mutex; // For protecting non-atomic operations

  int prior_action_idx;
  Network::ResultPtr network_result; // Contains both policy and value from network
  float end_node_value = 0.0f; // Only used for terminal nodes (game ended)

  Node(Node *parent_, float prior, Utils::STONE_COLOR turn, const Utils::Board &current_board, int action_idx)
      : parent(parent_), current_color(turn), opponent_color(turn == Utils::BLACK ? Utils::WHITE : Utils::BLACK),
        prior_p(prior), board_state(current_board), prior_action_idx(action_idx) {

    bool game_ended = false;
    float game_result = 0.0f;
    if (prior_action_idx != -1) { // this is not the initial board node
      auto [r, c] = Utils::index_to_coordinate(prior_action_idx);
      board_state[r][c] = opponent_color; // apply the move to the board state
      std::tie(game_ended, game_result) = ended();
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
  const Network::PolicyOut_T& policy() const {
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

  bool is_evaluated() const {
    return state.load() == NodeState::EXPANDED;
  }

  bool is_expanding() const {
    return state.load() == NodeState::EXPANDING;
  }

  bool is_unexpanded() const {
    return state.load() == NodeState::UNEXPANDED;
  }

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
    float u = Config::C_PUCT * child.prior_p * std::sqrt(static_cast<float>(this->visit_count.load())) / (1.0f + effective_visits);
    return q + u;
  }

  std::pair<int, Node *> select_child() const {
    auto legal_moves_vec = Utils::legal_moves(board_state);
    Node *best_child = nullptr;
    int best_action_idx = -1;
    float max_score = -std::numeric_limits<float>::infinity();

    int current_node_total_visits = this->visit_count.load();

    for (int i = 0; i < Config::BOARD_SQUARES; ++i) {
      if (!legal_moves_vec[i])
        continue;

      float score = (children[i] != nullptr)
                        ? child_score(*children[i].get())
                        : Config::C_PUCT * policy()[i] * std::sqrt(static_cast<float>(current_node_total_visits));

      if (score > max_score) {
        max_score = score;
        best_child = children[i].get();
        best_action_idx = i;
      }
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
    if (is_end_node) return; // End nodes don't need network evaluation
    
    // This should only be called after try_claim_for_expansion() returned true
    // The node should be in EXPANDING state
    assert(state.load() == NodeState::EXPANDING);
    
    // Move the network result directly (no copying!)
    network_result = Network::evaluate(board_state, current_color);
    
    // Mark as expanded - evaluation complete
    state.store(NodeState::EXPANDED);
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
class MCTSAgent {
private:
  std::unique_ptr<Node> root;
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

    // Selection phase
    std::vector<Node*> path;
    float backup_value = 0.0f;
    bool collision_occurred = false;

    while (!node->is_end_node) {
      path.push_back(node);
      
      // Check for collision: if node is being expanded by another thread
      if (node->is_expanding()) {
        // COLLISION DETECTED: Treat as immediate loss
        backup_value = -1.0f;
        collision_occurred = true;
        break;
      }
      
      auto [best_action_idx, next_node] = node->select_child();
      if (next_node != nullptr) {
        node = next_node;
      } else {
        // Try to claim this node for expansion
        if (!node->try_claim_for_expansion()) {
          // Another thread claimed it first - treat as collision
          backup_value = -1.0f;
          collision_occurred = true;
          break;
        }
        
        // Successfully claimed - create new child
        std::lock_guard<std::mutex> lock(node->node_mutex);
        auto &best_child_ptr = node->children[best_action_idx];
        if (best_child_ptr == nullptr) { // Double-check after lock
          best_child_ptr = std::make_unique<Node>(node, node->policy()[best_action_idx], 
                                                  node->opponent_color, node->board_state, best_action_idx);
        }
        node = best_child_ptr.get();
        path.push_back(node);
        break;
      }
    }

    // CRITICAL: Apply virtual loss FIRST to reserve the path
    for (Node* n : path) {
      n->virtual_loss_count.fetch_add(1);
    }

    if (!collision_occurred) {
      // Normal path: evaluate with network if needed
      if (!node->is_end_node && node->is_unexpanded()) {
        if (node->try_claim_for_expansion()) {
          node->evaluate_with_network();
        } else {
          // Collision during evaluation attempt
          backup_value = -1.0f;
          collision_occurred = true;
        }
      }
      
      if (!collision_occurred) {
        backup_value = node->value();
      }
    }

    // Backpropagation with real statistics
    node->backpropagate(backup_value);

    // Remove virtual loss from entire path
    for (Node* n : path) {
      n->virtual_loss_count.fetch_sub(1);
    }
  }

public:
  MCTSAgent(const Utils::Board &initial_board, Utils::STONE_COLOR player_color, int num_threads = 8)
      : root(std::make_unique<Node>(nullptr, 1.0f, player_color, initial_board, -1)),
        num_threads(num_threads) {
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
    for (int i = 0; i < Config::BOARD_SQUARES; ++i) {
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

  Utils::STONE_COLOR last_move_color() const { return root->current_color; }
  Utils::STONE_COLOR next_move_color() const { return root->opponent_color; }
  const Utils::Board &last_move_board() const { return root->board_state; }
  
  int get_simulations_completed() const { return simulations_completed.load(); }
  void set_num_threads(int new_num_threads) { num_threads = new_num_threads; }
}; // MCTSAgent

} // namespace MCTS
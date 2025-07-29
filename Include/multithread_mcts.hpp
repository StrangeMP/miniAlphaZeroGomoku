#pragma once

#include "hash128.hpp"
#include "multithreadhelpers.hpp"
#include "network.hpp"
#include "utils.hpp"
#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <random>
#include <unordered_map>

using namespace MultiThreadHelpers;

namespace MultiThreadMCTS {

// 前向声明
struct ThreadSafeNode;

// 测试友元访问器（全局作用域，提前声明）
class NodeTableTestAccessor;

// 全局节点表（transposition table）
class NodeTable {
public:
  using NodePtr = std::shared_ptr<ThreadSafeNode>;
  // 构造函数，支持指定分片数
  explicit NodeTable(size_t num_shards = 64)
      : num_shards_(num_shards), shards_(num_shards), shard_mutexes_(num_shards) {}
  // 查找或插入节点，线程安全
  NodePtr get_or_create(const Board &board, Utils::STONE_COLOR player, int move_number,
                        std::function<NodePtr()> node_factory);
  void print_stats() const;

private:
  size_t num_shards_;
  std::vector<std::unordered_map<Hash128, NodePtr>> shards_;
  mutable std::vector<std::mutex> shard_mutexes_;

  // KataGo风格的随机数生成器
  mutable std::mt19937_64 rng_{std::random_device{}()};
  mutable std::mutex rng_mutex_;

  // hash分片辅助
  size_t get_shard_idx(const Hash128 &hash) const { return (hash.hash0 ^ hash.hash1) % num_shards_; }

  // 生成随机数（线程安全）
  std::pair<uint64_t, uint64_t> generate_random_pair() const {
    std::lock_guard<std::mutex> lock(rng_mutex_);
    return {rng_(), rng_()};
  }

  friend class NodeTableTestAccessor;
};

// 测试友元访问器实现
class NodeTableTestAccessor {
public:
  // 获取所有分片（Hash128为key）
  static const std::vector<std::unordered_map<Hash128, NodeTable::NodePtr>> &get_shards(const NodeTable &table) {
    return table.shards_;
  }
  // 获取所有分片互斥锁
  static std::vector<std::mutex> &get_shard_mutexes(NodeTable &table) { return table.shard_mutexes_; }
};

// 多线程安全的Node结构
struct ThreadSafeNode {
  // 状态机定义
  enum class NodeState {
    UNEVALUATED, // 节点未评估
    EVALUATING,  // 正在评估中
    EXPANDED     // 已评估完成
  };

  size_t debug_id = 0; // 用于调试唯一标识
  ThreadSafeNode *parent;
  Utils::STONE_COLOR current_color;
  Utils::STONE_COLOR opponent_color;
  float prior_p;

  // 原子化的统计信息
  MultiThreadHelpers::AtomicNodeStats stats;
  MultiThreadHelpers::VirtualLossManager virtual_loss;

  // 状态机：原子状态变量
  std::atomic<NodeState> state{NodeState::UNEVALUATED};

  bool is_end_node = false;
  Board board_state;

  // 使用原子shared_ptr数组管理子节点，采用KataGo风格的原子操作
  std::array<std::atomic<std::shared_ptr<ThreadSafeNode>>, Config::BOARD_SQUARES> children;

  Vec<float, Config::BOARD_SQUARES + 1> pi;
  float value;
  int move_number = 0; // 新增：当前节点的步数

  // 构造函数

  ThreadSafeNode(ThreadSafeNode *parent_, float prior, Utils::STONE_COLOR turn, const Board &current_board,
                 int action_idx, Network &net, int move_number_ = 0)
      : parent(parent_), current_color(turn), opponent_color(turn == Utils::BLACK ? Utils::WHITE : Utils::BLACK),
        prior_p(prior), board_state(current_board), move_number(move_number_) {
    static size_t debug_id_counter = 1;
    debug_id = debug_id_counter++;

    // 初始化所有子节点指针为nullptr（原子shared_ptr需要特殊初始化）
    for (auto &child : children) {
      child.store(nullptr, std::memory_order_relaxed);
    }

    bool game_ended = false;
    float game_result = 0.0f;

    if (game_ended) {
      is_end_node = true;
      value = game_result;
      // 游戏结束的节点直接设置为EXPANDED，不需要神经网络评估
      state.store(NodeState::EXPANDED, std::memory_order_seq_cst);
    } else {
      // 去掉启发式初始化，等待神经网络评估
      // 初始化策略和价值为0，等待神经网络填充
      pi = decltype(pi){};
      value = 0.0f;

      // 非结束节点设置为UNEVALUATED，等待神经网络评估
      state.store(NodeState::UNEVALUATED, std::memory_order_seq_cst);
    }
  }

  // 析构函数 - 清理子节点
  ~ThreadSafeNode() {
    for (auto &child_ptr : children) {
      // shared_ptr自动管理，无需手动delete
    }
  }

  //========================================================================================
  // 线程安全的子节点选择 - MCTS选择阶段的核心函数
  // 功能：从当前节点的所有合法子节点中选择最佳节点进行扩展
  // 算法：使用UCB算法平衡探索和利用，支持已扩展和未扩展节点的选择
  // 线程安全：通过原子操作和状态机确保多线程环境下的安全性
  //========================================================================================
  std::pair<int, std::shared_ptr<ThreadSafeNode>> select_child(MultiThreadHelpers::MutexPool &mutex_pool) const {
    auto legal_moves_vec = Utils::legal_moves(board_state);
    std::shared_ptr<ThreadSafeNode> best_child = nullptr;
    int best_action_idx = -1;
    float max_score = -std::numeric_limits<float>::infinity();
    int min_virtual_loss = std::numeric_limits<int>::max();

    for (int i = 0; i < Config::BOARD_SQUARES; ++i) {
      if (!legal_moves_vec[i])
        continue;

      auto child = children[i].load(std::memory_order_acquire);
      int vloss = child ? child->virtual_loss.getVirtualLossCount() : 0;
      float score;
      if (child) {
        // 检查节点状态，跳过正在评估的节点
        if (child->isEvaluating()) {
          continue; // 跳过正在评估的节点
        }
        // 修正：虚拟损失已经在 child_score 中考虑了，这里不需要额外减去
        score = child_score(*child);
      } else {
        // 如果节点未评估，使用pi[i]（现在应该已经被神经网络填充）
        // 如果pi[i]仍然为0，说明节点还没有被评估，使用均匀分布作为后备
        float prior = (pi[i] > 0.0f) ? pi[i] : 1.0f;
        float u = Config::C_PUCT * prior * std::sqrt(std::max(1, get_visit_count_with_virtual_loss()));
        score = u;
      }
      if (score > max_score || (score == max_score && vloss < min_virtual_loss)) {
        max_score = score;
        best_child = child;
        best_action_idx = i;
        min_virtual_loss = vloss;
      }
    }
    return {best_action_idx, best_child};
  }

  //========================================================================================
  // 线程安全的子节点创建 - MCTS扩展阶段的核心函数
  // 功能：为指定的动作创建新的子节点，采用KataGo风格的原子操作
  // 线程安全：使用CAS操作确保只有一个线程能成功创建子节点
  // 节点复用：通过NodeTable实现节点复用，避免重复创建相同状态的节点
  //========================================================================================

  std::shared_ptr<ThreadSafeNode> create_child(int action_idx, Network &net, NodeTable &node_table,
                                               MultiThreadHelpers::MutexPool &mutex_pool) {
    if (action_idx < 0 || action_idx >= static_cast<int>(children.size())) {
      return nullptr;
    }

    // 使用CAS操作，采用KataGo风格的原子操作
    std::shared_ptr<ThreadSafeNode> expected = nullptr;
    auto child = children[action_idx].load(std::memory_order_acquire);
    if (child) {
      return child;
    }

    Board next_board = board_state;
    if (action_idx != -1) {
      auto [r, c] = Utils::index_to_coordinate(action_idx);
      next_board[r][c] = opponent_color;
    }

    int child_move_number = this->move_number + 1;

    auto node_factory = [&]() {
      return std::make_shared<ThreadSafeNode>(this, pi[action_idx], opponent_color, next_board, action_idx, net,
                                              child_move_number);
    };
    auto new_child = node_table.get_or_create(next_board, opponent_color, child_move_number, node_factory);

    // 使用CAS操作原子地设置子节点
    if (children[action_idx].compare_exchange_strong(expected, new_child, std::memory_order_release,
                                                     std::memory_order_acquire)) {
      return new_child;
    } else {
      // 另一个线程已经设置了子节点，返回已存在的节点
      return children[action_idx].load(std::memory_order_acquire);
    }
  }

  //========================================================================================
  // 包装方法：创建子节点并立即评估 - MCTS扩展和评估阶段的组合函数
  // 功能：创建子节点后立即进行神经网络评估，实现扩展和评估的原子操作
  // 设计模式：使用回调函数模式，将评估逻辑与节点创建逻辑解耦
  // 状态管理：通过状态机确保只有未评估的节点才会被评估
  //========================================================================================

  std::shared_ptr<ThreadSafeNode> create_child_and_evaluate(int action_idx, Network &net, NodeTable &node_table,
                                                            MultiThreadHelpers::MutexPool &mutex_pool,
                                                            auto evaluation_callback) {

    // 先创建子节点
    auto new_child = create_child(action_idx, net, node_table, mutex_pool);
    if (!new_child) {
      return nullptr;
    }

    if (new_child->tryAcquireEvaluation()) {
      evaluation_callback(new_child);
    }

    return new_child;
  }

  // 线程安全的回传
  void backpropagate();

  // 计算子节点分数
  //========================================================================================
  // 计算子节点UCB分数 - MCTS选择算法的核心计算函数（包含虚拟损失）
  // 功能：计算子节点的UCB分数，包含利用项(Q值)和探索项(U值)
  // 公式：UCB = Q + U，其中Q是利用项，U是探索项
  // 虚拟损失：包含虚拟损失调整，用于多线程环境下的负载均衡
  //========================================================================================
  float child_score(const ThreadSafeNode &child) const {
    int child_visits = child.get_visit_count_with_virtual_loss();
    // 修正: 分母加max(1, x)保护
    float q = (child_visits == 0) ? 0.0f : -child.get_average_value_with_virtual_loss();
    float u = Config::C_PUCT * child.prior_p * std::sqrt(std::max(1, get_visit_count_with_virtual_loss())) /
              (1.0f + std::max(1, child_visits));
    return q + u;
  }

  // 获取访问次数（包含虚拟损失）
  int get_visit_count_with_virtual_loss() const {
    return virtual_loss.getTotalVisitsWithVirtualLoss(stats.getVisitCount());
  }

  // 获取平均值（包含虚拟损失）
  double get_average_value_with_virtual_loss() const {
    int total_visits = get_visit_count_with_virtual_loss();
    if (total_visits == 0)
      return 0.0;
    // 修正：确保类型匹配，使用 double 类型计算
    double real_value_sum = static_cast<double>(stats.getVisitCount()) * stats.getAverageValue();
    double total_value = virtual_loss.getTotalValueWithVirtualLoss(real_value_sum);
    return total_value / total_visits;
  }

  // 线程安全的统计更新
  void add_visit(double value, double weight = 1.0) { stats.addVisit(value, weight); }

  // 状态机操作方法
  // 尝试获取评估权限
  bool tryAcquireEvaluation() {
    NodeState expected = NodeState::UNEVALUATED;
    return state.compare_exchange_strong(expected, NodeState::EVALUATING, std::memory_order_seq_cst);
  }

  // 设置推理结果并完成评估
  void setInferenceResult(const Network::PolicyOut_T &policy, const Network::ValueOut_T &inference_value) {
    pi = policy;
    value = inference_value[0] - inference_value[1] + inference_value[2] / 2;
    state.store(NodeState::EXPANDED, std::memory_order_seq_cst);
  }

  // 检查是否正在评估
  bool isEvaluating() const { return state.load(std::memory_order_acquire) == NodeState::EVALUATING; }

  // 检查是否已评估完成
  bool isExpanded() const { return state.load(std::memory_order_acquire) == NodeState::EXPANDED; }

  // 获取当前状态（用于调试）
  NodeState getCurrentState() const { return state.load(std::memory_order_acquire); }

  // 判断游戏是否结束的方法（从原始Node复制）
  std::pair<bool, float> ended() const;
};

class MCTSAgent {
private:
  std::shared_ptr<ThreadSafeNode> root;
  Network net;
  NodeTable node_table;
  MutexPool mutex_pool;
  std::atomic<int> evaluation_failures{0};    // 添加失败计数
  std::atomic<int> successful_simulations{0}; // 添加成功计数
  MultiThreadHelpers::ThreadPool thread_pool;
  MultiThreadHelpers::MultiThreadConfig config;
  MultiThreadHelpers::SearchStats search_stats;
  NodeTable *node_table_ = nullptr;
  std::atomic<int> total_simulations_{0}; // 全局模拟计数器
  std::atomic<bool> should_stop_{false};  // 全局终止信号
public:
  std::shared_ptr<ThreadSafeNode> get_root() const { return root; }
  MultiThreadHelpers::MutexPool &get_mutex_pool() { return mutex_pool; }
  MCTSAgent(
      const Board &initial_board, Utils::STONE_COLOR player_color, Network &network, NodeTable &node_table,
      const MultiThreadHelpers::MultiThreadConfig &mt_config = MultiThreadHelpers::MultiThreadConfig::getDefault())
      : net(network), thread_pool(mt_config.num_search_threads), mutex_pool(mt_config.mutex_pool_size),
        config(mt_config),
        root(std::make_shared<ThreadSafeNode>(nullptr, 1.0f, player_color, initial_board, -1, net, 0)),
        node_table_(&node_table) {
    // 确保根节点也被神经网络评估
    if (!root->isExpanded()) {
      evaluateNodeWithNeuralNetwork(root);
    }
  }
  void run_mcts_single();
  int run_mcts_parallel_with_stop(int max_simulations, double max_seconds);
  void apply_move(int move_idx, NodeTable &node_table);
  int next_move_idx() const;
  const MultiThreadHelpers::SearchStats &get_search_stats() const { return search_stats; }
  Utils::STONE_COLOR last_move_color() const { return root->current_color; }
  Utils::STONE_COLOR next_move_color() const { return root->opponent_color; }
  const Board &last_move_board() const { return root->board_state; }

  // 添加获取统计信息的方法
  int getEvaluationFailures() const { return evaluation_failures.load(); }
  int getSuccessfulSimulations() const { return successful_simulations.load(); }

  // 统一的神经网络评估方法 - MCTS搜索流程通过此方法进行评估
  void evaluateNodeWithNeuralNetwork(std::shared_ptr<ThreadSafeNode> node);
};

} // namespace MultiThreadMCTS

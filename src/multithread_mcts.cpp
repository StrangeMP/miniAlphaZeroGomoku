#include "multithread_mcts.hpp"
#include "zobrist.hpp"

using namespace MultiThreadMCTS;
//========================================================================================
// ThreadSafeNode 实现
//========================================================================================

//========================================================================================
// 线程安全的回传 - MCTS回传阶段的核心函数
// 功能：将叶子节点的价值回传到根节点，更新路径上所有节点的统计信息
// 线程安全：使用原子操作更新访问次数和价值总和
// 虚拟损失：在回传过程中移除虚拟损失，恢复节点的真实统计信息
//========================================================================================
void ThreadSafeNode::backpropagate() {
  ThreadSafeNode *current = this;
  float backup_value = value;

  while (current != nullptr) {
    // 只有非根节点才需要移除虚拟损失
    // 根节点（parent == nullptr）不应该有虚拟损失
    if (current->parent != nullptr) {
      current->virtual_loss.removeVirtualLoss();
    }

    // 更新统计信息
    current->add_visit(backup_value);

    // 翻转值用于对手
    backup_value = -backup_value;
    current = current->parent;
  }
}

std::pair<bool, float> ThreadSafeNode::ended() const {
  // 检查连续pass的情况：如果当前节点和父节点都是pass move，则游戏结束，结果为平局
  if (is_pass_move && parent && parent->is_pass_move) {
    return {true, 0.0f}; // 平局
  }

  // 检查是否有五子连线（从原始实现复制）
  const auto &board = board_state;

  // 检查水平、垂直和对角线
  for (int row = 0; row < Config::BOARD_SIZE; ++row) {
    for (int col = 0; col < Config::BOARD_SIZE; ++col) {
      if (board[row][col] == Utils::EMPTY)
        continue;

      Utils::STONE_COLOR color = board[row][col];

      // 检查水平
      if (col <= Config::BOARD_SIZE - 5) {
        bool win = true;
        for (int k = 0; k < 5; ++k) {
          if (board[row][col + k] != color) {
            win = false;
            break;
          }
        }
        if (win) {
          return {true, (color == current_color) ? 1.0f : -1.0f};
        }
      }

      // 检查垂直
      if (row <= Config::BOARD_SIZE - 5) {
        bool win = true;
        for (int k = 0; k < 5; ++k) {
          if (board[row + k][col] != color) {
            win = false;
            break;
          }
        }
        if (win) {
          return {true, (color == current_color) ? 1.0f : -1.0f};
        }
      }

      // 检查对角线（左上到右下）
      if (row <= Config::BOARD_SIZE - 5 && col <= Config::BOARD_SIZE - 5) {
        bool win = true;
        for (int k = 0; k < 5; ++k) {
          if (board[row + k][col + k] != color) {
            win = false;
            break;
          }
        }
        if (win) {
          return {true, (color == current_color) ? 1.0f : -1.0f};
        }
      }

      // 检查对角线（右上到左下）
      if (row <= Config::BOARD_SIZE - 5 && col >= 4) {
        bool win = true;
        for (int k = 0; k < 5; ++k) {
          if (board[row + k][col - k] != color) {
            win = false;
            break;
          }
        }
        if (win) {
          return {true, (color == current_color) ? 1.0f : -1.0f};
        }
      }
    }
  }

  return {false, 0.0f};
}

//========================================================================================
// NodeTable 实现
//========================================================================================

//========================================================================================
// 获取或创建节点 - 节点复用的核心方法
// 功能：根据棋盘状态获取已存在的节点，或创建新节点
// 线程安全：使用分片锁确保多线程环境下的安全性
// 节点复用：避免重复创建相同状态的节点，节省内存
//========================================================================================
inline NodeTable::NodePtr NodeTable::get_or_create(const Board &board, Utils::STONE_COLOR player, int move_number,
                                                   std::function<NodePtr()> node_factory) {
  // 使用基础hash，冲突概率极低
  Hash128 hash = Zobrist::hash(board, player, move_number);

  size_t shard_idx = get_shard_idx(hash);
  {
    std::lock_guard<std::mutex> guard(shard_mutexes_[shard_idx]);
    auto &shard = shards_[shard_idx];
    auto it = shard.find(hash);
    if (it != shard.end()) {
      return it->second;
    }
    auto node = node_factory();
    shard[hash] = node;
    return node;
  }
}

//========================================================================================
// 打印统计信息 - 用于调试和性能分析
// 功能：打印节点表的统计信息，包括节点数量和分片分布
// 线程安全：使用分片锁确保多线程环境下的安全性
//========================================================================================
inline void NodeTable::print_stats() const {
  size_t total = 0;
  for (size_t shard_idx = 0; shard_idx < num_shards_; ++shard_idx) {
    std::lock_guard<std::mutex> lock(shard_mutexes_[shard_idx]);
    const auto &shard = shards_[shard_idx];
    total += shard.size();
  }
}

void MCTSAgent::run_mcts_single() {
  std::vector<std::shared_ptr<ThreadSafeNode>> path;
  try {
    auto node = root;
    path.push_back(node);
    int step = 0;
    while (!node->is_end_node) {
      auto [best_action_idx, next_node] = node->select_child(mutex_pool);
      if (best_action_idx == -1) {
        node->is_end_node = true;
        node->value = 0.0f;
        break;
      }
      if (next_node) {
        next_node->virtual_loss.addVirtualLoss();
        node = next_node;
        path.push_back(node);
      } else {
        // 使用新的包装方法：创建子节点并立即评估
        auto new_child = node->create_child_and_evaluate(best_action_idx, net, *node_table_, mutex_pool,
                                                         [this](std::shared_ptr<ThreadSafeNode> node_to_evaluate) {
                                                           return this->evaluateNodeWithNeuralNetwork(node_to_evaluate);
                                                         });

        if (!new_child) {
          break;
        }

        // 为新创建的节点添加虚拟损失
        new_child->virtual_loss.addVirtualLoss();
        node = new_child;
        path.push_back(node);

        // 检查评估是否成功
        if (!new_child->isExpanded()) {
          // 评估失败，按照KataGo的策略：放弃这次playout
          evaluation_failures.fetch_add(1, std::memory_order_relaxed); // 增加失败计数

          // 移除路径上所有节点的虚拟损失（只移除真正有虚拟损失的节点）
          if (path.size() > 1) {
            for (auto it = path.rbegin() + 1; it != path.rend(); ++it) {
              int current_vl = (*it)->virtual_loss.getVirtualLossCount();
              if (current_vl > 0) {
                (*it)->virtual_loss.removeVirtualLoss();
              }
            }
          }
          return; // 直接返回，不执行后续的backpropagate,整个playout被放弃，不进行回传
        }

        break;
      }
      ++step;
    }
    node->backpropagate();
    search_stats.incrementSimulations();
    successful_simulations.fetch_add(1, std::memory_order_relaxed); // 增加成功计数
  } catch (const std::exception &e) {
    // 注释掉select_child、create_child、NodeTable-get_or_create-reuse/new、异常等日志
    // MCTS_LOG << "[MCTS线程异常] std::exception: " << e.what() << "\n";
    // MCTS_LOG << "[MCTS线程异常] 线程id: " << std::this_thread::get_id() << std::endl;
    throw;
  } catch (...) {
    // 注释掉select_child、create_child、NodeTable-get_or_create-reuse/new、异常等日志
    // MCTS_LOG << "[MCTS线程未知异常] 线程id: " << std::this_thread::get_id() << std::endl;
    throw;
  }
  // 移除这个额外的虚拟损失清理循环，因为backpropagate()内部已经处理了虚拟损失
  // for (auto it = path.rbegin(); it != path.rend(); ++it) {
  //     (*it)->virtual_loss.removeVirtualLoss();
  // }
}
//========================================================================================
// 并行MCTS搜索 - 多线程并行执行MCTS搜索
// 功能：使用线程池并行执行多次MCTS搜索，支持最大模拟次数和时间限制
// 参数：max_simulations - 最大模拟次数，max_seconds - 最大搜索时间
// 返回值：实际执行的模拟次数
//========================================================================================
int MCTSAgent::run_mcts_parallel_with_stop(int max_simulations, double max_seconds) {
  total_simulations_ = 0;
  should_stop_ = false;
  std::vector<std::thread> threads;
  int num_threads = config.num_search_threads;
  std::vector<int> thread_counts(num_threads, 0);
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([this, max_simulations, &thread_counts, i]() {
      while (true) {
        if (should_stop_)
          break;
        int cur = total_simulations_.fetch_add(1, std::memory_order_relaxed);
        if (cur >= max_simulations) {
          should_stop_ = true;
          break;
        }
        run_mcts_single();
        thread_counts[i]++;
      }
    });
  }
  // 定时线程：到时后置should_stop_
  std::thread timer([this, max_seconds]() {
    auto start = std::chrono::steady_clock::now();
    while (!should_stop_) {
      auto now = std::chrono::steady_clock::now();
      double elapsed = std::chrono::duration<double>(now - start).count();
      if (elapsed >= max_seconds)
        break;
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    should_stop_ = true;
  });
  for (auto &t : threads)
    if (t.joinable())
      t.join();
  if (timer.joinable())
    timer.join();
  int total = 0;
  for (int c : thread_counts)
    total += c;
  return total;
}

//========================================================================================
// 统一的神经网络评估方法 - MCTS搜索流程的核心评估函数
// 功能：使用神经网络评估节点，获取策略和价值
// 状态管理：通过状态机确保评估的线程安全性
// 错误处理：评估失败时保持节点状态，允许重试
//========================================================================================
void MCTSAgent::evaluateNodeWithNeuralNetwork(std::shared_ptr<ThreadSafeNode> node) {
  if (node->state.load() == ThreadSafeNode::NodeState::UNEVALUATED) {
    Network::evaluate(node.get());
  }
}

//========================================================================================
// 应用移动 - 更新根节点和节点表
// 功能：根据移动索引更新根节点，并更新节点表
// 线程安全：使用原子操作确保多线程环境下的安全性
//========================================================================================
void MCTSAgent::apply_move(int move_idx, NodeTable &node_table) {
  // 不再需要锁，因为create_child现在使用原子操作
  auto child = root->children[move_idx].load(std::memory_order_acquire);
  if (!child) {
    // 使用新的包装方法：创建子节点并立即评估
    child = root->create_child_and_evaluate(move_idx, net, node_table, mutex_pool,
                                            [this](std::shared_ptr<ThreadSafeNode> node_to_evaluate) {
                                              return this->evaluateNodeWithNeuralNetwork(node_to_evaluate);
                                            });
  }
  root = child;
  root->parent = nullptr;
}

//========================================================================================
// 获取最佳移动索引 - 选择最受欢迎的移动
// 功能：从根节点的所有子节点中选择访问次数最多的移动
// 线程安全：使用原子操作确保多线程环境下的安全性
//========================================================================================
int MCTSAgent::next_move_idx() const {
  int max_visits = -1;
  int best_move_idx = -1;
  // 不再需要锁，因为children现在使用原子操作
  // 检查所有可能的移动，包括pass move (索引 BOARD_SQUARES)
  for (int i = 0; i <= Config::BOARD_SQUARES; ++i) {
    auto child = root->children[i].load(std::memory_order_acquire);
    if (child) {
      int visits = child->stats.getVisitCount();
      if (visits > max_visits) {
        max_visits = visits;
        best_move_idx = i;
      }
    }
  }
  return best_move_idx;
}

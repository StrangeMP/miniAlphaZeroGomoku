#pragma once

#include "mcts.hpp"
#include "multithreadhelpers.hpp"
#include "zobrist.hpp"
#include "hash128.hpp"
#include <atomic>
#include <mutex>
#include <memory>
#include <unordered_map>
#include <functional>
#include <iostream>


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
        : num_shards_(num_shards),
          shards_(num_shards),
          shard_mutexes_(num_shards) {}
    // 查找或插入节点，线程安全
    NodePtr get_or_create(const Board& board, AlphaGomoku::STONE_COLOR player, int move_number, std::function<NodePtr()> node_factory);
    // 可选：定期清理无用节点
    void garbage_collect();
    void print_stats() const;
private:
    size_t num_shards_;
    std::vector<std::unordered_map<Hash128, NodePtr>> shards_;
    mutable std::vector<std::mutex> shard_mutexes_;
    
    // KataGo风格的随机数生成器
    mutable std::mt19937_64 rng_{std::random_device{}()};
    mutable std::mutex rng_mutex_;
    
    // hash分片辅助
    size_t get_shard_idx(const Hash128& hash) const { return (hash.hash0 ^ hash.hash1) % num_shards_; }
    
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
    static const std::vector<std::unordered_map<Hash128, NodeTable::NodePtr>>& get_shards(const NodeTable& table) {
        return table.shards_;
    }
    // 获取所有分片互斥锁
    static std::vector<std::mutex>& get_shard_mutexes(NodeTable& table) {
        return table.shard_mutexes_;
    }
};

// 多线程安全的Node结构
struct ThreadSafeNode {
    // 状态机定义
    enum class NodeState {
        UNEVALUATED,    // 节点未评估
        EVALUATING,     // 正在评估中
        EXPANDED        // 已评估完成
    };
    
    size_t debug_id = 0; // 用于调试唯一标识
    ThreadSafeNode *parent;
    AlphaGomoku::STONE_COLOR current_color;
    AlphaGomoku::STONE_COLOR opponent_color;
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
    
    int prior_action_idx;
    Vec<float, Config::BOARD_SQUARES> pi;
    float value = 0.0f;
    int move_number = 0; // 新增：当前节点的步数
    
    // 构造函数
    template<typename NetworkType>
    ThreadSafeNode(ThreadSafeNode *parent_, float prior, AlphaGomoku::STONE_COLOR turn, 
                   const Board &current_board, int action_idx, NetworkType &net, int move_number_ = 0)
        : parent(parent_), current_color(turn),
          opponent_color(turn == Config::BLACK_STONE ? Config::WHITE_STONE : Config::BLACK_STONE), 
          prior_p(prior), board_state(current_board), prior_action_idx(action_idx), move_number(move_number_)
    {
        static size_t debug_id_counter = 1;
        debug_id = debug_id_counter++;
        
        // 初始化所有子节点指针为nullptr（原子shared_ptr需要特殊初始化）
        for (auto& child : children) {
            child.store(nullptr, std::memory_order_relaxed);
        }
        
        std::optional<std::pair<WEIGHT_T, WEIGHT_T>> last_move = {};
        bool game_ended = false;
        float game_result = 0.0f;
        
        if (prior_action_idx != -1) {
            auto [r, c] = Utils::index_to_coordinate(prior_action_idx);
            board_state[r][c] = opponent_color;
            last_move = {r, c};
            std::tie(game_ended, game_result) = ended();
        }
        
        if (game_ended) {
            is_end_node = true;
            value = game_result;
            // 游戏结束的节点直接设置为EXPANDED，不需要神经网络评估
            state.store(NodeState::EXPANDED, std::memory_order_seq_cst);
        } else {
            // 去掉启发式初始化，等待神经网络评估
            // 初始化策略和价值为0，等待神经网络填充
            for (auto& v : pi) v = 0.0f;
            value = 0.0f;
            
            // 非结束节点设置为UNEVALUATED，等待神经网络评估
            state.store(NodeState::UNEVALUATED, std::memory_order_seq_cst);
        }
    }
    
    // 析构函数 - 清理子节点
    ~ThreadSafeNode() {
        for (auto& child_ptr : children) {
            // shared_ptr自动管理，无需手动delete
        }
    }
    
    // 线程安全的子节点选择
    std::pair<int, std::shared_ptr<ThreadSafeNode>> select_child(MultiThreadHelpers::MutexPool& mutex_pool) const {
        auto legal_moves_vec = Utils::legal_moves(board_state);
        std::shared_ptr<ThreadSafeNode> best_child = nullptr;
        int best_action_idx = -1;
        float max_score = -std::numeric_limits<float>::infinity();
        int min_virtual_loss = std::numeric_limits<int>::max();
        
        for (int i = 0; i < Config::BOARD_SQUARES; ++i) {
            if (!legal_moves_vec[i]) continue;
            
            auto child = children[i].load(std::memory_order_acquire);
            int vloss = child ? child->virtual_loss.getVirtualLossCount() : 0;
            float score;
            if (child) {
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
    
    // 线程安全的子节点创建
    template<typename NetworkType>
    std::shared_ptr<ThreadSafeNode> create_child(int action_idx, NetworkType& net, NodeTable& node_table, MultiThreadHelpers::MutexPool& mutex_pool) {
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
            return std::make_shared<ThreadSafeNode>(this, pi[action_idx], opponent_color, next_board, action_idx, net, child_move_number);
        };
        auto new_child = node_table.get_or_create(next_board, opponent_color, child_move_number, node_factory);
        
        // 使用CAS操作原子地设置子节点
        if (children[action_idx].compare_exchange_strong(expected, new_child, std::memory_order_release, std::memory_order_acquire)) {
            return new_child;
        } else {
            // 另一个线程已经设置了子节点，返回已存在的节点
            return children[action_idx].load(std::memory_order_acquire);
        }
    }
    
    // 包装方法：创建子节点并立即评估（通过回调函数）
    template<typename NetworkType>
    std::shared_ptr<ThreadSafeNode> create_child_and_evaluate(
        int action_idx, 
        NetworkType& net, 
        NodeTable& node_table, 
        MultiThreadHelpers::MutexPool& mutex_pool,
        std::function<bool(std::shared_ptr<ThreadSafeNode>)> evaluation_callback) {
        
        // 先创建子节点
        auto new_child = create_child(action_idx, net, node_table, mutex_pool);
        if (!new_child) {
            return nullptr;
        }
        
        // 检查节点状态，只有未评估的节点才需要评估
        NodeState current_state = new_child->getCurrentState();
        
        if (current_state == NodeState::UNEVALUATED) {
            // 只有未评估的节点才进行评估
            bool evaluation_success = evaluation_callback(new_child);
            // 评估失败时，我们可以选择返回nullptr或者返回未评估的节点
            // 这里我们返回节点，让调用方决定如何处理
        }
        
        return new_child;
    }
    
    // 线程安全的回传
    void backpropagate();
    
    // 计算子节点分数（包含虚拟损失）
    float child_score(const ThreadSafeNode& child) const {
        int child_visits = child.get_visit_count_with_virtual_loss();
        // 修正: 分母加max(1, x)保护
        float q = (child_visits == 0) ? 0.0f : -child.get_average_value_with_virtual_loss();
        float u = Config::C_PUCT * child.prior_p * std::sqrt(std::max(1, get_visit_count_with_virtual_loss())) / (1.0f + std::max(1, child_visits));
        return q + u;
    }
    
    // 获取访问次数（包含虚拟损失）
    int get_visit_count_with_virtual_loss() const {
        return virtual_loss.getTotalVisitsWithVirtualLoss(stats.getVisitCount());
    }
    
    // 获取平均值（包含虚拟损失）
    double get_average_value_with_virtual_loss() const {
        int total_visits = get_visit_count_with_virtual_loss();
        if (total_visits == 0) return 0.0;
        // 修正：确保类型匹配，使用 double 类型计算
        double real_value_sum = static_cast<double>(stats.getVisitCount()) * stats.getAverageValue();
        double total_value = virtual_loss.getTotalValueWithVirtualLoss(real_value_sum);
        return total_value / total_visits;
    }
    
    // 线程安全的统计更新
    void add_visit(double value, double weight = 1.0) {
        stats.addVisit(value, weight);
    }
    
    // 状态机操作方法
    // 尝试获取评估权限
    bool tryAcquireEvaluation() {
        NodeState expected = NodeState::UNEVALUATED;
        return state.compare_exchange_strong(
            expected, 
            NodeState::EVALUATING, 
            std::memory_order_seq_cst
        );
    }
    
    // 完成评估
    void finishEvaluation() {
        state.store(NodeState::EXPANDED, std::memory_order_seq_cst);
    }
    
    // 检查是否正在评估
    bool isEvaluating() const {
        return state.load(std::memory_order_acquire) == NodeState::EVALUATING;
    }
    
    // 检查是否已评估完成
    bool isExpanded() const {
        return state.load(std::memory_order_acquire) == NodeState::EXPANDED;
    }
    
    // 获取当前状态（用于调试）
    NodeState getCurrentState() const {
        return state.load(std::memory_order_acquire);
    }
    
    // 判断游戏是否结束的方法（从原始Node复制）
    std::pair<bool, float> ended() const;
};

// 模板化多线程MCTS代理前向声明

template<typename NetworkType> class ThreadSafeMCTS_Agent_Tmpl;

// 模板化多线程MCTS代理

template<typename NetworkType>
class ThreadSafeMCTS_Agent_Tmpl {
private:
    std::shared_ptr<ThreadSafeNode> root;
    NetworkType& net;
    NodeTable node_table;
    MutexPool mutex_pool;
    std::atomic<int> evaluation_failures{0}; // 添加失败计数
    std::atomic<int> successful_simulations{0}; // 添加成功计数
    MultiThreadHelpers::ThreadPool thread_pool;
    MultiThreadHelpers::MultiThreadConfig config;
    MultiThreadHelpers::SearchStats search_stats;
    NodeTable* node_table_ = nullptr;
    std::atomic<int> total_simulations_{0}; // 全局模拟计数器
    std::atomic<bool> should_stop_{false};  // 全局终止信号
public:
    std::shared_ptr<ThreadSafeNode> get_root() const { return root; }
    MultiThreadHelpers::MutexPool& get_mutex_pool() { return mutex_pool; }
    ThreadSafeMCTS_Agent_Tmpl(const Board &initial_board, AlphaGomoku::STONE_COLOR player_color, 
                        NetworkType &network, 
                        NodeTable& node_table,
                        const MultiThreadHelpers::MultiThreadConfig& mt_config = MultiThreadHelpers::MultiThreadConfig::getDefault())
        : net(network), 
          thread_pool(mt_config.num_search_threads),
          mutex_pool(mt_config.mutex_pool_size),
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
    void apply_move(int move_idx, NodeTable& node_table);
    int next_move_idx() const;
    const MultiThreadHelpers::SearchStats& get_search_stats() const { return search_stats; }
    int last_move_idx() const { return root->prior_action_idx; }
    AlphaGomoku::STONE_COLOR last_move_color() const { return root->current_color; }
    AlphaGomoku::STONE_COLOR next_move_color() const { return root->opponent_color; }
    const Board &last_move_board() const { return root->board_state; }
    
    // 添加获取统计信息的方法
    int getEvaluationFailures() const { return evaluation_failures.load(); }
    int getSuccessfulSimulations() const { return successful_simulations.load(); }

    // 统一的神经网络评估方法 - MCTS搜索流程通过此方法进行评估
    bool evaluateNodeWithNeuralNetwork(std::shared_ptr<ThreadSafeNode> node);
};

// ================== 模板实现迁移 ==================

template<typename NetworkType>
void ThreadSafeMCTS_Agent_Tmpl<NetworkType>::run_mcts_single() {
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
                auto new_child = node->create_child_and_evaluate(
                    best_action_idx, 
                    net, 
                    *node_table_, 
                    mutex_pool,
                    [this](std::shared_ptr<ThreadSafeNode> node_to_evaluate) {
                        return this->evaluateNodeWithNeuralNetwork(node_to_evaluate);
                    }
                );
                
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
    } catch (const std::exception& e) {
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

template<typename NetworkType>
int ThreadSafeMCTS_Agent_Tmpl<NetworkType>::run_mcts_parallel_with_stop(int max_simulations, double max_seconds) {
    total_simulations_ = 0;
    should_stop_ = false;
    std::vector<std::thread> threads;
    int num_threads = config.num_search_threads;
    std::vector<int> thread_counts(num_threads, 0);
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([this, max_simulations, &thread_counts, i]() {
            while (true) {
                if (should_stop_) break;
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
    for (auto& t : threads) if (t.joinable()) t.join();
    if (timer.joinable()) timer.join();
    int total = 0;
    for (int c : thread_counts) total += c;
    return total;
}

template<typename NetworkType>
bool ThreadSafeMCTS_Agent_Tmpl<NetworkType>::evaluateNodeWithNeuralNetwork(std::shared_ptr<ThreadSafeNode> node) {
    // 1. 状态机检查：尝试获取评估权限
    if (!node->tryAcquireEvaluation()) {
        // 其他线程正在评估或已经评估完成
        return false;
    }
    
    try {
        // 2. 调用神经网络接口进行评估
        auto [policy, value] = net.evaluate(node->board_state, node->current_color);
        
        // 3. 更新节点信息
        node->pi = policy;
        node->value = value;
        
        // 4. 标记评估完成
        node->finishEvaluation();
        
        return true;
        
    } catch (const std::exception& e) {
        // 5. 如果评估失败，重置为UNEVALUATED状态，允许重试
        // 但为了避免无限重试，我们可以考虑添加重试次数限制
        node->state.store(ThreadSafeNode::NodeState::UNEVALUATED, std::memory_order_seq_cst);
        return false;
    }
}

template<typename NetworkType>
void ThreadSafeMCTS_Agent_Tmpl<NetworkType>::apply_move(int move_idx, NodeTable& node_table) {
    // 不再需要锁，因为create_child现在使用原子操作
    auto child = root->children[move_idx].load(std::memory_order_acquire);
    if (!child) {
        // 使用新的包装方法：创建子节点并立即评估
        child = root->create_child_and_evaluate(
            move_idx,
            net,
            node_table,
            mutex_pool,
            [this](std::shared_ptr<ThreadSafeNode> node_to_evaluate) {
                return this->evaluateNodeWithNeuralNetwork(node_to_evaluate);
            }
        );
    }
    root = child;
    root->parent = nullptr;
}

template<typename NetworkType>
int ThreadSafeMCTS_Agent_Tmpl<NetworkType>::next_move_idx() const {
    int max_visits = -1;
    int best_move_idx = -1;
    // 不再需要锁，因为children现在使用原子操作
    for (int i = 0; i < Config::BOARD_SQUARES; ++i) {
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
// ================== 模板实现迁移 END ==================

} // namespace MultiThreadMCTS

namespace MultiThreadMCTS {

    //========================================================================================
    // ThreadSafeNode 实现
    //========================================================================================
    
    namespace {
    std::atomic<size_t> global_node_id{1};
    }
    
    void ThreadSafeNode::backpropagate() {
        ThreadSafeNode* current = this;
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
        // 检查是否有五子连线（从原始实现复制）
        const auto& board = board_state;
        
        // 检查水平、垂直和对角线
        for (int row = 0; row < Config::BOARD_SIZE; ++row) {
            for (int col = 0; col < Config::BOARD_SIZE; ++col) {
                if (board[row][col] == Config::EMPTY_STONE) continue;
                
                AlphaGomoku::STONE_COLOR color = board[row][col];
                
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
    
    inline NodeTable::NodePtr NodeTable::get_or_create(const Board& board, AlphaGomoku::STONE_COLOR player, int move_number, std::function<NodePtr()> node_factory) {
        // 使用基础hash，冲突概率极低
        Hash128 hash = Zobrist::hash(board, player, move_number);
        
        size_t shard_idx = get_shard_idx(hash);
        {
            std::lock_guard<std::mutex> guard(shard_mutexes_[shard_idx]);
            auto& shard = shards_[shard_idx];
            auto it = shard.find(hash);
            if (it != shard.end()) {
                return it->second;
            }
            auto node = node_factory();
            shard[hash] = node;
            return node;
        }
    }
    
    inline void NodeTable::garbage_collect() {
        for (size_t shard_idx = 0; shard_idx < num_shards_; ++shard_idx) {
            std::lock_guard<std::mutex> lock(shard_mutexes_[shard_idx]);
            auto& shard = shards_[shard_idx];
            for (auto it = shard.begin(); it != shard.end(); ) {
                if (it->second.use_count() == 1) {
                    it = shard.erase(it);
                } else {
                    ++it;
                }
            }
        }
    }
    
    inline void NodeTable::print_stats() const {
        size_t total = 0;
        for (size_t shard_idx = 0; shard_idx < num_shards_; ++shard_idx) {
            std::lock_guard<std::mutex> lock(shard_mutexes_[shard_idx]);
            const auto& shard = shards_[shard_idx];
            total += shard.size();
        }
    }

} // namespace MultiThreadMCTS

#pragma once

#include "mcts.hpp"
#include "multithreadhelpers.hpp"
#include "zobrist.hpp"
#include <atomic>
#include <mutex>
#include <memory>
#include <unordered_map>
#include <functional>
#include <iostream>
#include <fstream>
extern std::ofstream mcts_debug_log;
#define MCTS_LOG mcts_debug_log
#define MCTS_LOG_FATAL std::cerr

using namespace MultiThreadHelpers;

namespace MultiThreadMCTS {

// 前向声明
struct ThreadSafeNode;

// 全局节点表（transposition table）
class NodeTable {
public:
    using NodePtr = std::shared_ptr<ThreadSafeNode>;
    // 查找或插入节点，线程安全
    NodePtr get_or_create(const Board& board, AlphaGomoku::STONE_COLOR player, std::function<NodePtr()> node_factory);
    // 可选：定期清理无用节点
    void garbage_collect();
    void print_stats() const;
private:
    std::unordered_map<size_t, NodePtr> table_;
    mutable std::mutex mutex_;
};

// 多线程安全的Node结构
struct ThreadSafeNode {
    size_t debug_id = 0; // 用于调试唯一标识
    ThreadSafeNode *parent;
    AlphaGomoku::STONE_COLOR current_color;
    AlphaGomoku::STONE_COLOR opponent_color;
    float prior_p;
    
    // 原子化的统计信息
    MultiThreadHelpers::AtomicNodeStats stats;
    MultiThreadHelpers::VirtualLossManager virtual_loss;
    
    bool is_end_node = false;
    Board board_state;
    
    // 使用shared_ptr数组管理子节点，并用mutex保护
    std::array<std::shared_ptr<ThreadSafeNode>, Config::BOARD_SQUARES> children;
    
    int prior_action_idx;
    Vec<float, Config::BOARD_SQUARES> pi;
    float value = 0.0f;
    
    // 构造函数
    template<typename NetworkType>
    ThreadSafeNode(ThreadSafeNode *parent_, float prior, AlphaGomoku::STONE_COLOR turn, 
                   const Board &current_board, int action_idx, NetworkType &net)
        : parent(parent_), current_color(turn),
          opponent_color(turn == Config::BLACK_STONE ? Config::WHITE_STONE : Config::BLACK_STONE), 
          prior_p(prior), board_state(current_board), prior_action_idx(action_idx)
    {
        static size_t debug_id_counter = 1;
        debug_id = debug_id_counter++;
        
        // 初始化所有子节点指针为nullptr
        for (auto& child : children) {
            child = nullptr;
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
        } else {
            // 启发式先验分布：中心点最高，对角线次之，其余均匀
            float sum = 0.0f;
            int center = Config::BOARD_SIZE / 2;
            for (int i = 0; i < Config::BOARD_SQUARES; ++i) {
                auto [r, c] = Utils::index_to_coordinate(i);
                if (r == center && c == center) {
                    pi[i] = 10.0f; // 中心点权重最高
                } else if (r == c || r + c == Config::BOARD_SIZE - 1) {
                    pi[i] = 3.0f; // 对角线权重次之
                } else {
                    pi[i] = 1.0f; // 其余均匀
                }
                sum += pi[i];
            }
            for (auto& v : pi) v /= sum; // 归一化
            value = 0.0f;
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
            std::mutex& lock = mutex_pool.getMutex(this);
            std::lock_guard<std::mutex> guard(lock);
            auto child = children[i];
            int vloss = child ? child->virtual_loss.getVirtualLossCount() : 0;
            float score;
            if (child) {
                score = child_score(*child) - vloss * 1e-3f;
            } else {
                // 修正: sqrt和分母都加max(1, x)保护
                float u = Config::C_PUCT * pi[i] * std::sqrt(std::max(1, get_visit_count_with_virtual_loss()));
                score = u;
            }
            if (score > max_score || (score == max_score && vloss < min_virtual_loss)) {
                max_score = score;
                best_child = child;
                best_action_idx = i;
                min_virtual_loss = vloss;
            }
        }
        MCTS_LOG << "[select_child] node debug_id=" << debug_id << " best_action_idx=" << best_action_idx << " best_child=" << (best_child ? best_child.get() : nullptr) << std::endl;
        return {best_action_idx, best_child};
    }
    
    // 线程安全的子节点创建
    template<typename NetworkType>
    std::shared_ptr<ThreadSafeNode> create_child(int action_idx, NetworkType& net, NodeTable& node_table, MultiThreadHelpers::MutexPool& mutex_pool) {
        if (action_idx < 0 || action_idx >= static_cast<int>(children.size())) {
            MCTS_LOG << "[create_child-FATAL] 非法action_idx: " << action_idx << " node: " << this << std::endl;
            return nullptr;
        }
        std::mutex& lock = mutex_pool.getMutex(this);
        std::lock_guard<std::mutex> guard(lock);
        auto child = children[action_idx];
        if (child) {
            MCTS_LOG << "[create_child] node debug_id=" << debug_id << " action_idx=" << action_idx << " 已存在 children[action_idx]=" << child.get() << std::endl;
            return child;
        }
        Board next_board = board_state;
        if (action_idx != -1) {
            auto [r, c] = Utils::index_to_coordinate(action_idx);
            next_board[r][c] = opponent_color;
        }
        auto node_factory = [&]() {
            return std::make_shared<ThreadSafeNode>(this, pi[action_idx], opponent_color, next_board, action_idx, net);
        };
        auto new_child = node_table.get_or_create(next_board, opponent_color, node_factory);
        if (children[action_idx] == nullptr) {
            children[action_idx] = new_child;
            MCTS_LOG << "[create_child] node debug_id=" << debug_id << " action_idx=" << action_idx << " 分配新 children[action_idx]=" << new_child.get() << std::endl;
            return new_child;
        } else {
            MCTS_LOG << "[create_child] node debug_id=" << debug_id << " action_idx=" << action_idx << " 竞争后 children[action_idx]=" << children[action_idx].get() << std::endl;
            return children[action_idx];
        }
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
        double total_value = virtual_loss.getTotalValueWithVirtualLoss(stats.getVisitCount() * stats.getAverageValue());
        return total_value / total_visits;
    }
    
    // 线程安全的统计更新
    void add_visit(double value, double weight = 1.0) {
        stats.addVisit(value, weight);
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
    NetworkType &net;
    std::shared_ptr<ThreadSafeNode> root;
    MultiThreadHelpers::ThreadPool thread_pool;
    MultiThreadHelpers::MutexPool mutex_pool;
    MultiThreadHelpers::MultiThreadConfig config;
    MultiThreadHelpers::SearchStats search_stats;
    NodeTable* node_table_ = nullptr;
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
          root(std::make_shared<ThreadSafeNode>(nullptr, 1.0f, player_color, initial_board, -1, net)),
          node_table_(&node_table) {}
    void run_mcts_single();
    void run_mcts_parallel(int num_simulations);
    void apply_move(int move_idx, NodeTable& node_table);
    int next_move_idx() const;
    const MultiThreadHelpers::SearchStats& get_search_stats() const { return search_stats; }
    int last_move_idx() const { return root->prior_action_idx; }
    AlphaGomoku::STONE_COLOR last_move_color() const { return root->current_color; }
    AlphaGomoku::STONE_COLOR next_move_color() const { return root->opponent_color; }
    const Board &last_move_board() const { return root->board_state; }
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
                auto new_child = node->create_child(best_action_idx, net, *node_table_, mutex_pool);
                if (!new_child) {
                    break;
                }
                new_child->virtual_loss.addVirtualLoss();
                node = new_child;
                path.push_back(node);
                break;
            }
            ++step;
        }
        node->backpropagate();
        search_stats.incrementSimulations();
    } catch (const std::exception& e) {
        MCTS_LOG << "[MCTS线程异常] std::exception: " << e.what() << "\n";
        MCTS_LOG << "[MCTS线程异常] 线程id: " << std::this_thread::get_id() << std::endl;
        throw;
    } catch (...) {
        MCTS_LOG << "[MCTS线程未知异常] 线程id: " << std::this_thread::get_id() << std::endl;
        throw;
    }
    for (auto it = path.rbegin(); it != path.rend(); ++it) {
        (*it)->virtual_loss.removeVirtualLoss();
        // 已彻底移除removeVirtualLoss/while缁撴潫/pi_filled日志
    }
}

template<typename NetworkType>
void ThreadSafeMCTS_Agent_Tmpl<NetworkType>::run_mcts_parallel(int num_simulations) {
    std::vector<std::future<void>> futures;
    int simulations_per_thread = num_simulations / config.num_search_threads;
    int remaining_simulations = num_simulations % config.num_search_threads;
    for (size_t i = 0; i < config.num_search_threads; ++i) {
        int thread_simulations = simulations_per_thread + (i < remaining_simulations ? 1 : 0);
        auto future = thread_pool.enqueue([this, thread_simulations]() {
            try {
                for (int sim = 0; sim < thread_simulations; ++sim) {
                    run_mcts_single();
                }
            } catch (const std::exception& e) {
                MCTS_LOG << "[MCTS并发线程异常] std::exception: " << e.what() << "\n";
                MCTS_LOG << "[MCTS并发线程异常] 线程id: " << std::this_thread::get_id() << std::endl;
                throw;
            } catch (...) {
                MCTS_LOG << "[MCTS并发线程未知异常] 线程id: " << std::this_thread::get_id() << std::endl;
                throw;
            }
        });
        futures.push_back(std::move(future));
    }
    for (auto& future : futures) {
        try {
            future.get();
        } catch (const std::exception& e) {
            MCTS_LOG << "[MCTS主线程异常] std::exception: " << e.what() << "\n";
        } catch (...) {
            MCTS_LOG << "[MCTS主线程未知异常]" << std::endl;
        }
    }
}

template<typename NetworkType>
void ThreadSafeMCTS_Agent_Tmpl<NetworkType>::apply_move(int move_idx, NodeTable& node_table) {
    std::mutex& lock = mutex_pool.getMutex(static_cast<const void*>(root.get()));
    std::lock_guard<std::mutex> guard(lock);
    auto child = root->children[move_idx];
    if (!child) {
        Board next_board = root->board_state;
        if (move_idx != -1) {
            auto [r, c] = Utils::index_to_coordinate(move_idx);
            next_board[r][c] = root->opponent_color;
        }
        auto node_factory = [&]() {
            return std::make_shared<ThreadSafeNode>(nullptr, 1.0f, root->opponent_color, next_board, move_idx, net);
        };
        child = node_table.get_or_create(next_board, root->opponent_color, node_factory);
        root->children[move_idx] = child;
        MCTS_LOG << "[apply_move] 新建子节点 move_idx=" << move_idx << " debug_id=" << child->debug_id << std::endl;
    } else {
        MCTS_LOG << "[apply_move] 复用子节点 move_idx=" << move_idx << " debug_id=" << child->debug_id << std::endl;
    }
    root = child;
    root->parent = nullptr;
    MCTS_LOG << "[apply_move] 切换根节点 root debug_id=" << root->debug_id << std::endl;
}

template<typename NetworkType>
int ThreadSafeMCTS_Agent_Tmpl<NetworkType>::next_move_idx() const {
    int max_visits = -1;
    int best_move_idx = -1;
    std::mutex& lock = mutex_pool.getMutex(static_cast<const void*>(root.get()));
    std::lock_guard<std::mutex> guard(lock);
    for (int i = 0; i < Config::BOARD_SQUARES; ++i) {
        auto child = root->children[i];
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
            // 移除虚拟损失
            current->virtual_loss.removeVirtualLoss();
            
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
    
    NodeTable::NodePtr NodeTable::get_or_create(const Board& board, AlphaGomoku::STONE_COLOR player, std::function<NodePtr()> node_factory) {
        size_t hash = Zobrist::hash(board, player);
        {
            std::lock_guard<std::mutex> guard(mutex_);
            auto it = table_.find(hash);
            if (it != table_.end()) {
                MCTS_LOG << "[NodeTable-get_or_create-复用] hash=" << hash << " node=" << it->second.get() << " debug_id=" << it->second->debug_id << " board[0][0]=" << board[0][0] << std::endl;
                return it->second;
            }
            auto node = node_factory();
            table_[hash] = node;
            MCTS_LOG << "[NodeTable-get_or_create-分配新节点] hash=" << hash << " node=" << node.get() << " debug_id=" << node->debug_id << " board[0][0]=" << board[0][0] << std::endl;
            return node;
        }
    }
    
    void NodeTable::garbage_collect() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto it = table_.begin(); it != table_.end(); ) {
            if (it->second.use_count() == 1) {
                it = table_.erase(it);
            } else {
                ++it;
            }
        }
    }
    
    void NodeTable::print_stats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::cout << "[DEBUG][NodeTable] 当前节点数: " << table_.size() << std::endl;
        for (const auto& kv : table_) {
            std::cout << "  hash=" << kv.first << " debug_id=" << (kv.second ? kv.second->debug_id : 0)
                      << " use_count=" << (kv.second ? kv.second.use_count() : 0) << std::endl;
        }
    }
    
    } // namespace MultiThreadMCTS

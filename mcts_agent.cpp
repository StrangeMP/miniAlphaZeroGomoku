#include <iostream>
#include <vector>
#include <array>
#include <memory> // 用于 std::shared_ptr, std::make_shared, std::enable_shared_from_this
#include <cmath>
#include <algorithm>
#include <random>
#include <numeric> // 用于 std::iota
#include <cstdint>
#include <limits> // 需要 std::numeric_limits
#include <optional> // 用于 std::optional - 确保 C++17 已启用
#include <functional> // 用于 std::function - 需要 C++11 或更高版本
#include"compute.hpp" // 假设此文件包含 Matrix, Tensor, AZNet 定义
#include "nnet.hpp" // 假设此文件包含神经网络定义


// --- 配置参数 (来自 config.py 和 mcts.py) ---
namespace Config {
    constexpr int BOARD_SIZE = 15; // 棋盘大小
    constexpr int BOARD_SQUARES = BOARD_SIZE * BOARD_SIZE;

    // MCTS 超参数
    constexpr float C_PUCT = 5.0f;              // PUCT 常数 (已改为 float 类型)
    constexpr int SIMULATION_TIMES = 400;       // 模拟次数
    constexpr float INITIAL_TAU = 1.0f;         // 初始 tau (已改为 float 类型)
    constexpr float TAU_DECAY = 0.8f;           // tau 衰减 (来自 config.py 顶层) (已改为 float 类型)
    constexpr float EPSILON = 0.25f;            // epsilon (用于 Dirichlet 噪声) (已改为 float 类型)
    constexpr float ALPHA_DIRICHLET = 0.03f;    // alpha (用于 Dirichlet 噪声) (已改为 float 类型)
    constexpr bool USE_DIRICHLET = false;       // 是否使用 Dirichlet 噪声
    constexpr int VIRTUAL_LOSS = 10;            // 虚拟损失
    constexpr int CAREFUL_STAGE = 6;            // 谨慎阶段

    // NN 配置 (新增自 usenet.cpp)
    constexpr int NN_INPUT_CHANNELS = 3;        // NN 输入张量的通道数
    constexpr int NN_RES_FILTERS = 64;          // 示例: 残差块中的滤波器数量
    constexpr int NN_POLICY_FILTERS = 2;        // 示例: 策略头卷积层中的滤波器数量
    constexpr int NN_VALUE_FILTERS = 1;         // 示例: 价值头卷积层中的滤波器数量
    constexpr int NN_RES_BLOCKS = 5;            // 示例: ResNet 中的残差块数量

    // 棋子表示
    constexpr int8_t EMPTY_STONE = 0;
    constexpr int8_t BLACK_STONE = 1;
    constexpr int8_t WHITE_STONE = -1;
}

// 类型别名，用于 NN 和棋盘 (新增自 usenet.cpp)
// 假设 Matrix, Tensor, AZNet 在包含的 compute.hpp/nnet.hpp 中定义
using MatrixType = Matrix<int8_t, Config::BOARD_SIZE, Config::BOARD_SIZE>;
// 假设 Tensor 有一个构造函数或方法来访问通道，例如 tensor.channel(i) 或 tensor.data_for_channel(i)
// 目前，如果它是一个类似 3D 数组的结构或类似的，我们将假设直接访问，如 tensor[channel_idx]。
// 如果 compute.hpp 对 Tensor 的定义不同，这部分需要匹配该定义。
using InputTensor = Tensor<float, Config::NN_INPUT_CHANNELS, Config::BOARD_SIZE, Config::BOARD_SIZE>; // NN 输入是 float 类型
using ActualNNType = AZNet<float, Config::NN_RES_FILTERS, Config::NN_POLICY_FILTERS, Config::NN_VALUE_FILTERS, Config::NN_INPUT_CHANNELS, Config::BOARD_SIZE, Config::BOARD_SIZE, Config::NN_RES_BLOCKS>;


// --- 实用函数 (来自 utils.py) ---
namespace Utils {
    struct Coordinate {
        int r, c;
    };

    Coordinate index_to_coordinate(int index, int board_size) {
        return {index / board_size, index % board_size};
    }

    int coordinate_to_index(Coordinate coord, int board_size) {
        return coord.r * board_size + coord.c;
    }

    int coordinate_to_index(int r, int c, int board_size) {
        return r * board_size + c;
    }

    // 生成合法走子向量 (1 表示合法, 0 表示非法)
    std::vector<int> board_to_legal_vec(const MatrixType& board) { // 已改为 MatrixType
        std::vector<int> legal_vec(Config::BOARD_SQUARES, 0);
        for (int i = 0; i < Config::BOARD_SQUARES; ++i) {
            Coordinate coord = index_to_coordinate(i, Config::BOARD_SIZE);
            if (board[coord.r][coord.c] == Config::EMPTY_STONE) {
                legal_vec[i] = 1;
            }
        }
        return legal_vec;
    }

    // 辅助函数，将棋盘状态转换为 NN 输入张量 (新增自 usenet.cpp)
    InputTensor board_to_tensor_cpp(
        const MatrixType& board_state,
        int8_t player_to_play,
        std::optional<Utils::Coordinate> last_action_coord // 需要 C++17 支持 std::optional
    ) {
        InputTensor tensor_data{}; // 零初始化

        int8_t opponent_color = -player_to_play;

        // 通道 0: 当前玩家的棋子 (值为 1.0f)
        // 假设 Tensor 访问方式为 tensor_data[channel_idx][row][col]
        for (int r = 0; r < Config::BOARD_SIZE; ++r) {
            for (int c = 0; c < Config::BOARD_SIZE; ++c) {
                if (board_state[r][c] == player_to_play) {
                    tensor_data[0][r][c] = 1.0f; // 修正后的 Tensor 访问
                }
            }
        }

        // 通道 1: 对手玩家的棋子 (值为 1.0f)
        for (int r = 0; r < Config::BOARD_SIZE; ++r) {
            for (int c = 0; c < Config::BOARD_SIZE; ++c) {
                if (board_state[r][c] == opponent_color) {
                    tensor_data[1][r][c] = 1.0f; // 修正后的 Tensor 访问
                }
            }
        }

        // 通道 2: 最后一步落子位置 (值为 1.0f)
        if (last_action_coord.has_value()) {
            Utils::Coordinate coord = last_action_coord.value();
            if (coord.r >= 0 && coord.r < Config::BOARD_SIZE && coord.c >= 0 && coord.c < Config::BOARD_SIZE) {
                tensor_data[2][coord.r][coord.c] = 1.0f; // 修正后的 Tensor 访问
            }
        }
        return tensor_data;
    }
} // namespace Utils

// --- 游戏结束检查占位符 (新增，来自 usenet.cpp) ---
struct GameEndStatus {
    bool is_end;
    float outcome_value; // 从刚走棋的玩家角度看的价值
};


// --- 节点结构 (来自 mcts.py 和你的设计) ---
struct Node : std::enable_shared_from_this<Node> {
    std::shared_ptr<Node> parent;
    int8_t color_to_play;
    float prior_p;
    int visit_count; // 修改: 移除 std::atomic
    float value_sum; // 修改: 移除 std::atomic
    bool is_expanded;
    bool is_end_node;             // 是否是终止节点
    float game_result_if_end;    // 如果是终止节点, 游戏结果 (-1, 0, 1) (已改为 float 类型)

    MatrixType board_state; // 此节点的棋盘状态 (已改为 MatrixType)
    std::array<std::shared_ptr<Node>, Config::BOARD_SQUARES> children;

    // 用于 NN 交互的新成员 (来自 usenet.cpp Node)
    int action_idx_that_led_to_this_node; // 从父节点到此节点的动作索引，根节点为 -1
    Vec<float, Config::BOARD_SQUARES> raw_nn_policy_for_children; // NN 对此状态子节点的原始策略 (已改为 Vec)
    float raw_nn_value; // NN 对此状态的原始价值 (当前玩家视角)


    Node(std::shared_ptr<Node> parent_node, float p, int8_t turn, const MatrixType& current_board, int action_idx = -1) // 将 p 改为 float 类型, board 改为 MatrixType, 添加 action_idx
        : parent(parent_node),
          color_to_play(turn),
          prior_p(p),
          visit_count(0),
          value_sum(0.0f), // 初始化 atomic float
          is_expanded(false),
          is_end_node(false),
          game_result_if_end(0.0f),
          board_state(current_board), // 复制棋盘状态
          action_idx_that_led_to_this_node(action_idx),
          raw_nn_value(0.0f)
    {
        std::fill(children.begin(), children.end(), nullptr);
        // raw_nn_policy_for_children 将在扩展期间填充
    }

    // Q(s,a) = W(s,a) / N(s,a)
    float get_q_value() const { // 已改为 float 类型
        int N = visit_count;
        if (N == 0) {
            return 0.0f;
        }
        // 对于 std::atomic<float>，在除法前加载
        return value_sum / static_cast<float>(N);
    }

    // U(s,a) = c_puct * P(s,a) * sqrt(sum_b N(s,b)) / (1 + N(s,a))
    // P(s,a) 是此节点的 prior_p (动作 'a' 从父节点 's' 导致此节点)
    // parent_total_visits_for_children 是父节点 's' 的 N(s)
    // visit_count 是此节点的 N(s,a)
    float get_u_value(float c_puct, int parent_total_visits_for_children) const { // 已改为 float 类型
        if (parent_total_visits_for_children == 0) { // 如果父节点未被访问，则不应发生
             return c_puct * prior_p / (1.0f + visit_count);
        }
        return c_puct * prior_p * std::sqrt(static_cast<float>(parent_total_visits_for_children)) / (1.0f + visit_count);
    }


    // 选择最佳子节点 (PUCT 算法)
    // legal_moves_vec: 标记哪些动作是合法的
    std::pair<int, std::shared_ptr<Node>> select_child(float c_puct, const std::vector<int>& legal_moves_vec) { // 已改为 float 类型
        std::shared_ptr<Node> best_child = nullptr;
        int best_action_idx = -1;
        float max_score = -std::numeric_limits<float>::infinity(); // 已改为 float 类型

        // 用于计算 *此* 节点子节点的 U 值的 parent_total_visits_for_children 是 this->visit_count
        int current_node_total_visits = this->visit_count;
        if (current_node_total_visits == 0 && !is_expanded) {
            // 节点尚未被访问/扩展，如果 SIMULATION_TIMES 为 0 或 1，根节点可能会发生这种情况。
            // 或者，如果在刚创建但尚未评估的节点上调用 select_child。
            // 后备方案：选择一个随机的合法走子或第一个合法走子。
            // 理想情况下，这种情况应由 MCTS 循环处理：如果选择了一个节点并且它是叶节点，则会对其进行扩展。
            // 如果是根节点且没有访问，则首先对其进行扩展。
            for (int i = 0; i < Config::BOARD_SQUARES; ++i) {
                if (legal_moves_vec[i]) {
                    // 如果节点未扩展且尚无策略，这是一个临时操作。
                    // MCTS 流程应确保在从子节点中选择之前进行扩展。
                    // 但是，如果强制选择，未扩展节点的子节点是未知的。
                    // 如果经常命中，这部分逻辑可能表示流程问题。
                    // 目前，假设如果此节点未扩展，我们尚不能选择子节点。
                    // 调用者应处理此问题 (例如，首先扩展此节点)。
                    // 或者，如果这是叶节点扩展 *之前* 的选择阶段：
                    // “子节点”是概念上的走子。
                    if (raw_nn_policy_for_children.empty() && is_leaf()) {
                         // 此节点是叶节点，其策略尚未评估。
                         // 这种情况应首先通过扩展当前节点来处理。
                         // 为确保鲁棒性，如果强制执行，则选择一个随机的合法走子。
                        std::vector<int> actual_legal_indices;
                        for(int k=0; k < Config::BOARD_SQUARES; ++k) if(legal_moves_vec[k]) actual_legal_indices.push_back(k);
                        if(!actual_legal_indices.empty()){
                            std::random_device rd_sel;
                            std::mt19937 gen_sel(rd_sel());
                            std::uniform_int_distribution<> distrib_sel(0, actual_legal_indices.size() - 1);
                            best_action_idx = actual_legal_indices[distrib_sel(gen_sel)];
                            // best_child 保持为 nullptr，因为它还不是真正的子节点
                            return {best_action_idx, nullptr};
                        } else {
                            return {-1, nullptr}; // 没有合法的走子
                        }
                    }
                }
            }
        }


        for (int i = 0; i < Config::BOARD_SQUARES; ++i) {
            if (!legal_moves_vec[i]) continue;

            float score;
            if (children[i]) { // 如果子节点存在
                score = children[i]->get_q_value() + children[i]->get_u_value(c_puct, current_node_total_visits);
            } else { // 如果子节点不存在 (来自已扩展父节点的潜在走子)
                // 此节点 (父节点) 必须已扩展，才能填充 raw_nn_policy_for_children。
                if (is_expanded && !raw_nn_policy_for_children.empty()) {
                    float prior_p_for_action_i = raw_nn_policy_for_children[i];
                    // 对于未访问/未创建的节点，Q 值为 0。N(s,a) 为 0。
                    float u_value_for_action_i = c_puct * prior_p_for_action_i * std::sqrt(static_cast<float>(current_node_total_visits)) / (1.0f + 0);
                    score = 0.0f + u_value_for_action_i;
                } else {
                    // 如果 MCTS 流程正确 (在从其策略中选择子节点之前扩展父节点)，则理想情况下不应命中此情况。
                    // 如果父节点未扩展，我们没有此动作的策略。
                    // 分配一个低分或作为错误处理。目前，跳过。
                    continue;
                }
            }

            if (score > max_score) {
                max_score = score;
                best_child = children[i]; // 如果子节点尚不存在，则为 nullptr
                best_action_idx = i;
            }
        }

        if (best_action_idx == -1 && !legal_moves_vec.empty()) {
            // 后备方案：如果没有选择有效的走子 (例如，所有得分均为 -inf，或者没有子节点且节点未扩展)
            // 这表示存在问题或边缘情况 (例如，没有有效走子的平局状态，尽管那时 legal_moves_vec 应该为空)
            // 或者，如果所有合法走子都导致某种程度上无效的评分状态。
            // 为确保鲁棒性，选择一个随机的合法走子。
            std::vector<int> actual_legal_indices;
            for(int i=0; i < Config::BOARD_SQUARES; ++i) if(legal_moves_vec[i]) actual_legal_indices.push_back(i);
            if(!actual_legal_indices.empty()){
                 std::random_device rd_fall;
                 std::mt19937 gen_fall(rd_fall());
                 std::uniform_int_distribution<> distrib_fall(0, actual_legal_indices.size() - 1);
                 best_action_idx = actual_legal_indices[distrib_fall(gen_fall)];
                 best_child = children[best_action_idx]; // 可能为 nullptr
            } else {
                 // std::cerr << "错误: select_child 中没有合法的走子可供选择!" << std::endl; // 保留错误用于调试
                 return {-1, nullptr}; // 完全没有合法的走子
            }
        }
        // 如果 best_child 为 nullptr 但 best_action_idx 有效，则表示我们为未创建的子节点选择了一个动作。
        // 调用者 (MCTS 循环) 随后将创建此子节点。
        return {best_action_idx, best_child};
    }

    // 移除了 policy_probs_placeholder
    // 移除了 Node::expand 方法，因为扩展由 mcts_expand_and_evaluate 处理


    // 反向传播 (来自 mcts.py Node.backup)
    void backup(float leaf_value_for_leaf_player) {
        std::shared_ptr<Node> current_sptr = shared_from_this();
        float value_to_backup = leaf_value_for_leaf_player;

        while (current_sptr != nullptr) {
            current_sptr->visit_count++;
            // value_sum 现在是普通的 float，可以直接累加
            current_sptr->value_sum += value_to_backup;

            value_to_backup *= -1; // 切换玩家视角
            current_sptr = current_sptr->parent;
        }
    }


    bool is_leaf() const {
        return !is_expanded;
    }
};


// 向前声明
float mcts_expand_and_evaluate(
    std::shared_ptr<Node> node_to_expand,
    ActualNNType& current_active_net, // 修改参数名以提高一致性
    // 传递一个可调用对象用于游戏结束检查
    std::function<GameEndStatus(const MatrixType&, Utils::Coordinate, int8_t)> check_game_end_func
);


// --- MCTS 类 (来自 mcts.py) ---
class MCTS_Agent {
public:
    std::shared_ptr<Node> root;
    int8_t agent_color; // AI 的执棋颜色
    float current_tau; // 已改为 float 类型
    ActualNNType net_w;       // 白棋视角的神经网络
    ActualNNType net_b;       // 黑棋视角下的神经网络
    ActualNNType* active_net; // 指向当前使用的网络


    MCTS_Agent(int8_t player_color) : agent_color(player_color), current_tau(Config::INITIAL_TAU), active_net(nullptr) {
        MatrixType initial_board{}; // 零初始化
        // 根节点的 prior_p 通常为 1.0 或未使用。color_to_play 为 BLACK。action_idx 为 -1。
        root = std::make_shared<Node>(nullptr, 1.0f, Config::BLACK_STONE, initial_board, -1);
        
        // TODO: 在此处加载 net_w 的权重 (例如，从为白棋训练的文件加载)
        // 假设存在类似 net_w.load_model("path_to_white_model.weights"); 的方法
        std::cout << "Placeholder: Load weights for WHITE player network (net_w)" << std::endl;

        // TODO: 在此处加载 net_b 的权重 (例如，从为黑棋训练的文件加载)
        // 假设存在类似 net_b.load_model("path_to_black_model.weights"); 的方法
        std::cout << "Placeholder: Load weights for BLACK player network (net_b)" << std::endl;

        if (agent_color == Config::WHITE_STONE) {
            active_net = &net_w;
            std::cout << "MCTS_Agent initialized for WHITE player, using net_w." << std::endl;
        } else if (agent_color == Config::BLACK_STONE) {
            active_net = &net_b;
            std::cout << "MCTS_Agent initialized for BLACK player, using net_b." << std::endl;
        } else {
            // 错误处理：未知的玩家颜色
            std::cerr << "Error: Unknown agent_color (" << static_cast<int>(agent_color)
                      << ") for MCTS_Agent. Active network not set." << std::endl;
            // 考虑设置一个默认网络或抛出异常，这里暂时不设置或可以指向一个默认网络
            // 例如: active_net = &net_b; // 默认为黑棋网络并打印警告
        }
    }

    void reset_tree() {
        MatrixType initial_board{};
        root = std::make_shared<Node>(nullptr, 1.0f, Config::BLACK_STONE, initial_board, -1);
        current_tau = Config::INITIAL_TAU;
        // active_net 的选择在构造时完成，通常在 reset_tree 时不需要更改agent_color
    }

    // predict_nn 已移除，mcts_expand_and_evaluate 处理 NN 调用。

    // 检查游戏是否结束 (你需要彻底实现它)
    // 返回 GameEndStatus: {is_end, outcome_value for last_player_color}
    // last_action_coord: 导致当前棋盘状态的走子坐标
    // last_player_color: 做出该走子的玩家
    GameEndStatus check_game_end(const MatrixType& board, Utils::Coordinate last_action_coord, int8_t last_player_color) {
        // 如果 last_player_color 是 EMPTY_STONE 或坐标无效，则无法根据最后一步走子确定获胜者。
        // 此函数假设 last_action_coord 是 last_player_color 刚落子的位置。
        if (last_player_color == Config::EMPTY_STONE || last_action_coord.r < 0 || last_action_coord.r >= Config::BOARD_SIZE || last_action_coord.c < 0 || last_action_coord.c >= Config::BOARD_SIZE) {
            // 如果棋盘已满，则检查是否平局，否则从这个角度看游戏尚未结束
            bool board_full = true;
            for (int r = 0; r < Config::BOARD_SIZE; ++r) {
                for (int c = 0; c < Config::BOARD_SIZE; ++c) {
                    if (board[r][c] == Config::EMPTY_STONE) {
                        board_full = false;
                        break;
                    }
                }
                if (!board_full) break;
            }
            if (board_full) return {true, 0.0f}; // 平局
            return {false, 0.0f}; // 无法确定获胜，游戏未结束或无效调用
        }

        const int B_SIZE = Config::BOARD_SIZE;
        const int R = last_action_coord.r;
        const int C = last_action_coord.c;
        const int STONES_TO_WIN = 5;

        // 方向：水平、垂直、对角线 (左上到右下)、反对角线 (右上到左下)
        const int dr[] = {0, 1, 1, 1}; // 行增量
        const int dc[] = {1, 0, 1, -1}; // 列增量

        for (int i = 0; i < 4; ++i) { // 遍历 4 个方向
            int count = 1; // 计算刚落下的棋子

            // 检查正方向
            for (int k = 1; k < STONES_TO_WIN; ++k) {
                int nr = R + k * dr[i];
                int nc = C + k * dc[i];
                if (nr >= 0 && nr < B_SIZE && nc >= 0 && nc < B_SIZE && board[nr][nc] == last_player_color) {
                    count++;
                } else {
                    break;
                }
            }

            // 检查负方向
            for (int k = 1; k < STONES_TO_WIN; ++k) {
                int nr = R - k * dr[i];
                int nc = C - k * dc[i];
                if (nr >= 0 && nr < B_SIZE && nc >= 0 && nc < B_SIZE && board[nr][nc] == last_player_color) {
                    count++;
                } else {
                    break;
                }
            }

            if (count >= STONES_TO_WIN) {
                return {true, 1.0f}; // last_player_color 获胜
            }
        }

        // 检查是否平局 (棋盘已满)
        bool board_full = true;
        for (int r = 0; r < B_SIZE; ++r) {
            for (int c = 0; c < B_SIZE; ++c) {
                if (board[r][c] == Config::EMPTY_STONE) {
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
        return {false, 0.0f};
    }


    // 执行一次 MCTS 模拟 (选择、扩展、评估、反向传播)
    void run_one_simulation() {
        if (!active_net) { // 增加对 active_net 的检查
            std::cerr << "Error: Active network is not set in MCTS_Agent. Cannot run simulation." << std::endl;
            return;
        }

        std::shared_ptr<Node> current_node = root;
        // std::vector<std::shared_ptr<Node>> path; // 在此版本的反向传播中未明确使用路径
        // path.push_back(current_node);

        // 1. 选择
        while (current_node->is_expanded && !current_node->is_end_node) {
            std::vector<int> legal_moves = Utils::board_to_legal_vec(current_node->board_state);
            if (std::all_of(legal_moves.begin(), legal_moves.end(), [](int i){ return i == 0; })) {
                // 没有合法的走子，但节点尚未标记为 end_node。视为终止状态 (例如，如果不是赢/输，则为平局)
                // 如果 check_game_end 没有覆盖所有终止状态，则可能会发生这种情况。
                // 为确保鲁棒性，我们假设 mcts_expand_and_evaluate 中的 check_game_end 会处理它。
                // 或者，我们可以在这里标记它。
                GameEndStatus end_status = check_game_end(current_node->board_state, {-1,-1}/*虚拟坐标*/, current_node->color_to_play);
                current_node->is_end_node = true;
                current_node->game_result_if_end = (current_node->color_to_play == Config::BLACK_STONE) ? end_status.outcome_value : -end_status.outcome_value; // 调整为当前玩家视角
                break;
            }

            auto selection = current_node->select_child(Config::C_PUCT, legal_moves);
            int best_action_idx = selection.first;
            std::shared_ptr<Node> next_node = selection.second;

            if (best_action_idx == -1) { // 没有有效的子节点可供选择 (如果存在合法走子，则应该很少见)
                // 这意味着终止状态或问题。
                // 根据游戏规则标记为结束节点。
                 GameEndStatus end_status = check_game_end(current_node->board_state, {-1,-1}/*虚拟坐标*/, current_node->color_to_play);
                 current_node->is_end_node = true;
                 current_node->game_result_if_end = (current_node->color_to_play == Config::BLACK_STONE) ? end_status.outcome_value : -end_status.outcome_value;
                break;            }

            if (next_node) {
                current_node = next_node;
            } else { // 子节点不存在，需要创建它 (它是根据策略选择的)
                MatrixType next_board_state = current_node->board_state;
                Utils::Coordinate move_coord = Utils::index_to_coordinate(best_action_idx, Config::BOARD_SIZE);
                next_board_state[move_coord.r][move_coord.c] = current_node->color_to_play;
                int8_t next_player_color = (current_node->color_to_play == Config::BLACK_STONE) ? Config::WHITE_STONE : Config::BLACK_STONE;

                // 新子节点的 prior_p 来自父节点 (current_node) 的原始 NN 策略
                float prior_p_for_new_node = 0.1f; // 占位符：应从 current_node->raw_nn_policy_for_children[best_action_idx] 获取
                // 确保 current_node 已扩展且 raw_nn_policy_for_children 已填充
                if (current_node->is_expanded && !current_node->raw_nn_policy_for_children.empty() && best_action_idx < current_node->raw_nn_policy_for_children.size()) {
                    prior_p_for_new_node = current_node->raw_nn_policy_for_children[best_action_idx];
                } else {
                    // 处理错误或使用默认先验概率
                    // std::cerr << "警告: 无法获取新节点的先验概率。父节点可能未正确扩展。" << std::endl;
                }

                current_node->children[best_action_idx] = std::make_shared<Node>(current_node, prior_p_for_new_node, next_player_color, next_board_state, best_action_idx);
                current_node = current_node->children[best_action_idx];
            }
            // path.push_back(current_node);
        }

        // 2. 扩展和评估 (如果尚未终止)
        float leaf_value; // 从 current_node->color_to_play 的角度看的价值

        if (current_node->is_end_node) {
            leaf_value = current_node->game_result_if_end;
            // game_result_if_end 是从 *叶节点轮到的玩家* 的角度看的。
            // 我们需要从 current_node->color_to_play 的角度看的价值。
            // backup 函数会处理视角的转换，所以这里传递叶节点玩家视角的结果即可。
        } else {
            // 扩展叶节点
            // leaf_value 是从 node_to_expand->color_to_play 的角度看的价值
            leaf_value = mcts_expand_and_evaluate(current_node, *active_net, // 修改处：使用 *active_net
                [this](const MatrixType& board, Utils::Coordinate coord, int8_t color) {
                    return this->check_game_end(board, coord, color);
                }
            );
        }
        // 3. 反向传播
        // mcts_expand_and_evaluate 返回的价值是从 current_node->color_to_play 的角度看的
        current_node->backup(leaf_value);
    }

    // 用于 check_game_end 的静态包装器，以便在 mcts_expand_and_evaluate 不是成员函数时传递给它
    // 如果 mcts_expand_and_evaluate 是一个自由函数，这有点像一个变通方法。
    // 如果 mcts_expand_and_evaluate 可以是成员函数或接受 std::function，则更简洁。
    // 目前，假设 mcts_expand_and_evaluate 是一个自由函数，我们传递 MCTS_Agent 的 check_game_end。
    /*
    static GameEndStatus check_game_end_static_wrapper(const MatrixType& board, Utils::Coordinate last_action_coord, int8_t last_player_color, MCTS_Agent* agent_instance) {
        if (agent_instance) {
            return agent_instance->check_game_end(board, last_action_coord, last_player_color);
        }
        // 如果实例为 null，则回退或出错
        return {false, 0.0f};
    }
    */


}; // MCTS_Agent 类


// 函数：使用神经网络扩展叶节点，创建其子节点，
// 并返回节点的评估值。
// (新增，改编自 usenet.cpp)
float mcts_expand_and_evaluate(
    std::shared_ptr<Node> node_to_expand,
    ActualNNType& current_active_net, // 确保定义处的参数名也一致
    std::function<GameEndStatus(const MatrixType&, Utils::Coordinate, int8_t)> check_game_end_func
)
{
    int8_t current_player_at_node = node_to_expand->color_to_play;

    // 0. 检查游戏是否已在此节点结束 (在选择阶段可能未捕获到)
    // 假设 node_to_expand->action_idx_that_led_to_this_node 和 node_to_expand->parent 可用于获取上一步操作
    Utils::Coordinate last_action_coord = {-1, -1}; // 默认无效坐标
    int8_t player_who_made_last_move = Config::EMPTY_STONE;

    if(node_to_expand->parent){
        // 如果有父节点，则导致此节点的动作是由父节点的玩家做出的
        player_who_made_last_move = node_to_expand->parent->color_to_play;
        if(node_to_expand->action_idx_that_led_to_this_node != -1){
            last_action_coord = Utils::index_to_coordinate(node_to_expand->action_idx_that_led_to_this_node, Config::BOARD_SIZE);
        }
    }
    // 否则，如果是根节点，则没有“上一步操作”导致它。

    GameEndStatus end_status = check_game_end_func(node_to_expand->board_state, last_action_coord, player_who_made_last_move);

    if (end_status.is_end) {
        node_to_expand->is_expanded = true; // 技术上已“评估”，即使未通过 NN
        node_to_expand->is_end_node = true;
        // end_status.outcome_value 是从 player_who_made_last_move 的角度看的。
        // 我们需要将其转换为 node_to_expand->color_to_play (当前轮到的玩家) 的角度。
        if (player_who_made_last_move == current_player_at_node) {
            // 这不应该发生，因为如果轮到 current_player_at_node，那么 player_who_made_last_move 应该是对手。
            // 除非是游戏开始时的特殊情况或根节点。
            // 对于非根节点，如果 player_who_made_last_move == current_player_at_node，则逻辑有误。
            // 假设 player_who_made_last_move 是对手。
            node_to_expand->game_result_if_end = -end_status.outcome_value;
        } else {
            // player_who_made_last_move 是对手，所以 outcome_value 是对手的价值。
            // 当前玩家的价值是 -outcome_value。
            node_to_expand->game_result_if_end = -end_status.outcome_value;
        }
         if (player_who_made_last_move == Config::EMPTY_STONE && end_status.outcome_value == 0.0f) { // 例如，棋盘已满平局，没有最后走棋的玩家
            node_to_expand->game_result_if_end = 0.0f;
        }
        return node_to_expand->game_result_if_end;
    }

    // 1. 将棋盘状态转换为 NN 输入张量
    // last_action_coord_for_tensor 应为导致 *当前* board_state 的动作。
    // 这与上面用于 check_game_end 的 last_action_coord 相同。
    InputTensor nn_input = Utils::board_to_tensor_cpp(node_to_expand->board_state, current_player_at_node, last_action_coord);

    // 2. 通过神经网络进行推理
    auto nn_output_pair = current_active_net.feed(nn_input);
    const auto& raw_policy_from_nn = nn_output_pair.first; // 类型为 Vec<float, Config::BOARD_SQUARES>
    float value_from_nn = nn_output_pair.second;      // 标量值 (从 current_player_at_node 的角度)

    // 存储原始 NN 输出到节点
    node_to_expand->raw_nn_policy_for_children = raw_policy_from_nn;
    node_to_expand->raw_nn_value = value_from_nn;

    // 3. 标记节点为已扩展
    node_to_expand->is_expanded = true;

    // 4. (可选) 创建子节点，如果 MCTS 策略是立即创建它们
    // 或者，子节点可以在选择阶段根据存储的 raw_nn_policy_for_children 按需创建。
    // 当前实现似乎是在选择阶段按需创建。
    // 这里我们只确保策略已存储。

    // 应用 Dirichlet 噪声 (如果启用)
    if (Config::USE_DIRICHLET && node_to_expand->parent == nullptr) { // 通常仅应用于根节点
        Vec<float, Config::BOARD_SQUARES> dirichlet_noise{}; // 改为 Vec 并零初始化
        std::gamma_distribution<float> gamma(Config::ALPHA_DIRICHLET, 1.0f);
        std::random_device rd;
        std::mt19937 gen(rd());
        float sum_noise = 0.0f;
        for (int i = 0; i < Config::BOARD_SQUARES; ++i) {
            dirichlet_noise[i] = gamma(gen);
            sum_noise += dirichlet_noise[i];
        }
        if (sum_noise > 0) { // 避免除以零
            for (int i = 0; i < Config::BOARD_SQUARES; ++i) {
                node_to_expand->raw_nn_policy_for_children[i] =
                    (1.0f - Config::EPSILON) * node_to_expand->raw_nn_policy_for_children[i] +
                    Config::EPSILON * (dirichlet_noise[i] / sum_noise);
            }
        }
    }


    // 5. 返回从 NN 获得的评估值
    // 该值是从 current_player_at_node (即 node_to_expand->color_to_play) 的角度看的。
    return value_from_nn;
}

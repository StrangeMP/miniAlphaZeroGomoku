#include "compute.hpp"
#include "json.hpp"
#include "network.hpp"
#include "qnnet.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

using json = nlohmann::json; // 在全局作用域或 main 之前使用 using 声明

// --- 配置参数 (来自 config.py 和 mcts.py) ---
namespace Config {
constexpr int BOARD_SIZE = 15; // 棋盘大小
constexpr int BOARD_SQUARES = BOARD_SIZE * BOARD_SIZE;

// MCTS 超参数
constexpr float C_PUCT = 5.0f;           // PUCT 常数 (已改为 float 类型)
constexpr int SIMULATION_TIMES = 100;    // 模拟次数
constexpr float INITIAL_TAU = 1.0f;      // 初始 tau (已改为 float 类型)
constexpr float TAU_DECAY = 0.8f;        // tau 衰减 (来自 config.py 顶层) (已改为 float 类型)
constexpr float EPSILON = 0.25f;         // epsilon (用于 Dirichlet 噪声) (已改为 float 类型)
constexpr float ALPHA_DIRICHLET = 0.03f; // alpha (用于 Dirichlet 噪声) (已改为 float 类型)
constexpr bool USE_DIRICHLET = false;    // 是否使用 Dirichlet 噪声
constexpr int VIRTUAL_LOSS = 10;         // 虚拟损失
constexpr int CAREFUL_STAGE = 6;         // 谨慎阶段

// 棋子表示
WEIGHT_T EMPTY_STONE = AlphaGomoku::Black::Q_ZERO;
WEIGHT_T BLACK_STONE = AlphaGomoku::Black::Q_ONE;
WEIGHT_T WHITE_STONE = AlphaGomoku::Black::Q_NEG_ONE;
} // namespace Config

using MatrixType = Matrix<WEIGHT_T, Config::BOARD_SIZE, Config::BOARD_SIZE>;
using ActualNNType = AlphaGomoku::GodNet;
using InputTensor = ActualNNType::InputTensor;

// --- 实用函数 (来自 utils.py) ---
namespace Utils {
struct Coordinate {
  int r, c;
};

Coordinate index_to_coordinate(int index, int board_size) { return {index / board_size, index % board_size}; }

int coordinate_to_index(Coordinate coord, int board_size) { return coord.r * board_size + coord.c; }

int coordinate_to_index(int r, int c, int board_size) { return r * board_size + c; }

// 生成合法走子向量 (1 表示合法, 0 表示非法)
std::vector<int> board_to_legal_vec(const MatrixType &board) { // 已改为 MatrixType
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
InputTensor board_to_tensor_cpp(const MatrixType &board_state, WEIGHT_T player_to_play,
                                Utils::Coordinate last_action_coord, // 直接传递坐标
                                bool last_action_valid               // 新增布尔标志
) {
  InputTensor tensor_data{}; // 零初始化

  WEIGHT_T opponent_color = -player_to_play;

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
  if (last_action_valid) { // 使用布尔标志检查
    if (last_action_coord.r >= 0 && last_action_coord.r < Config::BOARD_SIZE && last_action_coord.c >= 0 &&
        last_action_coord.c < Config::BOARD_SIZE) {
      tensor_data[2][last_action_coord.r][last_action_coord.c] = 1.0f;
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
  WEIGHT_T color_to_play;
  float prior_p;
  int visit_count;
  float value_sum;
  bool is_expanded;
  bool is_end_node;         // 是否是终止节点
  float game_result_if_end; // 如果是终止节点, 游戏结果 (-1, 0, 1) (已改为 float 类型)

  MatrixType board_state; // 此节点的棋盘状态 (已改为 MatrixType)
  std::array<std::shared_ptr<Node>, Config::BOARD_SQUARES> children;

  // 用于 NN 交互的新成员 (来自 usenet.cpp Node)
  int action_idx_that_led_to_this_node;                         // 从父节点到此节点的动作索引，根节点为 -1
  Vec<float, Config::BOARD_SQUARES> raw_nn_policy_for_children; // NN 对此状态子节点的原始策略 (已改为 Vec)
  float raw_nn_value;                                           // NN 对此状态的原始价值 (当前玩家视角)

  Node(std::shared_ptr<Node> parent_node, float p, WEIGHT_T turn, const MatrixType &current_board,
       int action_idx = -1) // 将 p 改为 float 类型, board 改为 MatrixType, 添加 action_idx
      : parent(parent_node), color_to_play(turn), prior_p(p), visit_count(0), value_sum(0.0f), // 初始化 atomic float
        is_expanded(false), is_end_node(false), game_result_if_end(0.0f), board_state(current_board), // 复制棋盘状态
        action_idx_that_led_to_this_node(action_idx), raw_nn_value(0.0f) {
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
    if (parent_total_visits_for_children == 0) {                                // 如果父节点未被访问，则不应发生
      return c_puct * prior_p / (1.0f + visit_count);
    }
    return c_puct * prior_p * std::sqrt(static_cast<float>(parent_total_visits_for_children)) / (1.0f + visit_count);
  }

  // 选择最佳子节点 (PUCT 算法)
  // legal_moves_vec: 标记哪些动作是合法的
  std::pair<int, std::shared_ptr<Node>> select_child(float c_puct,
                                                     const std::vector<int> &legal_moves_vec) { // 已改为 float 类型
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
            for (int k = 0; k < Config::BOARD_SQUARES; ++k)
              if (legal_moves_vec[k])
                actual_legal_indices.push_back(k);
            if (!actual_legal_indices.empty()) {
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
      if (!legal_moves_vec[i])
        continue;

      float score;
      if (children[i]) { // 如果子节点存在
        score = -children[i]->get_q_value() + children[i]->get_u_value(c_puct, current_node_total_visits);
      } else { // 如果子节点不存在 (来自已扩展父节点的潜在走子)
        // 此节点 (父节点) 必须已扩展，才能填充 raw_nn_policy_for_children。
        if (is_expanded && !raw_nn_policy_for_children.empty()) {
          float prior_p_for_action_i = raw_nn_policy_for_children[i];
          // 对于未访问/未创建的节点，Q 值为 0。N(s,a) 为 0。
          float u_value_for_action_i =
              c_puct * prior_p_for_action_i * std::sqrt(static_cast<float>(current_node_total_visits)) / (1.0f + 0);
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
      for (int i = 0; i < Config::BOARD_SQUARES; ++i)
        if (legal_moves_vec[i])
          actual_legal_indices.push_back(i);
      if (!actual_legal_indices.empty()) {
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

  bool is_leaf() const { return !is_expanded; }
};

// 向前声明
float mcts_expand_and_evaluate(
    std::shared_ptr<Node> node_to_expand,
    ActualNNType &current_active_net, // 修改参数名以提高一致性
    // 传递一个可调用对象用于游戏结束检查
    std::function<GameEndStatus(const MatrixType &, Utils::Coordinate, WEIGHT_T)> check_game_end_func);

// --- MCTS 类 (来自 mcts.py) ---
class MCTS_Agent {
public:
  std::shared_ptr<Node> root;
  WEIGHT_T agent_color;                       // AI 的执棋颜色
  float current_tau;                          // 已改为 float 类型
  AlphaGomoku::White::QuantizedNetwork net_w; // 白棋视角的神经网络
  AlphaGomoku::Black::QuantizedNetwork net_b; // 黑棋视角下的神经网络
  ActualNNType *active_net;                   // 指向当前使用的网络

  MCTS_Agent(WEIGHT_T player_color) : agent_color(player_color), current_tau(Config::INITIAL_TAU), active_net(nullptr) {
    MatrixType initial_board{}; // 零初始化
    // 根节点的 prior_p 通常为 1.0 或未使用。color_to_play 为 BLACK。action_idx 为 -1。
    root = std::make_shared<Node>(nullptr, 1.0f, Config::BLACK_STONE, initial_board, -1);
    if (agent_color == Config::WHITE_STONE) {
      active_net = &net_w;
    } else if (agent_color == Config::BLACK_STONE) {
      active_net = &net_b;
    } else {
      throw std::invalid_argument("Unknown agent_color for MCTS_Agent: " + std::to_string(agent_color));
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
  GameEndStatus check_game_end(const MatrixType &board, Utils::Coordinate last_action_coord,
                               WEIGHT_T last_player_color) {
    // 如果 last_player_color 是 EMPTY_STONE 或坐标无效，则无法根据最后一步走子确定获胜者。
    // 此函数假设 last_action_coord 是 last_player_color 刚落子的位置。
    if (last_player_color == Config::EMPTY_STONE || last_action_coord.r < 0 ||
        last_action_coord.r >= Config::BOARD_SIZE || last_action_coord.c < 0 ||
        last_action_coord.c >= Config::BOARD_SIZE) {
      // 如果棋盘已满，则检查是否平局，否则从这个角度看游戏尚未结束
      bool board_full = true;
      for (int r = 0; r < Config::BOARD_SIZE; ++r) {
        for (int c = 0; c < Config::BOARD_SIZE; ++c) {
          if (board[r][c] == Config::EMPTY_STONE) {
            board_full = false;
            break;
          }
        }
        if (!board_full)
          break;
      }
      if (board_full)
        return {true, 0.0f}; // 平局
      return {false, 0.0f};  // 无法确定获胜，游戏未结束或无效调用
    }

    const int B_SIZE = Config::BOARD_SIZE;
    const int R = last_action_coord.r;
    const int C = last_action_coord.c;
    const int STONES_TO_WIN = 5;

    // 方向：水平、垂直、对角线 (左上到右下)、反对角线 (右上到左下)
    const int dr[] = {0, 1, 1, 1};  // 行增量
    const int dc[] = {1, 0, 1, -1}; // 列增量

    for (int i = 0; i < 4; ++i) { // 遍历 4 个方向
      int count = 1;              // 计算刚落下的棋子

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

    // 1. 选择
    while (current_node->is_expanded && !current_node->is_end_node) {
      std::vector<int> legal_moves = Utils::board_to_legal_vec(current_node->board_state);
      if (std::all_of(legal_moves.begin(), legal_moves.end(), [](int i) { return i == 0; })) {
        // 没有合法的走子，但节点尚未标记为 end_node。视为终止状态 (例如，如果不是赢/输，则为平局)
        // 如果 check_game_end 没有覆盖所有终止状态，则可能会发生这种情况。
        // 为确保鲁棒性，我们假设 mcts_expand_and_evaluate 中的 check_game_end 会处理它。
        // 或者，我们可以在这里标记它。
        GameEndStatus end_status =
            check_game_end(current_node->board_state, {-1, -1} /*虚拟坐标*/, current_node->color_to_play);
        current_node->is_end_node = true;
        current_node->game_result_if_end = (current_node->color_to_play == Config::BLACK_STONE)
                                               ? end_status.outcome_value
                                               : -end_status.outcome_value; // 调整为当前玩家视角
        break;
      }

      auto selection = current_node->select_child(Config::C_PUCT, legal_moves);
      int best_action_idx = selection.first;
      std::shared_ptr<Node> next_node = selection.second;

      if (best_action_idx == -1) { // 没有有效的子节点可供选择 (如果存在合法走子，则应该很少见)
                                   // 这意味着终止状态或问题。
                                   // 根据游戏规则标记为结束节点。
        GameEndStatus end_status =
            check_game_end(current_node->board_state, {-1, -1} /*虚拟坐标*/, current_node->color_to_play);
        current_node->is_end_node = true;
        current_node->game_result_if_end =
            (current_node->color_to_play == Config::BLACK_STONE) ? end_status.outcome_value : -end_status.outcome_value;
        break;
      }

      if (next_node) {
        current_node = next_node;
      } else { // 子节点不存在，需要创建它 (它是根据策略选择的)
        MatrixType next_board_state = current_node->board_state;
        Utils::Coordinate move_coord = Utils::index_to_coordinate(best_action_idx, Config::BOARD_SIZE);
        next_board_state[move_coord.r][move_coord.c] = current_node->color_to_play;
        WEIGHT_T next_player_color =
            (current_node->color_to_play == Config::BLACK_STONE) ? Config::WHITE_STONE : Config::BLACK_STONE;

        // 新子节点的 prior_p 来自父节点 (current_node) 的原始 NN 策略
        float prior_p_for_new_node = 0.1f;
        // 确保 current_node 已扩展且 raw_nn_policy_for_children 已填充
        if (current_node->is_expanded && !current_node->raw_nn_policy_for_children.empty() &&
            best_action_idx < current_node->raw_nn_policy_for_children.size()) {
          prior_p_for_new_node = current_node->raw_nn_policy_for_children[best_action_idx];
        } else {
          // 处理错误或使用默认先验概率
          // std::cerr << "警告: 无法获取新节点的先验概率。父节点可能未正确扩展。" << std::endl;
        }

        current_node->children[best_action_idx] = std::make_shared<Node>(
            current_node, prior_p_for_new_node, next_player_color, next_board_state, best_action_idx);
        current_node = current_node->children[best_action_idx];
      }
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
                                            [this](const MatrixType &board, Utils::Coordinate coord, WEIGHT_T color) {
                                              return this->check_game_end(board, coord, color);
                                            });
    }
    // 3. 反向传播
    // mcts_expand_and_evaluate 返回的价值是从 current_node->color_to_play 的角度看的
    current_node->backup(leaf_value);
  }

}; // MCTS_Agent 类

// 函数：使用神经网络扩展叶节点，创建其子节点，
// 并返回节点的评估值。
// (新增，改编自 usenet.cpp)
float mcts_expand_and_evaluate(
    std::shared_ptr<Node> node_to_expand,
    ActualNNType &current_active_net, // 确保定义处的参数名也一致
    std::function<GameEndStatus(const MatrixType &, Utils::Coordinate, WEIGHT_T)> check_game_end_func) {
  WEIGHT_T current_player_at_node = node_to_expand->color_to_play;

  Utils::Coordinate last_action_coord = {-1, -1};
  WEIGHT_T player_who_made_last_move = Config::EMPTY_STONE;
  bool is_last_action_valid_for_tensor = false; // For board_to_tensor_cpp

  if (node_to_expand->parent) {
    // 如果有父节点，则导致此节点的动作是由父节点的玩家做出的
    player_who_made_last_move = node_to_expand->parent->color_to_play;
    if (node_to_expand->action_idx_that_led_to_this_node != -1) {
      last_action_coord =
          Utils::index_to_coordinate(node_to_expand->action_idx_that_led_to_this_node, Config::BOARD_SIZE);
      is_last_action_valid_for_tensor = true; // last_action_coord is now valid
    }
  }
  // 否则，如果是根节点，则没有“上一步操作”导致它。

  GameEndStatus end_status =
      check_game_end_func(node_to_expand->board_state, last_action_coord, player_who_made_last_move);

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
    if (player_who_made_last_move == Config::EMPTY_STONE &&
        end_status.outcome_value == 0.0f) { // 例如，棋盘已满平局，没有最后走棋的玩家
      node_to_expand->game_result_if_end = 0.0f;
    }
    return node_to_expand->game_result_if_end;
  }

  // 1. 将棋盘状态转换为 NN 输入张量
  // last_action_coord_for_tensor 应为导致 *当前* board_state 的动作。
  // 这与上面用于 check_game_end 的 last_action_coord 相同。
  InputTensor nn_input = Utils::board_to_tensor_cpp(node_to_expand->board_state, current_player_at_node,
                                                    last_action_coord, is_last_action_valid_for_tensor);

  // 2. 通过神经网络进行推理
  auto nn_output_pair = current_active_net.feed(nn_input);
  const auto &raw_policy_from_nn = nn_output_pair.first; // 类型为 Vec<float, Config::BOARD_SQUARES>
  float value_from_nn = nn_output_pair.second;           // 标量值 (从 current_player_at_node 的角度)

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
    Vec<float, Config::BOARD_SQUARES> dirichlet_noise{};            // 改为 Vec 并零初始化
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

// Helper to place stone and switch player, used in board reconstruction
// 用于棋盘重建的辅助函数，放置棋子并切换玩家
void place_stone_on_board(MatrixType &board, int r, int c, WEIGHT_T &current_player_color) {
  if (r >= 0 && r < Config::BOARD_SIZE && c >= 0 && c < Config::BOARD_SIZE) {
    if (board[r][c] == Config::EMPTY_STONE) { // 仅在空位时落子
      board[r][c] = current_player_color;
      current_player_color = (current_player_color == Config::BLACK_STONE) ? Config::WHITE_STONE : Config::BLACK_STONE;
    }
  }
}

WEIGHT_T ai_fixed_color = Config::BLACK_STONE; // AI固定的执棋颜色 (例如，总是执黑)
MCTS_Agent agent(ai_fixed_color);              // 创建MCTS代理实例
int main() {
  std::ios_base::sync_with_stdio(false); // 关闭C++标准流与C标准流的同步，提高cin/cout效率
  std::cin.tie(NULL);                    // 解除cin与cout的绑定，进一步提高效率

  Utils::Coordinate ai_black_first_move_coord = {-1, -1}; // 记录AI执黑时第一步的落子，用于换手
  bool opponent_requested_swap_vs_ai_black = false;       // 标记对手(白方)是否在AI(黑方)第一步后请求换手
  bool ai_white_performed_swap = false;                   // 标记AI(白方)是否已经执行了换手操作

  // --- 全局游戏状态 ---
  MatrixType game_board_state{};                                // 全局维护的棋盘状态
  WEIGHT_T player_for_next_move_on_board = Config::BLACK_STONE; // 全局维护的轮到下棋的玩家
  int processed_requests_count = 0;                             // 已处理的requests数量
  int processed_responses_count = 0;                            // 已处理的responses数量
  // --- 全局游戏状态结束 ---

  int turn_id_counter = 0; // 回合计数器

  while (true) {                         // 主游戏循环
    json response_json;                  // 准备发送给裁判的JSON响应
    std::string line;                    // 用于存储从标准输入读取的行
    if (!std::getline(std::cin, line)) { // 从标准输入读取一行
      break;                             // 如果读取失败 (例如，输入结束)，则退出循环
    }
    if (line.empty()) { // 如果读取的行是空的
      continue;         // 继续下一次循环
    }

    json input_json; // 用于存储解析后的输入JSON
    try {
      input_json = json::parse(line); // 解析JSON字符串
    } catch (json::parse_error &e) {  // 如果JSON解析失败
      json error_response;
      error_response["response"]["x"] = -1;
      error_response["response"]["y"] = -1;
      error_response["debug"] = "解析输入JSON错误: " + std::string(e.what());
      std::cout << error_response.dump() << std::endl; // 发送错误响应
      continue;                                        // 继续下一次循环
    }
    response_json["debug"] = "MCTS Agent C++: 正在处理回合.";

    // --- 同步/增量更新棋盘状态 ---
    int current_total_requests = input_json["requests"].size();
    int current_total_responses = input_json["responses"].size();

    // 检查是否需要完全重建棋盘状态 (例如AI重启或历史记录不匹配)
    // AI即将做出一个响应，所以期望的 processed_responses_count 应该是 current_total_responses
    // 期望的 processed_requests_count 应该是 current_total_requests
    bool needs_full_reconstruction = false;
    if (processed_requests_count != current_total_requests || processed_responses_count != current_total_responses) {
      // 如果AI是开局 (processed_responses_count == 0, current_total_responses == 0)
      // 并且对手也还未下子 (processed_requests_count == 0, current_total_requests == 0), 则不需要重建。
      // 或者AI是白棋，对手下了第一步 (processed_requests_count == 0, current_total_requests == 1,
      // processed_responses_count == 0, current_total_responses == 0)
      // 这种情况下，我们不立即认为需要完全重建，而是先尝试处理单个新request。
      // 只有当计数器已经有值，但与输入JSON不匹配时，才更倾向于完全重建。
      if ((processed_requests_count > 0 || processed_responses_count > 0) || // 如果AI不是刚启动
          (current_total_requests > 1 || current_total_responses > 0)        // 或者历史记录已经不止一步
      ) {
        // 更精确的条件：如果AI的回合数 (由responses决定) 与已处理的responses不符，
        // 或者对手的步数 (由requests决定) 与已处理的requests不符。
        // 并且，这种不符不是因为AI即将下一步棋 (requests 比 responses 多一个)。
        if (current_total_responses != processed_responses_count ||
            (current_total_requests != processed_requests_count &&
             current_total_requests != processed_requests_count + 1)) {
          needs_full_reconstruction = true;
          response_json["debug"] = response_json["debug"].get<std::string>() + " 检测到状态不同步，执行棋盘完全重建.";
        }
      }
    }

    if (needs_full_reconstruction) {
      game_board_state = MatrixType{};                     // 重置棋盘
      player_for_next_move_on_board = Config::BLACK_STONE; // 重新开始计数玩家
      processed_requests_count = 0;
      processed_responses_count = 0;

      for (int i = 0; i < current_total_requests; ++i) {
        if (i < current_total_responses) { // 处理成对的 request 和 response
          if (input_json["requests"][i].count("x") && input_json["requests"][i]["x"].get<int>() != -1) {
            place_stone_on_board(game_board_state, input_json["requests"][i]["x"].get<int>(),
                                 input_json["requests"][i]["y"].get<int>(), player_for_next_move_on_board);
          }
          processed_requests_count++;

          if (input_json["responses"][i].count("x") && input_json["responses"][i]["x"].get<int>() != -1) {
            place_stone_on_board(game_board_state, input_json["responses"][i]["x"].get<int>(),
                                 input_json["responses"][i]["y"].get<int>(), player_for_next_move_on_board);
          }
          processed_responses_count++;
        } else { // 处理最后一个 request (如果 requests 比 responses 多)
          if (input_json["requests"][i].count("x") && input_json["requests"][i]["x"].get<int>() != -1) {
            place_stone_on_board(game_board_state, input_json["requests"][i]["x"].get<int>(),
                                 input_json["requests"][i]["y"].get<int>(), player_for_next_move_on_board);
          }
          processed_requests_count++;
          // 此时 player_for_next_move_on_board 已经是轮到AI了
        }
      }
    }

    // 处理当前回合对手的落子 (如果requests比responses多一个，并且我们没有进行完全重建)
    // 或者说，如果 processed_requests_count < current_total_requests
    Utils::Coordinate last_opponent_action = {-1, -1};
    if (current_total_requests > processed_requests_count) {
      // 对手刚下了一步，这是最新的request
      int new_request_idx = current_total_requests - 1; // 这是正确的，因为 processed_requests_count 是已处理的数量
      if (input_json["requests"][new_request_idx].count("x") &&
          input_json["requests"][new_request_idx]["x"].get<int>() != -1) {
        last_opponent_action = {input_json["requests"][new_request_idx]["x"].get<int>(),
                                input_json["requests"][new_request_idx]["y"].get<int>()};
        place_stone_on_board(game_board_state, last_opponent_action.r, last_opponent_action.c,
                             player_for_next_move_on_board);
        response_json["debug"] = response_json["debug"].get<std::string>() + " 对手落子: (" +
                                 std::to_string(last_opponent_action.r) + "," + std::to_string(last_opponent_action.c) +
                                 ").";
      } else if (input_json["requests"][new_request_idx].count("x") &&
                 input_json["requests"][new_request_idx]["x"].get<int>() == -1 &&
                 input_json["requests"][new_request_idx]["y"].get<int>() == -1) {
        // 对手发送了 -1, -1 (例如换手请求)
        last_opponent_action = {-1, -1};
        response_json["debug"] = response_json["debug"].get<std::string>() + " 对手发送了 (-1,-1).";
        // player_for_next_move_on_board 此时不应改变，因为没有实际落子
      }
      processed_requests_count = current_total_requests; // 更新已处理的requests计数
    } else if (current_total_requests == current_total_responses && current_total_requests == 0 &&
               processed_requests_count == 0) {
      // AI执黑开局，且是第一次处理 (processed_requests_count 为 0)
      response_json["debug"] = response_json["debug"].get<std::string>() + " AI执黑开局.";
    }

    // AI的回合数由 current_total_responses 决定 (AI即将做出第 current_total_responses 个response)
    // turn_id_counter 用于换手逻辑中判断是否是第一/二回合等。
    turn_id_counter = current_total_responses;

    MatrixType current_turn_board_state = game_board_state; // 将当前维护的棋盘状态复制一份用于本回合决策
    WEIGHT_T player_for_next_move_this_turn = player_for_next_move_on_board; // 轮到下棋的玩家
    // --- 同步/增量更新棋盘状态结束 ---

    WEIGHT_T ai_search_color = player_for_next_move_this_turn;

    // --- 换手逻辑 (SWAP LOGIC) ---
    Utils::Coordinate ai_decision_move = {-1, -1}; // AI最终决定的落子，默认为无效/弃权
    bool ai_sends_swap_signal_this_turn = false;   // 标记AI本回合是否因为换手规则发送 (-1,-1)

    if (ai_fixed_color == Config::BLACK_STONE) {                       // 如果AI固定执黑
      if (turn_id_counter == 0 && ai_black_first_move_coord.r == -1) { // AI执黑，游戏的第一步
        // AI作为黑方下第一手棋，MCTS正常运行
        response_json["debug"] = response_json["debug"].get<std::string>() + " AI (黑棋) 下第一手.";
        ai_search_color = Config::BLACK_STONE;
      } else if (turn_id_counter == 1 && !opponent_requested_swap_vs_ai_black) {
        if (last_opponent_action.r == -1 && last_opponent_action.c == -1) {
          if (ai_black_first_move_coord.r != -1) {
            current_turn_board_state[ai_black_first_move_coord.r][ai_black_first_move_coord.c] = Config::WHITE_STONE;
            // game_board_state 也需要同步这个变化，因为它代表真实棋盘
            game_board_state[ai_black_first_move_coord.r][ai_black_first_move_coord.c] = Config::WHITE_STONE;

            opponent_requested_swap_vs_ai_black = true;
            ai_search_color = Config::BLACK_STONE;
            response_json["debug"] = response_json["debug"].get<std::string>() + " AI黑棋: 对手换手. 我在 (" +
                                     std::to_string(ai_black_first_move_coord.r) + "," +
                                     std::to_string(ai_black_first_move_coord.c) +
                                     ") 的第一颗子现在是白棋. 我继续执黑.";
            // 此时 player_for_next_move_on_board 应该已经是 BLACK (AI的回合)
            // 因为对手发送-1,-1时，place_stone_on_board 没有执行，颜色不变。
            player_for_next_move_this_turn = Config::BLACK_STONE; // 确保本回合搜索颜色正确
          } else {
            response_json["debug"] =
                response_json["debug"].get<std::string>() + " 错误: 对手请求换手，但AI黑棋的第一步未被记录.";
          }
        } else { // 对手正常落子
          ai_search_color = Config::BLACK_STONE;
        }
      } else { // 其他情况
        ai_search_color = Config::BLACK_STONE;
      }
    } else { // AI固定执白
      if (turn_id_counter == 0 && !ai_white_performed_swap) {
        if (last_opponent_action.r != -1) { // 黑棋下了有效的第一步
          current_turn_board_state[last_opponent_action.r][last_opponent_action.c] = Config::WHITE_STONE;
          game_board_state[last_opponent_action.r][last_opponent_action.c] = Config::WHITE_STONE;

          // AI执白，对手(黑)下了第一步后，player_for_next_move_on_board 经place_stone_on_board变为WHITE (AI的回合)
          // AI现在执行换手，将黑子变白，然后发送-1,-1，回合交还给黑方。
          player_for_next_move_on_board = Config::BLACK_STONE;  // 全局状态更新：换手后轮到黑方
          player_for_next_move_this_turn = Config::BLACK_STONE; // AI发送-1,-1后，也轮到黑方 (MCTS不应搜索)

          ai_sends_swap_signal_this_turn = true;
          ai_white_performed_swap = true;
          response_json["debug"] = response_json["debug"].get<std::string>() + " AI白棋: 对黑棋在 (" +
                                   std::to_string(last_opponent_action.r) + "," +
                                   std::to_string(last_opponent_action.c) +
                                   ") 的落子执行换手. 发送 -1,-1. 现在轮到黑棋.";
        } else {
          response_json["debug"] =
              response_json["debug"].get<std::string>() + " 错误: AI执白，但黑棋的第一步是-1,-1或缺失.";
          ai_decision_move = {0, 0}; // 后备错误走法
        }
      } else {
        // AI执白，且不是换手回合，所以AI应该搜索为白棋
        ai_search_color = Config::WHITE_STONE;
      }
    }

    // MCTS agent的根节点使用 current_turn_board_state
    agent.root = std::make_shared<Node>(nullptr, 1.0f, ai_search_color, current_turn_board_state, -1);
    if (ai_search_color == Config::BLACK_STONE)
      agent.active_net = &agent.net_b;
    else
      agent.active_net = &agent.net_w;

    if (ai_sends_swap_signal_this_turn) {
      ai_decision_move = {-1, -1};
    } else {
      // ... (MCTS 搜索逻辑，与之前类似, 使用 agent.root->board_state 即 current_turn_board_state) ...
      // 确保这里的调试警告逻辑仍然合理
      if (ai_fixed_color != agent.root->color_to_play &&
          !(ai_fixed_color == Config::BLACK_STONE && opponent_requested_swap_vs_ai_black) &&
          !(ai_fixed_color == Config::WHITE_STONE && ai_white_performed_swap && turn_id_counter == 0)) {
        response_json["debug"] = response_json["debug"].get<std::string>() + " 警告: MCTS搜索颜色 (" +
                                 std::to_string(agent.root->color_to_play) + ") 与AI固定颜色 (" +
                                 std::to_string(ai_fixed_color) + ") 不一致，可能在换手后出现.";
      }

      // 执行指定次数的MCTS模拟
      for (int sim = 0; sim < Config::SIMULATION_TIMES; ++sim) {
        agent.run_one_simulation();
      }

      int best_action_idx = -1;  // 最佳走法的索引
      long long max_visits = -1; // 最大访问次数
      std::vector<int> legal_moves_for_root =
          Utils::board_to_legal_vec(agent.root->board_state); // 获取根节点的合法走法
      bool found_move = false;                                // 是否找到了走法

      // 从根节点的子节点中选择访问次数最多的合法走法
      for (int i = 0; i < Config::BOARD_SQUARES; ++i) {
        if (agent.root->children[i] && legal_moves_for_root[i]) { // 如果子节点存在且该走法合法
          if (agent.root->children[i]->visit_count > max_visits) {
            max_visits = agent.root->children[i]->visit_count;
            best_action_idx = i;
            found_move = true;
          }
        }
      }

      // 后备逻辑1: 如果MCTS没有找到被访问过的子节点，但存在合法走法，则选择第一个合法走法
      if (!found_move &&
          std::any_of(legal_moves_for_root.begin(), legal_moves_for_root.end(), [](int x) { return x == 1; })) {
        for (int i = 0; i < Config::BOARD_SQUARES; ++i) {
          if (legal_moves_for_root[i]) {
            best_action_idx = i;
            response_json["debug"] =
                response_json["debug"].get<std::string>() + " 后备: MCTS未找到已访问的子节点，选择第一个合法走法.";
            break;
          }
        }
        if (best_action_idx != -1)
          found_move = true;
      }

      if (found_move) {                                                                     // 如果找到了走法
        ai_decision_move = Utils::index_to_coordinate(best_action_idx, Config::BOARD_SIZE); // 将索引转换为坐标
        if (ai_fixed_color == Config::BLACK_STONE && turn_id_counter == 0) {                // 如果AI执黑且是第一步
          ai_black_first_move_coord = ai_decision_move;                                     // 记录第一步的落子
        }
      } else { // 如果通过MCTS或后备1仍未找到走法
        response_json["debug"] = response_json["debug"].get<std::string>() + " MCTS或后备未能找到合法走法.";
        // 后备逻辑2: 尝试寻找棋盘上任何一个合法的走法
        bool any_legal_exists = false;
        for (int i = 0; i < Config::BOARD_SQUARES; ++i) {
          if (legal_moves_for_root[i]) {
            ai_decision_move = Utils::index_to_coordinate(i, Config::BOARD_SIZE);
            response_json["debug"] = response_json["debug"].get<std::string>() +
                                     " 紧急后备: 选择棋盘上第一个可用的合法走法，索引 " + std::to_string(i);
            any_legal_exists = true;
            break;
          }
        }
        if (!any_legal_exists) {       // 如果棋盘上确实没有任何合法走法
          ai_decision_move = {-1, -1}; // 决策为弃权
          response_json["debug"] = response_json["debug"].get<std::string>() + " 棋盘上不存在合法走法.";
        }
      }
    }

    // 如果AI本回合实际落子了 (不是发送-1,-1的换手信号)
    if (!ai_sends_swap_signal_this_turn && ai_decision_move.r != -1) {
      // 更新全局棋盘状态和下一个玩家
      // place_stone_on_board 已经在上面处理对手棋步时或重建时更新了 player_for_next_move_on_board
      // 这里是AI落子，所以 player_for_next_move_on_board 会再次切换
      place_stone_on_board(game_board_state, ai_decision_move.r, ai_decision_move.c, player_for_next_move_on_board);
      processed_responses_count++; // AI 做出了一次有效响应
    } else if (ai_sends_swap_signal_this_turn) {
      // AI发送了换手信号，也算一次响应
      processed_responses_count++;
    } else if (ai_decision_move.r == -1 && !ai_sends_swap_signal_this_turn) {
      // AI没有有效落子 (例如棋盘已满或MCTS未能找到走法)，但也算作一次响应
      // 这种情况通常意味着游戏结束或出现问题，player_for_next_move_on_board 可能不需要改变，
      // 或者如果游戏规则允许“pass”，则会改变。这里我们假设不改变。
      processed_responses_count++;
      response_json["debug"] = response_json["debug"].get<std::string>() + " AI本回合无有效落子(非换手).";
    }

    response_json["response"]["x"] = ai_decision_move.r; // 设置响应中的x坐标
    response_json["response"]["y"] = ai_decision_move.c; // 设置响应中的y坐标
    std::cout << response_json.dump() << std::endl;      // 将JSON响应打印到标准输出
  }

  return 0; // 程序正常结束
}
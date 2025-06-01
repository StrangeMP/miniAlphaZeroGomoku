#include "compute.hpp"
#include "json.hpp"
#include "network.hpp"
#include "qnnet.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
using json = nlohmann::json; // 在全局作用域或 main 之前使用 using 声明

// --- 配置参数
namespace Config {
constexpr int BOARD_SIZE = 15; // 棋盘大小
constexpr int BOARD_SQUARES = BOARD_SIZE * BOARD_SIZE;

// MCTS 超参数
constexpr float C_PUCT = 5.0f;           // PUCT 常数 (已改为 float 类型)
constexpr int SIMULATION_TIMES = 100;    // 模拟次数
//constexpr float INITIAL_TAU = 1.0f;      // 初始 tau (已改为 float 类型)
//constexpr float TAU_DECAY = 0.8f;        // tau 衰减 (来自 config.py 顶层) (已改为 float 类型)
//constexpr int VIRTUAL_LOSS = 10;         // 虚拟损失
//constexpr int CAREFUL_STAGE = 6;         // 谨慎阶段

// NN 配置 (新增自 usenet.cpp)
constexpr int NN_INPUT_CHANNELS = 3; // NN 输入张量的通道数
constexpr int NN_RES_FILTERS = 64;   // 示例: 残差块中的滤波器数量
constexpr int NN_POLICY_FILTERS = 2; // 示例: 策略头卷积层中的滤波器数量
constexpr int NN_VALUE_FILTERS = 1;  // 示例: 价值头卷积层中的滤波器数量
constexpr int NN_RES_BLOCKS = 5;     // 示例: ResNet 中的残差块数量

// 棋子表示
auto EMPTY_STONE = AlphaGomoku::EMPTY;
auto BLACK_STONE = AlphaGomoku::BLACK;
auto WHITE_STONE = AlphaGomoku::WHITE;
} // namespace Config

// 类型别名，用于 NN 和棋盘 (新增自 usenet.cpp)
// 假设 Matrix, Tensor, AZNet 在包含的 compute.hpp/nnet.hpp 中定义
using MatrixType = Matrix<AlphaGomoku::STONE_COLOR, Config::BOARD_SIZE, Config::BOARD_SIZE>;
// 假设 Tensor 有一个构造函数或方法来访问通道，例如 tensor.channel(i) 或 tensor.data_for_channel(i)
// 目前，如果它是一个类似 3D 数组的结构或类似的，我们将假设直接访问，如 tensor[channel_idx]。
// 如果 compute.hpp 对 Tensor 的定义不同，这部分需要匹配该定义。
using InputTensor =
    Tensor<WEIGHT_T, Config::NN_INPUT_CHANNELS, Config::BOARD_SIZE, Config::BOARD_SIZE>; // NN 输入是 float 类型
using ActualNNType = AlphaGomoku::GodNet;

// --- 实用函数 (来自 utils.py) ---
namespace Utils {
struct Coordinate {
  int r, c;
};

Coordinate index_to_coordinate(int index, int board_size) { return {index / board_size, index % board_size}; }

int coordinate_to_index(Coordinate coord, int board_size) { return coord.r * board_size + coord.c; }

int coordinate_to_index(int r, int c, int board_size) { return r * board_size + c; }

// 生成合法走子向量 (1 表示合法, 0 表示非法)
std::array<int, Config::BOARD_SQUARES> board_to_legal_vec(const MatrixType &board) { // 已改为 MatrixType 和 std::array
  std::array<int, Config::BOARD_SQUARES> legal_vec{}; // 使用 {} 初始化为0
  for (int i = 0; i < Config::BOARD_SQUARES; ++i) {
    Coordinate coord = index_to_coordinate(i, Config::BOARD_SIZE);
    if (board[coord.r][coord.c] == Config::EMPTY_STONE) {
      legal_vec[i] = 1;
    }
  }
  return legal_vec;
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
  AlphaGomoku::STONE_COLOR color_to_play;
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

  Node(std::shared_ptr<Node> parent_node, float p, AlphaGomoku::STONE_COLOR turn, const MatrixType &current_board,
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
                                                     const std::array<int, Config::BOARD_SQUARES> &legal_moves_vec) {
    std::shared_ptr<Node> best_child = nullptr;
    int best_action_idx = -1;
    float max_score = -std::numeric_limits<float>::infinity();

    int current_node_total_visits = this->visit_count;

    for (int i = 0; i < Config::BOARD_SQUARES; ++i) {
      if (!legal_moves_vec[i])
        continue;

      float score;
      if (children[i]) { // 如果子节点存在
        score = children[i]->get_q_value() + children[i]->get_u_value(c_puct, current_node_total_visits);
      } else { // 如果子节点不存在 (来自已扩展父节点的潜在走子)
        if (is_expanded && !raw_nn_policy_for_children.empty()) {
          // 确保索引 i 在 raw_nn_policy_for_children 的有效范围内
          if (i < raw_nn_policy_for_children.size()) {
             float prior_p_for_action_i = raw_nn_policy_for_children[i];
             float u_value_for_action_i =
                c_puct * prior_p_for_action_i * std::sqrt(static_cast<float>(current_node_total_visits)) / (1.0f + 0);
             score = 0.0f + u_value_for_action_i;
          } else {
            // 如果索引越界，则不应发生，跳过此动作
            continue;
          }
        } else {
          // 如果父节点未扩展或策略为空，则无法计算此潜在走子的分数，跳过。
          continue;
        }
      }

      if (score > max_score) {
        max_score = score;
        best_child = children[i]; 
        best_action_idx = i;
      }
    }

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

// --- MCTS 类 (来自 mcts.py) ---
class MCTS_Agent {
public:
  std::shared_ptr<Node> root;
  AlphaGomoku::STONE_COLOR agent_color;       // AI 的执棋颜色
  //float current_tau;        // 已改为 float 类型
  AlphaGomoku::Network net;
  MCTS_Agent(AlphaGomoku::STONE_COLOR player_color) : agent_color(player_color), net() {
    MatrixType initial_board{}; // 零初始化
    root = std::make_shared<Node>(nullptr, 1.0f, Config::BLACK_STONE, initial_board, -1);
    //update_active_net(); 
  }

  void reset_tree_to_board(const MatrixType& board_state, AlphaGomoku::STONE_COLOR player_to_move) { 
    root = std::make_shared<Node>(nullptr, 1.0f, player_to_move, board_state, -1);
    //current_tau = Config::INITIAL_TAU;
  }

  void update_root_after_move(int action_idx, const MatrixType& new_board_state, AlphaGomoku::STONE_COLOR next_player_color) {
    if (action_idx >= 0 && action_idx < Config::BOARD_SQUARES && root->children[action_idx]) {
      std::shared_ptr<Node> chosen_child = root->children[action_idx];
      chosen_child->parent = nullptr; 
      chosen_child->board_state = new_board_state; 
      chosen_child->color_to_play = next_player_color;
      root = chosen_child; 
    } else {
      reset_tree_to_board(new_board_state, next_player_color);
    }
  }

  // predict_nn 已移除，mcts_expand_and_evaluate 处理 NN 调用。

  // 检查游戏是否结束 
  // 返回 GameEndStatus: {is_end, outcome_value for last_player_color}
  // last_action_coord: 导致当前棋盘状态的走子坐标
  // last_player_color: 做出该走子的玩家
  GameEndStatus check_game_end(const MatrixType &board, Utils::Coordinate last_action_coord, AlphaGomoku::STONE_COLOR last_player_color) {
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
  void run_one_simulation(); // 保留现有声明

  // 将 mcts_expand_and_evaluate 声明为成员函数
  float mcts_expand_and_evaluate(std::shared_ptr<Node> node_to_expand);

}; // MCTS_Agent 类

// --- MCTS_Agent 方法实现 ---

// 执行一次 MCTS 模拟 (选择、扩展、评估、反向传播)
void MCTS_Agent::run_one_simulation() {
    std::shared_ptr<Node> current_node = root;

    // 1. 选择
    while (current_node->is_expanded && !current_node->is_end_node) {
      std::array<int, Config::BOARD_SQUARES> legal_moves = Utils::board_to_legal_vec(current_node->board_state); // 已改为 std::array
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
        AlphaGomoku::STONE_COLOR next_player_color =
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
      leaf_value = this->mcts_expand_and_evaluate(current_node); // 新的调用方式
    }
    // 3. 反向传播
    // mcts_expand_and_evaluate 返回的价值是从 current_node->color_to_play 的角度看的
    current_node->backup(leaf_value);
  }

// 函数：使用神经网络扩展叶节点，创建其子节点，
// 并返回节点的评估值。
float MCTS_Agent::mcts_expand_and_evaluate( // 新的成员函数定义
    std::shared_ptr<Node> node_to_expand) {
  AlphaGomoku::STONE_COLOR current_player_at_node = node_to_expand->color_to_play;

  Utils::Coordinate last_action_coord = {-1, -1};
  AlphaGomoku::STONE_COLOR player_who_made_last_move = Config::EMPTY_STONE;
  bool is_last_action_valid_for_nn = false; 

  if (node_to_expand->parent) {
    // 如果有父节点，则导致此节点的动作是由父节点的玩家做出的
    player_who_made_last_move = node_to_expand->parent->color_to_play;
    if (node_to_expand->action_idx_that_led_to_this_node != -1) {
      last_action_coord =
          Utils::index_to_coordinate(node_to_expand->action_idx_that_led_to_this_node, Config::BOARD_SIZE);
      is_last_action_valid_for_nn = true; 
    }
  }
  // 否则，如果是根节点，则没有“上一步操作”导致它。

  // GameEndStatus end_status = check_game_end_func(node_to_expand->board_state, last_action_coord, player_who_made_last_move); // 旧的调用方式
  GameEndStatus end_status = this->check_game_end(node_to_expand->board_state, last_action_coord, player_who_made_last_move); // 直接调用成员函数

  if (end_status.is_end) {
    node_to_expand->is_expanded = true; // 技术上已“评估”，即使未通过 NN
    node_to_expand->is_end_node = true;
    // end_status.outcome_value 是从 player_who_made_last_move 的角度看的。
    // 我们需要将其转换为 node_to_expand->color_to_play (当前轮到的玩家) 的角度。
    
    // player_who_made_last_move 是对手，所以 outcome_value 是对手的价值。
    // 当前玩家的价值是 -outcome_value。
    node_to_expand->game_result_if_end = -end_status.outcome_value;
    
    if (player_who_made_last_move == Config::EMPTY_STONE &&
        end_status.outcome_value == 0.0f) { // 例如，棋盘已满平局，没有最后走棋的玩家
      node_to_expand->game_result_if_end = 0.0f;
    }
    return node_to_expand->game_result_if_end;
  }

  // 1. 准备神经网络输入并进行推理
  std::optional<std::pair<WEIGHT_T, WEIGHT_T>> last_move_for_nn_feed_typed = std::nullopt;
  if (is_last_action_valid_for_nn) {
      last_move_for_nn_feed_typed = std::make_pair(
          static_cast<WEIGHT_T>(last_action_coord.r),
          static_cast<WEIGHT_T>(last_action_coord.c)
      );
  }

  // 2. 通过 AlphaGomoku::Network 进行推理
  // auto nn_output_ref_pair = agent_network.feed( // 旧的调用方式
  auto nn_output_ref_pair = this->net.feed( // 直接使用成员变量 net
      node_to_expand->board_state,
      last_move_for_nn_feed_typed, // 使用转换后的类型
      current_player_at_node // current_player_at_node 已经是 STONE_COLOR
  );
  
  // RetType 是 std::pair<Vec<SCALE_T, BOARD_SIZE * BOARD_SIZE> &, SCALE_T &>
  // 我们需要将引用内容复制到节点的成员中。
  node_to_expand->raw_nn_policy_for_children = nn_output_ref_pair.first;
  node_to_expand->raw_nn_value = nn_output_ref_pair.second;
  float value_from_nn = nn_output_ref_pair.second; // 从 current_player_at_node 的角度

  // 3. 标记节点为已扩展
  node_to_expand->is_expanded = true;

  // 4. (可选) 创建子节点 - 当前实现在选择阶段按需创建
  // 这里不需要在扩展时创建子节点，因为它们将在选择阶段被创建。

  // 5. 返回从 NN 获得的评估值
  // 该值是从 current_player_at_node (即 node_to_expand->color_to_play) 的角度看的。
  return value_from_nn;
}

// Helper to place stone and switch player, used in board reconstruction
// 用于棋盘重建的辅助函数，放置棋子并切换玩家
void place_stone_on_board(MatrixType &board, int r, int c, AlphaGomoku::STONE_COLOR &current_player_color) {
  if (r >= 0 && r < Config::BOARD_SIZE && c >= 0 && c < Config::BOARD_SIZE) {
    if (board[r][c] == Config::EMPTY_STONE) { // 仅在空位时落子
      board[r][c] = current_player_color;
      current_player_color = (current_player_color == Config::BLACK_STONE) ? Config::WHITE_STONE : Config::BLACK_STONE;
    }
  }
}

MCTS_Agent agent(Config::BLACK_STONE);
int main() {
  std::ios_base::sync_with_stdio(false); 
  std::cin.tie(NULL);                    

  Utils::Coordinate ai_black_first_move_coord = {-1, -1}; 
  bool opponent_requested_swap_vs_ai_black = false;       
  bool ai_white_performed_swap = false;                   

  MatrixType game_board_state{};                              
  AlphaGomoku::STONE_COLOR player_for_next_move_on_board = Config::BLACK_STONE; 
  int processed_requests_count = 0;                           
  int processed_responses_count = 0;                          
  int turn_id_counter = 0; 

  while (true) {                         
    json response_json;                  
    std::string line;                    
    if (!std::getline(std::cin, line)) { 
      break;                             
    }
    if (line.empty()) { 
      continue;         
    }

    json input_json; 
    try {
        input_json = json::parse(line); 
    } catch (const json::parse_error& e) {
        // std::cerr << "JSON parse error: " << e.what() << std::endl;
        continue; // Or handle error appropriately
    }
    
    int current_total_requests = 0;
    if (input_json.contains("requests") && input_json["requests"].is_array()) {
        current_total_requests = input_json["requests"].size();
    }
    int current_total_responses = 0;
    if (input_json.contains("responses") && input_json["responses"].is_array()){
        current_total_responses = input_json["responses"].size();
    }


    Utils::Coordinate last_opponent_action_coord = {-2, -2}; 
    int last_opponent_action_idx = -1; 

    if (current_total_requests > processed_requests_count) {
      int new_request_idx = current_total_requests - 1; 
      if (input_json["requests"][new_request_idx].count("x") &&
          input_json["requests"][new_request_idx]["x"].get<int>() != -1) {
        last_opponent_action_coord = {input_json["requests"][new_request_idx]["x"].get<int>(),
                                input_json["requests"][new_request_idx]["y"].get<int>()};
        last_opponent_action_idx = Utils::coordinate_to_index(last_opponent_action_coord, Config::BOARD_SIZE);
        place_stone_on_board(game_board_state, last_opponent_action_coord.r, last_opponent_action_coord.c,
                             player_for_next_move_on_board);
        agent.update_root_after_move(last_opponent_action_idx, game_board_state, static_cast<AlphaGomoku::STONE_COLOR>(player_for_next_move_on_board));

      } else if (input_json["requests"][new_request_idx].count("x") &&
                 input_json["requests"][new_request_idx]["x"].get<int>() == -1 &&
                 input_json["requests"][new_request_idx]["y"].get<int>() == -1) {
        last_opponent_action_coord = {-1, -1};
      }
      processed_requests_count = current_total_requests;
    }

    turn_id_counter = current_total_responses;
    AlphaGomoku::STONE_COLOR player_for_next_move_this_turn = player_for_next_move_on_board;
    Utils::Coordinate ai_decision_move = {-2, -2};
    bool ai_sends_swap_signal_this_turn = false;  
    bool agent_color_changed_this_turn = false; // Renamed for clarity

    if (agent.agent_color == Config::BLACK_STONE && 
        turn_id_counter == 0 &&                
        last_opponent_action_coord.r != -1 &&        
        last_opponent_action_coord.c != -1 &&
        !opponent_requested_swap_vs_ai_black && 
        !ai_white_performed_swap) {             
        agent.agent_color = Config::WHITE_STONE;
        // agent.update_active_net(); 
        agent_color_changed_this_turn = true; 
    }

    if (!opponent_requested_swap_vs_ai_black && turn_id_counter == 1) { 
            if (last_opponent_action_coord.r == -1 && last_opponent_action_coord.c == -1) { 
                if (ai_black_first_move_coord.r != -1 && ai_black_first_move_coord.c != -1) { 
                    agent.agent_color = Config::WHITE_STONE; 
                    // agent.update_active_net(); 
                    opponent_requested_swap_vs_ai_black = true; 
                    player_for_next_move_this_turn = Config::WHITE_STONE; 
                    player_for_next_move_on_board = Config::WHITE_STONE; 
                    agent_color_changed_this_turn = true;
                }
            }
    } else if (agent.agent_color == Config::WHITE_STONE && !ai_white_performed_swap) { 
        if (turn_id_counter == 0 && player_for_next_move_this_turn == Config::WHITE_STONE) { 
            if (last_opponent_action_coord.r != -1 && last_opponent_action_coord.c != -1) { 
                ai_sends_swap_signal_this_turn = true; 
                ai_decision_move = {-1, -1}; 
                agent.agent_color = Config::BLACK_STONE;    
                // agent.update_active_net(); 
                ai_white_performed_swap = true;          
                player_for_next_move_on_board = Config::WHITE_STONE;
                agent_color_changed_this_turn = true; 
            }
         }
    }
        
    if (agent_color_changed_this_turn) {
        agent.reset_tree_to_board(game_board_state, player_for_next_move_on_board);
    }

    int best_action_idx_for_ai = -1; // Store AI's chosen action index

    if (!ai_sends_swap_signal_this_turn) { 
      //保底检查（根节点是不是当前棋盘状态）
      // 如果 agent.root 已经是当前棋盘状态，则不需要重置。 
      /*  
      if (agent.root->board_state != game_board_state || agent.root->color_to_play != player_for_next_move_this_turn) {
            agent.reset_tree_to_board(game_board_state, static_cast<AlphaGomoku::STONE_COLOR>(player_for_next_move_this_turn));
        }
      */
        for (int sim = 0; sim < Config::SIMULATION_TIMES; ++sim) {
            agent.run_one_simulation();
        }

        long long max_visits = -1; 
        std::array<int, Config::BOARD_SQUARES> legal_moves_for_root =
            Utils::board_to_legal_vec(agent.root->board_state); 
        bool found_move = false;                               

        for (int i = 0; i < Config::BOARD_SQUARES; ++i) {
            if (agent.root->children[i] && legal_moves_for_root[i]) { 
                if (agent.root->children[i]->visit_count > max_visits) {
                    max_visits = agent.root->children[i]->visit_count;
                    best_action_idx_for_ai = i; // Use the AI's action index variable
                    found_move = true;
                }
            }
        }
      
        if (found_move) { 
            ai_decision_move = Utils::index_to_coordinate(best_action_idx_for_ai, Config::BOARD_SIZE);
            if (agent.agent_color == Config::BLACK_STONE && turn_id_counter == 0 && !opponent_requested_swap_vs_ai_black && !ai_white_performed_swap) {
                ai_black_first_move_coord = ai_decision_move;
            }
        } else { 
            if (best_action_idx_for_ai == -1) { // 确保如果上面没有找到移动，这里也不会意外地有一个旧值
                 ai_decision_move = {-1, -1}; // 明确表示没有选择移动
            }
        }
    
        if (ai_decision_move.r != -1) {
            place_stone_on_board(game_board_state, ai_decision_move.r, ai_decision_move.c, player_for_next_move_on_board);
            agent.update_root_after_move(best_action_idx_for_ai, game_board_state, static_cast<AlphaGomoku::STONE_COLOR>(player_for_next_move_on_board));
        }
    }

    processed_responses_count++; 

    response_json["response"]["x"] = ai_decision_move.r; 
    response_json["response"]["y"] = ai_decision_move.c; 
    //std::cout << response_json.dump() << std::endl; 
    std::cout << ">>>BOTZONE_REQUEST_KEEP_RUNNING<<<" << std::endl; // Uncomment if required by platform
    std::cout << std::flush; // Ensure output is sent
  }

  return 0; 
}
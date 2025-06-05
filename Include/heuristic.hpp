#pragma once
#include "config.hpp"
#include "mcts.hpp"
#include <sstream>

// 检查指定位置是否有活三或冲五模式
// 返回值: 0-无特殊模式, 1-活三, 2-冲五
inline int check_threat_pattern(const Board &board, int r, int c, AlphaGomoku::STONE_COLOR stone_color) {
  if (board[r][c] != Config::EMPTY_STONE) {
    return 0; // 位置已经有棋子，无法形成威胁
  }

  // 临时放置一个棋子来测试威胁
  Board test_board = board;
  test_board[r][c] = stone_color;

  const int B_SIZE = Config::BOARD_SIZE;
  const int STONES_TO_WIN = 5;

  // 方向：水平、垂直、对角线(左上到右下)、反对角线(右上到左下)
  const int dr[] = {0, 1, 1, 1};  // 行增量
  const int dc[] = {1, 0, 1, -1}; // 列增量

  for (int dir = 0; dir < 4; ++dir) {
    // 检查每个方向的连续棋子数量和空位
    int total_length = 1; // 包括刚放的棋子
    int open_ends = 0;    // 开放端数量

    // 正方向检查
    int pos_count = 0;
    for (int k = 1; k < STONES_TO_WIN; ++k) {
      int nr = r + k * dr[dir];
      int nc = c + k * dc[dir];
      if (nr >= 0 && nr < B_SIZE && nc >= 0 && nc < B_SIZE) {
        if (test_board[nr][nc] == stone_color) {
          pos_count++;
          total_length++;
        } else if (test_board[nr][nc] == Config::EMPTY_STONE) {
          open_ends++;
          break;
        } else {
          break; // 遇到对方棋子
        }
      } else {
        break; // 出界
      }
    }

    // 负方向检查
    int neg_count = 0;
    for (int k = 1; k < STONES_TO_WIN; ++k) {
      int nr = r - k * dr[dir];
      int nc = c - k * dc[dir];
      if (nr >= 0 && nr < B_SIZE && nc >= 0 && nc < B_SIZE) {
        if (test_board[nr][nc] == stone_color) {
          neg_count++;
          total_length++;
        } else if (test_board[nr][nc] == Config::EMPTY_STONE) {
          open_ends++;
          break;
        } else {
          break; // 遇到对方棋子
        }
      } else {
        break; // 出界
      }
    }

    // 判断威胁类型
    if (total_length == 5) {
      return 2; // 冲五 (_oooo/oooo_/oo_oo/o_ooo/ooo_o)
    }
    if (total_length == 4 && open_ends == 2) {
      return 1; // 活三 (_ooo_/_o_oo_/_oo_o_)
    }
  }

  return 0; // 没有找到威胁
}

// 检查棋盘上给定颜色的所有可能威胁
// 返回值: 包含所有威胁位置和类型的向量
inline std::vector<std::pair<Utils::Coordinate, int>> find_all_threats(const Board &board,
                                                                       AlphaGomoku::STONE_COLOR stone_color) {
  std::vector<std::pair<Utils::Coordinate, int>> threats;

  for (int r = 0; r < Config::BOARD_SIZE; ++r) {
    for (int c = 0; c < Config::BOARD_SIZE; ++c) {
      if (board[r][c] == Config::EMPTY_STONE) {
        int threat_type = check_threat_pattern(board, r, c, stone_color);
        if (threat_type > 0) {
          threats.push_back({{r, c}, threat_type});
        }
      }
    }
  }

  return threats;
}

// Helper function to find the best legal action for a specific threat type
// Returns std::optional<int> containing the action_idx if a legal move is found.
// Logs the decision process to debug_stream.
static std::optional<int> find_best_legal_action_for_threat_type(
    const std::vector<std::pair<Utils::Coordinate, int>>& threats,
    int target_threat_type,
    const std::array<bool, Config::BOARD_SQUARES>& legal_moves,
    const MCTS_Agent& mcts_agent,
    std::ostringstream& debug_stream,
    const std::string& log_success_prefix,
    const std::string& log_illegal_suffix_part) {

    int best_action_candidate_idx = -1;
    int current_max_visits = -1;

    for (const auto& threat : threats) {
        if (threat.second == target_threat_type) {
            int threat_idx = Utils::coordinate_to_index(threat.first);
            
            // Consider only if the threat index is valid and MCTS has a child node for it
            if (threat_idx >= 0 && threat_idx < Config::BOARD_SQUARES &&
                mcts_agent.root && mcts_agent.root->children[threat_idx]) { // Assuming children is array-like and index is checked
                int visits = mcts_agent.root->children[threat_idx]->visit_count;
                if (best_action_candidate_idx == -1 || visits > current_max_visits) {
                    current_max_visits = visits;
                    best_action_candidate_idx = threat_idx;
                }
            }
        }
    }

    if (best_action_candidate_idx != -1) {
        Utils::Coordinate coord = Utils::index_to_coordinate(best_action_candidate_idx);
        if (legal_moves[best_action_candidate_idx]) {
            debug_stream << log_success_prefix << " (" << coord.first << "," << coord.second << "); ";
            return best_action_candidate_idx;
        } else {
            debug_stream << log_success_prefix << " (" << coord.first << "," << coord.second << ")"
                         << log_illegal_suffix_part << "; ";
            return std::nullopt; // Found a candidate, but it's illegal
        }
    }
    return std::nullopt; // No suitable threat of this type found or none with MCTS data
}

// 启发式决策函数：根据当前棋盘状态、威胁分析和MCTS结果做出最优决策
// 返回决定的动作索引和调试信息
inline std::pair<int, std::string>
make_heuristic_decision(const Board &board_state, AlphaGomoku::STONE_COLOR agent_color,
                        AlphaGomoku::STONE_COLOR enemy_color, int mcts_best_action_idx,
                        const std::array<bool, Config::BOARD_SQUARES> &legal_moves, const MCTS_Agent &mcts_agent) {

  std::ostringstream debug_info;
  debug_info << "MCTS_Agent decision: (" << Utils::index_to_coordinate(mcts_best_action_idx).first << ", "
             << Utils::index_to_coordinate(mcts_best_action_idx).second << "); ";

  int final_action_idx = mcts_best_action_idx; // 默认使用MCTS找到的最佳点
  bool heuristic_move_made = false;

  auto my_threats = find_all_threats(board_state, agent_color);
  auto enemy_threats = find_all_threats(board_state, enemy_color);

  // 优先级一：我方有冲五，则选择使用胜率大的点进攻
  std::optional<int> p1_action = find_best_legal_action_for_threat_type(
      my_threats, 2, legal_moves, mcts_agent, debug_info,
      "[DEBUG] 选择我方冲五", "，但该点不合法");
  if (p1_action) {
      final_action_idx = *p1_action;
      heuristic_move_made = true;
  }

  // 优先级二：如果对方有冲五，必须防守 (only if P1 didn't result in a move)
  if (!heuristic_move_made) {
      std::optional<int> p2_action = find_best_legal_action_for_threat_type(
          enemy_threats, 2, legal_moves, mcts_agent, debug_info,
          "[DEBUG] 选择防守对方冲五", "，但该点不合法");
      if (p2_action) {
          final_action_idx = *p2_action;
          heuristic_move_made = true;
      }
  }

  // 优先级三：如果我方有活三，尝试进攻 (only if P1 & P2 didn't result in a move)
  if (!heuristic_move_made) {
      std::optional<int> p3_action = find_best_legal_action_for_threat_type(
          my_threats, 1, legal_moves, mcts_agent, debug_info,
          "[DEBUG] 选择我方活三进攻", "，但该点不合法");
      if (p3_action) {
          final_action_idx = *p3_action;
          heuristic_move_made = true;
      }
  }

  // 优先级四：如果对方有活三，则选择胜率大的点防守 (only if P1, P2, P3 didn't result in a move)
  if (!heuristic_move_made) {
      std::optional<int> p4_action = find_best_legal_action_for_threat_type(
          enemy_threats, 1, legal_moves, mcts_agent, debug_info,
          "[DEBUG] 选择防守对方活三", "，但该点不合法");
      if (p4_action) {
          final_action_idx = *p4_action;
          heuristic_move_made = true;
      }
  }

  // 优先级五：敌我均不能将军，正常下棋（使用MCTS的结果）
  if (!heuristic_move_made && final_action_idx == mcts_best_action_idx) {
      // This log indicates that MCTS default was used because no heuristic override was successful.
      // The specific reasons (e.g. "tried X but illegal") would have been logged by the helper.
      debug_info << "[DEBUG] 无特殊威胁可利用或均不合法，使用MCTS最佳点; ";
  }

  // 添加威胁信息到调试输出
  if (!my_threats.empty()) {
    debug_info << " 我方威胁: ";
    for (const auto &threat : my_threats) {
      debug_info << "(" << threat.first.first << "," << threat.first.second << ":"
                 << (threat.second == 1 ? "活三" : "冲五") << ") ";
    }
  }

  if (!enemy_threats.empty()) {
    debug_info << " 对方威胁: ";
    for (const auto &threat : enemy_threats) {
      debug_info << "(" << threat.first.first << "," << threat.first.second << ":"
                 << (threat.second == 1 ? "活三" : "冲五") << ") ";
    }
  }

  return {final_action_idx, debug_info.str()};
}
/*
 * 打谱工具GomokuGameRecord 类方法说明：
 * 自动在构造时自动调取系统时间设置为开局时间，不需要手动设置
 *
 * 构造函数：
 * - GomokuGameRecord() - 默认构造函数，使用默认信息初始化
 * - GomokuGameRecord(black, white) - 使用指定信息初始化
 *
 * 游戏信息设置：
 * - setGameInfo(black, white) - 设置对局双方，黑方总是先手
 * - setResult(result_code) - 设置游戏结果：1=先手胜，2=后手胜
 * - setEventInfo(event_name, place) - 设置比赛信息和地点 - 默认已经设好了不需要调用这个
 *
 * 落子操作：
 * - addMove(color, index) - 使用index添加落子
 * - addMove(color, x, y) - 使用坐标添加落子
 *
 * 撤销重做：
 * - undoLastMove() - (悔棋）撤销上一步，返回是否成功
 * - redoLastMove() - （撤销悔棋）重做上一步，返回是否成功
 *
 * 数据管理：
 * - clear() - 清空所有落子记录
 * - clearUndoStack() - 清空撤销栈
 *
 * 查询功能：
 * - getCurrentStep() - 获取当前步数
 * - getMoves() - 获取所有落子记录
 * - getUndoStackSize() - 获取撤销栈大小
 * - getMoveAtStep(step) - 获取指定步数的落子
 * - getLastMove() - 获取最后一步落子
 * - isEmpty() - 检查是否为空
 *
 * 输出功能（重要）：
 * - toString() - 输出完整棋谱格式
 * - getStatistics() - 获取统计信息
 * - saveToFile(filename) - 保存到文件
 *
 * 内部方法：
 * - setCurrentDateTime() - 设置当前日期时间
 */

#pragma once

#include "config.hpp"
#include "utils.hpp"
#include <algorithm>
#include <chrono>
#include <ctime>
#include <deque>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>


namespace GomokuRecord {

// 棋谱单元结构
struct MoveRecord {
  int color; // Utils::BLACK 或 Utils::WHITE
  char col;  // 'A'~'O'（15路棋盘）
  int row;   // 1~15
  int index; // 原始index (0~224)
  int step;  // 步数

  MoveRecord(int color_, int index_, int step_) : color(color_), index(index_), step(step_) {
    auto [x, y] = Utils::index_to_coordinate(index);
    col = 'A' + x;
    row = y + 1;
  }

  MoveRecord(int color_, int x, int y, int step_) : color(color_), step(step_) {
    col = 'A' + x;
    row = y + 1;
    index = Utils::coordinate_to_index({x, y});
  }
};

class GomokuGameRecord {
private:
  std::string game_type = "C5";
  std::string black_team;
  std::string white_team;
  std::string result;
  std::string datetime_place;
  std::string event_name;
  std::deque<MoveRecord> moves;
  std::deque<MoveRecord> undo_stack;

public:
  // 默认构造函数
  GomokuGameRecord() { setCurrentDateTime(); }

  // 构造函数：设置对局双方
  GomokuGameRecord(const std::string &black, const std::string &white) : black_team(black), white_team(white) {
    setCurrentDateTime();
  }

  // 设置游戏信息
  void setGameInfo(const std::string &black, const std::string &white) {
    black_team = black;
    white_team = white;
  }

  // 设置结果
  void setResult(int result_code) {
    if (result_code == 1) {
      result = "先手胜";
    } else if (result_code == 2) {
      result = "后手胜";
    } else {
      // 默认设置为未定
      result = "未定";
    }
  }

  // 设置比赛信息
  void setEventInfo(const std::string &event_name_, const std::string &place = "") {
    event_name = event_name_;
    if (!place.empty()) {
      datetime_place += " " + place;
    }
  }

  // 添加落子（使用index）
  void addMove(int color, int index) {
    if (index >= 0 && index < Config::BOARD_SQUARES) {
      moves.emplace_back(color, index, moves.size() + 1);
      clearUndoStack(); // 新落子后清空撤销栈
    }
  }

  // 添加落子（使用坐标）
  void addMove(int color, int x, int y) {
    if (x >= 0 && x < Config::BOARD_SIZE && y >= 0 && y < Config::BOARD_SIZE) {
      moves.emplace_back(color, x, y, moves.size() + 1);
      clearUndoStack(); // 新落子后清空撤销栈
    }
  }

  // 撤销上一步
  bool undoLastMove() {
    if (moves.empty())
      return false;

    undo_stack.push_back(moves.back());
    moves.pop_back();
    return true;
  }

  // 重做上一步
  bool redoLastMove() {
    if (undo_stack.empty())
      return false;

    moves.push_back(undo_stack.back());
    undo_stack.pop_back();
    return true;
  }

  // 清空所有落子记录
  void clear() {
    moves.clear();
    undo_stack.clear();
  }

  // 清空撤销栈
  void clearUndoStack() { undo_stack.clear(); }

  // 获取当前步数
  size_t getCurrentStep() const { return moves.size(); }

  // 获取所有落子记录
  const std::deque<MoveRecord> &getMoves() const { return moves; }

  // 获取撤销栈大小
  size_t getUndoStackSize() const { return undo_stack.size(); }

  // 获取指定步数的落子
  MoveRecord getMoveAtStep(int step) const {
    if (step > 0 && step <= static_cast<int>(moves.size())) {
      return moves[step - 1];
    }
    return MoveRecord(Utils::EMPTY, -1, -1);
  }

  // 获取最后一步落子
  MoveRecord getLastMove() const {
    if (!moves.empty()) {
      return moves.back();
    }
    return MoveRecord(Utils::EMPTY, -1, -1);
  }

  // 检查是否为空
  bool isEmpty() const { return moves.empty(); }

  // 输出完整棋谱格式
  std::string toString() const {
    std::ostringstream oss;
    oss << "{[C5][";
    oss << black_team << "][";
    oss << white_team << "][";
    oss << result << "][";
    oss << datetime_place << "][";
    oss << event_name << "]";

    // 添加落子信息，用分号分隔
    for (size_t i = 0; i < moves.size(); ++i) {
      const auto &move = moves[i];
      char color_char = (move.color == Utils::BLACK) ? 'B' : 'W';
      oss << ";" << color_char << "(" << move.col << "," << move.row << ")";
    }
    oss << "}";
    return oss.str();
  }

  // 获取统计信息
  std::string getStatistics() const {
    std::ostringstream oss;
    oss << "总步数: " << moves.size() << "\n";
    oss << "黑方落子: " << std::count_if(moves.begin(), moves.end(), [](const MoveRecord &m) {
      return m.color == Utils::BLACK;
    }) << "\n";
    oss << "白方落子: " << std::count_if(moves.begin(), moves.end(), [](const MoveRecord &m) {
      return m.color == Utils::WHITE;
    }) << "\n";
    oss << "撤销栈大小: " << undo_stack.size();
    return oss.str();
  }

  // 保存到文件
  bool saveToFile(const std::string &filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
      return false;
    }
    file << toString();
    file.close();
    return true;
  }

  // 设置当前日期时间
  void setCurrentDateTime() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::tm *tm = std::localtime(&time_t);

    std::ostringstream oss;
    oss << std::put_time(tm, "%Y-%m-%d %H:%M");
    oss << " 祁门县";
    datetime_place = oss.str();
    event_name = "2025 CCGC";
  }
};

} // namespace GomokuRecord
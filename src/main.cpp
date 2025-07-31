#include "config.hpp"
#include "gomoku_record.hpp"
#include "heuristic.hpp"
#include "mcts.hpp"
#include "network.hpp"
#include "utils.hpp"
#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <optional>
#include <print>
#include <queue>
#include <string>
#include <thread>
#include <windows.h>

using namespace MCTS;
using namespace GomokuRecord;

// Timer class for cumulative AI thinking display
class ThinkingTimer {
private:
  std::atomic<bool> running{false};
  std::thread timer_thread;
  std::chrono::high_resolution_clock::time_point game_start_time;
  std::chrono::milliseconds total_thinking_time{0};
  std::atomic<bool> is_thinking{false};
  std::chrono::high_resolution_clock::time_point thinking_start_time;
  HWND timer_window = nullptr;
  std::thread window_thread;

  static ThinkingTimer *instance;

  static LRESULT CALLBACK WindowProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    if (instance) {
      return instance->HandleWindowMessage(hwnd, msg, wParam, lParam);
    }
    return DefWindowProcW(hwnd, msg, wParam, lParam);
  }

  LRESULT HandleWindowMessage(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
      case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
      case WM_PAINT: {
        PAINTSTRUCT ps;
        HDC hdc = BeginPaint(hwnd, &ps);
        RECT rect;
        GetClientRect(hwnd, &rect);
        SetBkMode(hdc, OPAQUE);
        SetTextColor(hdc, RGB(255, 255, 255));
        SetBkColor(hdc, RGB(0, 0, 0));

        // 获取当前时间并显示
        auto now = std::chrono::high_resolution_clock::now();
        auto total_milliseconds = total_thinking_time.count();

        // 如果正在思考，加上当前思考时间
        if (is_thinking) {
          auto current_thinking = std::chrono::duration_cast<std::chrono::milliseconds>(now - thinking_start_time);
          total_milliseconds = (total_thinking_time + current_thinking).count();
        }

        auto total_seconds = total_milliseconds / 1000;
        auto minutes = total_seconds / 60;
        auto seconds = total_seconds % 60;
        auto milliseconds = total_milliseconds % 1000;

        std::wstring time_text = std::to_wstring(minutes) + L":" + (seconds < 10 ? L"0" : L"") +
                                 std::to_wstring(seconds) + L"." + (milliseconds < 100 ? L"0" : L"") +
                                 (milliseconds < 10 ? L"0" : L"") + std::to_wstring(milliseconds);
        DrawTextW(hdc, time_text.c_str(), -1, &rect, DT_CENTER | DT_VCENTER | DT_SINGLELINE);
        EndPaint(hwnd, &ps);
        return 0;
      }
    }
    return DefWindowProcW(hwnd, msg, wParam, lParam);
  }

public:
  void startGame() {
    running = true;
    game_start_time = std::chrono::high_resolution_clock::now();
    total_thinking_time = std::chrono::milliseconds{0};

    // 设置静态实例指针
    instance = this;

    // 创建独立的时间显示窗口
    window_thread = std::thread([this]() {
      // 注册窗口类
      WNDCLASSEXW wc = {};
      wc.cbSize = sizeof(WNDCLASSEXW);
      wc.lpfnWndProc = WindowProc;
      wc.hInstance = GetModuleHandle(nullptr);
      wc.lpszClassName = L"TimerWindow";
      wc.hbrBackground = CreateSolidBrush(RGB(0, 0, 0)); // 黑色背景
      wc.hCursor = LoadCursor(nullptr, IDC_ARROW);

      RegisterClassExW(&wc);

      // 创建窗口
      timer_window = CreateWindowExW(WS_EX_TOPMOST | WS_EX_TOOLWINDOW, L"TimerWindow", L"AI Thinking Time",
                                     WS_POPUP | WS_VISIBLE | WS_CAPTION, 100, 100, 200, 80, nullptr, nullptr,
                                     GetModuleHandle(nullptr), nullptr);

      // 消息循环
      MSG msg;
      while (running && GetMessage(&msg, nullptr, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
      }

      if (timer_window) {
        DestroyWindow(timer_window);
        timer_window = nullptr;
      }
    });

    // 计时器线程
    timer_thread = std::thread([this]() {
      while (running) {
        // 每秒重绘窗口以更新时间显示
        if (timer_window) {
          InvalidateRect(timer_window, nullptr, TRUE);
          UpdateWindow(timer_window);
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1000)); // 每秒更新一次
      }
    });
  }

  void startThinking() {
    is_thinking = true;
    thinking_start_time = std::chrono::high_resolution_clock::now();
  }

  void stopThinking() {
    if (is_thinking) {
      is_thinking = false;
      auto now = std::chrono::high_resolution_clock::now();
      auto current_thinking = std::chrono::duration_cast<std::chrono::milliseconds>(now - thinking_start_time);
      total_thinking_time += current_thinking;
    }
  }

  void stopGame() {
    running = false;
    is_thinking = false;

    // 关闭窗口
    if (timer_window) {
      PostMessage(timer_window, WM_DESTROY, 0, 0);
    }

    // 等待线程结束
    if (timer_thread.joinable()) {
      timer_thread.join();
    }
    if (window_thread.joinable()) {
      window_thread.join();
    }
  }

  // 获取总思考时间（秒）
  double getTotalThinkingTime() { return total_thinking_time.count() / 1000.0; }
};

// 定义静态成员变量
ThinkingTimer *ThinkingTimer::instance = nullptr;

// Game settings structure
struct GameSettings {
  int thread_count = 4;
  float c_puct = 2.0f;
  int think_time_ms = 1000;
  std::string player_name = "Player";
  std::string ai_name = "AI";

  // Save settings to file
  void saveToFile(const std::string &filename = "game_settings.txt") {
    std::ofstream file(filename);
    if (file.is_open()) {
      file << thread_count << std::endl;
      file << c_puct << std::endl;
      file << think_time_ms << std::endl;
      file << player_name << std::endl;
      file << ai_name << std::endl;
      file.close();
    }
  }

  // Load settings from file
  void loadFromFile(const std::string &filename = "game_settings.txt") {
    std::ifstream file(filename);
    if (file.is_open()) {
      file >> thread_count;
      file >> c_puct;
      file >> think_time_ms;
      file >> player_name;
      file >> ai_name;
      file.close();
    }
  }
};

// Game interface class
class GameInterface {
private:
  GameSettings settings;
  enum MenuState { MAIN_MENU, SETTINGS, GAME_SETUP, GAME_PLAYING };
  MenuState current_state;
  ThinkingTimer thinking_timer;

  // Game state
  struct GameState {
    bool my_turn_first;
    bool game_ended;
    GomokuGameRecord record;
    int consecutive_passes;
    std::unique_ptr<MCTSAgent> agent;
    Network net;
  } game;

public:
  GameInterface() : current_state(MAIN_MENU) { settings.loadFromFile(); }

  // Display aligned board
  void print_board(const Utils::Board &board) {
    // Clean the output layout before printing
#if defined(_WIN32) || defined(_WIN64)
    std::system("cls");
#else
    std::system("clear");
#endif

    std::print("\n");
    std::print("   ");
    for (int i = 0; i < Config::BOARD_SIZE; i++) {
      std::print("{:2} ", i + 1);
    }
    std::print("\n");

    for (int i = 0; i < Config::BOARD_SIZE; i++) {
      std::print("{:2} ", i + 1);
      for (int j = 0; j < Config::BOARD_SIZE; j++) {
        char cell = '.';
        if (board[i][j] == Utils::BLACK)
          cell = 'X';
        else if (board[i][j] == Utils::WHITE)
          cell = 'O';
        std::print(" {} ", cell);
      }
      std::print("{:2}\n", i + 1);
    }

    std::print("    ");
    for (int i = 0; i < Config::BOARD_SIZE; i++) {
      std::print("{:2} ", static_cast<char>('A' + i));
    }
    std::print("\n\n");
  }
  // Main menu
  void show_main_menu() {
    std::println("\n=== Gomoku Game ===");
    std::println("1. Start Game");
    std::println("2. Settings");
    std::println("3. Exit");
    std::print("Choose: ");

    int choice;
    std::cin >> choice;

    switch (choice) {
      case 1:
        current_state = GAME_SETUP;
        break;
      case 2:
        current_state = SETTINGS;
        break;
      case 3:
        std::println("Goodbye!");
        exit(0);
      default:
        std::println("Invalid choice!");
    }
  }

  // Settings menu
  void show_settings_menu() {
    while (true) {
      std::println("\n=== Settings Menu ===");
      std::println("Current settings:");
      std::println("1. Thread count: {}", settings.thread_count);
      std::println("2. C_PUCT value: {}", settings.c_puct);
      std::println("3. Thinking time (ms): {}", settings.think_time_ms);
      std::println("4. Player name: {}", settings.player_name);
      std::println("5. AI name: {}", settings.ai_name);
      std::println("6. Return to main menu");
      std::print("Choose a setting to modify: ");

      int choice;
      std::cin >> choice;

      switch (choice) {
        case 1:
          std::print("Enter new thread count: ");
          std::cin >> settings.thread_count;
          break;
        case 2:
          std::print("Enter new C_PUCT value: ");
          std::cin >> settings.c_puct;
          break;
        case 3:
          std::print("Enter new thinking time (ms): ");
          std::cin >> settings.think_time_ms;
          break;
        case 4:
          std::print("Enter player name: ");
          std::cin >> settings.player_name;
          break;
        case 5:
          std::print("Enter AI name: ");
          std::cin >> settings.ai_name;
          break;
        case 6:
          settings.saveToFile();
          current_state = MAIN_MENU;
          return;
        default:
          std::println("Invalid choice!");
      }
    }
  }

  // AI换手判断函数 (占位符)
  bool ai_should_swap(const Utils::Board &board) {
    auto result = Network::evaluate(board, Utils::WHITE);
    return result->second < -0.0f;
  }

  // AI第五手N个落子位置
  std::vector<int> ai_get_n_moves(const Utils::Board &board, int n) {
    auto root = &game.agent->root_node();
    assert(root->current_color == Utils::BLACK);
    auto cmp = [root](int idx1, int idx2) {
      return root->children[idx1]->visit_count.load() < root->children[idx2]->visit_count.load();
    };
    std::priority_queue<int, std::vector<int>, decltype(cmp)> pq(cmp);
    for (int i = 0; i < Config::BOARD_SQUARES; ++i) {
      if (root->children[i]) {
        pq.push(i);
      }
    }
    std::vector<int> top_moves;
    for (int i = 0; i < n && !pq.empty(); ++i) {
      top_moves.push_back(pq.top());
      pq.pop();
    }
    // The following logic handles the very rare case that the top n moves are not enough
    if (top_moves.size() < n) {
      // Pick rest moves from legal moves in the central 5x5 area
      // Find legal moves in the central 5x5 area
      int center = Config::BOARD_SIZE / 2;
      int half = 2; // 5x5 area: center-2 to center+2
      const auto &legal_moves = root->legal_moves;
      for (int dr = -half; dr <= half; ++dr) {
        for (int dc = -half; dc <= half; ++dc) {
          int r = center + dr;
          int c = center + dc;
          if (r >= 0 && r < Config::BOARD_SIZE && c >= 0 && c < Config::BOARD_SIZE) {
            int idx = r * Config::BOARD_SIZE + c;
            if (legal_moves[idx]) {
              // Avoid duplicates
              if (std::find(top_moves.begin(), top_moves.end(), idx) == top_moves.end()) {
                top_moves.push_back(idx);
                if (top_moves.size() == n)
                  break;
              }
            }
          }
        }
        if (top_moves.size() == n)
          break;
      }
      // If still not enough, fill from any remaining legal moves
      if (top_moves.size() < n) {
        for (int i = 0; i < Config::BOARD_SQUARES; ++i) {
          if (legal_moves[i]) {
            if (std::find(top_moves.begin(), top_moves.end(), i) == top_moves.end()) {
              top_moves.push_back(i);
              if (top_moves.size() == n)
                break;
            }
          }
        }
      }
    }
    return top_moves;
  }

  // AI选择玩家提供的N个位置中的哪一个
  int ai_choose_from_player_moves(const Utils::Board &board, const std::vector<int> &player_moves) {
    auto root = &game.agent->root_node();
    assert(root->current_color == Utils::BLACK);
    auto cmp = [root](int idx1, int idx2) -> bool {
      // priority queue maintains the second operand near heap top when the cmp returns true
      if (!root->children[idx2]) { // if the second operand is not a child, it is a bad move for black, but a good move
                                   // for white
        return true;
      } else if (!root->children[idx1]) {
        return false;
      } else {
        return root->children[idx1]->visit_count.load() > root->children[idx2]->visit_count.load();
      }
    };
    std::priority_queue<int, std::vector<int>, decltype(cmp)> pq(cmp);
    for (int i = 0; i < player_moves.size(); ++i) {
      pq.push(player_moves[i]);
    }
    return pq.top();
  }

  // 坐标转换和验证函数
  struct CoordinateResult {
    bool valid;
    int x, y;
    int index;
    std::string error_message;
  };

  // 解析字符串坐标为坐标和索引
  CoordinateResult parse_coordinate(const std::string &input) {
    CoordinateResult result = {false, -1, -1, -1, ""};

    if (input.length() < 2) {
      result.error_message = "Invalid input format! Please use format: Letter + Number (e.g., H8).";
      return result;
    }

    char col = input[0];
    int row;
    try {
      row = std::stoi(input.substr(1));
    } catch (const std::exception &) {
      result.error_message = "Invalid row number! Please enter a valid number.";
      return result;
    }

    int x = col - 'A';
    int y = row - 1;

    if (x < 0 || x >= Config::BOARD_SIZE || y < 0 || y >= Config::BOARD_SIZE) {
      result.error_message = "Invalid coordinates! Please enter valid coordinates (e.g., A1-O15).";
      return result;
    }

    result.valid = true;
    result.x = x;
    result.y = y;
    result.index = y * Config::BOARD_SIZE + x;
    return result;
  }

  // 将索引转换为字符串坐标
  std::string index_to_coordinate(int index) {
    if (index < 0 || index > Config::BOARD_SQUARES) {
      return "INVALID";
    } else if (index == Config::BOARD_SQUARES) {
      return "PASS";
    }
    int r = index / Config::BOARD_SIZE;
    int c = index % Config::BOARD_SIZE;
    char col = 'A' + c;
    return std::string(1, col) + std::to_string(r + 1);
  }

  // 验证数字输入是否在指定范围内
  bool validate_number_input(int &value, int min_val, int max_val, const std::string &prompt) {
    bool valid_input = false;
    while (!valid_input) {
      std::print("{}: ", prompt);
      std::cin >> value;

      if (value >= min_val && value <= max_val) {
        valid_input = true;
      } else {
        std::println("Invalid number! Please enter a number between {} and {}.", min_val, max_val);
      }
    }
    return true;
  }

  // 获取用户输入的坐标
  CoordinateResult get_user_coordinate(const Utils::Board &board, const std::string &prompt) {
    CoordinateResult result;
    bool valid_coordinate = false;

    while (!valid_coordinate) {
      std::string coord_input;
      std::print("{}: ", prompt);
      std::cin >> coord_input;

      result = parse_coordinate(coord_input);
      if (!result.valid) {
        std::println("{}", result.error_message);
        continue;
      }

      if (board[result.y][result.x] == Utils::EMPTY) {
        valid_coordinate = true;
      } else {
        std::println("Position already occupied! Please choose another position.");
      }
    }

    return result;
  }

  // 显示坐标列表
  void display_coordinate_list(const std::vector<int> &moves, const std::string &title) {
    std::println("{}", title);
    for (size_t i = 0; i < moves.size(); i++) {
      std::print("{}. {}\t", i + 1, index_to_coordinate(moves[i]));
    }
    std::print("\n");
  }

  // 清屏函数
  void clear_screen() {
#if defined(_WIN32) || defined(_WIN64)
    std::system("cls");
#else
    std::system("clear");
#endif
  }

  // Apply fixed opening moves to board
  void apply_fixed_opening(Utils::Board &board, int opening_choice) {
    // Clear board first
    for (auto &row : board)
      for (auto &cell : row)
        cell = Utils::EMPTY;

    switch (opening_choice) {
      case 1:                       // 疏星局 (B:H8,W:H9;B:J10)
        board[7][7] = Utils::BLACK; // H8
        board[8][7] = Utils::WHITE; // H9
        board[9][9] = Utils::BLACK; // J10
        break;
      case 2:                       // 长星局 (B:H8;W:I9;B:J10)
        board[7][7] = Utils::BLACK; // H8
        board[8][8] = Utils::WHITE; // I9
        board[9][9] = Utils::BLACK; // J10
        break;
      case 3:                       // 流星局 (B:H8;W:I9;B:J6)
        board[7][7] = Utils::BLACK; // H8
        board[8][8] = Utils::WHITE; // I9
        board[5][9] = Utils::BLACK; // J6
        break;
      case 4: // 自定义开局
        std::println("\n=== Custom Opening Setup ===");
        std::println("First move is fixed at H8 (Black)");
        std::println("Please input 2 additional coordinates (e.g., I9 J10):");
        std::println("Format: Letter + Number (e.g., I9, J10)");

        // Store the 3 coordinates (first is fixed at H8)
        int coordinates[3][2];
        coordinates[0][0] = 7; // H8 row
        coordinates[0][1] = 7; // H8 column

        // Input 2 additional coordinates
        for (int i = 1; i < 3; i++) {
          auto coord_result = get_user_coordinate(board, "Enter coordinate " + std::to_string(i + 1));
          coordinates[i][0] = coord_result.y;
          coordinates[i][1] = coord_result.x;
        }

        // Apply moves: Black (H8), White, Black
        board[coordinates[0][0]][coordinates[0][1]] = Utils::BLACK; // First move (Black) at H8
        board[coordinates[1][0]][coordinates[1][1]] = Utils::WHITE; // Second move (White)
        board[coordinates[2][0]][coordinates[2][1]] = Utils::BLACK; // Third move (Black)

        std::println("Custom opening applied! (H8 + your 2 coordinates)");
        break;
    }
  }

  std::optional<int> get_win_point(Node *node) {
    auto threats = find_all_threats(node->board_state, node->current_color);
    const auto &legal_moves = node->legal_moves;
    for (auto [coord, threat_type] : threats) {
      if (threat_type == 2) {
        auto idx = Utils::coordinate_to_index(coord);
        if (legal_moves[idx]) {
          return idx;
        }
      }
    }
    return std::nullopt;
  }

  // Game setup
  void show_game_setup() {
    std::println("\n=== Game Setup ===");
    std::println("Choose first turn:");
    std::println("1. Player first");
    std::println("2. AI first");
    std::print("Choose: ");

    int choice;
    std::cin >> choice;

    game.my_turn_first = (choice == 1);
    game.game_ended = false;
    game.consecutive_passes = 0;

    // Initialize game
    Utils::Board board{};
    for (auto &row : board)
      for (auto &cell : row)
        cell = Utils::EMPTY;

    // If AI goes first, show fixed opening menu
    if (!game.my_turn_first) {
      std::println("\n=== Fixed Opening Selection ===");
      std::println("Choose a fixed opening:");
      std::println("1. 疏星局 (best opening) (B:H8,W:H9;B:J10)");
      std::println("2. 长星局 (B:H8;W:I9;B:J10)");
      std::println("3. 流星局 (B:H8;W:I9;B:J6)");
      std::println("4. 自定义开局(input 2 coordinates)");
      std::print("Choose: ");

      int opening_choice;
      std::cin >> opening_choice;

      if (opening_choice >= 1 && opening_choice <= 4) {
        apply_fixed_opening(board, opening_choice);
        if (opening_choice != 4) {
          std::println("Fixed opening applied!");
        }
      }
    } else {
      std::println("You go first, please input 2 coordinates");
      apply_fixed_opening(board, 4);
    }

    // 1. 打印当前开局棋盘
    std::println("\n=== Current Opening Board ===");
    print_board(board);

    // 2. 换手逻辑
    bool should_swap = false;
    if (game.my_turn_first) {
      // AI先手，使用AI换手判断函数
      should_swap = ai_should_swap(board);
      std::println("AI decides to {} colors.", (should_swap ? "swap" : "not swap"));
      std::this_thread::sleep_for(std::chrono::seconds(10));
    } else {
      // 玩家先手，询问玩家是否换手
      std::print("Do you want to swap colors? (y/n): ");
      std::string swap_input;
      std::cin >> swap_input;
      should_swap = (swap_input == "y" || swap_input == "Y");
      std::println("You decided to {} colors.", (should_swap ? "swap" : "not swap"));
    }

    // 3. 如果要换手，交换双方的颜色
    if (should_swap) {
      game.my_turn_first = !game.my_turn_first;
      std::print("Colors swapped! ");
      if (game.my_turn_first) {
        std::println("You will play as Black.");
      } else {
        std::println("AI will play as Black.");
      }
    }
    // 当前应该先手是白棋
    game.agent = std::make_unique<MCTSAgent>(board, Utils::WHITE, settings.thread_count);

    // Initialize game record
    game.record = GomokuGameRecord(settings.player_name, settings.ai_name);

    // 启动游戏计时器
    thinking_timer.startGame();

    current_state = GAME_PLAYING;
  }

  // Game loop
  void game_loop() {
    Utils::STONE_COLOR current_player = Utils::WHITE;
    int move_count = 4;

    std::println("\n=== Game Start ===");
    std::println("Instructions:");
    std::println("- Place stone: Enter coordinates (e.g., H8)");
    std::println("- Pass: Enter 'pass'");
    std::println("- Undo: Enter 'undo'");
    std::println("- Quit: Enter 'quit'\n");

    print_board(game.agent->last_move_board());
    float win_rate = 0.0f;
    int move_idx = -1;
    while (!game.game_ended && move_count < Config::BOARD_SQUARES) {
      const Utils::Board &root_board = game.agent->last_move_board();
      bool is_my_turn = (current_player == Utils::BLACK && game.my_turn_first) ||
                        (current_player == Utils::WHITE && !game.my_turn_first);
      std::println("move_count: {}", move_count);
      if (is_my_turn) {
        // Player's turn
        std::print("Your turn ({}): ", (current_player == Utils::BLACK ? "Black" : "White"));

        // 五手N打
        if (move_count == 5) {
          // 第五手特殊处理：玩家显示N个落子位置供AI选择
          std::println("\n=== Fifth Move Selection (Player's Turn) ===");

          // 1. 让玩家输入N是几
          int num_positions;
          validate_number_input(num_positions, 2, 5, "How many positions do you want to offer? (2-5)");

          // 2. 让玩家输入这N个坐标
          std::vector<int> player_moves;
          std::println("Please enter {} coordinates:", num_positions);

          for (int i = 0; i < num_positions; i++) {
            auto coord_result = get_user_coordinate(root_board, "Enter coordinate " + std::to_string(i + 1));
            player_moves.push_back(coord_result.index);
          }

          // 显示玩家提供的N个位置
          display_coordinate_list(player_moves, "You offered " + std::to_string(num_positions) + " positions:");

          // 3. 调用函数返回AI选择哪一个位置落子
          int ai_choice = ai_choose_from_player_moves(root_board, player_moves);

          // 4. 应用这个坐标然后继续游戏
          if (ai_choice != -1) {
            std::println("AI chose position: {}", index_to_coordinate(ai_choice));

            game.agent->apply_move(ai_choice);
            game.record.addMove(current_player == Utils::BLACK ? Utils::BLACK : Utils::WHITE, ai_choice);
            game.consecutive_passes = 0;

            // 继续到下一个玩家
            current_player = (current_player == Utils::BLACK ? Utils::WHITE : Utils::BLACK);
            move_count++;
            continue;
          } else {
            std::println("AI failed to choose a valid position!");
            continue;
          }
        }

        std::string input;
        std::cin >> input;

        if (input == "quit") {
          std::println("Game over");
          break;
        }

        if (input == "undo") {
          game.agent->undo_last_move();
          continue;
        }

        if (input == "pass") {
          std::println("You chose to pass");
          game.agent->apply_move(Network::PASS_IDX);
          game.consecutive_passes++;
          game.record.addMove(current_player == Utils::BLACK ? Utils::BLACK : Utils::WHITE, -1);

          if (game.consecutive_passes >= 2) {
            std::println("Two consecutive passes, game is a draw!");
            game.game_ended = true;
            game.record.setResult(0); // Draw
            break;
          }
        } else {
          // Parse coordinates
          auto coord_result = parse_coordinate(input);
          if (const auto &root_node = game.agent->root_node(); root_node.current_color == Utils::BLACK) {
            if (!root_node.legal_moves[coord_result.index]) {
              std::println("Illegal move!");
              game.game_ended = true;
              // Set game record result
              if (game.my_turn_first) {
                game.record.setResult(current_player == Utils::BLACK ? 1 : 2);
              } else {
                game.record.setResult(current_player == Utils::WHITE ? 1 : 2);
              }
              break;
            }
          }
          if (!coord_result.valid) {
            std::println("{}", coord_result.error_message);
            continue;
          }

          if (root_board[coord_result.y][coord_result.x] == Utils::EMPTY) {
            game.agent->apply_move(coord_result.index);
            game.record.addMove(current_player == Utils::BLACK ? Utils::BLACK : Utils::WHITE, coord_result.index);
            game.consecutive_passes = 0;
            std::println("You placed stone: {}", input);
          } else {
            std::println("Position already occupied!");
            continue;
          }
        }
      } else {
        // AI's turn
        std::println("AI is thinking...");
        // 启动思考计时器
        thinking_timer.startThinking();

        auto start = std::chrono::high_resolution_clock::now();
        // Calculate simulations based on thinking time (rough estimate)
        int target_simulations = 1600;
        game.agent->run_mcts(target_simulations);
        int sim_count = game.agent->get_simulations_completed();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        // 停止思考计时器
        thinking_timer.stopThinking();

        move_idx = game.agent->next_move_idx();
        auto win_point = get_win_point(&game.agent->root_node());
        if (win_point) {
          move_idx = *win_point;
        }

        win_rate = game.agent->root_node().value();
        // 五手N打
        if (move_count == 5) {
          // 第五手特殊处理：AI显示N个落子位置供玩家选择
          std::println("\n=== Fifth Move Selection ===");

          // 让玩家输入位置数量
          int num_positions;
          validate_number_input(num_positions, 2, 5, "How many positions do you want? (2-5)");

          std::println("AI suggests {} possible moves:", num_positions);

          std::vector<int> n_moves = ai_get_n_moves(game.agent->last_move_board(), num_positions);

          // 显示N个位置
          display_coordinate_list(n_moves, "AI suggests " + std::to_string(num_positions) + " possible moves:");

          // 让玩家选择
          int choice;
          validate_number_input(choice, 1, num_positions,
                                "Please choose a move (1-" + std::to_string(num_positions) + ")");

          move_idx = n_moves[choice - 1];
          std::println("You chose move {}.", choice);
        }
        if (move_idx == Network::PASS_IDX) {
          std::println("AI chose to pass");
          game.agent->apply_move(Network::PASS_IDX);
          game.consecutive_passes++;
          game.record.addMove(current_player == Utils::BLACK ? Utils::BLACK : Utils::WHITE, Network::PASS_IDX);

          if (game.consecutive_passes >= 2) {
            std::println("Two consecutive passes, game is a draw!");
            game.game_ended = true;
            game.record.setResult(0); // Draw
            break;
          }
        } else {
          game.agent->apply_move(move_idx);
          int r = move_idx / Config::BOARD_SIZE;
          int c = move_idx % Config::BOARD_SIZE;
          char col = 'A' + c;
          std::println("AI placed stone: {}{} | Time: {}ms | Simulations: {}", col, r + 1, duration.count(), sim_count);
          game.record.addMove(current_player == Utils::BLACK ? Utils::BLACK : Utils::WHITE, move_idx);
          game.consecutive_passes = 0;
        }
      }

      print_board(game.agent->last_move_board());
      std::println("AI placed at {}", index_to_coordinate(move_idx));
      std::println("AI Win Rate: {}", win_rate);
      std::println("New Root Node Visit Count: {}", game.agent->root_node().visit_count.load());

      // Check win
      if (check_win(game.agent->last_move_board(), current_player)) {
        clear_screen();
        std::string winner = (current_player == Utils::BLACK ? "Black" : "White");
        std::println("{} wins!", winner);
        game.game_ended = true;

        // Set game record result
        if (game.my_turn_first) {
          game.record.setResult(current_player == Utils::BLACK ? 1 : 2);
        } else {
          game.record.setResult(current_player == Utils::WHITE ? 1 : 2);
        }
        break;
      }

      current_player = (current_player == Utils::BLACK ? Utils::WHITE : Utils::BLACK);
      move_count++;
    }

    if (!game.game_ended) {
      std::println("Game is a draw!");
      game.record.setResult(0);
    }

    // 游戏结束，先清屏
    clear_screen();

    // 显示最终棋盘
    std::println("\n=== Final Board ===");
    print_board(game.agent->last_move_board());

    // 显示游戏结果
    if (game.game_ended) {
      if (game.record.getCurrentStep() > 0) {
        auto last_move = game.record.getLastMove();
        if (last_move.color == Utils::BLACK) {
          std::println("Game Result: 先手胜 (Black wins)");
        } else {
          std::println("Game Result: 后手胜 (White wins)");
        }
      }
    } else {
      std::println("Game Result: 平局 (Draw)");
    }

    // 停止游戏计时器并显示总思考时间
    thinking_timer.stopGame();
    double total_time = thinking_timer.getTotalThinkingTime();
    int total_seconds = static_cast<int>(total_time);
    int minutes = total_seconds / 60;
    int seconds = total_seconds % 60;
    std::println("\n=== Game Summary ===");
    std::println("Total AI thinking time: {:02}:{:02} (MM:SS)", minutes, seconds);

    // Save game record
    save_game_record();

    std::println("\nPress any key to return to main menu...");
    std::cin.ignore();
    std::cin.get();
    current_state = MAIN_MENU;
  }

  // Check win
  bool check_win(const Utils::Board &board, Utils::STONE_COLOR player) {
    int dr[4] = {0, 1, 1, 1};
    int dc[4] = {1, 0, 1, -1};

    for (int r = 0; r < Config::BOARD_SIZE; ++r) {
      for (int c = 0; c < Config::BOARD_SIZE; ++c) {
        if (board[r][c] != player)
          continue;

        for (int d = 0; d < 4; ++d) {
          int cnt = 1;
          for (int k = 1; k < 5; ++k) {
            int nr = r + dr[d] * k, nc = c + dc[d] * k;
            if (nr < 0 || nr >= Config::BOARD_SIZE || nc < 0 || nc >= Config::BOARD_SIZE)
              break;
            if (board[nr][nc] == player)
              cnt++;
            else
              break;
          }
          for (int k = 1; k < 5; ++k) {
            int nr = r - dr[d] * k, nc = c - dc[d] * k;
            if (nr < 0 || nr >= Config::BOARD_SIZE || nc < 0 || nc >= Config::BOARD_SIZE)
              break;
            if (board[nr][nc] == player)
              cnt++;
            else
              break;
          }
          if (cnt >= 5)
            return true;
        }
      }
    }
    return false;
  }

  // Save game record
  void save_game_record() {
    std::string result_str = "";
    if (game.record.getCurrentStep() > 0) {
      auto last_move = game.record.getLastMove();
      if (last_move.color == Utils::BLACK) {
        result_str = "先手胜";
      } else {
        result_str = "后手胜";
      }
    }

    std::string filename = "C5-" + settings.player_name + " vs " + settings.ai_name + "-" + result_str + ".txt";
    game.record.saveToFile(filename);

    std::println("\n=== Game Record ===");
    std::println("{}", game.record.toString());
    std::println("\nGame record saved to: {}", filename);
  }

  // Main loop
  void run() {
    while (true) {
      switch (current_state) {
        case MAIN_MENU:
          show_main_menu();
          break;
        case SETTINGS:
          show_settings_menu();
          break;
        case GAME_SETUP:
          show_game_setup();
          break;
        case GAME_PLAYING:
          game_loop();
          break;
      }
    }
  }
};

int main() {
  GameInterface game;
  game.run();
  return 0;
}

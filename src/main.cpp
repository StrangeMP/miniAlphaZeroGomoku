#include "config.hpp"
#include "gomoku_record.hpp"
#include "mcts.hpp"
#include "network.hpp"
#include <atomic>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <print>
#include <string>
#include <thread>

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

public:
  void startGame() {
    running = true;
    game_start_time = std::chrono::high_resolution_clock::now();
    total_thinking_time = std::chrono::milliseconds{0};
    timer_thread = std::thread([this]() {
      while (running) {
        if (is_thinking) {
          auto now = std::chrono::high_resolution_clock::now();
          auto current_thinking =
              std::chrono::duration_cast<std::chrono::milliseconds>(now - game_start_time) - total_thinking_time;
          auto total_milliseconds = (total_thinking_time + current_thinking).count();
          auto total_seconds = total_milliseconds / 1000;
          auto minutes = total_seconds / 60;
          auto seconds = total_seconds % 60;

          // 在屏幕右上角显示累计计时器 (分钟:秒钟格式)
          std::print("\r\x1B[1;60HAI Total: {:02}:{:02}", minutes, seconds);
          std::cout << std::flush;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1000)); // 每秒更新一次
      }
    });
  }

  void startThinking() { is_thinking = true; }

  void stopThinking() {
    is_thinking = false;
    auto now = std::chrono::high_resolution_clock::now();
    auto current_thinking =
        std::chrono::duration_cast<std::chrono::milliseconds>(now - game_start_time) - total_thinking_time;
    total_thinking_time += current_thinking;
  }

  void stopGame() {
    running = false;
    is_thinking = false;
    if (timer_thread.joinable()) {
      timer_thread.join();
    }
    // 清除计时器显示
    std::print("\r\x1B[1;60H{}\r", std::string(20, ' '));
    std::cout << std::flush;
  }

  // 获取总思考时间（秒）
  double getTotalThinkingTime() { return total_thinking_time.count() / 1000.0; }
};

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
      std::print("{:2} ", static_cast<char>('A' + i));
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

    std::print("   ");
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
    // TODO: 实现AI换手判断逻辑
    // 这里应该分析棋盘局势，判断是否应该换手
    // 暂时返回false作为占位符
    return false;
  }

  // AI第五手N个落子位置函数 (占位符)
  std::vector<int> ai_get_n_moves(const Utils::Board &board, int n) {
    // TODO: 实现AI第五手N个落子位置逻辑
    // 这里应该分析棋盘局势，返回n个最佳的落子位置
    // 暂时返回n个默认位置作为占位符
    std::vector<int> moves;
    for (int i = 0; i < n; i++) {
      moves.push_back(60 + i); // 示例位置
    }
    return moves;
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
          bool valid_input = false;
          while (!valid_input) {
            std::string input;
            std::print("Enter coordinate {}: ", i + 1);
            std::cin >> input;

            if (input.length() >= 2) {
              char col = input[0];
              int row = std::stoi(input.substr(1));

              int x = col - 'A';
              int y = row - 1;

              if (x >= 0 && x < Config::BOARD_SIZE && y >= 0 && y < Config::BOARD_SIZE) {
                coordinates[i][0] = y;
                coordinates[i][1] = x;
                valid_input = true;
              } else {
                std::println("Invalid coordinate! Please enter a valid coordinate (e.g., A1-O15).");
              }
            } else {
              std::println("Invalid input format! Please use format: Letter + Number (e.g., I9).");
            }
          }
        }

        // Apply moves: Black (H8), White, Black
        board[coordinates[0][0]][coordinates[0][1]] = Utils::BLACK; // First move (Black) at H8
        board[coordinates[1][0]][coordinates[1][1]] = Utils::WHITE; // Second move (White)
        board[coordinates[2][0]][coordinates[2][1]] = Utils::BLACK; // Third move (Black)

        std::println("Custom opening applied! (H8 + your 2 coordinates)");
        break;
    }
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
    if (!game.my_turn_first) {
      // AI先手，使用AI换手判断函数
      should_swap = ai_should_swap(board);
      std::println("AI decides to {} colors.", (should_swap ? "swap" : "not swap"));
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
    while (!game.game_ended && move_count < Config::BOARD_SQUARES) {
      const Utils::Board &root_board = game.agent->last_move_board();
      bool is_my_turn = (current_player == Utils::BLACK && game.my_turn_first) ||
                        (current_player == Utils::WHITE && !game.my_turn_first);

      if (is_my_turn) {
        // Player's turn
        std::print("Your turn ({}): ", (current_player == Utils::BLACK ? "Black" : "White"));
        std::string input;
        std::cin >> input;

        if (input == "quit") {
          std::println("Game over");
          break;
        }

        if (input == "undo") {
          if (game.record.undoLastMove()) {
            std::println("Undo successful");
            // Reinitialize agent state, simplified handling
            continue;
          } else {
            std::println("Cannot undo");
            continue;
          }
        }

        if (input == "pass") {
          std::println("You chose to pass");
          game.agent->apply_move(-1);
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
          if (input.length() >= 2) {
            char col = input[0];
            int row = std::stoi(input.substr(1));

            int x = col - 'A';
            int y = row - 1;

            if (x >= 0 && x < Config::BOARD_SIZE && y >= 0 && y < Config::BOARD_SIZE) {
              if (root_board[y][x] == Utils::EMPTY) {
                int move_idx = y * Config::BOARD_SIZE + x;
                game.agent->apply_move(move_idx);
                game.record.addMove(current_player == Utils::BLACK ? Utils::BLACK : Utils::WHITE, move_idx);
                game.consecutive_passes = 0;
                std::println("You placed stone: {}", input);
              } else {
                std::println("Position already occupied!");
                continue;
              }
            } else {
              std::println("Invalid coordinates!");
              continue;
            }
          } else {
            std::println("Invalid input!");
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
        int target_simulations = 800;
        game.agent->run_mcts(target_simulations);
        int sim_count = game.agent->get_simulations_completed();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        // 停止思考计时器
        thinking_timer.stopThinking();

        int move_idx = game.agent->next_move_idx();
        win_rate = game.agent->root->value();
        // 五手N打
        if (move_idx == 5) {
          // 第五手特殊处理：AI显示N个落子位置供玩家选择
          std::println("\n=== Fifth Move Selection ===");

          // 让玩家输入位置数量
          int num_positions;
          bool valid_input = false;
          while (!valid_input) {
            std::print("How many positions do you want? (2-5): ");
            std::cin >> num_positions;

            if (num_positions >= 2 && num_positions <= 5) {
              valid_input = true;
            } else {
              std::println("Invalid number! Please enter a number between 2 and 5.");
            }
          }

          std::println("AI suggests {} possible moves:", num_positions);

          std::vector<int> n_moves = ai_get_n_moves(game.agent->last_move_board(), num_positions);

          // 显示N个位置
          for (int i = 0; i < num_positions; i++) {
            int r = n_moves[i] / Config::BOARD_SIZE;
            int c = n_moves[i] % Config::BOARD_SIZE;
            char col = 'A' + c;
            std::print("{}. {}{}\t", i + 1, col, r + 1);
          }
          std::print("\n");
          // 让玩家选择
          std::print("Please choose a move (1-{}): ", num_positions);
          int choice;
          std::cin >> choice;

          if (choice >= 1 && choice <= num_positions) {
            move_idx = n_moves[choice - 1];
            std::println("You chose move {}.", choice);
          } else {
            std::println("Invalid choice! Using first move.");
            move_idx = n_moves[0];
          }
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
      std::println("AI Win Rate: {}", win_rate);
      std::println("New Root Node Visit Count: {}", game.agent->root->visit_count.load());

      // Check win
      if (check_win(game.agent->last_move_board(), current_player)) {
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
        result_str = "First player wins";
      } else {
        result_str = "Second player wins";
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

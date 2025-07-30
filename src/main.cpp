#include "config.hpp"
#include "gomoku_record.hpp"
#include "mcts.hpp"
#include "network.hpp"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <atomic>

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
          auto current_thinking = std::chrono::duration_cast<std::chrono::milliseconds>(now - game_start_time) - total_thinking_time;
          auto total_milliseconds = (total_thinking_time + current_thinking).count();
          auto total_seconds = total_milliseconds / 1000;
          auto minutes = total_seconds / 60;
          auto seconds = total_seconds % 60;
          
          // 在屏幕右上角显示累计计时器 (分钟:秒钟格式)
          std::cout << "\r\x1B[1;60H" << "AI Total: " << std::setfill('0') << std::setw(2) << minutes << ":" << std::setw(2) << seconds << std::flush;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1000)); // 每秒更新一次
      }
    });
  }
  
  void startThinking() {
    is_thinking = true;
  }
  
  void stopThinking() {
    is_thinking = false;
    auto now = std::chrono::high_resolution_clock::now();
    auto current_thinking = std::chrono::duration_cast<std::chrono::milliseconds>(now - game_start_time) - total_thinking_time;
    total_thinking_time += current_thinking;
  }
  
  void stopGame() {
    running = false;
    is_thinking = false;
    if (timer_thread.joinable()) {
      timer_thread.join();
    }
    // 清除计时器显示
    std::cout << "\r\x1B[1;60H" << std::string(20, ' ') << "\r" << std::flush;
  }
  
  // 获取总思考时间（秒）
  double getTotalThinkingTime() {
    return total_thinking_time.count() / 1000.0;
  }
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
    std::cout << "\n";
    std::cout << "   ";
    for (int i = 0; i < Config::BOARD_SIZE; i++) {
      std::cout << std::setw(2) << (char)('A' + i) << " ";
    }
    std::cout << "\n";

    for (int i = 0; i < Config::BOARD_SIZE; i++) {
      std::cout << std::setw(2) << (i + 1) << " ";
      for (int j = 0; j < Config::BOARD_SIZE; j++) {
        char cell = '.';
        if (board[i][j] == Utils::BLACK)
          cell = 'X';
        else if (board[i][j] == Utils::WHITE)
          cell = 'O';
        std::cout << " " << cell << " ";
      }
      std::cout << std::setw(2) << (i + 1) << "\n";
    }

    std::cout << "   ";
    for (int i = 0; i < Config::BOARD_SIZE; i++) {
      std::cout << std::setw(2) << (char)('A' + i) << " ";
    }
    std::cout << "\n\n";
  }

  // Main menu
  void show_main_menu() {
    std::cout << "\n=== Gomoku Game ===\n";
    std::cout << "1. Start Game\n";
    std::cout << "2. Settings\n";
    std::cout << "3. Exit\n";
    std::cout << "Choose: ";

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
        std::cout << "Goodbye!\n";
        exit(0);
      default:
        std::cout << "Invalid choice!\n";
    }
  }

  // Settings menu
  void show_settings_menu() {
    while (true) {
      std::cout << "\n=== Settings Menu ===\n";
      std::cout << "Current settings:\n";
      std::cout << "1. Thread count: " << settings.thread_count << "\n";
      std::cout << "2. C_PUCT value: " << settings.c_puct << "\n";
      std::cout << "3. Thinking time (ms): " << settings.think_time_ms << "\n";
      std::cout << "4. Player name: " << settings.player_name << "\n";
      std::cout << "5. AI name: " << settings.ai_name << "\n";
      std::cout << "6. Return to main menu\n";
      std::cout << "Choose a setting to modify: ";

      int choice;
      std::cin >> choice;

      switch (choice) {
        case 1:
          std::cout << "Enter new thread count: ";
          std::cin >> settings.thread_count;
          break;
        case 2:
          std::cout << "Enter new C_PUCT value: ";
          std::cin >> settings.c_puct;
          break;
        case 3:
          std::cout << "Enter new thinking time (ms): ";
          std::cin >> settings.think_time_ms;
          break;
        case 4:
          std::cout << "Enter player name: ";
          std::cin >> settings.player_name;
          break;
        case 5:
          std::cout << "Enter AI name: ";
          std::cin >> settings.ai_name;
          break;
        case 6:
          settings.saveToFile();
          current_state = MAIN_MENU;
          return;
        default:
          std::cout << "Invalid choice!\n";
      }
    }
  }

  // AI换手判断函数 (占位符)
  bool ai_should_swap(const Utils::Board& board) {
    // TODO: 实现AI换手判断逻辑
    // 这里应该分析棋盘局势，判断是否应该换手
    // 暂时返回false作为占位符
    return false;
  }

  // AI第五手N个落子位置函数 (占位符)
  std::vector<int> ai_get_n_moves(const Utils::Board& board, int n) {
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
  void apply_fixed_opening(Utils::Board& board, int opening_choice) {
    // Clear board first
    for (auto &row : board)
      for (auto &cell : row)
        cell = Utils::EMPTY;
    
    switch (opening_choice) {
      case 1: // 疏星局 (B:H8,W:H9;B:J10)
        board[7][7] = Utils::BLACK;   // H8
        board[8][7] = Utils::WHITE;   // H9
        board[9][9] = Utils::BLACK;   // J10
        break;
      case 2: // 长星局 (B:H8;W:I9;B:J10)
        board[7][7] = Utils::BLACK;   // H8
        board[8][8] = Utils::WHITE;   // I9
        board[9][9] = Utils::BLACK;   // J10
        break;
      case 3: // 流星局 (B:H8;W:I9;B:J6)
        board[7][7] = Utils::BLACK;   // H8
        board[8][8] = Utils::WHITE;   // I9
        board[5][9] = Utils::BLACK;   // J6
        break;
      case 4: // 自定义开局
        std::cout << "\n=== Custom Opening Setup ===\n";
        std::cout << "First move is fixed at H8 (Black)\n";
        std::cout << "Please input 2 additional coordinates (e.g., I9 J10):\n";
        std::cout << "Format: Letter + Number (e.g., I9, J10)\n";
        
        // Store the 3 coordinates (first is fixed at H8)
        int coordinates[3][2];
        coordinates[0][0] = 7;  // H8 row
        coordinates[0][1] = 7;  // H8 column
        
        // Input 2 additional coordinates
        for (int i = 1; i < 3; i++) {
          bool valid_input = false;
          while (!valid_input) {
            std::string input;
            std::cout << "Enter coordinate " << (i + 1) << ": ";
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
                std::cout << "Invalid coordinate! Please enter a valid coordinate (e.g., A1-O15).\n";
              }
            } else {
              std::cout << "Invalid input format! Please use format: Letter + Number (e.g., I9).\n";
            }
          }
        }
        
        // Apply moves: Black (H8), White, Black
        board[coordinates[0][0]][coordinates[0][1]] = Utils::BLACK;  // First move (Black) at H8
        board[coordinates[1][0]][coordinates[1][1]] = Utils::WHITE;  // Second move (White)
        board[coordinates[2][0]][coordinates[2][1]] = Utils::BLACK;  // Third move (Black)
        
        std::cout << "Custom opening applied! (H8 + your 2 coordinates)\n";
        break;
    }
  }

  // Game setup
  void show_game_setup() {
    std::cout << "\n=== Game Setup ===\n";
    std::cout << "Choose first turn:\n";
    std::cout << "1. Player first\n";
    std::cout << "2. AI first\n";
    std::cout << "Choose: ";

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
      std::cout << "\n=== Fixed Opening Selection ===\n";
      std::cout << "Choose a fixed opening:\n";
      std::cout << "1. 疏星局 (best opening) (B:H8,W:H9;B:J10)\n";
      std::cout << "2. 长星局 (B:H8;W:I9;B:J10)\n";
      std::cout << "3. 流星局 (B:H8;W:I9;B:J6)\n";
      std::cout << "4. 自定义开局(input 2 coordinates)\n";
      std::cout << "Choose: ";
      
      int opening_choice;
      std::cin >> opening_choice;
      
      if (opening_choice >= 1 && opening_choice <= 4) {
        apply_fixed_opening(board, opening_choice);
        if (opening_choice != 4) {
          std::cout << "Fixed opening applied!\n";
        }
      }
    }
    else{
      std::cout << "You go first, please input 2 coordinates\n";
      apply_fixed_opening(board, 4);
    }

    // 1. 打印当前开局棋盘
    std::cout << "\n=== Current Opening Board ===\n";
    print_board(board);

    // 2. 换手逻辑
    bool should_swap = false;
    if (!game.my_turn_first) {
      // AI先手，使用AI换手判断函数
      should_swap = ai_should_swap(board);
      std::cout << "AI decides to " << (should_swap ? "swap" : "not swap") << " colors.\n";
    } else {
      // 玩家先手，询问玩家是否换手
      std::cout << "Do you want to swap colors? (y/n): ";
      std::string swap_input;
      std::cin >> swap_input;
      should_swap = (swap_input == "y" || swap_input == "Y");
      std::cout << "You decided to " << (should_swap ? "swap" : "not swap") << " colors.\n";
    }

    // 3. 如果要换手，交换双方的颜色
    if (should_swap) {
      game.my_turn_first = !game.my_turn_first;
      std::cout << "Colors swapped! ";
      if (game.my_turn_first) {
        std::cout << "You will play as Black.\n";
      } else {
        std::cout << "AI will play as Black.\n";
      }
    }
    //当前应该先手是白棋
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

    std::cout << "\n=== Game Start ===\n";
    std::cout << "Instructions:\n";
    std::cout << "- Place stone: Enter coordinates (e.g., H8)\n";
    std::cout << "- Pass: Enter 'pass'\n";
    std::cout << "- Undo: Enter 'undo'\n";
    std::cout << "- Quit: Enter 'quit'\n\n";

    print_board(game.agent->last_move_board());

    while (!game.game_ended && move_count < Config::BOARD_SQUARES) {
      const Utils::Board &root_board = game.agent->last_move_board();
      bool is_my_turn = (current_player == Utils::BLACK && game.my_turn_first) ||
                        (current_player == Utils::WHITE && !game.my_turn_first);

      if (is_my_turn) {
        // Player's turn
        std::cout << "Your turn (" << (current_player == Utils::BLACK ? "Black" : "White") << "): ";
        std::string input;
        std::cin >> input;

        if (input == "quit") {
          std::cout << "Game over\n";
          break;
        }

        if (input == "undo") {
          if (game.record.undoLastMove()) {
            std::cout << "Undo successful\n";
            // Reinitialize agent state, simplified handling
            continue;
          } else {
            std::cout << "Cannot undo\n";
            continue;
          }
        }

        if (input == "pass") {
          std::cout << "You chose to pass\n";
          game.agent->apply_move(-1);
          game.consecutive_passes++;
          game.record.addMove(current_player == Utils::BLACK ? Utils::BLACK : Utils::WHITE, -1);

          if (game.consecutive_passes >= 2) {
            std::cout << "Two consecutive passes, game is a draw!\n";
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
                std::cout << "You placed stone: " << input << "\n";
              } else {
                std::cout << "Position already occupied!\n";
                continue;
              }
            } else {
              std::cout << "Invalid coordinates!\n";
              continue;
            }
          } else {
            std::cout << "Invalid input!\n";
            continue;
          }
        }
      } else {
        // AI's turn
        std::cout << "AI is thinking...\n";
        
        // 启动思考计时器
        thinking_timer.startThinking();
        
        auto start = std::chrono::high_resolution_clock::now();
        // Calculate simulations based on thinking time (rough estimate)
        int target_simulations = std::max(1000, settings.think_time_ms * 10);
        game.agent->run_mcts(target_simulations);
        int sim_count = game.agent->get_simulations_completed();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        // 停止思考计时器
        thinking_timer.stopThinking();

        int move_idx = game.agent->next_move_idx();
        //五手N打
        if(move_idx == 5){
          // 第五手特殊处理：AI显示N个落子位置供玩家选择
          std::cout << "\n=== Fifth Move Selection ===\n";
          
          // 让玩家输入位置数量
          int num_positions;
          bool valid_input = false;
          while (!valid_input) {
            std::cout << "How many positions do you want? (2-5): ";
            std::cin >> num_positions;
            
            if (num_positions >= 2 && num_positions <= 5) {
              valid_input = true;
            } else {
              std::cout << "Invalid number! Please enter a number between 2 and 5.\n";
            }
          }
          
          std::cout << "AI suggests " << num_positions << " possible moves:\n";
          
          std::vector<int> n_moves = ai_get_n_moves(game.agent->last_move_board(), num_positions);
          
          // 显示N个位置
          for (int i = 0; i < num_positions; i++) {
            int r = n_moves[i] / Config::BOARD_SIZE;
            int c = n_moves[i] % Config::BOARD_SIZE;
            char col = 'A' + c;
            std::cout << (i + 1) << ". " << col << (r + 1) << "\t";
          }
          std::cout << std::endl;
          // 让玩家选择
          std::cout << "Please choose a move (1-" << num_positions << "): ";
          int choice;
          std::cin >> choice;
          
          if (choice >= 1 && choice <= num_positions) {
            move_idx = n_moves[choice - 1];
            std::cout << "You chose move " << choice << ".\n";
          } else {
            std::cout << "Invalid choice! Using first move.\n";
            move_idx = n_moves[0];
          }
        }
        if (move_idx == -1) {
          std::cout << "AI chose to pass\n";
          game.agent->apply_move(-1);
          game.consecutive_passes++;
          game.record.addMove(current_player == Utils::BLACK ? Utils::BLACK : Utils::WHITE, -1);

          if (game.consecutive_passes >= 2) {
            std::cout << "Two consecutive passes, game is a draw!\n";
            game.game_ended = true;
            game.record.setResult(0); // Draw
            break;
          }
        } else {
          game.agent->apply_move(move_idx);
          int r = move_idx / Config::BOARD_SIZE;
          int c = move_idx % Config::BOARD_SIZE;
          char col = 'A' + c;
          std::cout << "AI placed stone: " << col << (r + 1) << " | Time: " << duration.count() << "ms | Simulations: " << sim_count
                    << "\n";
          game.record.addMove(current_player == Utils::BLACK ? Utils::BLACK : Utils::WHITE, move_idx);
          game.consecutive_passes = 0;
        }
      }

      print_board(game.agent->last_move_board());

      // Check win
      if (check_win(game.agent->last_move_board(), current_player)) {
        std::string winner = (current_player == Utils::BLACK ? "Black" : "White");
        std::cout << winner << " wins!\n";
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
      std::cout << "Game is a draw!\n";
      game.record.setResult(0);
    }

    // 停止游戏计时器并显示总思考时间
    thinking_timer.stopGame();
    double total_time = thinking_timer.getTotalThinkingTime();
    int total_seconds = static_cast<int>(total_time);
    int minutes = total_seconds / 60;
    int seconds = total_seconds % 60;
    std::cout << "\n=== Game Summary ===\n";
    std::cout << "Total AI thinking time: " << std::setfill('0') << std::setw(2) << minutes << ":" << std::setw(2) << seconds << " (MM:SS)\n";

    // Save game record
    save_game_record();

    std::cout << "\nPress any key to return to main menu...\n";
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

    std::cout << "\n=== Game Record ===\n";
    std::cout << game.record.toString() << "\n";
    std::cout << "\nGame record saved to: " << filename << "\n";
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

#include "config.hpp"
#include "gomoku_record.hpp"
#include "multithread_mcts.hpp"
#include "network.hpp"
#include "zobrist.hpp"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

using namespace MultiThreadMCTS;
using namespace GomokuRecord;

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

  // Game state
  struct GameState {
    bool my_turn_first;
    bool game_ended;
    GomokuGameRecord record;
    int consecutive_passes;
    std::unique_ptr<MCTSAgent> agent;
    NodeTable node_table;
    Network net;
  } game;

public:
  GameInterface() : current_state(MAIN_MENU) { settings.loadFromFile(); }

  // Display aligned board
  void print_board(const Board &board) {
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
    Board board{};
    for (auto &row : board)
      for (auto &cell : row)
        cell = Utils::EMPTY;

    Zobrist::init();

    MultiThreadHelpers::MultiThreadConfig config;
    config.num_search_threads = settings.thread_count;

    game.agent = std::make_unique<MCTSAgent>(board, Utils::BLACK, game.net, game.node_table, config);

    // Initialize game record
    game.record = GomokuGameRecord(settings.player_name, settings.ai_name);

    current_state = GAME_PLAYING;
  }

  // Game loop
  void game_loop() {
    Utils::STONE_COLOR current_player = Utils::BLACK;
    int move_count = 0;

    std::cout << "\n=== Game Start ===\n";
    std::cout << "Instructions:\n";
    std::cout << "- Place stone: Enter coordinates (e.g., H8)\n";
    std::cout << "- Pass: Enter 'pass'\n";
    std::cout << "- Undo: Enter 'undo'\n";
    std::cout << "- Quit: Enter 'quit'\n\n";

    print_board(game.agent->get_root()->board_state);

    while (!game.game_ended && move_count < Config::BOARD_SQUARES) {
      Board &root_board = game.agent->get_root()->board_state;
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
          game.agent->apply_move(-1, game.node_table);
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
                game.agent->apply_move(move_idx, game.node_table);
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
        auto start = std::chrono::high_resolution_clock::now();
        int sim_count = game.agent->run_mcts_parallel_with_stop(20000000, settings.think_time_ms / 1000.0);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        int move_idx = game.agent->next_move_idx();

        if (move_idx == -1) {
          std::cout << "AI chose to pass\n";
          game.agent->apply_move(-1, game.node_table);
          game.consecutive_passes++;
          game.record.addMove(current_player == Utils::BLACK ? Utils::BLACK : Utils::WHITE, -1);

          if (game.consecutive_passes >= 2) {
            std::cout << "Two consecutive passes, game is a draw!\n";
            game.game_ended = true;
            game.record.setResult(0); // Draw
            break;
          }
        } else {
          game.agent->apply_move(move_idx, game.node_table);
          int r = move_idx / Config::BOARD_SIZE;
          int c = move_idx % Config::BOARD_SIZE;
          char col = 'A' + c;
          std::cout << "AI placed stone: " << col << (r + 1) << " | Time: " << duration.count() << "ms | Simulations: " << sim_count
                    << "\n";
          game.record.addMove(current_player == Utils::BLACK ? Utils::BLACK : Utils::WHITE, move_idx);
          game.consecutive_passes = 0;
        }
      }

      print_board(game.agent->get_root()->board_state);

      // Check win
      if (check_win(game.agent->get_root()->board_state, current_player)) {
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

    // Save game record
    save_game_record();

    std::cout << "\nPress any key to return to main menu...\n";
    std::cin.ignore();
    std::cin.get();
    current_state = MAIN_MENU;
  }

  // Check win
  bool check_win(const Board &board, Utils::STONE_COLOR player) {
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
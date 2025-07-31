#pragma once
#include "mcts.hpp"
#include "config.hpp"
#include "utils.hpp"
#include <iostream>

/*
 * Example demonstrating pass move handling in MCTS
 * 
 * This file shows how the MCTS implementation correctly handles:
 * 1. Pass moves as valid actions (index Network::PASS_IDX = 225)
 * 2. Board state inheritance for pass moves
 * 3. Player color switching on pass moves
 * 4. Consecutive pass detection leading to draw
 */

namespace PassMoveExample {

// Example showing basic pass move functionality
inline void demonstrate_pass_move() {
    // Create a nearly full board to encourage pass moves
    Utils::Board board;
    for (int i = 0; i < Config::BOARD_SIZE; ++i) {
        for (int j = 0; j < Config::BOARD_SIZE; ++j) {
            if (i == 0 && j == 0) {
                board[i][j] = Utils::EMPTY; // Leave one empty spot
            } else {
                board[i][j] = ((i + j) % 2 == 0) ? Utils::BLACK : Utils::WHITE;
            }
        }
    }
    
    MCTS::MCTSAgent agent(board, Utils::BLACK, 4);
    agent.run_mcts(100);
    
    int best_move = agent.next_move_idx();
    
    // The agent might choose to pass if the remaining move is not favorable
    if (agent.is_pass_move(best_move)) {
        std::cout << "Agent chose to pass" << std::endl;
        
        // Apply the pass move
        agent.apply_move(best_move);
        
        // Now it's White's turn - if White also passes, game ends in draw
        agent.run_mcts(100);
        int white_move = agent.next_move_idx();
        
        if (agent.is_pass_move(white_move)) {
            std::cout << "Both players passed - game ends in draw" << std::endl;
        }
    }
}

// Example showing pass move in action selection
inline void demonstrate_pass_in_selection() {
    Utils::Board empty_board;
    for (int i = 0; i < Config::BOARD_SIZE; ++i) {
        for (int j = 0; j < Config::BOARD_SIZE; ++j) {
            empty_board[i][j] = Utils::EMPTY;
        }
    }
    
    MCTS::MCTSAgent agent(empty_board, Utils::BLACK, 1);
    agent.run_mcts(1000);
    
    // Even on empty board, pass is a valid option (though likely low-scored)
    std::cout << "Move descriptions for different indices:" << std::endl;
    std::cout << "Move 0: " << agent.get_move_description(0) << std::endl;
    std::cout << "Move 112: " << agent.get_move_description(112) << std::endl;
    std::cout << "Pass move: " << agent.get_move_description(Network::PASS_IDX) << std::endl;
}

// Example showing consecutive pass detection
inline void demonstrate_consecutive_passes() {
    // Create a simple scenario where passes might occur
    Utils::Board board;
    for (int i = 0; i < Config::BOARD_SIZE; ++i) {
        for (int j = 0; j < Config::BOARD_SIZE; ++j) {
            board[i][j] = Utils::EMPTY;
        }
    }
    
    MCTS::MCTSAgent agent(board, Utils::BLACK);
    
    // Manually force a pass move scenario
    // In real games, this would happen naturally through MCTS selection
    
    // Apply a pass move
    agent.apply_move(Network::PASS_IDX);
    std::cout << "Black passed" << std::endl;
    
    // Apply another pass move  
    agent.apply_move(Network::PASS_IDX);
    std::cout << "White passed" << std::endl;
    
    // The second pass should trigger consecutive pass detection
    // The game state should now be considered ended with a draw
    std::cout << "Game should end in draw due to consecutive passes" << std::endl;
}

// Helper function to print board state
inline void print_board_state(const Utils::Board& board) {
    std::cout << "\nBoard state:" << std::endl;
    for (int i = 0; i < Config::BOARD_SIZE; ++i) {
        for (int j = 0; j < Config::BOARD_SIZE; ++j) {
            char symbol = '.';
            if (board[i][j] == Utils::BLACK) symbol = 'X';
            else if (board[i][j] == Utils::WHITE) symbol = 'O';
            std::cout << symbol << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

// Example showing complete game with pass moves
inline void demonstrate_complete_game_with_passes() {
    Utils::Board board;
    for (int i = 0; i < Config::BOARD_SIZE; ++i) {
        for (int j = 0; j < Config::BOARD_SIZE; ++j) {
            board[i][j] = Utils::EMPTY;
        }
    }
    
    MCTS::MCTSAgent agent(board, Utils::BLACK, 4);
    
    std::cout << "=== Game with Pass Move Support ===" << std::endl;
    print_board_state(board);
    
    int move_count = 0;
    while (move_count < 10) { // Limit for demonstration
        agent.run_mcts(200);
        int best_move = agent.next_move_idx();
        
        if (best_move == -1) {
            std::cout << "No valid moves available" << std::endl;
            break;
        }
        
        std::cout << "Move " << (move_count + 1) << ": " 
                  << (agent.last_move_color() == Utils::BLACK ? "Black" : "White")
                  << " plays " << agent.get_move_description(best_move) << std::endl;
        
        agent.apply_move(best_move);
        
        if (!agent.is_pass_move(best_move)) {
            print_board_state(agent.last_move_board());
        }
        
        move_count++;
    }
}

} // namespace PassMoveExample
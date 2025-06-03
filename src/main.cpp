#include "network.hpp"
#include <tuple>
#ifdef _BOTZONE_ONLINE
#include "nlohmann/json.hpp"
#else
#include "json.hpp"
#endif
#include "mcts.hpp"
#include <chrono>
#include <cstddef>
#include <iostream>
#include <ostream>
#include <string>
#include <utility>

using json = nlohmann::json;

void response(int action_idx, int sim_count, long long elapsed) {
  json response_json;
  std::tie(response_json["response"]["x"], response_json["response"]["y"]) = Utils::index_to_coordinate(action_idx);
  response_json["debug"] =
      "Completed " + std::to_string(sim_count) + " simulations in " + std::to_string(elapsed) + " ms.";
  std::cout << response_json.dump() << "\n";
  std::cout << "\n>>>BOTZONE_REQUEST_KEEP_RUNNING<<<\n";
  std::cout << std::flush;
}

auto get_next_move(MCTS_Agent &agent) {
  auto simulate = [](MCTS_Agent &agent) {
    auto start_time = std::chrono::high_resolution_clock::now();
    auto deadline = start_time + std::chrono::milliseconds(Config::TIME_FOR_SIMS); // Reserve 50ms buffer

    int count = 0;
    while (true) {
      auto current_time = std::chrono::high_resolution_clock::now();
      auto time_remaining = std::chrono::duration_cast<std::chrono::duration<double>>(deadline - current_time).count();

      if (time_remaining < Config::FORWARD_TIME_COST) {
        break;
      }

      agent.run_mcts();
      ++count;
    }
    auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time)
            .count();
    return std::pair{count, elapsed};
  };

  auto sim_info = simulate(agent);
  auto best_move_idx = agent.next_move_idx();
  return std::pair(sim_info, best_move_idx);
}

auto parse_request() {
  std::string line;
  std::getline(std::cin, line);
  json input_json = json::parse(line);

  Utils::Coordinate action;

  if (input_json.find("requests") != input_json.end()) {
    action = {input_json["requests"][0]["x"].get<int>(), input_json["requests"][0]["y"].get<int>()};
  } else {
    action = {input_json["x"].get<int>(), input_json["y"].get<int>()};
  }

  return action;
}

MCTS_Agent handle_first_turn(AlphaGomoku::Network &net, int &sim_count, long long &elapsed) {
  Utils::Coordinate first_move = parse_request();
  AlphaGomoku::STONE_COLOR player_color;
  Board initial_board{};
  if (first_move.first == -1) {
    player_color = AlphaGomoku::STONE_COLOR::BLACK;
  } else {
    player_color = AlphaGomoku::STONE_COLOR::WHITE;
    initial_board[first_move.first][first_move.second] = AlphaGomoku::BLACK;
  }
  MCTS_Agent agent(initial_board, player_color, net);
  auto [sim_info, action_idx] = get_next_move(agent);
  std::tie(sim_count, elapsed) = sim_info;
  agent.apply_move(action_idx);
  return std::move(agent);
}

void handle_second_turn(MCTS_Agent &agent) {
  auto move = parse_request();
  if (!(agent.last_move_color() == AlphaGomoku::STONE_COLOR::WHITE && move.first == -1)) {
    agent.apply_move(Utils::coordinate_to_index(move));
  }
  auto [sim_info, action_idx] = get_next_move(agent);
  agent.apply_move(action_idx);
  response(action_idx, sim_info.first, sim_info.second);
}

AlphaGomoku::Network net;
int main() {
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(NULL);

  int sim_count = 0;
  long long elapsed = 0;
  auto agent = handle_first_turn(net, sim_count, elapsed);
  response(agent.root->prior_action_idx, sim_count, elapsed);
  handle_second_turn(agent);

  while (true) {
    auto move = parse_request();
    agent.apply_move(Utils::coordinate_to_index(move));
    auto [sim_info, action_idx] = get_next_move(agent);
    agent.apply_move(action_idx);
    response(action_idx, sim_info.first, sim_info.second);
  }
  return 0;
}
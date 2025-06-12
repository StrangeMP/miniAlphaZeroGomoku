#include "network.hpp"
#include <random>
#include <tuple>
#ifdef _BOTZONE_ONLINE
#include "nlohmann/json.hpp"
#else
#include "json.hpp"
#endif
#include "heuristic.hpp"
#include "mcts.hpp"
#include <chrono>
#include <cstddef>
#include <iostream>
#include <ostream>
#include <string>
#include <utility>

using json = nlohmann::json;

struct DecisionInfo {
  Utils::Coordinate action;
  int sim_count;
  long long elapsed_time;
  std::string heuristic_info;
};

void response(DecisionInfo info) {
  json response_json;
  std::tie(response_json["response"]["x"], response_json["response"]["y"]) = info.action;
  response_json["debug"] = info.heuristic_info + "; Completed " + std::to_string(info.sim_count) + " simulations in " +
                           std::to_string(info.elapsed_time) + " ms.";
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
  }; // returns {sim_count, elapsed_time}

  auto [sim_count, elapsed] = simulate(agent);
  auto [heuristic_move_idx, heuristic_info] =
      make_heuristic_decision(agent.last_move_board(), agent.last_move_color(), agent.next_move_color(),
                              agent.next_move_idx(), Utils::legal_moves(agent.last_move_board()), agent);
  return DecisionInfo{Utils::index_to_coordinate(heuristic_move_idx), sim_count, elapsed, std::move(heuristic_info)};
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

MCTS_Agent handle_first_turn(AlphaGomoku::Network &net) {
  Utils::Coordinate first_move = parse_request();
  auto is_first_move_around_center = [](const Utils::Coordinate move) {
    if (move.first == -1)
      return false; // Invalid move
    int center = Config::BOARD_SIZE / 2;
    int dx = std::abs(move.first - center);
    int dy = std::abs(move.second - center);
    return dx <= 2 && dy <= 2; // Within 2 squares of center
  };
  AlphaGomoku::STONE_COLOR player_color = AlphaGomoku::STONE_COLOR::BLACK;
  MCTS_Agent agent(Board{}, player_color, net);
  auto decision_info = get_next_move(agent);
  if (first_move.first == -1) { // We are playing black
    agent.apply_move(Utils::coordinate_to_index(decision_info.action));
  } else { // we are playing white
    agent.apply_move(Utils::coordinate_to_index(first_move));
    // always swap our color to have the first move
    decision_info.action = {-1, -1};
    decision_info.heuristic_info = "换手; ";
  }
  response(std::move(decision_info));
  return agent;
}

void handle_second_turn(MCTS_Agent &agent) {
  auto move = parse_request();
  if (!(agent.next_move_color() == AlphaGomoku::STONE_COLOR::BLACK && move.first == -1)) {
    agent.apply_move(Utils::coordinate_to_index(move));
  }
  auto decision_info = get_next_move(agent);
  agent.apply_move(Utils::coordinate_to_index(decision_info.action));
  response(std::move(decision_info));
}

AlphaGomoku::Network net;
int main() {
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(NULL);

  auto agent = handle_first_turn(net);
  handle_second_turn(agent);

  while (true) {
    auto move = parse_request();
    agent.apply_move(Utils::coordinate_to_index(move));
    auto decision_info = get_next_move(agent);
    agent.apply_move(Utils::coordinate_to_index(decision_info.action));
    response(std::move(decision_info));
  }
  return 0;
}
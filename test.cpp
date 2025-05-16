#include "compute.hpp"
#include "nnet.hpp"
#include <cstddef>
#include <cstdint>
#include <iostream>

constexpr size_t Filters = 32;
constexpr size_t Channels = 7;
constexpr size_t Height = 15;
constexpr size_t Width = 15;
constexpr size_t Blocks = 4;
using param_t = int8_t;

AZNet<param_t, Filters, Channels, Height, Width, Blocks> net;

int main() {
  Tensor<param_t, Channels, Height, Width> input;
  auto [prob, v] = net.feed(input);
  std::cout << "Probabilities: " << prob << "\n";
  std::cout << "Value: " << v << "\n";
  return 0;
}
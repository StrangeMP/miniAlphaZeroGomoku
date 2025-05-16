#pragma once

#include "compute.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <utility>

struct Sigmoid {
  template <typename T> static T call(const T &x) { return 1 / (1 + std::exp(-x)); }
};

struct Tanh {
  template <typename T> static T call(const T &x) { return std::tanh(x); }
};

struct ReLU {
  template <typename T> static T call(const T &x) { return std::max(T(0), x); }
};

struct Linear {
  template <typename T> static T call(const T &x) { return x; }
};

template <typename T, size_t L> struct Neuron {
  Vec<T, L> weights;
  T bias;

  Neuron() : weights{}, bias{0} {}

  Neuron(const Vec<T, L> &w, const T &b) : weights{w}, bias{b} {}

  Neuron(const Neuron &n) = default;
  Neuron(Neuron &&n) = default;
  Neuron &operator=(const Neuron &n) {
    if (this != &n) {
      weights = n.weights;
      bias = n.bias;
    }
    return *this;
  }

  Neuron &operator=(Neuron &&n) {
    if (this != &n) {
      weights = std::move(n.weights);
      bias = std::move(n.bias);
    }
    return *this;
  }

  T activate(const Vec<T, L> &input) { return weights.dot(input) + bias; }
};

template <typename T, size_t Params, size_t Filters> struct LayerBase {
  std::array<Neuron<T, Params>, Filters> neurons;

  LayerBase() = default;

  LayerBase(const std::array<Neuron<T, Params>, Filters> &n) : neurons{n} {}

  LayerBase(const LayerBase &l) = default;
  LayerBase(LayerBase &&l) = default;
  LayerBase &operator=(const LayerBase &l) = default;
  LayerBase &operator=(LayerBase &&l) = default;

  template <typename TensorWindow> void feed(Vec<T, Params> &input, TensorWindow w) {
    size_t i = 0;
    for (auto &neuron : neurons) {
      w.at(i, 0, 0) = neuron.activate(input);
      ++i;
    }
  }
};

template <typename T, size_t Filters, size_t InputChannels, size_t InputHeight, size_t InputWidth,
          bool SamePadding = false, size_t KernelHeight = 3, size_t KernelWidth = 3, size_t Stride = 1>
struct ConvLayer {
  LayerBase<T, InputChannels * KernelHeight * KernelWidth, Filters> layer;
  using InputTensor = Tensor<T, InputChannels, InputHeight, InputWidth>;

  static constexpr size_t OutputHeight =
      SamePadding ? ceil_div(InputHeight, Stride) : (InputHeight - KernelHeight) / Stride + 1;
  static constexpr size_t OutputWidth =
      SamePadding ? ceil_div(InputWidth, Stride) : (InputWidth - KernelWidth) / Stride + 1;
  Tensor<T, Filters, OutputHeight, OutputWidth> _output;

  void feed(InputTensor &input) {
    for (size_t out_y; out_y < OutputHeight; ++out_y) {
      for (size_t out_x = 0; out_x < OutputWidth; ++out_x) {
        int in_y, in_x;
        if constexpr (SamePadding) {
          // Calculate padding needed
          int total_pad_h = std::max(0, static_cast<int>((OutputHeight - 1) * Stride + KernelHeight - InputHeight));
          int total_pad_w = std::max(0, static_cast<int>((OutputWidth - 1) * Stride + KernelWidth - InputWidth));

          int pad_top = total_pad_h / 2;
          int pad_left = total_pad_w / 2;

          // Apply padding offset to input position
          in_y = static_cast<int>(out_y * Stride) - pad_top;
          in_x = static_cast<int>(out_x * Stride) - pad_left;
        } else {
          in_y = out_y * Stride;
          in_x = out_x * Stride;
        }

        auto receptive_field = make_TensorWindow<KernelHeight, KernelWidth, InputChannels>(input, in_y, in_x, 0);
        auto vec = receptive_field.flatten();
        auto entry = make_TensorWindow<1, 1, Filters>(_output, out_y, out_x, 0);
        layer.feed(vec, entry);
      }
    }
  }
};

template <typename T, size_t Channels, size_t Height, size_t Width> struct NormLayer {
  std::array<T, Channels> gamma;
  std::array<T, Channels> beta;
  std::array<T, Channels> mean;
  std::array<T, Channels> var;
  static constexpr T epsilon = 1e-5;

  Tensor<T, Channels, Height, Width> _output;

  NormLayer() {
    for (size_t i = 0; i < Channels; ++i) {
      gamma[i] = T(1);
      beta[i] = T(0);
      mean[i] = T(0);
      var[i] = T(1);
    }
  }

  NormLayer(const std::array<T, Channels> &g, const std::array<T, Channels> &b, const std::array<T, Channels> &m,
            const std::array<T, Channels> &v)
      : gamma(g), beta(b), mean(m), var(v) {}

  void feed(const Tensor<T, Channels, Height, Width> &input) {
    for (size_t c = 0; c < Channels; ++c) {
      for (size_t h = 0; h < Height; ++h) {
        for (size_t w = 0; w < Width; ++w) {
          // Apply normalization formula
          T normalized = (input[c][h][w] - mean[c]) / std::sqrt(var[c] + epsilon);
          _output[c][h][w] = gamma[c] * normalized + beta[c];
        }
      }
    }
  }
};

// Activation layer to apply activation function
template <typename T, size_t Channels, size_t Height, size_t Width, typename Activation = ReLU> struct ActivationLayer {
  Tensor<T, Channels, Height, Width> _output;

  void feed(const Tensor<T, Channels, Height, Width> &input) {
    for (size_t c = 0; c < Channels; ++c) {
      for (size_t h = 0; h < Height; ++h) {
        for (size_t w = 0; w < Width; ++w) {
          _output[c][h][w] = Activation::call(input[c][h][w]);
        }
      }
    }
  }
};

template <typename T, size_t Filters, size_t Channels, size_t Height, size_t Width, bool SamePadding = false,
          typename A = ReLU, size_t KernelHeight = 3, size_t KernelWidth = 3, size_t Stride = 1>
struct ResBlock {
  ConvLayer<T, Filters, Channels, Height, Width, SamePadding, KernelHeight, KernelWidth, Stride> conv;
  NormLayer<T, Filters, Height, Width> norm;
  ActivationLayer<T, Filters, Height, Width, A> activation;
  using InputTensor = typename decltype(conv)::InputTensor;
  using OutputTensor = decltype(conv._output);

  void feed(InputTensor &input) {
    conv.feed(input);
    norm.feed(conv._output);
    activation.feed(norm._output);
  }

  OutputTensor &output() { return activation._output; }
  const OutputTensor &output() const { return activation._output; }
};

template <typename T, size_t Filters, size_t Channels, size_t Height, size_t Width, size_t Blocks,
          typename Activation = ReLU, size_t KernelHeight = 3, size_t KernelWidth = 3, size_t Stride = 1>
struct ResNet {
  ConvLayer<T, Filters, Channels, Height, Width, true> conv;
  std::array<ResBlock<T, Filters, Filters, Height, Width, true, Activation, KernelHeight, KernelWidth, Stride>, Blocks>
      blocks;
  using InputTensor = typename decltype(conv)::InputTensor;
  using OutputTensor = typename decltype(blocks)::value_type::OutputTensor;

  void feed(InputTensor &input) {
    conv.feed(input);
    OutputTensor *block_input = &conv._output;
    for (auto &block : blocks) {
      block.feed(*block_input);
      block_input = &block.output();
    }
  }

  OutputTensor &output() { return blocks.back().output(); }
  const OutputTensor &output() const { return blocks.back().output(); }
};
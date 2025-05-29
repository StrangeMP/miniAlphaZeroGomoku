#pragma once

#include "compute.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <type_traits>
#include <utility>

struct Sigmoid {
  template <typename T> static T call(const T &x) {
    return 1 / (1 + std::exp(-x));
  }
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

template <typename T, size_t L> Vec<float, L> softmax(const Vec<T, L> &v) {
  Vec<float, L> result;
  T max_val = *std::max_element(v.begin(), v.end());
  T sum = T(0);
  for (size_t i = 0; i < L; ++i) {
    result[i] = std::exp(v[i] - max_val);
    sum += result[i];
  }
  for (size_t i = 0; i < L; ++i) {
    result[i] /= sum;
  }
  return result;
}

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

template <typename T, size_t Params, size_t Filters> struct BasicLayer {
  std::array<Neuron<T, Params>, Filters> neurons;

  BasicLayer() = default;

  BasicLayer(const std::array<Neuron<T, Params>, Filters> &n) : neurons{n} {}

  BasicLayer(const BasicLayer &l) = default;
  BasicLayer(BasicLayer &&l) = default;
  BasicLayer &operator=(const BasicLayer &l) = default;
  BasicLayer &operator=(BasicLayer &&l) = default;

  template <size_t D = 0, typename TensorWindow = void>
  void feed(Vec<T, Params> &input, TensorWindow w) {
    size_t i = 0;
    for (auto &neuron : neurons) {
      switch (D) {
      case 0:
        w.at(i, 0, 0) = neuron.activate(input);
        break;
      case 1:
        w.at(0, i, 0) = neuron.activate(input);
        break;
      case 2:
        w.at(0, 0, i) = neuron.activate(input);
        break;
      default:
        throw std::out_of_range("Invalid dimension for feed");
      }
      ++i;
    }
  }
};

template <typename T, size_t Filters, size_t InputChannels, size_t InputHeight,
          size_t InputWidth, bool SamePadding = true, size_t KernelHeight = 3,
          size_t KernelWidth = 3, size_t Stride = 1>
struct ConvLayer {
  BasicLayer<T, InputChannels * KernelHeight * KernelWidth, Filters> layer;
  using InputTensor = Tensor<T, InputChannels, InputHeight, InputWidth>;

  static constexpr size_t OutputHeight =
      SamePadding ? ceil_div(InputHeight, Stride)
                  : (InputHeight - KernelHeight) / Stride + 1;
  static constexpr size_t OutputWidth =
      SamePadding ? ceil_div(InputWidth, Stride)
                  : (InputWidth - KernelWidth) / Stride + 1;
  Tensor<T, Filters, OutputHeight, OutputWidth> _output;
  using OutputTensor = decltype(_output);

  void feed(InputTensor &input) {
    for (size_t out_y = 0; out_y < OutputHeight; ++out_y) {
      for (size_t out_x = 0; out_x < OutputWidth; ++out_x) {
        int in_y, in_x;
        if constexpr (SamePadding) {
          // Calculate padding needed
          int total_pad_h =
              std::max(0, static_cast<int>((OutputHeight - 1) * Stride +
                                           KernelHeight - InputHeight));
          int total_pad_w =
              std::max(0, static_cast<int>((OutputWidth - 1) * Stride +
                                           KernelWidth - InputWidth));

          int pad_top = total_pad_h / 2;
          int pad_left = total_pad_w / 2;

          // Apply padding offset to input position
          in_y = static_cast<int>(out_y * Stride) - pad_top;
          in_x = static_cast<int>(out_x * Stride) - pad_left;
        } else {
          in_y = out_y * Stride;
          in_x = out_x * Stride;
        }

        auto receptive_field =
            make_TensorWindow<InputChannels, KernelHeight, KernelWidth>(
                input, in_y, in_x, 0);
        auto vec = receptive_field.flatten();
        auto entry = make_TensorWindow<Filters, 1, 1>(_output, out_y, out_x, 0);
        layer.feed(vec, entry);
      }
    }
  }

  decltype(_output) &output() { return _output; }
  const decltype(_output) &output() const { return _output; }
};

template <typename T, size_t Channels, size_t Height, size_t Width>
struct NormLayer {
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

  NormLayer(const std::array<T, Channels> &g, const std::array<T, Channels> &b,
            const std::array<T, Channels> &m, const std::array<T, Channels> &v)
      : gamma(g), beta(b), mean(m), var(v) {}

  void feed(const Tensor<T, Channels, Height, Width> &input) {
    for (size_t c = 0; c < Channels; ++c) {
      for (size_t h = 0; h < Height; ++h) {
        for (size_t w = 0; w < Width; ++w) {
          // Apply normalization formula
          T normalized =
              (input[c][h][w] - mean[c]) / std::sqrt(var[c] + epsilon);
          _output[c][h][w] = gamma[c] * normalized + beta[c];
        }
      }
    }
  }

  decltype(_output) &output() { return _output; }
  const decltype(_output) &output() const { return _output; }
};

// Activation layer to apply activation function
template <typename T, size_t Channels, size_t Height, size_t Width,
          typename Activation = ReLU>
struct ActivationLayer {
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

  decltype(_output) &output() { return _output; }
  const decltype(_output) &output() const { return _output; }
};

template <typename T, size_t Channels, size_t Height, size_t Width,
          typename Activation = ReLU>
struct FusedNormActivationLayer {
  Tensor<T, Channels, Height, Width> _output;

  std::array<T, Channels> gamma;
  std::array<T, Channels> beta;
  std::array<T, Channels> mean;
  std::array<T, Channels> var;
  static constexpr T epsilon = 1e-5;

  FusedNormActivationLayer() {
    for (size_t i = 0; i < Channels; ++i) {
      gamma[i] = T(1);
      beta[i] = T(0);
      mean[i] = T(0);
      var[i] = T(1);
    }
  }

  void feed(Tensor<T, Channels, Height, Width> &input) {
    for (size_t c = 0; c < Channels; ++c) {
      for (size_t h = 0; h < Height; ++h) {
        for (size_t w = 0; w < Width; ++w) {
          // Apply normalization formula
          T normalized =
              (input[c][h][w] - mean[c]) / std::sqrt(var[c] + epsilon);
          _output[c][h][w] = Activation::call(gamma[c] * normalized + beta[c]);
        }
      }
    }
  }

  decltype(_output) &output() { return _output; }
  const decltype(_output) &output() const { return _output; }
};

template <typename T, size_t Channels, size_t Height, size_t Width,
          bool SamePadding = true, typename Activation = ReLU,
          size_t KernelHeight = 3, size_t KernelWidth = 3, size_t Stride = 1>
struct ResBlock {
  ConvLayer<T, Channels, Channels, Height, Width, SamePadding, KernelHeight,
            KernelWidth, Stride>
      conv1, conv2;
  FusedNormActivationLayer<T, Channels, Height, Width> na1;
  NormLayer<T, Channels, Height, Width> norm2;
  ActivationLayer<T, Channels, Height, Width, Activation> activate2;

  using conv1_t = decltype(conv1);
  using InputTensor = typename conv1_t::InputTensor;
  using OutputTensor = typename conv1_t::OutputTensor;

  void feed(InputTensor &input) {
    conv1.feed(input);
    na1.feed(conv1._output);
    conv2.feed(na1._output);
    norm2.feed(conv2._output);
    conv2._output += input; // Residual connection
    activate2.feed(norm2._output);
  }

  OutputTensor &output() { return activate2.output(); }
  const OutputTensor &output() const { return activate2.output(); }
};

template <typename T, size_t Filters, size_t Height, size_t Width,
          size_t Blocks, typename Activation = ReLU, size_t KernelHeight = 3,
          size_t KernelWidth = 3, size_t Stride = 1>
struct ResNet {
  using BlockType = ResBlock<T, Filters, Height, Width, true, Activation,
                             KernelHeight, KernelWidth, Stride>;
  std::array<BlockType, Blocks> blocks;
  using InputTensor = typename BlockType::InputTensor;
  using OutputTensor = typename decltype(blocks)::value_type::OutputTensor;

  void feed(InputTensor &input) {
    OutputTensor *block_input = &input;
    for (auto &block : blocks) {
      block.feed(*block_input);
      block.output() += *block_input; // Residual connection
      block_input = &block.output();
    }
  }

  OutputTensor &output() { return blocks.back().output(); }
  const OutputTensor &output() const { return blocks.back().output(); }
};

template <typename InType, typename OutType, size_t Filters, size_t Channels,
          size_t Height, size_t Width, typename Activation = ReLU>
struct DenseLayer {
  BasicLayer<OutType, Channels * Height * Width, Filters> layer;
  Tensor<OutType, 1, 1, Filters> _output;

  void feed(const Tensor<InType, Channels, Height, Width> &input) {
    auto vec = input.flatten();
    auto entry = make_TensorWindow<1, 1, Filters>(_output, 0, 0, 0);
    if constexpr (std::is_same_v<InType, OutType>) {
      layer.template feed<2>(vec, entry);
    } else {
      auto vec_cast = Vec<OutType, Channels * Height * Width>(vec);
      layer.template feed<2>(vec_cast, entry);
    }
    auto &output = _output[0][0];
    for (auto &x : output) {
      x = Activation::call(x);
    }
  }

  decltype(_output) &output() { return _output; }
  const decltype(_output) &output() const { return _output; }

  auto &out_vector() { return _output[0][0]; }
  const auto &out_vector() const { return _output[0][0]; }
};

template <typename T, size_t ResFilters, size_t PolicyFilters,
          size_t ValueFilters, size_t Channels, size_t Height, size_t Width,
          size_t Blocks>
struct AZNet {
  using value_t = float;
  using prob_t = Vec<value_t, Height * Width>;
  ConvLayer<T, ValueFilters, Channels, Height, Width> conv1;
  FusedNormActivationLayer<T, ValueFilters, Height, Width> na1;
  ResNet<T, ResFilters, Height, Width, Blocks> resnet;

  // Policy head
  ConvLayer<T, PolicyFilters, ResFilters, Height, Width> pi_conv;
  FusedNormActivationLayer<T, PolicyFilters, Height, Width, ReLU> pi_na;
  DenseLayer<T, value_t, Height * Width, PolicyFilters, Height, Width> pi_final;

  // Value head
  ConvLayer<T, ValueFilters, ResFilters, Height, Width, false, 1, 1, 1> v_conv;
  FusedNormActivationLayer<T, ValueFilters, Height, Width> v_na;
  DenseLayer<T, value_t, 1, ValueFilters, Height, Width> v_final;

  Vec<value_t, Height * Width> pi;
  value_t v;

  using InputTensor = typename decltype(conv1)::InputTensor;

  auto feed(InputTensor &input) {
    conv1.feed(input);
    na1.feed(conv1.output());
    resnet.feed(na1.output());
    auto &resnet_output = resnet.output();

    // Policy head
    pi_conv.feed(resnet_output);
    pi_na.feed(pi_conv.output());
    pi_final.feed(pi_na.output());
    pi = softmax(pi_final.out_vector());

    // Value head
    v_conv.feed(resnet_output);
    v_na.feed(v_conv.output());
    v_final.feed(v_na.output());
    v = Tanh::call(v_final.out_vector()[0]);

    return output();
  }

  std::pair<const prob_t &, const value_t &> output() const { return {pi, v}; }
};
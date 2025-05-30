#pragma once

#include "compute.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

using SCALE_T = float;
using ZP_T = uint8_t;
using WEIGHT_T = uint8_t;
using BIAS_T = int32_t;
inline constexpr int WEIGHT_MIN = std::numeric_limits<WEIGHT_T>::min();
inline constexpr int WEIGHT_MAX = std::numeric_limits<WEIGHT_T>::max();

template <typename T> constexpr int32_t const_round_half_away_from_zero(T val) {
  // std::round behavior: rounds halves away from zero.
  // e.g., 2.5f -> 3, -2.5f -> -3
  // 0.0f -> 0
  return (val >= 0.0f) ? static_cast<int32_t>(val + 0.5f) : static_cast<int32_t>(val - 0.5f);
}

template <SCALE_T SCALE, ZP_T ZP> constexpr uint8_t quantize(SCALE_T x) {
  static_assert(SCALE != 0, "SCALE must not be zero");
  return static_cast<WEIGHT_T>(std::clamp(static_cast<int32_t>(std::round(x / SCALE)) + ZP, WEIGHT_MIN, WEIGHT_MAX));
}

template <SCALE_T SCALE, ZP_T ZP> constexpr SCALE_T dequantize(WEIGHT_T x) {
  return (static_cast<SCALE_T>(x) - ZP) * SCALE;
}

template <SCALE_T InputScale, ZP_T InputZP, SCALE_T WeightScale, ZP_T WeightZP, SCALE_T OutputScale, ZP_T OutputZP,
          size_t Filters, size_t InputChannels, size_t InputHeight, size_t InputWidth, size_t KernelHeight = 3,
          size_t KernelWidth = 3>
struct QLinearConv {
  Vec<Tensor<int32_t, InputChannels, KernelHeight, KernelWidth>, Filters> w_minus_wzp;
  const Vec<BIAS_T, Filters> &biases;
  Tensor<WEIGHT_T, Filters, InputHeight, InputWidth> _output;

  constexpr QLinearConv(const Vec<Tensor<WEIGHT_T, InputChannels, KernelHeight, KernelWidth>, Filters> &weights_,
                        const Vec<BIAS_T, Filters> &biases_)
      : biases(biases_) {
    for (int f = 0; f < Filters; ++f) {
      for (int c = 0; c < InputChannels; ++c) {
        for (size_t kh = 0; kh < KernelHeight; ++kh) {
          for (size_t kw = 0; kw < KernelWidth; ++kw) {
            // Pre-subtract weight zero point
            w_minus_wzp[f][c][kh][kw] = static_cast<int32_t>(weights_[f][c][kh][kw]) - static_cast<int32_t>(WeightZP);
          }
        }
      }
    }
  }

  using OutputTensor = decltype(_output);
  using InputTensor = Tensor<WEIGHT_T, InputChannels, InputHeight, InputWidth>;

  void feed(const InputTensor &input) {
    static constexpr double M = (InputScale * WeightScale) / OutputScale;

    constexpr int pad_h = KernelHeight / 2;
    constexpr int pad_w = KernelWidth / 2;

    for (int f = 0; f < Filters; ++f) {
      for (int h = 0; h < InputHeight; ++h) {
        for (int w = 0; w < InputWidth; ++w) {
          int32_t acc = biases[f];

          for (int c = 0; c < InputChannels; ++c) {
            for (int kh = 0; kh < KernelHeight; ++kh) {
              for (int kw = 0; kw < KernelWidth; ++kw) {
                int32_t ih = static_cast<int32_t>(h) + static_cast<int32_t>(kh) - pad_h;
                int32_t iw = static_cast<int32_t>(w) + static_cast<int32_t>(kw) - pad_w;

                // padding
                int32_t x_q = InputZP; // Default to zero point if out of bounds
                if (ih >= 0 && ih < InputHeight && iw >= 0 && iw < InputWidth) {
                  x_q = input[c][ih][iw];
                }

                acc += (x_q - InputZP) * w_minus_wzp[f][c][kh][kw];
              }
            }
          }

          // Apply bias and requantization
          auto prod = static_cast<double>(acc) * M;
          _output[f][h][w] = std::clamp(static_cast<int32_t>(std::round(prod)) + OutputZP,
                                        static_cast<int32_t>(std::numeric_limits<WEIGHT_T>::min()),
                                        static_cast<int32_t>(std::numeric_limits<WEIGHT_T>::max()));
        }
      }
    }
  }
  decltype(_output) &output() { return _output; }
  const decltype(_output) &output() const { return _output; }
};

// QElemWise: Element-wise addition and multiplication on quantized tensors, inplace applied to the first operand.
template <SCALE_T a_scale, ZP_T a_zp, SCALE_T b_scale, ZP_T b_zp, SCALE_T c_scale, ZP_T c_zp, size_t C, size_t H,
          size_t W>
struct QElemWise {
  using InputTensor = Tensor<WEIGHT_T, C, H, W>;
  using OutputTensor = Tensor<WEIGHT_T, C, H, W>;

  static void add(InputTensor &a, const InputTensor &b) {
    static constexpr SCALE_T Sac = a_scale / c_scale;
    static constexpr SCALE_T Sbc = b_scale / c_scale;
    for (int c = 0; c < C; ++c) {
      for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
          // C = (A_scale * (A - A_zero_point) + B_scale * (B - B_zero_point))/C_scale + C_zero_point
          SCALE_T result = static_cast<SCALE_T>(static_cast<int32_t>(a[c][h][w]) - a_zp) * Sac +
                           static_cast<SCALE_T>(static_cast<int32_t>(b[c][h][w]) - b_zp) * Sbc;
          int32_t out_q = static_cast<int32_t>(std::round(result)) + c_zp;
          a[c][h][w] = static_cast<WEIGHT_T>(std::clamp(out_q, WEIGHT_MIN, WEIGHT_MAX));
        }
      }
    }
  }

  static void mul(InputTensor &a, const InputTensor &b) {
    static constexpr SCALE_T M = a_scale * b_scale / c_scale;

    for (int c = 0; c < C; ++c) {
      for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
          int32_t a_q = static_cast<int32_t>(a[c][h][w]) - a_zp;
          int32_t b_q = static_cast<int32_t>(b[c][h][w]) - b_zp;
          int32_t prod = a_q * b_q;
          int32_t out_q = static_cast<int32_t>(std::round(prod * M)) + c_zp;
          a[c][h][w] = static_cast<WEIGHT_T>(std::clamp(out_q, WEIGHT_MIN, WEIGHT_MAX));
        }
      }
    }
  }
};

// BNop: Batch Normalization Operation, doing column wise tensor * scalar multiplication or addition
template <SCALE_T a_scale, ZP_T a_zp, SCALE_T b_scale, ZP_T b_zp, SCALE_T c_scale, ZP_T c_zp, size_t C, size_t H,
          size_t W, bool IsMul = true>
struct BNop {
  Vec<SCALE_T, W> Constants;

  constexpr BNop(const Vec<WEIGHT_T, W> &constants) : Constants(constants) {
    constexpr SCALE_T M = IsMul ? (a_scale * b_scale / c_scale) : (b_scale / c_scale);
    for (int i = 0; i < constants.size(); ++i) {
      SCALE_T b_val_minus_zp = Constants[i] - static_cast<SCALE_T>(b_zp);
      Constants[i] = b_val_minus_zp * M;
    }
  }

  using InputTensor = Tensor<WEIGHT_T, C, H, W>;
  void mul(InputTensor &input) {
    for (int w = 0; w < W; ++w) {
      for (int c = 0; c < C; c++) {
        for (int h = 0; h < H; ++h) {
          int32_t a_q = input[c][h][w] - a_zp;
          SCALE_T prod = static_cast<SCALE_T>(a_q) * Constants[w];
          int32_t out_q = static_cast<int32_t>(std::round(prod)) + c_zp;
          input[c][h][w] = static_cast<WEIGHT_T>(std::clamp(out_q, WEIGHT_MIN, WEIGHT_MAX));
        }
      }
    }
  }

  void add(InputTensor &input) {
    static constexpr SCALE_T Sac = a_scale / c_scale;
    for (int w = 0; w < W; ++w) {
      for (int c = 0; c < C; c++) {
        for (int h = 0; h < H; ++h) {
          int a_q = input[c][h][w] - a_zp;
          SCALE_T result = static_cast<SCALE_T>(a_q) * Sac + Constants[w];
          int32_t out_q = static_cast<int32_t>(std::round(result)) + c_zp;
          input[c][h][w] = static_cast<WEIGHT_T>(std::clamp(out_q, WEIGHT_MIN, WEIGHT_MAX));
        }
      }
    }
  }
};

// MatMul: Column-wise apply matrix multiplication on a weight matrix with a vector flattened from a tensor
template <SCALE_T a_scale, ZP_T a_zp, SCALE_T b_scale, ZP_T b_zp, SCALE_T c_scale, ZP_T c_zp, size_t FlattenLen,
          size_t Filters, bool Requant = true>
struct QGemm {
  using InputVector = Vec<WEIGHT_T, FlattenLen>;
  using InputMatrix = Matrix<WEIGHT_T, FlattenLen, Filters>;
  using OutputVector = Vec<std::conditional_t<Requant, WEIGHT_T, SCALE_T>, Filters>;
  using WeightsMatrix = Matrix<int32_t, FlattenLen, Filters>;
  using BiasVector = Vec<int32_t, Filters>;

  WeightsMatrix _weights_minus_wzp;
  const BiasVector &_biases;
  OutputVector _output;

  constexpr QGemm(const Matrix<uint8_t, FlattenLen, Filters> &weights, const BiasVector &biases) : _biases(biases) {
    for (int f = 0; f < Filters; ++f) {
      for (int i = 0; i < FlattenLen; ++i) {
        _weights_minus_wzp[i][f] = static_cast<int32_t>(weights[i][f]) - static_cast<int32_t>(b_zp);
      }
    }
  }

  void feed(const InputVector &vec) {
    for (int j = 0; j < Filters; ++j) {
      int32_t acc = _biases[j];
      for (int i = 0; i < FlattenLen; ++i) {
        acc += static_cast<int32_t>(vec[i] - a_zp) * _weights_minus_wzp[i][j];
      }
      SCALE_T dequantized_acc_value = static_cast<SCALE_T>(acc) * (a_scale * b_scale);
      _output[j] = Requant ? quantize<c_scale, c_zp>(dequantized_acc_value) : dequantized_acc_value;
    }
  }

  const OutputVector &output() const { return _output; }
};

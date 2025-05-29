#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <initializer_list>
#include <numeric>
#include <ostream>
#include <string>
#ifdef DEBUG
#include <format>
#endif

template <typename T, size_t L> struct Vec : std::array<T, L> {
  Vec() = default;

  using std::array<T, L>::array;
  constexpr Vec(std::initializer_list<T> il) {
    size_t i = 0;
    for (const auto &v : il) {
      (*this)[i] = v;
      ++i;
    }
  }

  template <typename U> constexpr Vec(const Vec<U, L> &v) {
    for (size_t i = 0; i < L; ++i) {
      (*this)[i] = static_cast<T>(v[i]);
    }
  }

  constexpr T dot(const Vec &v) const { return std::inner_product(this->begin(), this->end(), v.begin(), T{0}); }

  constexpr Vec &operator+=(const Vec &v) {
    for (size_t i = 0; i < L; ++i) {
      (*this)[i] += v[i];
    }
    return *this;
  }

  constexpr Vec &operator=(const Vec &v) { return static_cast<Vec &>(std::array<T, L>::operator=(v)); }

  operator std::string() const {
    std::string s = "[";
    for (size_t i = 0; i < L; ++i) {
      s += std::to_string((*this)[i]);
      if (i != L - 1) {
        s += ", ";
      }
    }
    s += "]";
    return s;
  }
};

template <typename T, size_t R, size_t C> struct Matrix : Vec<Vec<T, C>, R> {
  constexpr Matrix &operator+=(const Matrix &m) {
    for (size_t i = 0; i < R; ++i) {
      (*this)[i] += m[i];
    }
    return *this;
  }

  constexpr Matrix &operator=(const Matrix &m) { return static_cast<Matrix &>(Vec<Vec<T, C>, R>::operator=(m)); }

  operator std::string() const {
    std::string s = "[";
    for (size_t i = 0; i < R; ++i) {
      s += static_cast<std::string>((*this)[i]);
      if (i != R - 1) {
        s += ", ";
      }
    }
    s += "]";
    return s;
  }

  constexpr Matrix &operator=(const Vec<Vec<T, C>, R> &m) {
    return static_cast<Matrix &>(Vec<Vec<T, C>, R>::operator=(m));
  }

  constexpr Matrix &operator=(const std::array<T, R * C> &m) {
    return static_cast<Matrix &>(Vec<Vec<T, C>, R>::operator=(m));
  }

  using Vec<Vec<T, C>, R>::Vec;
  using Vec<Vec<T, C>, R>::operator=;

  constexpr Matrix(std::initializer_list<std::initializer_list<T>> il) {
    size_t i = 0;
    for (const auto &r : il) {
      (*this)[i] = Vec<T, C>(r);
      ++i;
    }
  }

  constexpr Vec<T, R * C> flatten() const {
    Vec<T, R * C> r{};
    for (size_t i = 0; i < R; ++i) {
      for (size_t j = 0; j < C; ++j) {
        r[i * C + j] = (*this)[i][j];
      }
    }
    return r;
  }
};

template <typename T, size_t L> std::ostream &operator<<(std::ostream &os, const Vec<T, L> &v) {
  os << static_cast<std::string>(v);
  return os;
}

template <typename T, size_t R, size_t C> std::ostream &operator<<(std::ostream &os, const Matrix<T, R, C> &m) {
  os << static_cast<std::string>(m);
  return os;
}

template <typename T, size_t Channels, size_t Height, size_t Width>
struct Tensor : std::array<Matrix<T, Height, Width>, Channels> {
  using std::array<Matrix<T, Height, Width>, Channels>::array;

  constexpr Tensor(const Tensor &t) = default;
  constexpr Tensor(std::initializer_list<std::initializer_list<std::initializer_list<T>>> il) {
    size_t i = 0;
    for (const auto &c : il) {
      (*this)[i] = Matrix<T, Height, Width>(c);
      ++i;
    }
  }

  using base = std::array<Matrix<T, Height, Width>, Channels>;
  Tensor &operator+=(const Tensor &t) {
    for (size_t i = 0; i < Channels; ++i) {
      (*this)[i] += t[i];
    }
    return *this;
  }

  Vec<T, Channels * Height * Width> flatten() const;

  auto window();

  using value_t = T;
  static constexpr size_t CHANNELS = Channels;
  static constexpr size_t HEIGHT = Height;
  static constexpr size_t WIDTH = Width;
  static constexpr size_t SIZE = Channels * Height * Width;
  static constexpr size_t SIZE2D = Height * Width;
};

template <size_t Delta_Dep, size_t Delta_Row, size_t Delta_Col, typename T, size_t Channels, size_t Height,
          size_t Width>
struct TensorWindow {
  Tensor<T, Channels, Height, Width> *const data;
  const int Row, Col, Dep; // Global offsets

  TensorWindow(Tensor<T, Channels, Height, Width> &data, int row, int col, int dep)
      : data(&data), Row(row), Col(col), Dep(dep) {}

  void check_access(size_t depth, size_t row, size_t column) const {
    auto d = static_cast<int>(depth) + Dep;
    auto r = static_cast<int>(row) + Row;
    auto c = static_cast<int>(column) + Col;
    bool valid = (d >= 0 && d < Channels) && (r >= 0 && r < Height) && (c >= 0 && c < Width);
    if (!valid) {
      std::string errmsg =
          std::format("TensorWindow: Access out of bounds {{d({}) >= 0 && d({}) < Channels({}), r({}) >= 0 && "
                      "r({}) < Height({}), c({}) >= 0 && c({}) < Width({})}} evaluated to false",
                      d, d, Channels, r, r, Height, c, c, Width);

      throw std::out_of_range(errmsg);
    }
  }

  T &at(size_t depth, size_t row, size_t column) {
#ifdef DEBUG
    check_access(depth, row, column);
#endif
    return (*data)[depth + Dep][row + Row][column + Col];
  }

  const T &at(size_t depth, size_t row, size_t column) const {
#ifdef DEBUG
    check_access(depth, row, column);
#endif
    return (*data)[depth + Dep][row + Row][column + Col];
  }

  Vec<T, Delta_Row * Delta_Col * Delta_Dep> flatten() const {
    Vec<T, Delta_Row * Delta_Col * Delta_Dep> v{}; // Zero-initialized

    static_assert(Delta_Dep <= Channels && Delta_Row <= Height && Delta_Col <= Width,
                  "TensorWindow: Delta_Dep, Delta_Row, and Delta_Col must be less than or equal to Channels, Height, "
                  "and Width respectively");

    // Handle completely out-of-bounds cases
    if (Dep >= static_cast<int>(Channels) || Row >= static_cast<int>(Height) || Col >= static_cast<int>(Width) ||
        Dep + static_cast<int>(Delta_Dep) <= 0 || Row + static_cast<int>(Delta_Row) <= 0 ||
        Col + static_cast<int>(Delta_Col) <= 0) {
      throw std::out_of_range("TensorWindow: Access out of bounds");
    }

    // Calculate valid ranges
    const size_t MAX_DEPTH = std::min(Delta_Dep, Channels - std::max(0, Dep));
    const size_t MAX_ROW = std::min(Delta_Row, Height - std::max(0, Row));
    const size_t MAX_COL = std::min(Delta_Col, Width - std::max(0, Col));
    const size_t D_INIT = std::max(0, -Dep);
    const size_t R_INIT = std::max(0, -Row);
    const size_t C_INIT = std::max(0, -Col);

    const size_t R_I_INCREM = (MAX_ROW < Delta_Row) ? (Delta_Row - MAX_ROW) * Delta_Col : 0;
    const size_t C_I_INCREM = (MAX_COL < Delta_Col) ? (Delta_Col - MAX_COL) : 0;

    size_t i = 0;
    for (size_t d = D_INIT; d < MAX_DEPTH; ++d) {
      for (size_t r = R_INIT; r < MAX_ROW; ++r) {
        // Process valid columns in this row
        for (size_t c = C_INIT; c < MAX_COL; ++c) {
          v[i++] = at(d, r, c);
        }

        // Skip out-of-bounds columns
        i += C_I_INCREM;
      }

      // Skip out-of-bounds rows
      i += R_I_INCREM;
    }

    return v;
  }
};

template <size_t Delta_Dep, size_t Delta_Row, size_t Delta_Col, typename T, size_t Channels, size_t Height,
          size_t Width>
auto make_TensorWindow(Tensor<T, Channels, Height, Width> &data, size_t row, size_t col, size_t dep) {
  return TensorWindow<Delta_Dep, Delta_Row, Delta_Col, T, Channels, Height, Width>(data, row, col, dep);
}

template <typename T, size_t Channels, size_t Height, size_t Width> auto Tensor<T, Channels, Height, Width>::window() {
  return make_TensorWindow<Channels, Height, Width, T, Channels, Height, Width>(*this, 0, 0, 0);
}

template <typename T, size_t Channels, size_t Height, size_t Width>
Vec<T, Channels * Height * Width> Tensor<T, Channels, Height, Width>::flatten() const {
  return const_cast<Tensor *>(this)->window().flatten();
}

template <typename T> constexpr T ceil_div(T numerator, T denominator) {
  return (numerator + denominator - 1) / denominator;
}
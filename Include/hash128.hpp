#pragma once
#include <cstdint>
#include <string>
#include <ostream>
#include <functional>

struct Hash128 {
  uint64_t hash0;
  uint64_t hash1;

  Hash128();
  Hash128(uint64_t hash0, uint64_t hash1);

  bool operator<(const Hash128 other) const;
  bool operator>(const Hash128 other) const;
  bool operator<=(const Hash128 other) const;
  bool operator>=(const Hash128 other) const;
  bool operator==(const Hash128 other) const;
  bool operator!=(const Hash128 other) const;

  Hash128 operator^(const Hash128 other) const;
  Hash128 operator|(const Hash128 other) const;
  Hash128 operator&(const Hash128 other) const;
  Hash128& operator^=(const Hash128 other);
  Hash128& operator|=(const Hash128 other);
  Hash128& operator&=(const Hash128 other);

  friend std::ostream& operator<<(std::ostream& out, const Hash128 other) {
    out << std::hex << other.hash0 << ":" << other.hash1;
    return out;
  }
  std::string toString() const {
    char buf[40];
    snprintf(buf, sizeof(buf), "%016llx:%016llx", (unsigned long long)hash0, (unsigned long long)hash1);
    return std::string(buf);
  }
  static Hash128 ofString(const std::string& s);
  static Hash128 mixInt(Hash128 h, int64_t t);
};

inline Hash128::Hash128() : hash0(0), hash1(0) {}
inline Hash128::Hash128(uint64_t h0, uint64_t h1) : hash0(h0), hash1(h1) {}
inline bool Hash128::operator==(const Hash128 other) const { return hash0 == other.hash0 && hash1 == other.hash1; }
inline bool Hash128::operator!=(const Hash128 other) const { return hash0 != other.hash0 || hash1 != other.hash1; }
inline bool Hash128::operator>(const Hash128 other) const { if(hash1 > other.hash1) return true; if(hash1 < other.hash1) return false; return hash0 > other.hash0; }
inline bool Hash128::operator>=(const Hash128 other) const { if(hash1 > other.hash1) return true; if(hash1 < other.hash1) return false; return hash0 >= other.hash0; }
inline bool Hash128::operator<(const Hash128 other) const { if(hash1 < other.hash1) return true; if(hash1 > other.hash1) return false; return hash0 < other.hash0; }
inline bool Hash128::operator<=(const Hash128 other) const { if(hash1 < other.hash1) return true; if(hash1 > other.hash1) return false; return hash0 <= other.hash0; }
inline Hash128 Hash128::operator^(const Hash128 other) const { return Hash128(hash0 ^ other.hash0, hash1 ^ other.hash1); }
inline Hash128 Hash128::operator|(const Hash128 other) const { return Hash128(hash0 | other.hash0, hash1 | other.hash1); }
inline Hash128 Hash128::operator&(const Hash128 other) const { return Hash128(hash0 & other.hash0, hash1 & other.hash1); }
inline Hash128& Hash128::operator^=(const Hash128 other) { hash0 ^= other.hash0; hash1 ^= other.hash1; return *this; }
inline Hash128& Hash128::operator|=(const Hash128 other) { hash0 |= other.hash0; hash1 |= other.hash1; return *this; }
inline Hash128& Hash128::operator&=(const Hash128 other) { hash0 &= other.hash0; hash1 &= other.hash1; return *this; }
// mixInt实现
inline Hash128 Hash128::mixInt(Hash128 h, int64_t t) {
  uint64_t x = (uint64_t)t;
  h.hash0 ^= (x + 0x9e3779b97f4a7c15ULL + (h.hash0<<6) + (h.hash0>>2));
  h.hash1 ^= (x + 0x517cc1b727220a95ULL + (h.hash1<<6) + (h.hash1>>2));
  return h;
}

// 允许Hash128作为unordered_map的key
namespace std {
template<>
struct hash<Hash128> {
    size_t operator()(const Hash128& h) const noexcept {
        // 简单异或混合
        return std::hash<uint64_t>()(h.hash0) ^ (std::hash<uint64_t>()(h.hash1) << 1);
    }
};
} 
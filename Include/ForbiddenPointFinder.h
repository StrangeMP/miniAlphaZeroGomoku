#pragma once
#include "config.hpp"
#include "utils.hpp"
#include <stdint.h>

struct CForbiddenPointFinder {

public:
  CForbiddenPointFinder(const Utils::Board &board);

  bool isForbidden(int x, int y);
  void SetStone(int x, int y, Utils::STONE_COLOR cStone);

private:
  char cBoard[Config::BOARD_SIZE + 2][Config::BOARD_SIZE + 2];
  bool isForbiddenNoNearbyCheck(int x, int y);
  bool IsFive(int x, int y, int nColor);
  bool IsOverline(int x, int y);
  bool IsFive(int x, int y, int nColor, int nDir);
  bool IsFour(int x, int y, int nColor, int nDir);
  int IsOpenFour(int x, int y, int nColor, int nDir);
  bool IsOpenThree(int x, int y, int nColor, int nDir);
  bool IsDoubleFour(int x, int y);
  bool IsDoubleThree(int x, int y);
};

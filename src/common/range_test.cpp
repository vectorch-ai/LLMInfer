#include "range.h"

#include <gtest/gtest.h>

namespace llm {
TEST(RangeTest, EmptyRange) {
  for (int i : range<int>(0)) {
    FAIL() << "Empty range should not have any elements";
  }

  for (int i : range<int>(4, 2)) {
    FAIL() << "Empty range should not have any elements";
  }
}

// create test for basic cases
TEST(RangeTest, BasicRange) {
  std::vector<int> expected = {0, 1, 2, 3, 4};
  std::vector<int> actual;
  for (int i : range<int>(5)) {
    actual.push_back(i);
  }
  EXPECT_EQ(expected, actual);

  expected = {2, 3};
  actual.clear();
  for (int i : range<int>(2, 4)) {
    actual.push_back(i);
  }
  EXPECT_EQ(expected, actual);
}

}  // namespace llm
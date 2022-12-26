#include <catch2/catch_test_macros.hpp>

TEST_CASE("Foo", "[bar]") {
  REQUIRE(1 + 1 == 2);
  REQUIRE(2 + 2 == 4);
}

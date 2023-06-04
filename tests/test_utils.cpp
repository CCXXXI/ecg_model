#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "../src/utils.h"

TEST_CASE("test_load_model", "[utils]") {
    set_models_path("./assets/ecg_models/models/");

    auto filename = GENERATE("u_net.pt", "res_net.pt");
    auto model = load_model(filename);

    REQUIRE(!model.is_training());
}

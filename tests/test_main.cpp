#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

#include "../src/main.h"

TEST_CASE("test_infer", "[main]") {
    // set up
    set_models_path("./assets/ecg_models/models/");

    // get actual
    const torch::NoGradGuard no_grad;
    auto actual = infer(get_input(), 125);

    // get expected
    std::ifstream in("./assets/ecg_models/output/beats.json");
    auto expected = nlohmann::json::parse(in);

    // compare
    REQUIRE(actual.size() == expected.size());
    const auto n = static_cast<int>(actual.size());
    for (auto i = 0; i < n; ++i) {
        auto a = actual[i];
        auto e = expected[i];
        REQUIRE(a.label == e["label"]);
        REQUIRE(a.position * 1000 / fs == e["millisecondsSinceStart"]);
    }
}

cmake_minimum_required(VERSION 3.25)
project(ecg_model)

# standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# flags
string(APPEND CMAKE_CXX_FLAGS " -Wall -Wextra -Wpedantic -Werror")
string(APPEND CMAKE_CXX_FLAGS_PROFILE " --coverage")
string(APPEND CMAKE_CXX_FLAGS_DEBUG " -fsanitize=address -fsanitize=undefined")

# executable
add_executable(ecg_model_tests src/utils.cpp src/utils.h tests/test_utils.cpp)

# torch
set(Torch_DIR "./libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
string(APPEND CMAKE_CXX_FLAGS " ${TORCH_CXX_FLAGS}")
target_link_libraries(ecg_model_tests "${TORCH_LIBRARIES}")

# catch2
find_package(Catch2 3 REQUIRED)
target_link_libraries(ecg_model_tests Catch2::Catch2WithMain)

# assets
file(COPY assets DESTINATION ${CMAKE_BINARY_DIR})

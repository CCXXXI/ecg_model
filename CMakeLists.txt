cmake_minimum_required(VERSION 3.24)
project(ecg_model)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Werror")

set(CMAKE_CXX_FLAGS_PROFILE --coverage)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address -fsanitize=undefined")

add_executable(ecg_model_tests src/utils.cpp src/utils.h tests/test_utils.cpp)

set(Torch_DIR "./libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
target_link_libraries(ecg_model_tests "${TORCH_LIBRARIES}")

find_package(Catch2 3 REQUIRED)
target_link_libraries(ecg_model_tests Catch2::Catch2WithMain)

file(COPY resources DESTINATION ${CMAKE_BINARY_DIR})

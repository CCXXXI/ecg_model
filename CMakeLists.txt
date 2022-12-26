cmake_minimum_required(VERSION 3.24)
project(ecg_model)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Werror")

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fsanitize=undefined")
endif ()

add_executable(ecg_model_tests tests/test_forward.cpp)

add_library(ecg_model SHARED src/forward.hpp)
set_target_properties(ecg_model PROPERTIES LINKER_LANGUAGE CXX)

set(Torch_DIR "./libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
target_link_libraries(ecg_model_tests "${TORCH_LIBRARIES}")
target_link_libraries(ecg_model "${TORCH_LIBRARIES}")

find_package(Catch2 3 REQUIRED)
target_link_libraries(ecg_model_tests Catch2::Catch2WithMain)

file(COPY resources DESTINATION ${CMAKE_BINARY_DIR})

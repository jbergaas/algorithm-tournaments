cmake_minimum_required(VERSION 3.16)
project(Engine VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -O3 -march=native)
endif()

find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)

# Matrix Market reader, version 1.7.7, included in third_party/.
add_subdirectory(third_party/fast_matrix_market)

include_directories(${CMAKE_SOURCE_DIR}/include)

# OpenMP is used to compute Betweenness in parallel.
function(engine_executable name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE Threads::Threads OpenMP::OpenMP_CXX fast_matrix_market::fast_matrix_market)
endfunction()

engine_executable(engine src/main.cpp)
engine_executable(isolate_filter preprocessing/isolate_filter.cpp)

# One test program per engine header, plus one for the isolate filter.
set(ENGINE_TESTS graph spread game algorithms betweenness seeds dataset_identity batch isolate_filter)
foreach(test ${ENGINE_TESTS})
    engine_executable(test_${test} tools/test_${test}.cpp)
endforeach()
target_include_directories(test_isolate_filter PRIVATE "${CMAKE_SOURCE_DIR}/preprocessing")

# The tests run in test-work/ in the build directory, with copies of the graphs in tests/data/.
enable_testing()
set(ENGINE_TEST_WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/test-work")
set(ENGINE_TEST_DATA_DIRECTORY "${CMAKE_SOURCE_DIR}/tests/data")
file(MAKE_DIRECTORY "${ENGINE_TEST_WORKING_DIRECTORY}/data" "${CMAKE_BINARY_DIR}/tmp")
file(GLOB test_graphs "${ENGINE_TEST_DATA_DIRECTORY}/*.mtx")
file(COPY ${test_graphs} DESTINATION "${ENGINE_TEST_WORKING_DIRECTORY}/data")

function(engine_test name)
    add_test(NAME ${name} COMMAND ${ARGN})
    set_tests_properties(${name} PROPERTIES
        WORKING_DIRECTORY "${ENGINE_TEST_WORKING_DIRECTORY}"
        ENVIRONMENT "TMPDIR=${CMAKE_BINARY_DIR}/tmp")
endfunction()

foreach(test ${ENGINE_TESTS})
    engine_test(${test} test_${test})
endforeach()
# Betweenness on two real networks, checked against a single-thread reference version.
engine_test(betweenness_real_graphs test_betweenness --real-graphs data/Erdos02.mtx data/ca-GrQc.mtx)

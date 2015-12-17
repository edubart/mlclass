cmake_minimum_required(VERSION 2.8)
project(mlight)


option(DEBUG "Compile with debugging information" ON)
set(CXX_WARNS "-Wall -Wextra -Wno-unused-parameter -Wno-deprecated-declarations")
add_definitions(-DARMA_DONT_USE_WRAPPER -DDLIB_USE_BLAS -DDLIB_USE_LAPACK)

include_directories(${CMAKE_CURRENT_LIST_DIR}/src)

set(CMAKE_CXX_FLAGS "-std=c++11 ${CXX_WARNS} -O3 -march=native -ffast-math -fopenmp -rdynamic -g")

add_subdirectory(src/mlight)
add_subdirectory(src/tests)

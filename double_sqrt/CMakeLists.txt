cmake_minimum_required (VERSION 3.16)
project(DoubleSqrt)

find_package(Kokkos REQUIRED)

add_executable(double_sqrt double_sqrt.cpp)
target_link_libraries(double_sqrt Kokkos::kokkos)

cmake_minimum_required (VERSION 3.16)
project (Sort)

find_package(Kokkos REQUIRED)

add_executable(sort sort.cpp)
target_link_libraries(sort Kokkos::kokkos)

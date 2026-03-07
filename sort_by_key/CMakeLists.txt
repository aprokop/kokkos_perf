cmake_minimum_required (VERSION 3.16)
project (SortByKey)

find_package(Kokkos REQUIRED)

add_executable(sort_by_key sort_by_key.cpp)
target_link_libraries(sort_by_key Kokkos::kokkos)

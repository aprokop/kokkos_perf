cmake_minimum_required (VERSION 3.16)
project (CosmologyCenters)

find_package(Kokkos REQUIRED)

add_executable(cosmology_centers cosmology_centers.cpp)
target_link_libraries(cosmology_centers Kokkos::kokkos)

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_Sort.hpp>

#include <iostream>

#ifdef KOKKOS_ENABLE_CUDA
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#endif

template <typename ExecutionSpace, typename Keys, typename Values>
void sortByKeySort(ExecutionSpace const &space, Keys &keys, Values &values)
{
  Kokkos::Profiling::ScopedRegion guard("sortByKeySort");
  Kokkos::Experimental::sort_by_key(space, keys, values);
}

#ifdef KOKKOS_ENABLE_CUDA
template <typename Keys, typename Values>
void sortByKeyThrust(Kokkos::Cuda const &space, Keys &keys, Values &values)
{
  Kokkos::Profiling::ScopedRegion guard("sortByKeyThrust");

  auto const n = keys.size();
  auto const execution_policy = thrust::cuda::par.on(space.cuda_stream());

  thrust::sort_by_key(execution_policy, keys.data(), keys.data() + n,
                      values.data());
}
#endif

#ifdef KOKKOS_ENABLE_HIP
template <typename Keys, typename Values>
void sortByKeyThrust(Kokkos::HIP const &space, Keys &keys, Values &values)
{
  Kokkos::Profiling::ScopedRegion guard("sortByKeyROCm");

  auto const n = keys.size();
  auto const execution_policy = thrust::hip::par.on(space.hip_stream());

  thrust::sort_by_key(execution_policy, keys.data(), keys.data() + n,
                      values.data());
}
#endif

template <typename ExecutionSpace, typename View>
void constructProblem(ExecutionSpace const &space, int n, View &v)
{
  using MemorySpace = typename View::memory_space;

  Kokkos::Random_XorShift1024_Pool<MemorySpace> rand_pool(5374857);

  Kokkos::resize(Kokkos::WithoutInitializing, v, n);
  Kokkos::parallel_for(
      "construct_v", Kokkos::RangePolicy<ExecutionSpace>(space, 0, v.extent(0)),
      KOKKOS_LAMBDA(int i) {
        auto rand_gen = rand_pool.get_state();
        v(i) = rand_gen.urand();
        rand_pool.free_state(rand_gen);
      });

  space.fence();
}

template <typename Scalar, typename ExecutionSpace>
void main__(std::string const &name, ExecutionSpace const &space, int n)
{
  using MemorySpace = typename ExecutionSpace::memory_space;

  std::cout << " === " << name << " ===\n";

  Kokkos::View<Scalar *, MemorySpace> values_orig("values_orig", 0);
  constructProblem(space, n, values_orig);

  Kokkos::View<unsigned *, MemorySpace> permute_orig("permute_orig", n);
  Kokkos::parallel_for(
      "iota", Kokkos::RangePolicy<ExecutionSpace>(space, 0, n),
      KOKKOS_LAMBDA(int i) { permute_orig(i) = i; });

  Kokkos::View<Scalar *, MemorySpace> values(
      Kokkos::view_alloc(space, Kokkos::WithoutInitializing, "values"), n);

  Kokkos::View<unsigned *, MemorySpace> permute(
      Kokkos::view_alloc(space, Kokkos::WithoutInitializing, "permute"), n);

  Kokkos::Timer timer;

  Kokkos::deep_copy(space, values, values_orig);
  Kokkos::deep_copy(space, permute, permute_orig);
  space.fence();
  timer.reset();
  sortByKeySort(space, values, permute);
  space.fence();
  printf("Time [Kokkos]         : %7.3f\n", timer.seconds());

#ifdef KOKKOS_ENABLE_CUDA
  if constexpr (std::is_same_v<ExecutionSpace, Kokkos::Cuda>)
#endif
#ifdef KOKKOS_ENABLE_HIP
    if constexpr (std::is_same_v<ExecutionSpace, Kokkos ::HIP>)
#endif
    {
      Kokkos::deep_copy(space, values, values_orig);
      Kokkos::deep_copy(space, permute, permute_orig);
      space.fence();
      timer.reset();
      sortByKeyThrust(space, values, permute);
      space.fence();
      printf("Time [Thrust]         : %7.3f\n", timer.seconds());
    }
}

template <typename Scalar>
void main_(std::string const &name, int n)
{
#ifdef KOKKOS_ENABLE_SERIAL
  // main__<Scalar>(name + " [SERIAL]", Kokkos::Serial{}, n);
#endif
#ifdef KOKKOS_ENABLE_OPENMP
  // main__<Scalar>(name + " [OpenMP]", Kokkos::OpenMP{}, n);
#endif
#ifdef KOKKOS_ENABLE_CUDA
  main__<Scalar>(name + " [CUDA]", Kokkos::Cuda{}, n);
#endif
#ifdef KOKKOS_ENABLE_HIP
  main__<Scalar>(name + " [HIP]", Kokkos::HIP{}, n);
#endif
}

int main(int argc, char *argv[])
{
  Kokkos::ScopeGuard guard(argc, argv);

  int n = 30000000;

  for (int i = 1; i < argc; ++i)
  {
    if (!strcmp(argv[i], "-n"))
    {
      n = atoi(argv[++i]);
    }
    else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help"))
    {
      std::cout << "./sort -n <num_points>" << std::endl;
      return 0;
    }
    else
      throw std::runtime_error(std::string("Uknown option: ") + argv[i]);
  }

  printf("number of points          : %d\n", n);

  main_<unsigned>("unsigned", n);
  main_<int>("int", n);
  main_<long long>("long long", n);
  main_<float>("float", n);
  main_<double>("double", n);

  return 0;
}

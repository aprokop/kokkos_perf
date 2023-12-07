#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_Sort.hpp>

#include <iostream>

#ifdef KOKKOS_ENABLE_CUDA
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#endif

template <typename ExecutionSpace, typename ViewType>
std::pair<typename ViewType::non_const_value_type,
          typename ViewType::non_const_value_type>
minMax(ExecutionSpace &&space, ViewType const &v)
{
  static_assert(ViewType::rank == 1, "minMax requires a View of rank 1");
  auto const n = v.extent(0);
  using ValueType = typename ViewType::non_const_value_type;
  ValueType min_val;
  ValueType max_val;
  Kokkos::RangePolicy<std::decay_t<ExecutionSpace>> policy(
      std::forward<ExecutionSpace>(space), 0, n);
  Kokkos::parallel_reduce(
      "minMax", policy,
      KOKKOS_LAMBDA(int i, ValueType &local_min, ValueType &local_max) {
        auto const &val = v(i);
        if (val < local_min)
        {
          local_min = val;
        }
        if (local_max < val)
        {
          local_max = val;
        }
      },
      Kokkos::Min<ValueType>(min_val), Kokkos::Max<ValueType>(max_val));
  return std::make_pair(min_val, max_val);
}

template <typename ExecutionSpace, typename Permute, typename View>
void applyPermutation(ExecutionSpace const &space, Permute const &permute,
                      View const &view)
{
  static_assert(std::is_integral<typename Permute::value_type>::value);

  typename View::non_const_type scratch_view(
      Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                         "scratch" + view.label()),
      view.layout());
  Kokkos::deep_copy(space, scratch_view, view);
  Kokkos::parallel_for(
      "permute", Kokkos::RangePolicy<ExecutionSpace>(space, 0, view.size()),
      KOKKOS_LAMBDA(int i) { view(i) = scratch_view(permute(i)); });
}

template <typename ExecutionSpace, typename Keys, typename Values>
void sortByKeyBinSort(ExecutionSpace const &space, Keys &keys, Values &values)
{
  Kokkos::Profiling::ScopedRegion guard("sortByKeyBinSort");

  auto const n = keys.size();

  auto [min_val, max_val] = minMax(space, keys);

  using SizeType = unsigned int;
  using CompType = Kokkos::BinOp1D<Keys>;

  Kokkos::BinSort<Keys, CompType, typename Keys::device_type, SizeType>
      bin_sort(space, keys, CompType(n / 2, min_val, max_val), true);
  bin_sort.create_permute_vector(space);
  bin_sort.sort(space, keys);
  bin_sort.sort(space, values);
}

template <typename Keys>
struct CustomOperator
{
  Keys _keys;

  KOKKOS_FUNCTION bool operator()(int i, int j) const
  {
    return _keys(i) < _keys(j);
  }
};

template <typename ExecutionSpace, typename Keys, typename Values>
void sortByKeyKokkosSort(ExecutionSpace const &space, Keys &keys,
                         Values &values)
{
  Kokkos::Profiling::ScopedRegion guard("sortByKeyKokkosSort");

  Kokkos::sort(space, values, CustomOperator<Keys>{keys});
  applyPermutation(space, values, keys);
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

template <typename ExecutionSpace, typename View>
void constructProblem(ExecutionSpace const &exec_space, int n, View &v)
{
  using MemorySpace = typename View::memory_space;

  std::cout << "Constructing problem...";
  std::cout.flush();

  Kokkos::Random_XorShift1024_Pool<MemorySpace> rand_pool(5374857);

  Kokkos::resize(Kokkos::WithoutInitializing, v, n);
  Kokkos::parallel_for(
      "construct_v",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, v.extent(0)),
      KOKKOS_LAMBDA(int i) {
        auto rand_gen = rand_pool.get_state();
        v(i) = rand_gen.urand();
        rand_pool.free_state(rand_gen);
      });

  exec_space.fence();
  std::cout << "done" << std::endl;
}

int main(int argc, char *argv[])
{
  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using MemorySpace = typename ExecutionSpace::memory_space;

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

  ExecutionSpace exec_space;

  Kokkos::View<unsigned *, MemorySpace> values("values", 0);
  constructProblem(exec_space, n, values);

  Kokkos::View<unsigned *, MemorySpace> permute("permute", n);
  Kokkos::parallel_for(
      "iota", Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n),
      KOKKOS_LAMBDA(int i) { permute(i) = i; });

  Kokkos::Timer timer;

  exec_space.fence();
  timer.reset();
  sortByKeyBinSort(exec_space, values, permute);
  exec_space.fence();
  printf("Time [Kokkos::BinSort]: %7.3f\n", timer.seconds());

  exec_space.fence();
  timer.reset();
  sortByKeyKokkosSort(exec_space, values, permute);
  exec_space.fence();
  printf("Time [Kokkos::sort]   : %7.3f\n", timer.seconds());

#ifdef KOKKOS_ENABLE_CUDA
  exec_space.fence();
  timer.reset();
  sortByKeyThrust(exec_space, values, permute);
  exec_space.fence();
  printf("Time [Thrust]         : %7.3f\n", timer.seconds());
#endif

  return 0;
}

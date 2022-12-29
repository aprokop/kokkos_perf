#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <iostream>

struct Point
{
  float _data[3];

  KOKKOS_INLINE_FUNCTION
  constexpr float &operator[](unsigned int i) { return _data[i]; }

  KOKKOS_INLINE_FUNCTION
  constexpr const float &operator[](unsigned int i) const { return _data[i]; }
};

KOKKOS_INLINE_FUNCTION
float distance(Point const &a, Point const &b)
{
  float distance_squared = 0.0;
  for (int d = 0; d < 3; ++d)
  {
    float tmp = b[d] - a[d];
    distance_squared += tmp * tmp;
  }
  return std::sqrt(distance_squared);
}

template <typename ExecutionSpace, typename MemorySpace>
void exclusivePrefixSum(ExecutionSpace const &exec_space,
                        Kokkos::View<int *, MemorySpace> v)
{
  auto const n = v.extent(0);
  Kokkos::parallel_scan("exclusive_scan",
                        Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n),
                        KOKKOS_LAMBDA(int i, int &update, bool final_pass) {
                          auto const in = v(i);
                          if (final_pass)
                            v(i) = update;
                          update += in;
                        });
}

template <typename MemorySpace>
int lastElement(Kokkos::View<int *, MemorySpace> const &v)
{
  auto const n = v.extent(0);
  auto v_subview = Kokkos::subview(v, n - 1);
  auto v_host = Kokkos::create_mirror_view(v_subview);
  Kokkos::deep_copy(v_host, v_subview);
  return v_host();
}

template <typename ExecutionSpace, typename MemorySpace>
void constructProblem(ExecutionSpace const &exec_space, int num_points,
                      int num_halos, int halo_size, bool uniform_halo_sizes,
                      Kokkos::View<Point *, MemorySpace> &points,
                      Kokkos::View<int *, MemorySpace> &offsets,
                      Kokkos::View<int *, MemorySpace> &indices)
{
  std::cout << "Constructing problem...";
  std::cout.flush();

  Kokkos::resize(Kokkos::WithoutInitializing, points, num_points);
  Kokkos::parallel_for(
      "construct_points",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, points.extent(0)),
      KOKKOS_LAMBDA(int i) {
        points(i) = Point{0.f, 0.f, 0.f};
      });

  Kokkos::Random_XorShift1024_Pool<MemorySpace> rand_pool(5374857);

  Kokkos::resize(Kokkos::WithoutInitializing, offsets, num_halos + 1);
  Kokkos::parallel_for(
      "construct_offsets",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, offsets.extent(0)),
      KOKKOS_LAMBDA(int i) {
        if (uniform_halo_sizes)
        {
          offsets(i) = halo_size;
        }
        else
        {
          auto rand_gen = rand_pool.get_state();
          offsets(i) = rand_gen.urand() % (halo_size + 1);
          if (offsets(i) == 0)
            offsets(i) = 1;
          rand_pool.free_state(rand_gen);
        }
      });
  exclusivePrefixSum(exec_space, offsets);

  Kokkos::resize(Kokkos::WithoutInitializing, indices, lastElement(offsets));
  Kokkos::parallel_for(
      "construct_indices",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_halos),
      KOKKOS_LAMBDA(int halo_index) {
        auto rand_gen = rand_pool.get_state();

        uint32_t state = rand_gen.urand();

        auto lcg_parkmiller = [&state]() {
          uint64_t product = (uint64_t)state * 48271;
          uint32_t x = (product & 0x7fffffff) + (product >> 31);

          x = (x & 0x7fffffff) + (x >> 31);
          state = x;
          return state;
        };

        auto start = offsets(halo_index);
        auto end = offsets(halo_index + 1);
        for (int i = start; i < end; ++i)
          indices(i) = lcg_parkmiller() % num_points;

        rand_pool.free_state(rand_gen);
      });

  exec_space.fence();
  std::cout << "done" << std::endl;
  printf("#indices = %d\n", indices.extent_int(0));
}

template <typename ExecutionSpace, typename MemorySpace>
bool verify(ExecutionSpace const &exec_space,
            Kokkos::View<int *, MemorySpace> const &offsets,
            Kokkos::View<int *, MemorySpace> const &indices,
            Kokkos::View<int *, MemorySpace> const &min_potential_indices)
{
  auto const num_halos = offsets.extent_int(0) - 1;

  if (min_potential_indices.extent_int(0) != num_halos)
  {
    printf(
        "Size mismatch: num halos = %d, number of min potential indices = %d\n",
        num_halos, min_potential_indices.extent_int(0));
    return false;
  }

  int num_failures = 0;
  Kokkos::parallel_reduce(
      "verify", Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_halos),
      KOKKOS_LAMBDA(int halo_index, int &update) {
        int last_index_in_halo = offsets(halo_index + 1) - 1;

        int found_index = min_potential_indices(halo_index);
        int correct_index = indices(last_index_in_halo);
        if (found_index != correct_index)
        {
          printf("[%d]: %d vs %d\n", halo_index, found_index, correct_index);
          ++update;
        }
      },
      num_failures);

  return (num_failures == 0);
}

template <typename ExecutionSpace, typename MemorySpace>
Kokkos::View<int *, MemorySpace>
computeMinPotential1(ExecutionSpace const &exec_space,
                     Kokkos::View<Point *, MemorySpace> const &points,
                     Kokkos::View<int *, MemorySpace> const &offsets,
                     Kokkos::View<int *, MemorySpace> const &indices)
{
  auto const num_halos = offsets.extent(0) - 1;

  Kokkos::View<int *, MemorySpace> min_potential_indices(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "min_potential_indices"),
      num_halos);
  Kokkos::parallel_for(
      "compute_galaxy_centers_1",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_halos),
      KOKKOS_LAMBDA(int const halo_index) {
        auto start = offsets(halo_index);
        auto end = offsets(halo_index + 1);

        double min_potential =
            1.; // potential is always negative so this is max value
        int min_potential_index;

        // But in this example we set all points to {0, 0, 0} and modify the
        // formula so that we can check the answer. The minimum potential is
        // always going to be -(end-1), and min_index (end-1)
        for (int ii = start; ii < end; ++ii)
        {
          int i = indices(ii);
          double potential = -ii; // = 0 in real code

          Point point = points(i);

          for (int jj = start; jj < end; ++jj)
          {
            int j = indices(jj);

            constexpr float mass = 1.f;
            // The correct formula is
            //   potential -= mass * 1.f/distance;
            // But in this example we set all points to {0, 0, 0} and modify the
            // formula so that we can check the answer.
            potential -=
                mass *
                distance(point, points(j)); // rhs will be 0, just for testing
          }

          if (potential < min_potential)
          {
            min_potential = potential;
            min_potential_index = i;
          }
        }

        min_potential_indices(halo_index) = min_potential_index;
      });

  return min_potential_indices;
}

template <typename ExecutionSpace, typename MemorySpace>
Kokkos::View<int *, MemorySpace>
computeMinPotential2(ExecutionSpace const &exec_space,
                     Kokkos::View<Point *, MemorySpace> const &points,
                     Kokkos::View<int *, MemorySpace> const &offsets,
                     Kokkos::View<int *, MemorySpace> const &indices)
{
  auto const num_halos = offsets.extent(0) - 1;

  using TeamPolicy =
      Kokkos::TeamPolicy<ExecutionSpace, Kokkos::Schedule<Kokkos::Dynamic>>;

  Kokkos::View<int *, MemorySpace> min_potential_indices(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "min_potential_indices"),
      num_halos);
  Kokkos::parallel_for(
      "compute_galaxy_centers_2",
      TeamPolicy(exec_space, num_halos, Kokkos::AUTO),
      KOKKOS_LAMBDA(typename TeamPolicy::member_type const &team) {
        auto const halo_index = team.league_rank();

        auto start = offsets(halo_index);
        auto end = offsets(halo_index + 1);

        double min_potential =
            1.; // potential is always negative so this is max value
        int min_potential_index;

        // But in this example we set all points to {0, 0, 0} and modify the
        // formula so that we can check the answer. The minimum potential is
        // always going to be -(end-1), and min_index (end-1)
        for (int ii = start; ii < end; ++ii)
        {
          int i = indices(ii);
          double potential = -ii; // = 0 in real code

          Point point = points(i);

          double accumulated = 0.;
          Kokkos::parallel_reduce(
              Kokkos::TeamThreadRange(team, start, end),
              [&](int jj, double &update) {
                int j = indices(jj);

                constexpr float mass = 1.f;
                // The correct formula is
                //   potential -= mass * 1.f/distance;
                // But in this example we set all points to {0, 0, 0} and modify
                // the formula so that we can check the answer.
                update -=
                    mass *
                    distance(point,
                             points(j)); // rhs will be 0, just for testing
              },
              accumulated);
          potential += accumulated;

          Kokkos::single(Kokkos::PerTeam(team), [&]() {
            if (potential < min_potential)
            {
              min_potential = potential;
              min_potential_index = i;
            }
          });
        }

        Kokkos::single(Kokkos::PerTeam(team), [&]() {
          min_potential_indices(halo_index) = min_potential_index;
        });
      });

  return min_potential_indices;
}

struct PotentialPair
{
  int index = -1;
  double potential = 1.f;

private:
  friend KOKKOS_INLINE_FUNCTION bool operator<(PotentialPair const &a,
                                               PotentialPair const &b)
  {
    return a.potential < b.potential;
  }
};

template <typename ExecutionSpace, typename MemorySpace>
Kokkos::View<int *, MemorySpace>
computeMinPotential3(ExecutionSpace const &exec_space,
                     Kokkos::View<Point *, MemorySpace> const &points,
                     Kokkos::View<int *, MemorySpace> const &offsets,
                     Kokkos::View<int *, MemorySpace> const &indices)
{
  auto const num_halos = offsets.extent(0) - 1;
  auto const n = indices.extent(0);

  Kokkos::View<PotentialPair *, MemorySpace> min_potentials("min_potentials",
                                                            num_halos);
  Kokkos::parallel_for(
      "compute_galaxy_centers_3",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n),
      KOKKOS_LAMBDA(int const ii) {
        auto upper_bound = [](auto const &v, int x) {
          int first = 0;
          int last = v.extent_int(0);
          int count = last - first;
          while (count > 0)
          {
            int step = count / 2;
            int mid = first + step;
            if (!(x < v(mid)))
            {
              first = ++mid;
              count -= step + 1;
            }
            else
            {
              count = step;
            }
          }
          return first;
        };

        // Use binary search to find start and end of the halo this index
        // belongs to
        int halo_index = upper_bound(offsets, ii) - 1;

        if (!(offsets(halo_index) <= ii && ii < offsets(halo_index + 1)))
          printf("NANI? %d not in [%d, %d)\n", ii, offsets(halo_index),
                 offsets(halo_index + 1));

        auto start = offsets(halo_index);
        auto end = offsets(halo_index + 1);

        int i = indices(ii);

        double potential = -ii; // = 0 in real code

        Point point = points(i);
        for (int jj = start; jj < end; ++jj)
        {
          int j = indices(jj);

          constexpr float mass = 1.f;
          // The correct formula is
          //   potential -= mass * 1.f/distance;
          // But in this example we set all points to {0, 0, 0} and modify the
          // formula so that we can check the answer.
          potential -=
              mass *
              distance(point, points(j)); // rhs will be 0, just for testing
        }

        if (potential < min_potentials(halo_index).potential)
        { // this is just to avoid hammering atomics
          Kokkos::atomic_min(&min_potentials(halo_index),
                             PotentialPair{i, potential});
        }
      });

  Kokkos::View<int *, MemorySpace> min_potential_indices(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "min_potential_indices"),
      num_halos);
  Kokkos::parallel_for(
      "extract_min_potential_indices",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_halos),
      KOKKOS_LAMBDA(int const halo_index) {
        min_potential_indices(halo_index) = min_potentials(halo_index).index;
      });

  return min_potential_indices;
}

int main(int argc, char *argv[])
{
  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using MemorySpace = typename ExecutionSpace::memory_space;

  Kokkos::ScopeGuard guard(argc, argv);

  int num_points = 30000000;
  int num_halos = 1000000;
  int halo_size = 10000;
  bool uniform_halo_sizes = true;

  for (int i = 1; i < argc; ++i)
  {
    if (!strcmp(argv[i], "--num-points"))
      num_points = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--num-halos"))
      num_halos = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--halo-size"))
      halo_size = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--variable-halo-sizes"))
      uniform_halo_sizes = false;
    else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help"))
    {
      std::cout << "./cosmology_centers.exe [--num-halos <number_of_halos>] [--halo-size "
                   "<halo_size>] [--variable-halo-sizes]"
                << std::endl;
      return 0;
    }
    else
      throw std::runtime_error(std::string("Uknown option: ") + argv[i]);
  }

  printf("number of points          : %d\n", num_points);
  printf("number of halos           : %d\n", num_halos);
  printf("halo size                 : %d [%s]\n", halo_size,
         (uniform_halo_sizes ? "uniform" : "variable"));

  ExecutionSpace exec_space;

  Kokkos::View<Point *, MemorySpace> points("points", 0);
  Kokkos::View<int *, MemorySpace> offsets("offsets", 0);
  Kokkos::View<int *, MemorySpace> indices("indices", 0);
  constructProblem(exec_space, num_points, num_halos, halo_size,
                   uniform_halo_sizes, points, offsets, indices);

  Kokkos::Timer timer;

  Kokkos::View<int *, MemorySpace> min_potential_indices(
      "min_potential_indices", 0);
  for (int variant = 3; variant <= 3; ++variant)
  {
    exec_space.fence();
    timer.reset();
    switch (variant)
    {
    case 1:
      min_potential_indices =
          computeMinPotential1(exec_space, points, offsets, indices);
      break;
    case 2:
      min_potential_indices =
          computeMinPotential2(exec_space, points, offsets, indices);
      break;
    case 3:
      min_potential_indices =
          computeMinPotential3(exec_space, points, offsets, indices);
      break;
    }
    exec_space.fence();
    auto time = timer.seconds();

    printf("Time[%d]: %7.3f\n", variant, time);

    auto passed = verify(exec_space, offsets, indices, min_potential_indices);
    if (!passed)
      printf("Verification[%d] failed\n", variant);
  }

  return 0;
}

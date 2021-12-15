#include <Kokkos_Core.hpp>

#include <iostream>

int main(int argc, char *argv[])
{
  Kokkos::ScopeGuard guard(argc, argv);

  using scalar_type = double;
  using local_ordinal_type = int;
  using execution_space = Kokkos::DefaultExecutionSpace;
  using device_type =
      Kokkos::Device<execution_space, typename execution_space::memory_space>;

  srand(13721);

  int n = 100000;
  int num_loops = 10;
  int num_stuff = 1000;

  for (int i = 1; i < argc; i++)
  {
    if (!strcmp(argv[i], "-n"))
    {
      n = atoi(argv[++i]);
    }
    else if (!strcmp(argv[i], "-l") || !strcmp(argv[i], "--num_loops"))
    {
      num_loops = atoi(argv[++i]);
    }
    else if (!strcmp(argv[i], "--node"))
    {
    }
    else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help"))
    {
      std::cout
          << "./coalesce_drop.exe [-n <matrix_size>] [-l <number_of_loops>]"
          << std::endl;
      return 0;
    }
    else
    {
      throw std::runtime_error(std::string("Uknown option: ") + argv[i]);
    }
  }

  typedef Kokkos::RangePolicy<execution_space> range_policy;

  double r = 1.;

  Kokkos::View<double *, device_type> a("a", n);
  Kokkos::parallel_for(range_policy(0, n),
                       KOKKOS_LAMBDA(const int i) { a[i] = i; });

  // ------------------------------------------------------------------------------------
  {
    Kokkos::fence();
    Kokkos::Timer timer;

    double s = 0.;
    for (int i = 0; i < num_loops; i++)
      Kokkos::parallel_reduce(range_policy(0, n),
                              KOKKOS_LAMBDA(const int, double &v) {
                                for (int k = 0; k < num_stuff; k++)
                                  v += (a[k] <= r * r);
                              },
                              s);
    Kokkos::fence();
    double kernel_time = timer.seconds();

    printf("[s     <= r*r] %.2e (s)\n", kernel_time / num_loops);
  }
  // ------------------------------------------------------------------------------------
  {
    Kokkos::fence();
    Kokkos::Timer timer;

    double s = 0.;
    for (int i = 0; i < num_loops; i++)
      Kokkos::parallel_reduce(range_policy(0, n),
                              KOKKOS_LAMBDA(const int, double &v) {
                                for (int k = 0; k < num_stuff; k++)
                                  v += (sqrt(a[k]) <= r);
                              },
                              s);
    Kokkos::fence();
    double kernel_time = timer.seconds();

    printf("[sq(s) <= r  ] %.2e (s)\n", kernel_time / num_loops);
  }
  // ------------------------------------------------------------------------------------
  {
    Kokkos::fence();
    Kokkos::Timer timer;

    double s = 0.;
    for (int i = 0; i < num_loops; i++)
      Kokkos::parallel_reduce(range_policy(0, n),
                              KOKKOS_LAMBDA(const int, double &v) {
                                for (int k = 0; k < num_stuff; k++)
                                  v += (a[k] / r <= r);
                              },
                              s);
    Kokkos::fence();
    double kernel_time = timer.seconds();

    printf("[s/r   <= r  ] %.2e (s)\n", kernel_time / num_loops);
  }
  // ------------------------------------------------------------------------------------

  return 0;
}

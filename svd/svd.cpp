#include <Kokkos_Core.hpp>

#include <Kokkos_ArithTraits.hpp>

#include <cassert>
#include <chrono>
#include <iostream>

using execution_space   = Kokkos::DefaultExecutionSpace;
using device_type       = Kokkos::Device<execution_space,typename execution_space::memory_space>;
using scalar_type       = double;
using matrices_type     = Kokkos::View<scalar_type*,device_type>;

struct SVDFunctor {
public:
  using shared_matrix   = Kokkos::View<scalar_type**, typename execution_space::scratch_memory_space, Kokkos::MemoryUnmanaged>;
  using ATS             = Kokkos::ArithTraits<scalar_type>;
  using mag_type        = typename ATS::mag_type;
  using matrix_2x2_type = Kokkos::Array<Kokkos::Array<scalar_type,2>, 2>;

public:
  SVDFunctor(int n_, matrices_type::const_type As_, matrices_type pseudoAs_) : n(n_), As(As_), pseudoAs(pseudoAs_) { }

  KOKKOS_INLINE_FUNCTION
  int sgn(scalar_type x) const {
    auto zero = ATS::zero();
    return (zero < x) - (x < zero);
  }

  KOKKOS_INLINE_FUNCTION
  void givens_left(shared_matrix& A, scalar_type c, scalar_type s, int i, int k) const {
    auto n = A.extent(0);

    for (int j = 0; j < n; j++) {
      auto aij = A(i,j);
      auto akj = A(k,j);
      A(i,j) = c * aij - s * akj;
      A(k,j) = s * aij + c * akj;
    }
  }

  KOKKOS_INLINE_FUNCTION
  void givens_right(shared_matrix& A, scalar_type c, scalar_type s, int i, int k) const {
    auto n = A.extent(0);

    for (int j = 0; j < n; ++j) {
      auto aji = A(j,i);
      auto ajk = A(j,k);
      A(j,i) = c * aji - s * ajk;
      A(j,k) = s * aji + c * ajk;
    }
  }

  KOKKOS_INLINE_FUNCTION
  void trans_2x2(const matrix_2x2_type A, matrix_2x2_type& B) const {
    B = { { { A[0][0], A[1][0] }, { A[0][1], A[1][1] } } };
  }

  KOKKOS_INLINE_FUNCTION
  void trans_nxn(const shared_matrix A, shared_matrix& B) const {
    auto n = A.extent(0);

    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++)
        B(i,j) = A(j,i);
  }

  KOKKOS_INLINE_FUNCTION
  void mult_2x2(const matrix_2x2_type A, const matrix_2x2_type B, matrix_2x2_type& C) const {
    C =
    { { { A[0][0]*B[0][0] + A[0][1]*B[1][0], A[0][0]*B[0][1] + A[0][1]*B[1][1] },
        { A[1][0]*B[0][0] + A[1][1]*B[1][0], A[1][0]*B[0][1] + A[1][1]*B[1][1] } } };
  }

  KOKKOS_INLINE_FUNCTION
  void svd_2x2(const matrix_2x2_type A, matrix_2x2_type& U, matrix_2x2_type& E, matrix_2x2_type& V) const {
    matrix_2x2_type At, AAt, AtA;
    trans_2x2(A, At);
    mult_2x2(A, At, AAt);
    mult_2x2(At, A, AtA);

    // Find U such that U*A*A’*U’ = diag
    auto phi = 0.5 * atan2(AAt[0][1] + AAt[1][0], AAt[0][0] - AAt[1][1]);
    auto cphi = cos(phi);
    auto sphi = sin(phi);

    U = { { {cphi, -sphi}, { sphi, cphi } } };

    // Find W such that W’*A’*A*W = diag
    auto theta = 0.5*atan2(AtA[0][1] + AtA[1][0], AtA[0][0] - AtA[1][1]);
    auto ctheta = cos(theta);
    auto stheta = sin(theta);
    matrix_2x2_type W = { { {ctheta, -stheta}, { stheta, ctheta } } };

    // Find the singular values from U
    auto sum = AAt[0][0] + AAt[1][1];
    auto dif = sqrt((AAt[0][0] - AAt[1][1])*(AAt[0][0] - AAt[1][1]) + 4*AAt[0][1]*AAt[1][0]);
    E = { { { sqrt(0.5*(sum + dif)), 0 }, { 0, sqrt(0.5*(sum - dif)) } } };

    // Find the correction matrix for the right side
    matrix_2x2_type Ut, AW, S;
    mult_2x2(A, W, AW);
    trans_2x2(U, Ut);
    mult_2x2(Ut, AW, S);

    matrix_2x2_type C = { { { scalar_type(sgn(S[0][0])), 0.0 }, { 0.0, scalar_type(sgn(S[1][1])) } } };

    mult_2x2(W, C, V);

    // assert(check_svd(A, U, E, V) < 1e-14);
  }

  KOKKOS_INLINE_FUNCTION
  void argmax_off_diagonal(shared_matrix::const_type A, int& p, int& q) const {
    auto n = A.extent(0);

    p = q = -1;
    mag_type max = -1;

    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++)
        if (i != j && std::abs(A(i,j)) > max) {
          p = i; q = j;
          max = std::abs(A(i,j));
        }
  }

  KOKKOS_INLINE_FUNCTION
  mag_type norm_F_wo_diag(shared_matrix::const_type A) const {
    auto n = A.extent(0);

    mag_type norm = 0.0;
    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++)
        norm += ((i != j) ? A(i,j)*A(i,j) : 0);

    return std::sqrt(norm);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const typename Kokkos::TeamPolicy<execution_space>::member_type& thread) const {
    int matrix_id = thread.league_rank();

    auto A       = Kokkos::subview(As,       Kokkos::make_pair(static_cast<size_t>(matrix_id*n*n), static_cast<size_t>((matrix_id+1)*n*n)));
    auto pseudoA = Kokkos::subview(pseudoAs, Kokkos::make_pair(static_cast<size_t>(matrix_id*n*n), static_cast<size_t>((matrix_id+1)*n*n)));

    // Allocate (from scratch) and initialize
    shared_matrix E(thread.team_shmem(), n, n);
    shared_matrix U(thread.team_shmem(), n, n);
    shared_matrix V(thread.team_shmem(), n, n);

    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++) {
        E(i,j) = A(i*n+j);
      }
    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++) {
        U(i,j) = (i == j ? 1.0 : 0.0);
        V(i,j) = (i == j ? 1.0 : 0.0);
      }

    auto norm = norm_F_wo_diag(E);
    auto tol  = ATS::epsilon();

    int       num_iter = 0;
    const int max_iter = 1000;

    while (norm > tol && num_iter < max_iter) {
      // Find largest off-diagonal entry
      int p, q;
      argmax_off_diagonal(E, p, q);
      assert(p != -1 && q != -1);
      if (p > q) {
        auto t = p;
        p = q;
        q = t;
      }

      // Obtain left and right Givens rotations by using 2x2 SVD
      matrix_2x2_type Apq = { { {E(p,p), E(p,q)}, {E(q,p), E(q,q)} } }, L, D, R;

      svd_2x2(Apq, L, D, R);

      auto cl = L[0][0];
      auto sl = L[0][1];
      auto cr = R[0][0];
      auto sr = (sgn(R[0][1]) == sgn(R[1][0])) ? scalar_type(-R[0][1]) : scalar_type(R[0][1]);

      // Apply both Givens rotations to matrices that are converging to singular
      // values and singular vectors
      givens_left (E, cl, sl, p, q);
      givens_right(E, cr, sr, p, q);

      givens_right(U, cl, sl, p, q);
      givens_left (V, cr, sr, p, q);

      norm = norm_F_wo_diag(E);
      num_iter++;
    }
    assert(num_iter < max_iter);

    // Compute pseudo-inverse (pseudoA = V pseudoE U^T)
    // NOTE: the V stored above is actually V^T, but we don't transpose it,
    // instead we modify the MxM loop to do (pseudoA = V^T pseudoE U^T)
    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++) {
        scalar_type value = 0;
        for (int k = 0; k < n; k++)
          value += (E(k,k) ? V(k,i)*U(j,k)/E(k,k) : 0.0);
        pseudoA(i*n+j) = value;
      }
  }

  // amount of shared memory
  size_t team_shmem_size(int team_size) const {
    return 3*shared_matrix::shmem_size(n, n);    // U, E, V
  }

private:
  int n;
  matrices_type::const_type As;
  matrices_type pseudoAs;
};

void kernel_svd(int num_matrices, int N, typename matrices_type::const_type As, matrices_type& pseudoAs) {
  SVDFunctor svdFunctor(N, As, pseudoAs);
  Kokkos::parallel_for("main_loop", Kokkos::TeamPolicy<execution_space>(num_matrices, 1), svdFunctor);
}

void build_random_matrices(int num_matrices, int n, matrices_type& As) {
  As = matrices_type("As", num_matrices*n*n);

  auto As_host = Kokkos::create_mirror_view(As);
  for (int matrix_id = 0; matrix_id < num_matrices; matrix_id++) {
    auto A = Kokkos::subview(As_host, std::make_pair(matrix_id*n*n, (matrix_id+1)*n*n));

    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++)
        A(i*n+j) = scalar_type(rand())/INT_MAX - 0.5;
  }

  Kokkos::deep_copy(As, As_host);
}

scalar_type check_svd(int num_matrices, int n, matrices_type::const_type As, matrices_type::const_type pseudoAs) {
  auto As_host       = Kokkos::create_mirror_view(As);
  auto pseudoAs_host = Kokkos::create_mirror_view(pseudoAs);

  Kokkos::deep_copy(As_host,       As);
  Kokkos::deep_copy(pseudoAs_host, pseudoAs);

  scalar_type max_norm = 0.0;
  for (int matrix_id = 0; matrix_id < num_matrices; matrix_id++) {
    auto A       = Kokkos::subview(As_host,       Kokkos::make_pair(matrix_id*n*n, (matrix_id+1)*n*n));
    auto pseudoA = Kokkos::subview(pseudoAs_host, Kokkos::make_pair(matrix_id*n*n, (matrix_id+1)*n*n));

    scalar_type norm = 0.0;
    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++) {
        scalar_type t = 0.0;
        for (int k = 0; k < n; k++)
          t += A(i*n+k)*pseudoA(k*n+j);
        norm += (i != j ? t*t : (t-1)*(t-1));
      }
    norm = sqrt(norm);

    if (norm > max_norm)
      max_norm = norm;
  }

  return max_norm;
}

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);

  srand(1372);

  int N         = 4;
  int num_loops = 1;
  int num_matrices = 100;

  for (int i = 1; i < argc; i++) {
    if      (!strcmp(argv[i], "-N") )                                       { N = atoi(argv[++i]); }
    else if (!strcmp(argv[i], "-n") || !strcmp(argv[i], "--num_matrices"))  { num_matrices = atoi(argv[++i]); }
    else if (!strcmp(argv[i], "-l") || !strcmp(argv[i], "--num_loops"))     { num_loops = atoi(argv[++i]); }
    else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
      std::cout << "./svd.exe [-N <matrix_size>] [-n <num_matrices>] [-l <number_of_loops>]" << std::endl;
      Kokkos::finalize();
      return 0;
    } else {
      std::cerr << "Unknown option: " << argv[i] << std::endl;;
      Kokkos::finalize();
      return 1;
    }
  }

  std::cout << "Cmd line parameters:\n"
      << "\tN             = " << N << "\n"
      << "\tnum_matrices  = " << num_matrices << "\n"
      << "\tnum_loops     = " << num_loops << std::endl;

  matrices_type As;

  {
    execution_space::fence();
    auto start = std::chrono::high_resolution_clock::now();

    build_random_matrices(num_matrices, N, As);

    execution_space::fence();
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    printf("build: %.2e (s)\n", elapsed.count());
  }

  matrices_type pseudoAs("pseudoAs", As.extent(0));

  {
    execution_space::fence();
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_loops; i++)
      kernel_svd(num_matrices, N, As, pseudoAs);

    execution_space::fence();
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    printf("kernel_svd: %.2e (s)\n", elapsed.count() / num_loops);
  }

  // Check the solution
  auto max_norm = check_svd(num_matrices, N, As, pseudoAs);
  std::cout << "max_svd_norm = " << max_norm << std::endl;
  assert(max_norm < 1e-11);

  Kokkos::finalize();

  return 0;
}

#include <Kokkos_Core.hpp>
#include <impl/Kokkos_Timer.hpp>

#include <Kokkos_ArithTraits.hpp>

#include <cassert>
#include <iostream>

using execution_space   = Kokkos::DefaultExecutionSpace;
using device_type       = Kokkos::Device<execution_space,typename execution_space::memory_space>;
using scalar_type       = double;
using ATS               = Kokkos::ArithTraits<scalar_type>;
using mag_type          = typename ATS::mag_type;
using matrix_type       = Kokkos::View<scalar_type**,device_type>;
using matrix_2x2_type   = Kokkos::Array<Kokkos::Array<scalar_type,2>, 2>;

KOKKOS_INLINE_FUNCTION
int sgn(scalar_type x) {
  auto zero = ATS::zero();
  return (zero < x) - (x < zero);
}

mag_type matrix_compare(matrix_type A, matrix_type B) {
  mag_type norm = ATS::zero();

  auto n = A.extent(0);

  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++) {
      norm += (A(i,j) - B(i,j))*(A(i,j) - B(i,j));
    }
  return sqrt(norm);
}

void write(const std::string& name, matrix_type A) {
  auto n = A.extent(0);

  auto A_host = Kokkos::create_mirror_view(A);

  std::cout << name << std::endl;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++)
      std::cout << " " << A_host(i,j);
    std::cout << std::endl;
  }
}

void write(const std::string& name, matrix_2x2_type A) {
  std::cout << name << std::endl;
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++)
      std::cout << " " << A[i][j];
    std::cout << std::endl;
  }
}

KOKKOS_INLINE_FUNCTION
void givens_left(matrix_type& A, scalar_type c, scalar_type s, int i, int k) {
  auto n = A.extent(0);

  for (int j = 0; j < n; j++) {
    auto aij = A(i,j);
    auto akj = A(k,j);
    A(i,j) = c * aij - s * akj;
    A(k,j) = s * aij + c * akj;
  }
}

KOKKOS_INLINE_FUNCTION
void givens_right(matrix_type& A, scalar_type c, scalar_type s, int i, int k) {
  auto n = A.extent(0);

  for (int j = 0; j < n; ++j) {
    auto aji = A(j,i);
    auto ajk = A(j,k);
    A(j,i) = c * aji - s * ajk;
    A(j,k) = s * aji + c * ajk;
  }
}

KOKKOS_INLINE_FUNCTION
void trans_2x2(const matrix_2x2_type A, matrix_2x2_type& B) {
  B = { { { A[0][0], A[1][0] }, { A[0][1], A[1][1] } } };
}

KOKKOS_INLINE_FUNCTION
void mult_nxn(const matrix_type A, const matrix_type B, matrix_type& C) {
  auto n = A.extent(0);

  C = matrix_type("C", n, n);

  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      for (int k = 0; k < n; k++)
        C(i,j) += A(i,k)*B(k,j);
}

KOKKOS_INLINE_FUNCTION
void trans_nxn(const matrix_type A, matrix_type& B) {
  auto n = A.extent(0);

  B = matrix_type("B", n, n);

  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      B(i,j) = A(j,i);
}

KOKKOS_INLINE_FUNCTION
void mult_2x2(const matrix_2x2_type A, const matrix_2x2_type B, matrix_2x2_type& C) {
  C =
  { { { A[0][0]*B[0][0] + A[0][1]*B[1][0], A[0][0]*B[0][1] + A[0][1]*B[1][1] },
      { A[1][0]*B[0][0] + A[1][1]*B[1][0], A[1][0]*B[0][1] + A[1][1]*B[1][1] } } };
}

mag_type check_svd(const matrix_2x2_type A, const matrix_2x2_type U, const matrix_2x2_type S, const matrix_2x2_type V) {
  matrix_2x2_type Vt, SVt, USVt;
  trans_2x2(V, Vt);
  mult_2x2(S, Vt, SVt);
  mult_2x2(U, SVt, USVt);

  mag_type norm = ATS::zero();
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 2; j++) {
      norm += (A[i][j] - USVt[i][j])*(A[i][j] - USVt[i][j]);
    }
  return sqrt(norm);
}

mag_type check_svd(const matrix_type A, const matrix_type U, const matrix_type S, const matrix_type V) {
  auto n = A.extent(0);

  matrix_type Vt, SVt, USVt;
  trans_nxn(V, Vt);
  mult_nxn(S, Vt, SVt);
  mult_nxn(U, SVt, USVt);

  mag_type norm = ATS::zero();
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++) {
      norm += (A(i,j) - USVt(i,j))*(A(i,j) - USVt(i,j));
    }
  return sqrt(norm);
}

void svd_2x2(const matrix_2x2_type A, matrix_2x2_type& U, matrix_2x2_type& E, matrix_2x2_type& V) {
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

  assert(check_svd(A, U, E, V) < 1e-14);
}

KOKKOS_INLINE_FUNCTION
void argmax_off_diagonal(matrix_type A, int& p, int& q) {
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
mag_type norm_F_wo_diag(matrix_type A) {
  auto n = A.extent(0);

  mag_type norm = 0.0;
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      norm += ((i != j) ? A(i,j)*A(i,j) : 0);

  return std::sqrt(norm);
}

void kernel_svd(typename matrix_type::const_type A, matrix_type& U, matrix_type& E, matrix_type& V) {
  typedef Kokkos::RangePolicy<execution_space> range_policy;

  auto n = A.extent(0);

  E = matrix_type("E", n, n);
  U = matrix_type("U", n, n);
  V = matrix_type("V", n, n);

  Kokkos::deep_copy(E, A);

  // FIXME: Scale original matrix
  // auto norm_a = norm(E);
  // auto scale = norm_a > 0.0 ? norm_a : scalar_type(1.0);
  // scale_matrix(E, 1./scale);

  // Create U and V
  Kokkos::parallel_for("init_U_V", range_policy(0,n), KOKKOS_LAMBDA(const int i) {
    U(i,i) = 1;
    V(i,i) = 1;
  });

  auto norm = norm_F_wo_diag(E);
  auto tol  = ATS::epsilon();

  int       num_iter = 0;
  const int max_iter = 1000;

  while (norm > tol && num_iter < max_iter) {
    // Find largest off-diagonal entry
    int p, q;
    argmax_off_diagonal(E, p, q);
    assert(p != -1 && q != -1);
    if (p > q)
      std::swap(p, q);

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

  matrix_type Vt;
  trans_nxn(V, Vt);
  V = Vt;
}

matrix_type build_random_matrix(int n) {
  matrix_type A("A", n, n);

  auto A_host = Kokkos::create_mirror_view(A);
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      A_host(i,j) = scalar_type(rand())/INT_MAX - 0.5;

  Kokkos::deep_copy(A, A_host);

  return A;
}

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);

  srand(13721);

  int n         = 4;
  int num_loops = 1;

  for (int i = 1; i < argc; i++) {
    if      (!strcmp(argv[i], "-n") )                                       { n = atoi(argv[++i]); }
    else if (!strcmp(argv[i], "-l") || !strcmp(argv[i], "--num_loops"))     { num_loops = atoi(argv[++i]); }
    else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
      std::cout << "./svd.exe [-n <matrix_size>] [-l <number_of_loops>]" << std::endl;
      return 0;
    } else {
      throw std::runtime_error(std::string("Unknown option: ") + argv[i]);
    }
  }

  std::cout << "Cmd line parameters:\n"
      << "\tn             = " << n << "\n"
      << "\tnum_loops     = " << num_loops << std::endl;

  matrix_type A = build_random_matrix(n), U, E, V;

  execution_space::fence();
  Kokkos::Impl::Timer timer;

  for (int i = 0; i < num_loops; i++)
    kernel_svd(A, U, E, V);

  double kernel_time = timer.seconds();
  execution_space::fence();

  printf("kernel_svd: %.2e (s)\n", kernel_time / num_loops);

  // Check the solution
  assert(check_svd(A, U, E, V) < 1e-14);
  std::cout << "diff = " << check_svd(A, U, E, V) << std::endl;

  execution_space::finalize();

  Kokkos::finalize();

  return 0;
}

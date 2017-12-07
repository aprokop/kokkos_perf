#include <Kokkos_Core.hpp>
#include <Kokkos_ArithTraits.hpp>
#include <KokkosSparse_CrsMatrix.hpp>
#include <Kokkos_StaticCrsGraph.hpp>
#include <impl/Kokkos_Timer.hpp>

#include <iostream>

template<class local_ordinal_type, class device_type>
class LWGraph_kokkos {
public:
  typedef Kokkos::StaticCrsGraph<local_ordinal_type, Kokkos::LayoutLeft, device_type> local_graph_type;

public:
  LWGraph_kokkos(const local_graph_type& graph) : graph_(graph) { }
  ~LWGraph_kokkos() { }

private:
  //! Underlying graph (with label)
  const local_graph_type graph_;
};

template<class T1, class T2>
struct custom_pair {
  T1 first;
  T2 second;

  KOKKOS_INLINE_FUNCTION
  custom_pair() {
    first  = 0;
    second = 0;
  }
  KOKKOS_INLINE_FUNCTION
  custom_pair(const custom_pair& pair) {
    first  = pair.first;
    second = pair.second;
  }
  KOKKOS_INLINE_FUNCTION
  custom_pair(const volatile custom_pair& pair) {
    first  = pair.first;
    second = pair.second;
  }
  KOKKOS_INLINE_FUNCTION
  custom_pair& operator=(const custom_pair& pair) {
    first  = pair.first;
    second = pair.second;
    return *this;
  }
  KOKKOS_INLINE_FUNCTION
  custom_pair& operator=(const volatile custom_pair& pair) {
    first  = pair.first;
    second = pair.second;
    return *this;
  }
  KOKKOS_INLINE_FUNCTION
  custom_pair& operator+=(const custom_pair& pair) {
    first  += pair.first;
    second += pair.second;
    return *this;
  }
  KOKKOS_INLINE_FUNCTION
  custom_pair& operator+=(const volatile custom_pair& pair) volatile {
    first  += pair.first;
    second += pair.second;
    return *this;
  }
  KOKKOS_INLINE_FUNCTION
  bool operator==(const custom_pair& pair) {
    return ((first == pair.first) && (second == pair.second));
  }
};

template<class local_ordinal_type, class DiagViewType>
class ClassicalDropFunctor {
private:
  using scalar_type     = typename DiagViewType::value_type;
  using ATS             = Kokkos::ArithTraits<scalar_type>;
  using magnitudeType   = typename ATS::magnitudeType;

  DiagViewType  diag;   // corresponds to overlapped diagonal multivector (2D View)
  magnitudeType eps;

public:
  ClassicalDropFunctor(DiagViewType ghostedDiag, magnitudeType threshold) :
      diag(ghostedDiag),
      eps(threshold)
  { }

  // Return true if we drop, false if not
  KOKKOS_FORCEINLINE_FUNCTION
  bool operator()(local_ordinal_type row, local_ordinal_type col, scalar_type val) const {
    // We avoid square root by using squared values
    auto aiiajj = ATS::magnitude(diag(row, 0)) * ATS::magnitude(diag(col, 0));   // |a_ii|*|a_jj|
    auto aij2   = ATS::magnitude(val)          * ATS::magnitude(val);            // |a_ij|^2

    return (aij2 <= eps*eps * aiiajj);
  }
  KOKKOS_INLINE_FUNCTION
  scalar_type cacheDiagVal(local_ordinal_type i) const {
    return diag(i,0);
  }
  // Return true if we drop, false if not
  KOKKOS_FORCEINLINE_FUNCTION
  bool operator()(scalar_type cachedDiagVal, local_ordinal_type col, scalar_type val) const {
    // We avoid square root by using squared values
    auto aiiajj = ATS::magnitude(cachedDiagVal) * ATS::magnitude(diag(col, 0));   // |a_ii|*|a_jj|
    auto aij2   = ATS::magnitude(val)           * ATS::magnitude(val);            // |a_ij|^2

    return (aij2 <= eps*eps * aiiajj);
  }

};

template<class local_ordinal_type, class CoordsType>
class DistanceFunctor {
private:
  typedef typename CoordsType::value_type           scalar_type;
  typedef          Kokkos::ArithTraits<scalar_type>          ATS;
  typedef typename ATS::magnitudeType               magnitudeType;

public:
  typedef scalar_type value_type;

public:
  DistanceFunctor(CoordsType coords_) : coords(coords_) { }

  KOKKOS_INLINE_FUNCTION
  magnitudeType distance2(local_ordinal_type row, local_ordinal_type col) const {
    scalar_type d = ATS::zero(), s;
    for (size_t j = 0; j < coords.extent(1); j++) {
      s = coords(row,j) - coords(col,j);
      d += s*s;
    }
    return ATS::magnitude(d);
  }
private:
  CoordsType coords;
};

template<class local_ordinal_type, class GhostedViewType, class DistanceFunctor>
class DistanceLaplacianDropFunctor {
private:
  typedef typename GhostedViewType::value_type      scalar_type;
  typedef          Kokkos::ArithTraits<scalar_type>          ATS;
  typedef typename ATS::magnitudeType               magnitudeType;

public:
  DistanceLaplacianDropFunctor(GhostedViewType lapl_diag, DistanceFunctor distFunctor_, magnitudeType threshold) :
      diag(lapl_diag),
      distFunctor(distFunctor_),
      eps(threshold)
  { }

  // Return true if we drop, false if not
  KOKKOS_INLINE_FUNCTION
  bool operator()(local_ordinal_type row, local_ordinal_type col, scalar_type val) const {
    // We avoid square root by using squared values

    // We ignore incoming value of val as we operate on an auxiliary
    // distance Laplacian matrix
    typedef typename DistanceFunctor::value_type      dscalar_type;
    typedef          Kokkos::ArithTraits<dscalar_type>         dATS;
    auto fval = dATS::one() / distFunctor.distance2(row, col);

    auto aiiajj = ATS::magnitude(diag(row, 0)) * ATS::magnitude(diag(col, 0));   // |a_ii|*|a_jj|
    auto aij2   = ATS::magnitude(fval)         * ATS::magnitude(fval);           // |a_ij|^2

    return (aij2 <= eps*eps * aiiajj);
  }

private:
  GhostedViewType   diag;   // corresponds to overlapped diagonal multivector (2D View)
  DistanceFunctor   distFunctor;
  magnitudeType     eps;
};

template<class scalar_type, class local_ordinal_type, class MatrixType, class DropFunctorType>
class ScalarFunctor {
private:
  typedef typename MatrixType::StaticCrsGraphType   graph_type;
  typedef typename graph_type::row_map_type         rows_type;
  typedef typename graph_type::entries_type         cols_type;
  typedef typename MatrixType::values_type          vals_type;
  typedef          Kokkos::ArithTraits<scalar_type>          ATS;
  typedef typename ATS::magnitudeType               magnitudeType;

  typedef typename graph_type::execution_space          execution_space;
  typedef typename Kokkos::TeamPolicy<execution_space>  team_policy;
  typedef typename team_policy::member_type             team_member;

public:
  ScalarFunctor(MatrixType A_, DropFunctorType dropFunctor_,
                typename rows_type::non_const_type rows_,
                typename cols_type::non_const_type colsAux_,
                typename vals_type::non_const_type valsAux_,
                bool reuseGraph_, bool lumping_, scalar_type threshold_,
                const int rows_per_team_) :
      A(A_),
      dropFunctor(dropFunctor_),
      rows(rows_),
      colsAux(colsAux_),
      valsAux(valsAux_),
      reuseGraph(reuseGraph_),
      lumping(lumping_),
      rows_per_team(rows_per_team_)
  {
    zero = ATS::zero();
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const team_member& dev, local_ordinal_type& nnz) const {
    Kokkos::parallel_for(Kokkos::TeamThreadRange(dev, 0, rows_per_team), [&](const local_ordinal_type& loop) {
      const local_ordinal_type row = static_cast<local_ordinal_type>( dev.league_rank() ) * rows_per_team + loop;
      if (row >= A.numRows ())
        return;

      auto rowView    = A.rowConst(row);
      auto row_length = rowView.length;
      auto offset     = A.graph.row_map(row);

      local_ordinal_type diagID = -1;

      using reduction_type = custom_pair<local_ordinal_type,scalar_type>;

      auto diagCacheRow = dropFunctor.cacheDiagVal(row);

      reduction_type reduce_pair;
      // Rule of thumb: vector length to be 1/3 of row length
      Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(dev, row_length), [&] (const local_ordinal_type& colID, reduction_type& local_reduce_pair) {
        local_ordinal_type col = rowView.colidx(colID);
        scalar_type        val = rowView.value (colID);

        if (!dropFunctor(diagCacheRow, col, val) || row == col) {
          colsAux(offset+colID) = col;
          valsAux(offset+colID) = val;
          if (row == col)
            diagID = colID;

          local_reduce_pair.first++;            // row_nnz++

        } else {
          colsAux(offset+colID) = -1;
          valsAux(offset+colID) = zero;
          local_reduce_pair.second += val;      // diag += val
        }
      }, reduce_pair);

      if (diagID != -1) {
        // Only thread that has diagonal is here
        auto rownnz = reduce_pair.first;
        auto diag   = reduce_pair.second;

        // FIXME: How to assert on the device?
        rows(row+1) = rownnz;
        if (lumping) {
          // Add diag to the diagonal

          // NOTE_KOKKOS: valsAux was allocated with
          // ViewAllocateWithoutInitializing. This is not a problem here
          // because we explicitly set this value above.
          valsAux(offset+diagID) += diag;
        }

        nnz += rownnz;
      }
    });
  }

private:
  MatrixType                            A;
  DropFunctorType                       dropFunctor;

  const local_ordinal_type              rows_per_team;

  typename rows_type::non_const_type    rows;
  typename cols_type::non_const_type    colsAux;
  typename vals_type::non_const_type    valsAux;

  bool                                  reuseGraph;
  bool                                  lumping;
  scalar_type                           zero;
};

template<class scalar_type, class local_ordinal_type, class device_type>
void kernel_coalesce_drop_device(KokkosSparse::CrsMatrix<scalar_type, local_ordinal_type, device_type> A, const std::string& algo, scalar_type threshold, bool lumping) {
  using local_matrix_type = KokkosSparse::CrsMatrix<scalar_type, local_ordinal_type, device_type>;
  using kokkos_graph_type = typename LWGraph_kokkos<local_ordinal_type,device_type>::local_graph_type;
  using rows_type         = typename kokkos_graph_type::row_map_type::non_const_type;
  using cols_type         = typename kokkos_graph_type::entries_type::non_const_type;
  using vals_type         = typename local_matrix_type::values_type::non_const_type;
  using ATS               = Kokkos::ArithTraits<scalar_type>;
  using magnitude_type    = typename ATS::mag_type;

  using execution_space   = typename local_matrix_type::execution_space;

  auto numRows = A.numRows();
  auto nnzA    = A.nnz();
  auto rowsA   = A.graph.row_map;

  const bool reuseGraph = false;

  // FIXME: replace by ViewAllocateWithoutInitializing + setting a single value
  rows_type rows   (                                        "FA_rows",      numRows+1);
  cols_type colsAux(Kokkos::ViewAllocateWithoutInitializing("FA_aux_cols"), nnzA);
  vals_type valsAux(Kokkos::ViewAllocateWithoutInitializing("FA_aux_vals"), nnzA);

  local_ordinal_type nnzFA = 0;

  const int rows_per_team = 10;     // FIXME: make a cmd-line argument

  typedef Kokkos::RangePolicy<execution_space>  range_policy;
  typedef Kokkos::TeamPolicy<execution_space>   team_policy;

  if (algo == "classical") {
    // Construct overlapped matrix diagonal
    Kokkos::View<double**,device_type> ghostedDiagView("ghosted_diag", numRows, 1);             // MultiVector
    // FIXME: extract matrix diagonal

    ClassicalDropFunctor<local_ordinal_type, decltype(ghostedDiagView)> dropFunctor(ghostedDiagView, threshold);
    ScalarFunctor<scalar_type, local_ordinal_type, local_matrix_type, decltype(dropFunctor)>
        scalarFunctor(A, dropFunctor, rows, colsAux, valsAux, reuseGraph, lumping, threshold, rows_per_team);

    // Rule of thumb for GPU: vector length (3rd parameter to team policy) to be 1/3 of row length, and has to be a power of 2
    // 27 nonzeros in a row -> 8
    Kokkos::parallel_reduce("main_loop", team_policy(numRows, Kokkos::AUTO, 8), scalarFunctor, nnzFA);

  } else if (algo == "distance laplacian") {
#if 1
    throw std::runtime_error("disable for now");
#else
    // FIXME: create some coordinates
    Kokkos::View<scalar_type**,device_type> ghostedCoordsView("ghosted_coords", numRows, 3); // MultiVector

    DistanceFunctor<local_ordinal_type, decltype(ghostedCoordsView)> distFunctor(ghostedCoordsView);

    Kokkos::View<scalar_type*,device_type> lapl_diag("lapl_diag", numRows);

    // Construct Laplacian diagonal
    Kokkos::parallel_for("construct_lapl_diag", range_policy(0,numRows),
      KOKKOS_LAMBDA(const local_ordinal_type row) {
        auto rowView = A.graph.rowConst(row);
        auto length  = rowView.length;

        scalar_type d = ATS::zero();
        for (decltype(length) colID = 0; colID < length; colID++) {
          auto col = rowView(colID);
          if (row != col)
            d += ATS::one()/distFunctor.distance2(row, col);
        }
        lapl_diag(row,0) = d;
      });

    // Filter out entries
    DistanceLaplacianDropFunctor<local_ordinal_type, decltype(lapl_diag), decltype(distFunctor)>
        dropFunctor(lapl_diag, distFunctor, threshold);
    ScalarFunctor<scalar_type, local_ordinal_type, local_matrix_type, decltype(dropFunctor)>
        scalarFunctor(A, dropFunctor, rows, colsAux, valsAux, reuseGraph, lumping, threshold, rows_per_team);

    Kokkos::parallel_reduce("main_loop", team_policy(numRows, Kokkos::AUTO, 32), scalarFunctor, nnzFA);
#endif
  }
  // std::cout << "Number of dropped entries = " << (1.0 - (double)nnzFA/nnzA) << std::endl;

#if 0
  // parallel_scan (exclusive)
  Kokkos::parallel_scan("compress_rows", range_policy(0,numRows+1),
    KOKKOS_LAMBDA(const local_ordinal_type i, local_ordinal_type& update, const bool& final_pass) {
      update += rows(i);
      if (final_pass)
        rows(i) = update;
  });

  // Compress cols (and optionally vals)
  // We use a trick here: we moved all remaining elements to the beginning
  // of the original row in the main loop, so we don't need to check for
  // INVALID here, and just stop when achieving the new number of elements
  // per row.
  cols_type cols(Kokkos::ViewAllocateWithoutInitializing("FA_cols"), nnzFA);
  vals_type vals(Kokkos::ViewAllocateWithoutInitializing("FA_vals"), nnzFA);
  // Kokkos::parallel_reduce("main_loop", team_policy(numRows, Kokkos::AUTO, 32), scalarFunctor, nnzFA);
  Kokkos::parallel_for("compress_cols_and_vals", range_policy(0,numRows),
    KOKKOS_LAMBDA(const local_ordinal_type i) {
      local_ordinal_type rowStart  = rows(i);
      local_ordinal_type rowAStart = rowsA(i);
      size_t rownnz = rows(i+1) - rows(i);
      for (size_t j = 0; j < rownnz; j++) {
        cols(rowStart+j) = colsAux(rowAStart+j);
        vals(rowStart+j) = colsAux(rowAStart+j);
      }
  });
#else
  cols_type cols(Kokkos::ViewAllocateWithoutInitializing("FA_cols"), nnzFA);
  vals_type vals(Kokkos::ViewAllocateWithoutInitializing("FA_vals"), nnzFA);
#endif

  kokkos_graph_type kokkosGraph(cols, rows);
}

template<class scalar_type, class local_ordinal_type, class device_type>
KokkosSparse::CrsMatrix<scalar_type, local_ordinal_type, device_type>
kernel_construct(local_ordinal_type numRows) {
  const int nnzPerRow  = 27;
  auto numCols         = numRows;
  auto nnz             = nnzPerRow*numRows;

  auto varianz_nel_row = 0.2*nnzPerRow;
  auto width_row       = 0.01*numRows;

  auto rowPtr = new local_ordinal_type[numRows+1];
  rowPtr[0] = 0;
  for (int row = 0; row < numRows; row++) {
    int varianz = (1.0*rand()/INT_MAX-0.5)*varianz_nel_row;

    rowPtr[row+1] = rowPtr[row] + nnzPerRow + varianz;
  }
  nnz = rowPtr[numRows];

  auto colInd = new local_ordinal_type[nnz];
  auto values = new scalar_type       [nnz];
  for (int row = 0; row < numRows; row++) {
    for (int k = rowPtr[row]; k < rowPtr[row+1]; k++) {
      int pos = row + (1.0*rand()/INT_MAX-0.5)*width_row;

      if (pos <  0)       pos += numCols;
      if (pos >= numCols) pos -= numCols;
      colInd[k] = pos;
      values[k] = 100.0*rand()/INT_MAX - 50.0;
    }
  }

  typedef KokkosSparse::CrsMatrix<scalar_type, local_ordinal_type, device_type> local_matrix_type;

  return local_matrix_type("A", numRows, numCols, nnz, values, rowPtr, colInd, false/*pad*/);
}

template<class scalar_type, class local_ordinal_type, class device_type>
int main_(int argc, char **argv) {
  int n         = 100000;
  int num_loops = 10;

  bool lumping = true;
  scalar_type eps = 0.05;
  std::string algo = "classical";

  for (int i = 1; i < argc; i++) {
    if      (strcmp(argv[i], "-n") == 0)                                   { n    = atoi(argv[++i]); continue; }
    else if (strcmp(argv[i], "-l") == 0 || strcmp(argv[i], "--num_loops") == 0) { num_loops = atoi(argv[++i]); continue; }
    else if (strcmp(argv[i], "--node") == 0)                               { i++; continue; }
    else if (strcmp(argv[i], "--lumping") == 0)                            { i++; lumping = true; continue; }
    else if (strcmp(argv[i], "--no-lumping") == 0)                         { i++; lumping = false; continue; }
    else if (strcmp(argv[i], "-a") == 0 || strcmp(argv[i], "--algo") == 0) { algo = argv[++i]; continue; }
    else if (strcmp(argv[i], "-e") == 0 || strcmp(argv[i], "--eps") == 0)  { eps = atof(argv[++i]); continue; }
    else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
      std::cout << "./coalesce_drop.exe [-n <matrix_size>] [-l <number_of_loops>]" << std::endl;
      return 0;
    } else {
      throw std::runtime_error(std::string("Uknown option: ") + argv[i]);
    }
  }
  if (algo != "classical" && algo != "distance_laplacian")
    throw std::runtime_error("Unknown algo: \"" + algo + "\"");

  std::cout << "n = " << n << ", num_loops = " << num_loops << ", algo = " << algo << ", eps = " << eps << ", lumping = " << lumping << std::endl;

  typedef typename device_type::execution_space execution_space;

  auto A = kernel_construct<scalar_type, local_ordinal_type, device_type>(n);

  {
    execution_space::fence();
    Kokkos::Impl::Timer timer;

    for (int i = 0; i < num_loops; i++)
      kernel_coalesce_drop_device(A, algo, eps, lumping);

    double kernel_time = timer.seconds();
    execution_space::fence();

    printf("kernel_coalesce_drop: %.2e (s)\n", kernel_time / num_loops);
  }

  execution_space::finalize();

  return 0;
}

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);

  srand(13721);

  std::string node;
  for (int i = 1; i < argc; i++) {
    if (strncmp(argv[i], "--node=", 7) == 0) {
      std::cout << "Use --node <node> instead of --node=<node>" << std::endl;
      Kokkos::finalize();
      return 0;

    } else if (strncmp(argv[i], "--node", 6) == 0)
      node = argv[++i];
  }

  std::cout << "node = " << (node == "" ? "default" : node) << std::endl;

  if (node == "") {
    return main_<double,int,Kokkos::DefaultExecutionSpace>(argc, argv);

  } else if (node == "serial") {
#ifdef KOKKOS_HAVE_SERIAL
    return main_<double,int,Kokkos::Serial>(argc, argv);
#else
    std::cout << "Error: Serial node type is disabled" << std::endl;
#endif

  } else if (node == "openmp") {
#ifdef KOKKOS_HAVE_OPENMP
    return main_<double,int,Kokkos::OpenMP>(argc, argv);
#else
    std::cout << "Error: OpenMP node type is disabled" << std::endl;
#endif

  } else if (node == "cuda") {
#ifdef KOKKOS_HAVE_CUDA
    return main_<double,int,Kokkos::Cuda>(argc, argv);
#else
    std::cout << "Error: CUDA node type is disabled" << std::endl;
#endif
  }

  Kokkos::finalize();

  return 0;
}

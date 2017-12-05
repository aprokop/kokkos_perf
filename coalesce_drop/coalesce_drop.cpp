#include <Kokkos_Core.hpp>
#include <Kokkos_ArithTraits.hpp>
#include <KokkosSparse_CrsMatrix.hpp>
#include <Kokkos_StaticCrsGraph.hpp>
#include <impl/Kokkos_Timer.hpp>

#include <iostream>

template<class scalar_type, class local_ordinal_type, class device_type>
class LWGraph_kokkos {
public:
  typedef Kokkos::StaticCrsGraph<local_ordinal_type, Kokkos::LayoutLeft, device_type>   local_graph_type;
  typedef Kokkos::View<bool*, device_type>                                              boundary_nodes_type;

public:
  LWGraph_kokkos(const local_graph_type& graph) : graph_(graph) { }
  ~LWGraph_kokkos() { }

  //! Set boolean array indicating which rows correspond to Dirichlet boundaries.
  KOKKOS_INLINE_FUNCTION
  void setBoundaryNodeMap(const boundary_nodes_type bndry) {
    dirichletBoundaries_ = bndry;
  }

private:
  //! Underlying graph (with label)
  const local_graph_type graph_;

  //! Boolean array marking Dirichlet rows.
  boundary_nodes_type dirichletBoundaries_;
};

template<class local_ordinal_type, class GhostedViewType>
class ClassicalDropFunctor {
private:
  typedef typename GhostedViewType::value_type      scalar_type;
  typedef          Kokkos::ArithTraits<scalar_type>          ATS;
  typedef typename ATS::magnitudeType               magnitudeType;

  GhostedViewType   diag;   // corresponds to overlapped diagonal multivector (2D View)
  magnitudeType     eps;

public:
  ClassicalDropFunctor(GhostedViewType ghostedDiag, magnitudeType threshold) :
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

template<class scalar_type, class local_ordinal_type, class MatrixType, class BndViewType, class DropFunctorType>
class ScalarFunctor {
private:
  typedef typename MatrixType::StaticCrsGraphType   graph_type;
  typedef typename graph_type::row_map_type         rows_type;
  typedef typename graph_type::entries_type         cols_type;
  typedef typename MatrixType::values_type          vals_type;
  typedef          Kokkos::ArithTraits<scalar_type>          ATS;
  typedef typename ATS::magnitudeType               magnitudeType;

public:
  ScalarFunctor(MatrixType A_, BndViewType bndNodes_, DropFunctorType dropFunctor_,
                typename rows_type::non_const_type rows_,
                typename cols_type::non_const_type colsAux_,
                typename vals_type::non_const_type valsAux_,
                bool reuseGraph_, bool lumping_, scalar_type threshold_) :
      A(A_),
      bndNodes(bndNodes_),
      dropFunctor(dropFunctor_),
      rows(rows_),
      colsAux(colsAux_),
      valsAux(valsAux_),
      reuseGraph(reuseGraph_),
      lumping(lumping_)
  {
    rowsA = A.graph.row_map;
    zero = ATS::zero();
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const local_ordinal_type row, local_ordinal_type& nnz) const {
    auto rowView = A.rowConst(row);
    auto length  = rowView.length;
    auto offset  = rowsA(row);

    scalar_type diag = zero;
    local_ordinal_type rownnz = 0;
    local_ordinal_type diagID = -1;
    for (decltype(length) colID = 0; colID < length; colID++) {
      local_ordinal_type col = rowView.colidx(colID);
      scalar_type val = rowView.value (colID);

      if (!dropFunctor(row, col, rowView.value(colID)) || row == col) {
        colsAux(offset+rownnz) = col;

        local_ordinal_type valID = (reuseGraph ? colID : rownnz);
        valsAux(offset+valID) = val;
        if (row == col)
          diagID = valID;

        rownnz++;

      } else {
        // Rewrite with zeros (needed for reuseGraph)
        valsAux(offset+colID) = zero;
        diag += val;
      }
    }
    // FIXME: How to assert on the device?
    // assert(diagIndex != -1);
    rows(row+1) = rownnz;
    // if (lumping && diagID != -1)
    if (lumping) {
      // Add diag to the diagonal

      // NOTE_KOKKOS: valsAux was allocated with
      // ViewAllocateWithoutInitializing. This is not a problem here
      // because we explicitly set this value above.
      valsAux(offset+diagID) += diag;
    }

    // If the only element remaining after filtering is diagonal, mark node as boundary
    // FIXME: this should really be replaced by the following
    //    if (indices.size() == 1 && indices[0] == row)
    //        boundaryNodes[row] = true;
    // We do not do it this way now because there is no framework for distinguishing isolated
    // and boundary nodes in the aggregation algorithms
    bndNodes(row) = (rownnz == 1);

    nnz += rownnz;
  }

private:
  MatrixType                            A;
  BndViewType                           bndNodes;
  DropFunctorType                       dropFunctor;

  rows_type                             rowsA;

  typename rows_type::non_const_type    rows;
  typename cols_type::non_const_type    colsAux;
  typename vals_type::non_const_type    valsAux;

  bool                                  reuseGraph;
  bool                                  lumping;
  scalar_type                                    zero;
};

template<class scalar_type, class local_ordinal_type, class device_type>
void kernel_coalesce_drop_device(KokkosSparse::CrsMatrix<scalar_type, local_ordinal_type, device_type> A) {
  typedef typename LWGraph_kokkos<scalar_type,local_ordinal_type,device_type>::local_graph_type                 kokkos_graph_type;
  typedef KokkosSparse::CrsMatrix<scalar_type, local_ordinal_type, device_type>   local_matrix_type;
  typedef typename kokkos_graph_type::row_map_type::non_const_type  rows_type;
  typedef typename kokkos_graph_type::entries_type::non_const_type  cols_type;
  typedef typename local_matrix_type::values_type::non_const_type   vals_type;
  typedef          Kokkos::ArithTraits<scalar_type>                 ATS;
  typedef typename ATS::mag_type                                    magnitude_type;
  typedef Kokkos::View<bool*, device_type>                          boundary_nodes_type;

    typedef Kokkos::RangePolicy<typename local_matrix_type::execution_space> range_type;

  auto numRows = A.numRows();
  auto nnzA    = A.nnz();
  auto rowsA   = A.graph.row_map;

  scalar_type threshold = 0.05;

  bool reuseGraph = false; // FIXME
  bool lumping    = true;  // FIXME
  if (lumping)
    std::cout << "Lumping dropped entries" << std::endl;

  // FIXME_KOKKOS: replace by ViewAllocateWithoutInitializing + setting a single value
  rows_type rows   ("FA_rows",     numRows+1);
  cols_type colsAux(Kokkos::ViewAllocateWithoutInitializing("FA_aux_cols"), nnzA);
  vals_type valsAux(Kokkos::ViewAllocateWithoutInitializing("FA_aux_vals"), nnzA);

  typename boundary_nodes_type::non_const_type bndNodes(Kokkos::ViewAllocateWithoutInitializing("boundary_nodes"), numRows);

  local_ordinal_type nnzFA = 0;

  // FIXME: make it command-line argument
  std::string algo = "classical";
  if (algo == "classical") {
    // Construct overlapped matrix diagonal
    Kokkos::View<double*,device_type> ghostedDiagView("ghosted_diag", numRows);
    // FIXME: fill in ghostedDiag with the matrix diag values

    // Filter out entries
    {
      // FIXME: timers
      // SubFactoryMonitor m2(*this, "MainLoop", currentLevel);

      ClassicalDropFunctor<local_ordinal_type, decltype(ghostedDiagView)> dropFunctor(ghostedDiagView, threshold);
      ScalarFunctor<scalar_type, local_ordinal_type, local_matrix_type, decltype(bndNodes), decltype(dropFunctor)>
          scalarFunctor(A, bndNodes, dropFunctor, rows, colsAux, valsAux, reuseGraph, lumping, threshold);

      Kokkos::parallel_reduce("MueLu:CoalesceDropF:Build:scalar_filter:main_loop", range_type(0,numRows),
                              scalarFunctor, nnzFA);
    }

  } else if (algo == "distance laplacian") {
      // FIXME: create some coordinates
    Kokkos::View<scalar_type*[3],device_type> ghostedCoordsView("ghosted_coords", numRows);

    DistanceFunctor<local_ordinal_type, decltype(ghostedCoordsView)> distFunctor(ghostedCoordsView);

    Kokkos::View<scalar_type*,device_type> lapl_diag("lap_diag", numRows);
      // Construct Laplacian diagonal
      {
        // FIXME timers
        // SubFactoryMonitor m2(*this, "Local Laplacian diag construction", currentLevel);

        auto kokkosGraph = A.graph;

        Kokkos::parallel_for("laplacian_diag", range_type(0,numRows),
          KOKKOS_LAMBDA(const local_ordinal_type row) {
            auto rowView = kokkosGraph.rowConst(row);
            auto length  = rowView.length;

            scalar_type d = ATS::zero();
            for (decltype(length) colID = 0; colID < length; colID++) {
              auto col = rowView(colID);
              if (row != col)
                d += ATS::one()/distFunctor.distance2(row, col);
            }
            lapl_diag(row,0) = d;
          });
      }

      // Filter out entries
      {
        // FIXME: time this section
        // SubFactoryMonitor m2(*this, "MainLoop", currentLevel);
        DistanceLaplacianDropFunctor<local_ordinal_type, decltype(lapl_diag), decltype(distFunctor)>
            dropFunctor(lapl_diag, distFunctor, threshold);
        ScalarFunctor<scalar_type, local_ordinal_type, local_matrix_type, decltype(bndNodes), decltype(dropFunctor)>
            scalarFunctor(A, bndNodes, dropFunctor, rows, colsAux, valsAux, reuseGraph, lumping, threshold);

        Kokkos::parallel_reduce("main_loop", range_type(0,numRows), scalarFunctor, nnzFA);
      }
    }

  {
    // FIXME: time this section

    // parallel_scan (exclusive)
    Kokkos::parallel_scan("compress_rows", range_type(0,numRows+1),
      KOKKOS_LAMBDA(const local_ordinal_type i, local_ordinal_type& update, const bool& final_pass) {
        update += rows(i);
        if (final_pass)
          rows(i) = update;
    });
  }

  // Compress cols (and optionally vals)
  // We use a trick here: we moved all remaining elements to the beginning
  // of the original row in the main loop, so we don't need to check for
  // INVALID here, and just stop when achieving the new number of elements
  // per row.
  cols_type cols(Kokkos::ViewAllocateWithoutInitializing("FA_cols"), nnzFA);
  vals_type vals(Kokkos::ViewAllocateWithoutInitializing("FA_vals"), nnzFA);
  {
    // FIXME: time this section
    // SubFactoryMonitor m2(*this, "CompressColsAndVals", currentLevel);

    // Compress cols and vals

    Kokkos::parallel_for("compress_cols_and_vals", range_type(0,numRows),
      KOKKOS_LAMBDA(const local_ordinal_type i) {
        local_ordinal_type rowStart  = rows(i);
        local_ordinal_type rowAStart = rowsA(i);
        size_t rownnz = rows(i+1) - rows(i);
        for (size_t j = 0; j < rownnz; j++) {
          cols(rowStart+j) = colsAux(rowAStart+j);
          vals(rowStart+j) = colsAux(rowAStart+j);
        }
    });
  }

  kokkos_graph_type kokkosGraph(cols, rows);




  // Stage 0: detect Dirichlet rows
  boundary_nodes_type boundaryNodes("boundaryNodes", numRows);

  double tol = 0.0;
  Kokkos::parallel_for("detect_d_rows", range_type(0, numRows),
    KOKKOS_LAMBDA(const local_ordinal_type row) {
      auto rowView = A.row(row);
      auto length  = rowView.length;

      boundaryNodes(row) = true;
      for (decltype(length) colID = 0; colID < length; colID++)
        if ((rowView.colidx(colID) != row) && (ATS::magnitude(rowView.value(colID)) > tol)) {
          boundaryNodes(row) = false;
          break;
        }
  });
}

  template<class scalar_type, class local_ordinal_type, class device_type>
  void kernel_coalesce_drop_serial(KokkosSparse::CrsMatrix<scalar_type, local_ordinal_type, device_type> A) {
    typedef KokkosSparse::CrsMatrix<scalar_type, local_ordinal_type, device_type>   local_matrix_type;
    typedef Kokkos::ArithTraits<scalar_type>                                  ATS;
    typedef typename ATS::mag_type                                            magnitude_type;
    typedef Kokkos::View<bool*, device_type>                                  boundary_nodes_type;

    auto numRows = A.numRows();

    scalar_type eps = 0.05;

    typedef Kokkos::RangePolicy<typename local_matrix_type::execution_space> range_type;

    // Stage 0: detect Dirichlet rows
    boundary_nodes_type boundaryNodes("boundaryNodes", numRows);
    for (int row = 0; row < numRows; row++) {
      auto rowView = A.row (row);
      auto length  = rowView.length;

      boundaryNodes(row) = true;
      for (decltype(length) colID = 0; colID < length; colID++)
        if ((rowView.colidx(colID) != row) && (ATS::magnitude(rowView.value(colID)) > 1e-13)) {
          boundaryNodes(row) = false;
          break;
        }
    }
  }

  template<class scalar_type, class local_ordinal_type, class device_type>
      KokkosSparse::CrsMatrix<scalar_type, local_ordinal_type, device_type>
      kernel_construct(local_ordinal_type numRows) {
        auto numCols         = numRows;
        auto nnz             = 10*numRows;

        auto varianz_nel_row = 0.2*nnz/numRows;
        auto width_row       = 0.01*numRows;

        auto elements_per_row = nnz/numRows;

        auto rowPtr = new local_ordinal_type[numRows+1];
        rowPtr[0] = 0;
        for (int row = 0; row < numRows; row++) {
          int varianz = (1.0*rand()/INT_MAX-0.5)*varianz_nel_row;

          rowPtr[row+1] = rowPtr[row] + elements_per_row + varianz;
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
        int n    = 100000;
        int loop = 10;

        for (int i = 1; i < argc; i++) {
          if      (strcmp(argv[i], "-n") == 0)                                   { n    = atoi(argv[++i]); continue; }
          else if (strcmp(argv[i], "-l") == 0 || strcmp(argv[i], "--loop") == 0) { loop = atoi(argv[++i]); continue; }
          else if (strcmp(argv[i], "--node") == 0)                               { i++; continue; }
          else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            std::cout << "./coalesce_drop.exe [-n <matrix_size>] [-l <number_of_loops>]" << std::endl;
            return 0;
          } else {
            throw std::runtime_error(std::string("Uknown option: ") + argv[i]);
          }
        }

        typedef KokkosSparse::CrsMatrix<scalar_type, local_ordinal_type, device_type> local_matrix_type;
        typedef typename device_type::execution_space execution_space;

        local_matrix_type A = kernel_construct<scalar_type, local_ordinal_type, device_type>(n);

        execution_space::fence();
        Kokkos::Impl::Timer timer;

#ifdef KOKKOS_HAVE_SERIAL
        if (typeid(device_type) == typeid(Kokkos::Serial)) {
          for (int i = 0; i < loop; i++)
            kernel_coalesce_drop_serial(A);

        } else {
#else
          {
#endif
            for (int i = 0; i < loop; i++)
              kernel_coalesce_drop_device(A);
          }

          double kernel_time = timer.seconds();

          execution_space::fence();

          printf("kernel_coalesce_drop: %.2e (s)\n", kernel_time / loop);

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

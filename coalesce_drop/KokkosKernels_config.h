/* ---------------------------------------------
Makefile constructed configuration:
Tue Dec  5 13:12:25 MST 2017
----------------------------------------------*/
#ifndef KOKKOSKERNELS_CONFIG_H_
#define KOKKOSKERNELS_CONFIG_H_

/* ---------------------------------------------
ETI Scalar Types:
   ---------------------------------------------*/
#define KOKKOSKERNELS_INST_DOUBLE
#if defined(KOKKOSKERNELS_INST_COMPLEX_DOUBLE)
#define KOKKOSKERNELS_INST_KOKKOS_COMPLEX_DOUBLE_
#endif
#if defined(KOKKOSKERNELS_INST_COMPLEX_FLOAT)
#define KOKKOSKERNELS_INST_KOKKOS_COMPLEX_FLOAT_
#endif

/* ---------------------------------------------
ETI Ordinal Types:
   ---------------------------------------------*/
#define KOKKOSKERNELS_INST_ORDINAL_INT
#define KOKKOSKERNELS_INST_ORDINAL_INT64_T

/* ---------------------------------------------
ETI Offset Types:
   ---------------------------------------------*/
#define KOKKOSKERNELS_INST_OFFSET_INT
#define KOKKOSKERNELS_INST_OFFSET_SIZE_T

/* ---------------------------------------------
ETI Layout Types:
   ---------------------------------------------*/
#define KOKKOSKERNELS_INST_LAYOUTLEFT
#define KOKKOSKERNELS_INST_LAYOUTRIGHT

/* ---------------------------------------------
ETI ExecutionSpace Types:
   ---------------------------------------------*/
#define KOKKOSKERNELS_INST_EXECSPACE_OPENMP

/* ---------------------------------------------
ETI Memory Space Types:
   ---------------------------------------------*/
#define KOKKOSKERNELS_INST_MEMSPACE_HOSTSPACE

/* ---------------------------------------------
Third Party Libraries:
   ---------------------------------------------*/
#if defined(KOKKOSKERNELS_ENABLE_TPL_MKL)
#if !defined(KOKKOSKERNELS_ENABLE_TPL_BLAS)
#define KOKKOSKERNELS_ENABLE_TPL_BLAS
#endif
#endif

/* ---------------------------------------------
Optional Settings:
   ---------------------------------------------*/

#ifndef KOKKOSKERNELS_IMPL_COMPILE_LIBRARY
#define KOKKOSKERNELS_IMPL_COMPILE_LIBRARY false
#endif

#endif // KOKKOSKERNELS_CONFIG_H_

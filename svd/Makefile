KOKKOS_PATH = ${HOME}/code/kokkos/kokkos
KOKKOSKERNELS_PATH = ${HOME}/code/kokkos/kokkos-kernels
KOKKOS_DEVICES = "Cuda"
EXE_NAME = "svd"
USE_KOKKOSKERNELS = "yes"

SRC = $(wildcard *.cpp)

default: build
	echo "Start Build"


ifneq (,$(findstring Cuda,$(KOKKOS_DEVICES)))
CXX = ${KOKKOS_PATH}/bin/nvcc_wrapper
EXE = ${EXE_NAME}.cuda
KOKKOS_ARCH = "SNB,Pascal61"
KOKKOS_CUDA_OPTIONS = "enable_lambda"
else
CXX = g++
EXE = ${EXE_NAME}.host
KOKKOS_ARCH = "SNB"
endif

CXXFLAGS = -O3
LINK = ${CXX}
LINKFLAGS =

DEPFLAGS = -M

OBJ = $(SRC:.cpp=.o)
LIB =

build: $(EXE)

include $(KOKKOS_PATH)/Makefile.kokkos

ifeq (${USE_KOKKOSKERNELS}, "no")
# Without KokkosKernels
$(EXE): $(OBJ) $(KOKKOS_LINK_DEPENDS)
	$(LINK) $(KOKKOS_LDFLAGS) $(LINKFLAGS) $(EXTRA_PATH) $(OBJ) $(KOKKOS_LIBS) $(LIB) -o $(EXE)

clean: kokkos-clean
	rm -f *.o *.cuda *.host

# Compilation rules

%.o:%.cpp $(KOKKOS_CPP_DEPENDS)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(EXTRA_INC) -c $<
else
# With KokkosKernels
include $(KOKKOSKERNELS_PATH)/Makefile.kokkos-kernels
$(EXE): $(OBJ) $(KOKKOS_LINK_DEPENDS) $(KOKKOSKERNELS_LINK_DEPENDS)
	$(LINK) $(KOKKOS_LDFLAGS) $(LINKFLAGS) $(EXTRA_PATH) $(OBJ) $(KOKKOSKERNELS_LIBS) $(KOKKOS_LIBS) $(LIB) -o $(EXE)

clean: kokkos-clean kokkoskernels-clean
	rm -f *.o *.cuda *.host

# Compilation rules

%.o:%.cpp $(KOKKOS_CPP_DEPENDS) $(KOKKOSKERNELS_CPP_DEPENDS)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOSKERNELS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(KOKKOSKERNELS_CXXFLAGS) $(CXXFLAGS) $(EXTRA_INC) -c $<
endif



test: $(EXE)
	./$(EXE)

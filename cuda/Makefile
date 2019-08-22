# This Makefile expects the following env. vars to be set:
#
# F90       - How to invoke the Fortran compiler
# F90FLAGS  - Flags to pass to the Fortran compiler
# OMPFLAGS  - Flags for compiling with OpenMP
# AR        - Command to use when creating an archive (.a)

# Location of the dl_timer and infrastucture code
SHARED_DIR = ../shared
TIMER_DIR = ${SHARED_DIR}/dl_timer
TIMER_INC = ${TIMER_DIR}/src
TIMER_LIB = ${TIMER_DIR}/dl_timer_lib.a
INF_DIR = ${SHARED_DIR}/dl_esm_inf/finite_difference
INF_INC = ${INF_DIR}/src
INF_LIB = ${INF_DIR}/src/dl_esm_inf_fd.a
NVCC_FLAGS = -O3 -std=c++14 -use_fast_math -arch=$(CUDA_ARCH) --ptxas-options=-v -lineinfo
CUDA_OBJS = cuda_nemolite2d.o

EXECS = nemolite2d

# API lib is an archive that must come at the end of the list of objects
# passed to the linker
COMMON_MODULES = ${INF_LIB}

.PHONY: all nemolite2d cuda_nemolite2d timer_lib inf_lib

all: $(EXECS)

# Normal targets
nemolite2d: inf_lib timer_lib
	${MAKE} MODULE_LIST="$(CUDA_OBJS) nemolite2d.o ${COMMON_MODULES}" nemolite2d.exe

timer_lib:
	${MAKE} -C ${TIMER_DIR} sm_lib

inf_lib:
	${MAKE} -C ${INF_DIR}

nemolite2d.o: $(COMMON_MODULES)

cuda_nemolite2d.o: $(COMMON_MODULES)

# Generic rules

%.exe: $(MODULE_LIST)
	$(F90) $(OMPFLAGS) -o $@ $(MODULE_LIST) $(TIMER_LIB) $(LDFLAGS) -L$(CUDA_LIB) -lcudart -lstdc++

%.o: %.f90
	$(F90) $(F90FLAGS) -I${INF_INC} -I${TIMER_INC} -c $<

%.o: %.F90
	$(F90) $(F90FLAGS) -I${INF_INC} -I${TIMER_INC} -c $<

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c -I$(CUDA_INC) $<

# Cleaning up

clean: clean_timer clean_inf
	rm -f *.o *.mod *.MOD *~

clean_timer:
	${MAKE} -C ${TIMER_DIR} clean

clean_inf:
	${MAKE} -C ${INF_DIR} clean

allclean: clean allclean_timer
	rm -f *.exe
	rm -rf *_wpl_dir
	rm -f *.opt *.optrpt gnu_opt_report.txt

allclean_timer:
	${MAKE} -C ${TIMER_DIR} allclean

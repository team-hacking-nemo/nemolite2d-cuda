# Top-level Makefile for NEMOLite2D port to CUDA.
# Includes both original Fortan code and CUDA C port.
#
# Picks-up the compiler and compiler flags from environment
# variables. See e.g. compiler_setup/gnu.sh

.PHONY: original cuda openacc clean allclean

# No default

original:
	${MAKE} -C ./original

cuda: original
	${MAKE} -C ./cuda

openacc: original
	${MAKE} -C ./openacc

clean:
	${MAKE} -C ./original clean
	${MAKE} -C ./openacc clean
	${MAKE} -C ./cuda clean

allclean:
	${MAKE} -C ./original allclean
	${MAKE} -C ./openacc allclean
	${MAKE} -C ./cuda allclean

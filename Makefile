# Top-level Makefile for NEMOLite2D port to CUDA.
# Includes both original Fortan code and CUDA C port.
#
# Picks-up the compiler and compiler flags from environment
# variables. See e.g. compiler_setup/gnu.sh

.PHONY: all nemolite_cpu

all: nemolite_cpu

# All manual targets for CPU versions of NEMOLite2D
nemolite_cpu:
	${MAKE} -C ./original

clean:
	${MAKE} -C ./original clean

allclean:
	${MAKE} -C ./original allclean

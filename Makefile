# Top-level Makefile for PSycloneBench benchmarks.
# Only supports those benchmarks that target the CPU (i.e. excluding
# OpenACC, OpenCL and Maxeler.)
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

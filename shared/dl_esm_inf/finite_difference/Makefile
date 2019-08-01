API_LIB = dl_esm_inf_fd.a

all: fd_lib

fd_lib:
	make -C src API_LIB="${API_LIB}"

install: fd_lib lib include
	cp src/${API_LIB} ./lib/.
	cp src/*.[Mm][Oo][Dd] ./include/.

lib:
	mkdir lib

include:
	mkdir include

clean:
	make -C src clean

distclean: clean
	rm -f lib/${API_LIB}
	rm -f include/*.[Mm][Oo][Dd]

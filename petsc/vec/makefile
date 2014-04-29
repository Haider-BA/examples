ALL: main
CFLAGS	=
FFLAGS	=
CPPFLAGS=
FPPFLAGS=
CLEANFILES = ex1
include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

ex1: ex1.o chkopts
	${CLINKER} -o ex1 ex1.o ${PETSC_DM_LIB}
	${RM} -f ex1.o

runex1:
	mpiexec -n 4 ./ex1

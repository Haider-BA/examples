ALL: main
CFLAGS	=
FFLAGS	=
CPPFLAGS=
FPPFLAGS=
CLEANFILES = ex1 ex2 ex3 ex4 ex5 ex6a ex6b ex7 ex8 ex9 ex10 ex11a ex11b ex12 ex13 ex13a ex14
include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

all: ex1 ex2 ex3 ex4 ex5

ex1: ex1.o chkopts
	${CLINKER} -o ex1 ex1.o ${PETSC_DM_LIB}
	${RM} -f ex1.o

ex2: ex2.o chkopts
	${CLINKER} -o ex2 ex2.o ${PETSC_DM_LIB}
	${RM} -f ex2.o

ex3: ex3.o chkopts
	${CLINKER} -o ex3 ex3.o ${PETSC_DM_LIB}
	${RM} -f ex3.o

ex4: ex4.o chkopts
	${CLINKER} -o ex4 ex4.o ${PETSC_DM_LIB}
	${RM} -f ex4.o

ex5: ex5.o chkopts
	${CLINKER} -o ex5 ex5.o ${PETSC_DM_LIB}
	${RM} -f ex5.o

ex6a: ex6a.o chkopts
	${CLINKER} -o ex6a ex6a.o ${PETSC_DM_LIB}
	${RM} -f ex6a.o

ex6b: ex6b.o chkopts
	${CLINKER} -o ex6b ex6b.o ${PETSC_DM_LIB}
	${RM} -f ex6b.o

ex7: ex7.o chkopts
	${CLINKER} -o ex7 ex7.o ${PETSC_DM_LIB}
	${RM} -f ex7.o

ex8: ex8.o chkopts
	${CLINKER} -o ex8 ex8.o ${PETSC_DM_LIB}
	${RM} -f ex8.o

ex9: ex9.o chkopts
	${CLINKER} -o ex9 ex9.o ${PETSC_DM_LIB}
	${RM} -f ex9.o

ex10: ex10.o chkopts
	${CLINKER} -o ex10 ex10.o ${PETSC_DM_LIB}
	${RM} -f ex10.o

ex11a: ex11a.o chkopts
	${CLINKER} -o ex11a ex11a.o ${PETSC_DM_LIB}
	${RM} -f ex11a.o

ex11b: ex11b.o chkopts
	${CLINKER} -o ex11b ex11b.o ${PETSC_DM_LIB}
	${RM} -f ex11b.o

ex12: ex12.o chkopts
	${CLINKER} -o ex12 ex12.o ${PETSC_DM_LIB}
	${RM} -f ex12.o

ex13: ex13.o chkopts
	${CLINKER} -o ex13 ex13.o ${PETSC_DM_LIB}
	${RM} -f ex13.o

ex13a: ex13a.o chkopts
	${CLINKER} -o ex13a ex13a.o ${PETSC_DM_LIB}
	${RM} -f ex13a.o

ex14: ex14.o chkopts
	${CLINKER} -o ex14 ex14.o ${PETSC_DM_LIB}
	${RM} -f ex14.o

ex15: ex15.o chkopts
	${CLINKER} -o ex15 ex15.o ${PETSC_DM_LIB}
	${RM} -f ex15.o

runex1:
	${MPIEXEC} -n 4 ./ex1

runex2:
	${MPIEXEC} -n 4 ./ex2

runex3:
	${MPIEXEC} -n 4 ./ex3

runex4:
	${MPIEXEC} -n 4 ./ex4

runex5:
	${MPIEXEC} -n 4 ./ex5

runex6a:
	${MPIEXEC} -n 4 ./ex6a

runex6b:
	${MPIEXEC} -n 4 ./ex6b

runex7:
	${MPIEXEC} -n 4 ./ex7

runex8:
	${MPIEXEC} -n 4 ./ex8

runex9:
	${MPIEXEC} -n 4 ./ex9

runex10:
	${MPIEXEC} -n 1 ./ex10

runex11a:
	${MPIEXEC} -n 4 ./ex11a

runex11b:
	${MPIEXEC} -n 4 ./ex11b

runex12:
	${MPIEXEC} -n 4 ./ex12

runex13:
	${MPIEXEC} -n 4 ./ex13

runex13a:
	${MPIEXEC} -n 4 ./ex13a

runex14:
	${MPIEXEC} -n 4 ./ex14

runex15:
	${MPIEXEC} -n 4 ./ex15

CC = ${MPICC}
CFLAGS = -Wextra -Wall -pedantic -O3
RM = rm -rf
EXEC = ${MPIEXEC}
ALL = ex00 ex01 ex02 ex03 ex04

all: ${ALL}

ex00: 00-helloWorld.o
	${CC} ${CFLAGS} -o ex00  $^
	${RM} $^

runex00:
	${EXEC} -n 4 ./ex00

ex01: 01-stringExchange.o
	${CC} ${CFLAGS} -o ex01 $^
	${RM} $^

runex01:
	${EXEC} -n 2 ./ex01

ex02: 02-broadcast.o
	${CC} ${CFLAGS} -o ex02 $^
	${RM} $^

runex02:
	${EXEC} -n 4 ./ex02

ex03: 03-collectiveFun.o
	${CC} ${CFLAGS} -o ex03 $^
	${RM} $^

runex03:
	${EXEC} -n 4 ./ex03

ex04: 04-reduce.o
	${CC} ${CFLAGS} -o ex04 $^
	${RM} $^

runex04:
	${EXEC} -n 4 ./ex04

clean:
	rm -rf ${ALL}

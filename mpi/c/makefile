CC = ${MPICC}
CFLAGS = -Wextra -Wall -pedantic -O3
RM = rm -rf
EXEC = ${MPIEXEC}
ALL = ex00 ex01

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

clean:
	rm -rf ${ALL}

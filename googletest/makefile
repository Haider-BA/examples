ALL: bin/main bin/test1 bin/test2

SRC = ${wildcard src/*.cpp}
OBJ = ${SRC:.cpp=.o}
GTEST_DIR = external/gtest-1.7.0

bin/main: ${OBJ}
	${CXX} $^ -o $@

run:
	bin/main

clean:
	rm -rf ${OBJ}
	rm -rf lib/*
	rm -rf bin/*

vars:
	@echo SRC: ${SRC}
	@echo OBJ: ${OBJ}

lib/libgtest.a:
	${CXX} -I ${GTEST_DIR}/include -I${GTEST_DIR} -pthread \
	-c ${GTEST_DIR}/src/gtest-all.cc -o lib/gtest-all.o
	ar -rv lib/libgtest.a lib/gtest-all.o

bin/test1: tests/ex1.cpp lib/libgtest.a src/Vector3d.o
	${CXX} -I ${GTEST_DIR}/include -I src -pthread $^ -o bin/test1

bin/test2: tests/ex2.cpp lib/libgtest.a src/Vector3d.o
	${CXX} -I ${GTEST_DIR}/include -I src -pthread $^ -o bin/test2

runtests:
	bin/test1
	bin/test2

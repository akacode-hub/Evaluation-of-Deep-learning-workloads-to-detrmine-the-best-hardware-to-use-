#! /bin/bash
#

mkdir include
mkdir bincpp

g++ -c -Wall -Og -std=c++11 -I $PWD/include linpack_bench.cpp
if [ $? -ne 0 ]; then
  echo "Compile error."
  exit
fi
#
g++ linpack_bench.o -lm
if [ $? -ne 0 ]; then
  echo "Load error."
  exit
fi
#
rm linpack_bench.o
#
chmod ugo+x a.out
mv a.out $PWD/bincpp/linpack_benchg
#
echo "Normal end of execution."

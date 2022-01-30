#! /bin/bash
#

mkdir include
mkdir bincpp

g++ -c -Wall -O -std=c++11 -I $PWD/include sparse_test.cpp
if [ $? -ne 0 ]; then
  echo "Compile error."
  exit
fi
#
g++ sparse_test.o -lm
if [ $? -ne 0 ]; then
  echo "Load error."
  exit
fi
#
rm sparse_test.o
#
chmod ugo+x a.out
mv a.out $PWD/bincpp/sparse_test
#
echo "Normal end of execution."

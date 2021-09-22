#! /bin/bash
#

mkdir include
mkdir bincpp

g++ -c -Wall -Og -std=c++11 -I $PWD/include memory_test.cpp
if [ $? -ne 0 ]; then
  echo "Compile error."
  exit
fi
#
g++ memory_test.o -lm
if [ $? -ne 0 ]; then
  echo "Load error."
  exit
fi
#
rm memory_test.o
#
chmod ugo+x a.out
mv a.out $PWD/bincpp/memory_test_g
#
echo "Normal end of execution."

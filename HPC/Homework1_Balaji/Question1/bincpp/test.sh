## declare an array variable
declare -a arr=("0" "1" "2" "3" "s" "fast" "g")

## now loop through the above array

### Linpack
for i in "${arr[@]}"
do
   echo "LINPACK BENCH $i"
   ./linpack_bench$i
   # or do whatever with individual element of the array
done

## Memory test
for i in "${arr[@]}"
do
   echo "MEMORY TEST $i"
   ./memory_test_$i 25 25
   # or do whatever with individual element of the array
done

## Sparse test
for i in "${arr[@]}"
do
   echo "SPARSE TEST $i"
   ./sparse_test_$i 500 500 100 0.5
   # or do whatever with individual element of the array
done
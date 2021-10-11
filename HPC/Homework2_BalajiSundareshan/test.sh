for i in 1 2 4 8 16 32 64 128 256 512
do
   for j in 1 2 3 4 5
   do
	   ./soe_omp 500000 $i
   done
done

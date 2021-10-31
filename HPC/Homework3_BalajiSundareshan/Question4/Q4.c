#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <time.h>
#include <stdlib.h>
#include <unistd.h>

#define DIM 10
#define MAXNUM 1000

void gen_random_vector(int vec[DIM]);
void gen_random_matrix(int mat[DIM][DIM]);
void print_matrix(int mat[DIM][DIM]);
void print_vector(int vec[DIM]);
void matrix_vector_multiply_omp(int inp_mat[DIM][DIM], int inp_vec[DIM], int * out_vec);
void matrix_vector_multiply_serial(int inp_mat[DIM][DIM], int inp_vec[DIM], int * out_vec);
void compare_serial_omp(int vec1[DIM], int vec2[DIM]);

void gen_random_vector(int vec[DIM]) {

   int i;
  
   for ( i = 0; i < DIM; ++i) {
      vec[i] = rand()%MAXNUM;
   }

}

void gen_random_matrix(int mat[DIM][DIM])
{
    int i, j;

    for (i = 0; i < DIM; i++) {
        for (j = 0; j < DIM; j++) {
            mat[i][j] = rand()%MAXNUM;
        }
   }
}

void print_matrix(int mat[DIM][DIM]){

    int i,j;

    for(i=0;i<DIM;i++){
        for(j=0;j<DIM-1;j++){
            printf("%d, ",mat[i][j]);
        }
        printf("%d\n",mat[i][DIM-1]);
    }
}

void print_vector(int vec[DIM]){

    int i,j;

    for(i=0;i<DIM-1;i++){
        printf("%d, ",vec[i]);
    }
    printf("%d\n",vec[DIM-1]);
}

void matrix_vector_multiply_omp(int inp_mat[DIM][DIM], int inp_vec[DIM], int * out_vec){

    int thread_id;
    int i, j;

    #pragma omp parallel for schedule(static) private(i, j)    
    for(i=0;i<DIM;i++){
        thread_id = omp_get_thread_num();
        printf("Thread ID: %d works on row no: %d\n", thread_id, i);
        for(j=0;j<DIM;j++){
            inp_mat[i][j] = inp_mat[i][j]*inp_vec[j];
        }
    }

    int sum_ = 0;
    #pragma omp parallel for reduction(+:sum_) private(i, j)  
    for(i=0;i<DIM;i++){
        out_vec[i] = 0;
        sum_ = 0;
        for(j=0;j<DIM;j++){
            sum_ += inp_mat[i][j];
        }
        out_vec[i] = sum_;
    }
}

void matrix_vector_multiply_serial(int inp_mat[DIM][DIM], int inp_vec[DIM], int * out_vec){

    int i, j;
    for(i=0;i<DIM;i++){
        for(j=0;j<DIM;j++){
            inp_mat[i][j] = inp_mat[i][j]*inp_vec[j];
        }
    }

    for(i=0;i<DIM;i++){
        out_vec[i] = 0;
        for(j=0;j<DIM;j++){
            out_vec[i] += inp_mat[i][j];
        }
    }
}

void compare_serial_omp(int vec1[DIM], int vec2[DIM]){

    int diff = 0;
    int i;

    for(i=0;i<DIM;i++){
        diff += abs(vec1[i] - vec2[i]);
    }

    if(diff==0){
        printf("\nTest PASSED !!!!\n");
    }else{
        printf("\nTest FAILED !!!!\n");
    }
}

void matrixcopy(void * dest_matrix, void * src_matrix) 
{
  memcpy(dest_matrix, src_matrix, DIM*DIM*sizeof(int));
}


int main (int argc, char *argv[])
{

    srand( (unsigned)time( NULL ) );

    int inp_vector[DIM];
    int inp_matrix_serial[DIM][DIM];
    int inp_matrix_omp[DIM][DIM];
    int out_vector_omp[DIM];
    int out_vector_serial[DIM];
    struct timespec start, end;

    // generate inputs
    gen_random_vector(inp_vector);
    gen_random_matrix(inp_matrix_serial);
    matrixcopy(inp_matrix_omp, inp_matrix_serial);

    // print inputs
    printf("Input Vector: \n");
    print_vector(inp_vector);
    printf("\nInput Matrix: \n");
    print_matrix(inp_matrix_serial);

    // multiply serial and omp
    clock_gettime(CLOCK_MONOTONIC, &start);
    matrix_vector_multiply_omp(inp_matrix_omp, inp_vector, out_vector_omp);
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_taken_omp = (end.tv_sec - start.tv_sec);
    time_taken_omp += (end.tv_nsec - start.tv_nsec) / 1000000000.0;

    clock_gettime(CLOCK_MONOTONIC, &start);
    matrix_vector_multiply_serial(inp_matrix_serial, inp_vector, out_vector_serial);
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_taken_serial = (end.tv_sec - start.tv_sec);
    time_taken_serial += (end.tv_nsec - start.tv_nsec) / 1000000000.0;

    // print outputs
    printf("\nOutput Vector OPENMP: \n");
    print_vector(out_vector_omp);

    printf("\nOutput Vector SERIAL: \n");
    print_vector(out_vector_serial);

    // compare results
    compare_serial_omp(out_vector_omp, out_vector_serial);

    printf("\nTime taken for matrix-vector multiplication OPENMP: %f\n",time_taken_omp);
    printf("Time taken for matrix-vector multiplication SERIAL: %f\n",time_taken_serial);

    return 0;
}

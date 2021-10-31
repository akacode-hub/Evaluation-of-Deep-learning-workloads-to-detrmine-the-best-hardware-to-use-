#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <assert.h>

double calc_factorial(int num);
double calc_taylor_series_dfp(double x, int num_terms);
float calc_taylor_series_fp(float x, int num_terms);
double calc_taylor_series_dfp_vec(double x, int num_terms);
float calc_taylor_series_fp_vec(float x, int num_terms);

double calc_factorial(int num)
{   
    double val = 1.0;
    int i;
    for(i=2; i<=num; i++){
        val = val*i;
    }

    return val;

}

double calc_taylor_series_dfp(double x, int num_terms)
{

    double sum = 1.0;
    int i;
    for(i=1; i<num_terms; i++){
        sum += pow(x, i)/calc_factorial(i);
    }

    return sum;
}

float calc_taylor_series_fp(float x, int num_terms)
{

    float sum = 1.0;
    int i;
    for(i=1; i<num_terms; i++){
        sum += pow(x, i)/calc_factorial(i);
    }

    return sum;
}

double calc_taylor_series_dfp_vec(double x, int num_terms)
{

    double sum = 1.0;
    int i;
    for(i=1; i<num_terms; i++){
        sum += pow(x, i)/calc_factorial(i);
    }

    return sum;
}

float calc_taylor_series_fp_vec(float x, int num_terms)
{

    float sum = 1.0;
    int i;
    for(i=1; i<num_terms; i++){
        sum += pow(x, i)/calc_factorial(i);
    }

    return sum;
}

int main(int argc, char *argv[]){

    struct timespec start, end;
    srand(time(NULL));
    assert(("./Q1 <input number> <number of terms>", argc == 3));

    double x_dfp = atoi(argv[1]);
    int num_terms = atoi(argv[2]);

    // compute taylor series double
    clock_gettime(CLOCK_MONOTONIC, &start);

    double val_dfp = 1.0/x_dfp + calc_taylor_series_dfp(x_dfp, num_terms);

    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_taken = (end.tv_sec - start.tv_sec);
    time_taken += (end.tv_nsec - start.tv_nsec) / 1000000000.0;

    printf("value of f(x) where x, fx are double: %f\n", val_dfp);
    printf("Time taken: %f\n", time_taken);

    // compute taylor series float
    clock_gettime(CLOCK_MONOTONIC, &start);

    float x_fp = (float) x_dfp;
    float val_fp = 1.0/x_fp + calc_taylor_series_fp(x_fp, num_terms);

    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken = (end.tv_sec - start.tv_sec);
    time_taken += (end.tv_nsec - start.tv_nsec) / 1000000000.0;

    printf("value of f(x) where x, fx are float: %f\n", val_fp);
    printf("Time taken: %f\n", time_taken);

    return 0;

}
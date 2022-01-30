#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <assert.h>

double calc_factorial(int num);
void calc_factorial_terms(int num);
double calc_taylor_series_dfp(double x, int num_terms);
double calc_taylor_series_dfp_vec(double x, int num_terms);
void test_dp(double x_dfp, int num_terms);
void test_dp_vec(double x_dfp, int num_terms);

float calc_taylor_series_fp(float x, int num_terms);
float calc_taylor_series_fp_vec(float x, int num_terms);
void test_fp(float x_fp, int num_terms);
void test_fp_vec(float x_fp, int num_terms);

double * facts;
double * pows_dp;
float * pows_fp;

double calc_factorial(int num)
{   
    double val = 1.0;
    int i;
    for(i=2; i<=num; i++){
        val = val*i;
    }

    return val;
}

void calc_factorial_terms(int num)
{   
    facts = (double *)calloc(num+1, sizeof(double));
    int i;

    for(i=0; i<=num; i++){
        facts[i] = calc_factorial(i);
    }

}

void calc_pow_terms_dp(double x, int num)
{   
    pows_dp = (double *)calloc(num+1, sizeof(double));
    int i;

    for(i=0; i<=num; i++){
        pows_dp[i] = pow(x, i);
    }

}

void calc_pow_terms_fp(float x, int num)
{   
    pows_fp = (float *)calloc(num+1, sizeof(float));
    int i;

    for(i=0; i<=num; i++){
        pows_fp[i] = pow(x, i);
    }

}

double calc_taylor_series_dfp(double x, int num_terms)
{

    double sum = 0.0;
    int i;
    for(i=0; i<num_terms; i++){
        sum += pows_dp[i]/facts[i];
    }

    return sum;
}

double calc_taylor_series_dfp_vec(double x, int num_terms)
{

    int i;
    double taylor_terms[num_terms];
    for(i=0; i<num_terms; i++){
        taylor_terms[i] = pows_dp[i]/facts[i];
    }

    double sum = 0.0;
    for(i=0; i<num_terms; i++){
        sum += taylor_terms[i];
    }

    return sum;
}

float calc_taylor_series_fp(float x, int num_terms)
{

    float sum = 0.0;
    int i;
    for(i=0; i<num_terms; i++){
        sum += pows_fp[i]/facts[i];
    }

    return sum;
}

float calc_taylor_series_fp_vec(float x, int num_terms)
{

    int i;
    float taylor_terms[num_terms];
    for(i=0; i<num_terms; i++){
        taylor_terms[i] = pows_fp[i]/facts[i];
    }

    float sum = 0.0;
    for(i=0; i<num_terms; i++){
        sum += taylor_terms[i];
    }

    return sum;
}

void test_fp(float x_fp, int num_terms){

    struct timespec start, end;

    clock_gettime(CLOCK_MONOTONIC, &start);

    float val_fp = 1.0/x_fp + calc_taylor_series_fp(x_fp, num_terms);

    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_taken = (end.tv_sec - start.tv_sec);
    time_taken += (end.tv_nsec - start.tv_nsec) / 1000000000.0;

    printf("value of f(x) where x, fx are float: %f\n", val_fp);
    printf("Time taken: %f\n", time_taken);

}

void test_fp_vec(float x_fp, int num_terms){

    struct timespec start, end;

    clock_gettime(CLOCK_MONOTONIC, &start);

    float val_fp = 1.0/x_fp + calc_taylor_series_fp_vec(x_fp, num_terms);

    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_taken = (end.tv_sec - start.tv_sec);
    time_taken += (end.tv_nsec - start.tv_nsec) / 1000000000.0;

    printf("value of f(x) where x, fx are float vector: %f\n", val_fp);
    printf("Time taken: %f\n", time_taken);
    
}

void test_dp(double x_dfp, int num_terms){

    struct timespec start, end;

    clock_gettime(CLOCK_MONOTONIC, &start);

    double val_dfp = 1.0/x_dfp + calc_taylor_series_dfp(x_dfp, num_terms);

    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_taken = (end.tv_sec - start.tv_sec);
    time_taken += (end.tv_nsec - start.tv_nsec) / 1000000000.0;

    printf("value of f(x) where x, fx are double: %f\n", val_dfp);
    printf("Time taken: %f\n", time_taken);

}

void test_dp_vec(double x_dfp, int num_terms){

    struct timespec start, end;

    clock_gettime(CLOCK_MONOTONIC, &start);

    double val_dfp = 1.0/x_dfp + calc_taylor_series_dfp_vec(x_dfp, num_terms);

    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_taken = (end.tv_sec - start.tv_sec);
    time_taken += (end.tv_nsec - start.tv_nsec) / 1000000000.0;

    printf("value of f(x) where x, fx are double vector: %f\n", val_dfp);
    printf("Time taken: %f\n", time_taken);
    
}

int main(int argc, char *argv[]){

    
    srand(time(NULL));
    assert(("./Q1 <input number> <number of terms>", argc == 3));

    double x_dfp = atoi(argv[1]);
    int num_terms = atoi(argv[2]);

    printf("Input: %f\n", x_dfp);
    printf("Num_terms: %f\n", num_terms);

    calc_factorial_terms(num_terms);
    calc_pow_terms_dp(x_dfp, num_terms);
    
    // compute taylor series double
    test_dp(x_dfp, num_terms);
    test_dp_vec(x_dfp, num_terms);

    // compute taylor series float
    float x_fp = (float) x_dfp;
    calc_pow_terms_fp(x_fp, num_terms);
    test_fp(x_fp, num_terms);
    test_fp_vec(x_fp, num_terms);
    

    return 0;

}
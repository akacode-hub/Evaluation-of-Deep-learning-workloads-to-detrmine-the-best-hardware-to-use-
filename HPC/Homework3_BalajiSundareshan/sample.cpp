#include <iostream>
using namespace std;
#define VECTOR_SIZE 100000

int main() {
    float* vect1 = (float *) malloc(VECTOR_SIZE * sizeof(float));
    float* vect2 = (float *) malloc(VECTOR_SIZE * sizeof(float));
    float* vect3 = (float *) malloc(VECTOR_SIZE * sizeof(float));

    // Initialization
    for (size_t i=0; i < VECTOR_SIZE; i++) {
      vect1[i] = 5.0;
      vect2[i] = 2.0;
      vect3[i] = 0.0;
    }

    struct timespec start, end;

    clock_gettime(CLOCK_MONOTONIC, &start);

    for (size_t i=0; i < VECTOR_SIZE; i++) {
      vect3[i] = vect1[i] + vect2[i];
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_taken = (end.tv_sec - start.tv_sec);
    time_taken += (end.tv_nsec - start.tv_nsec) / 1000000000.0;
    cout << "time_taken: " << time_taken << endl; 
    cout << "Vect 3: " << vect3[VECTOR_SIZE - 1] << endl;
    return 0;
}
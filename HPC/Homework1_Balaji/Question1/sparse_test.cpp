# include <cstdlib>
# include <iostream>
# include <cmath>
# include <sys/time.h>
# include <cassert>

using namespace std;

int main (int argc, char *argv[]);
void multiply_matrix(int matrix_dim, float frac_sparse);
double ** sparse_matrix(int n, float frac_sparse);

int main (int argc, char *argv[])
{
  
  assert(("Two arguments (min matrix dimension, max matrix dimension, step and frac sparsity) must be provided !!", argc == 5));
  
  int min_matrix_dim = atoi(argv[1]);
  int max_matrix_dim = atoi(argv[2]);
  int step = atoi(argv[3]);
  float frac_sparse = stof(argv[4]);
  
  assert(("sparsity fraction should be less than/equal to 1", frac_sparse <= 1));
  assert(("number of steps should be greater than 1", step >= 1));

  cout << "Min Matrix dimension: "<<min_matrix_dim<<" * "<<min_matrix_dim<<"\n";
  cout << "Max Matrix dimension: "<<max_matrix_dim<<" * "<<max_matrix_dim<<"\n";
  cout << "Increment Step: "<<step<<"\n";
  cout << "Sparsity fraction: "<<frac_sparse<<"\n";

  int matrix_dim;
  for (matrix_dim=min_matrix_dim; matrix_dim<=max_matrix_dim; matrix_dim+=step)
  {
    multiply_matrix(matrix_dim, frac_sparse);
  }
  
  return 0;

}

double ** sparse_matrix(int n, float frac_sparse)

{   
    double** arr = 0;
    arr = new double*[n];
    double prob, aval;
    int i, j;

    for ( i = 0; i < n; i++ )
    {
        arr[i] = new double[n];
        for ( j = 0; j < n; j++ )
        {
        prob = ( double ) random ( ) / ( double ) RAND_MAX;
        aval = ( double ) random ( ) / ( double ) RAND_MAX;
        if (prob < frac_sparse){
            aval = 0.0;
        }
        arr[i][j] = aval;
        }
    }

    return arr;
}

void multiply_matrix(int matrix_dim, float frac_sparse)
{ 

  double** a;
  double** b;
  double c[matrix_dim][matrix_dim];

  int i, j, k;
  struct timeval  start, end;
  double time_elapsed;

  a = sparse_matrix(matrix_dim, frac_sparse);
  b = sparse_matrix(matrix_dim, frac_sparse);

  gettimeofday(&start, NULL);
  for ( i = 0; i < matrix_dim; i++ )
  {
    for ( j = 0; j < matrix_dim; j++ )
    {
      c[i][j] = 0.0;
      for ( k = 0; k < matrix_dim; k++ )
      {
        c[i][j] = c[i][j] + a[i][k] * b[k][j];
      }
    }
  }
  gettimeofday(&end, NULL);
  time_elapsed = ((double) ((double) (end.tv_usec - start.tv_usec) / 1000000 +
                    (double) (end.tv_sec - start.tv_sec)));
  cout<<" Matrix dim: "<<matrix_dim<<" Time elapsed: "<<time_elapsed<<"\n";

  delete [] a;
  delete [] b;

  return;

}
    
# include <cstdlib>
# include <iostream>
# include <cmath>
# include <ctime>
# include <cassert>

using namespace std;

int main (int argc, char *argv[]);
double ** sparse_matrix(int n, float frac_sparse);
float cpu_time( );

int main (int argc, char *argv[])
{
  
  assert(("Two arguments (matrix dimension and frac sparsity) must be provided !!", argc == 3));
  
  int matrix_dim = atoi(argv[1]);
  float frac_sparse = stof(argv[2]);

  assert(("sparsity fraction should be less than/equal to 1", frac_sparse <= 1));

  double** a;
  double** b;
  double c[matrix_dim][matrix_dim];
  int i, j, k;
  float time_elapsed, cpu_time_start, cpu_time_end;

  a = sparse_matrix(matrix_dim, frac_sparse);
  b = sparse_matrix(matrix_dim, frac_sparse);

  cpu_time_start = cpu_time( );

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
  
  cpu_time_end = cpu_time( );
  time_elapsed = cpu_time_end - cpu_time_start;
  cout << "Time elapsed for matrix dim "<<matrix_dim<<" with sparsity fraction "<<frac_sparse<<" = "<<time_elapsed<<"\n";
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

float cpu_time( )
{
  float value;

  value = ( float ) clock ( ) / ( float ) CLOCKS_PER_SEC;

  return value;
}
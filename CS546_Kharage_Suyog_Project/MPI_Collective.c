#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <mpi.h>
	
	typedef struct {float r; float i;} complex;
	static complex ctmp;
	#define C_SWAP(a,b) {ctmp=(a);(a)=(b);(b)=ctmp;}
	
	
	/* Size of matrix (NxN) */
	const int N = 512;
	
	
	int p, rank;
	#define ROOT 0
	
	int main (int argc, char **argv) {
	
	   MPI_Init(&argc, &argv);
	   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	   MPI_Comm_size(MPI_COMM_WORLD, &p);
	
	   const char* filename1 = argc == 3 ? argv[1] : "im1";
	   const char* filename2 = argc == 3 ? argv[2] : "im2";
	
	   if ( rank==0) 
		printf("\nMPI Collective operations\n");
	   if ( rank==0)
		 printf("For %d processors\n",p);
	   if ( rank==0)
		 printf("Using input files %s and %s\n\n",filename1, filename2);
	
	   int read_input_matrix ( const char* filename, complex matrix[N][N] );
	   int write_matrix ( const char* filename, complex matrix[N][N] );
	   void c_fft1d(complex *r, int n, int isign);
	 
	
	   int chunk = N / p; 
	   complex A[N][N], B[N][N], C[N][N];
	   int i, j;
	   complex tmp;
	   double time_init, time_end, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10;
	   MPI_Status status;
	
	
	   read_input_matrix (filename1, A);
	   read_input_matrix (filename2, B);
	
	   
	   if ( rank == ROOT )
	      time_init = MPI_Wtime();
	   if(rank==0){	
	   MPI_Scatter( &A[0][0], chunk*N, MPI_COMPLEX, MPI_IN_PLACE, chunk*N, MPI_COMPLEX, ROOT, MPI_COMM_WORLD );
	   MPI_Scatter( &B[0][0], chunk*N, MPI_COMPLEX, MPI_IN_PLACE, chunk*N, MPI_COMPLEX, ROOT, MPI_COMM_WORLD );
		}
	else{
           MPI_Scatter( &A[0][0], chunk*N, MPI_COMPLEX, &A[chunk*rank][0], chunk*N, MPI_COMPLEX, ROOT, MPI_COMM_WORLD );
	   MPI_Scatter( &B[0][0], chunk*N, MPI_COMPLEX, &B[chunk*rank][0], chunk*N, MPI_COMPLEX, ROOT, MPI_COMM_WORLD );
	}

	   if ( rank == ROOT ) t1 = MPI_Wtime();

	   for (i= chunk*rank ;i< chunk*(rank+1);i++) {
	         c_fft1d(A[i], N, -1);
	         c_fft1d(B[i], N, -1);
	   }
	   if ( rank == ROOT ) t2 = MPI_Wtime();
	   
	   if(rank==0){ 	
	   MPI_Gather( MPI_IN_PLACE, chunk*N, MPI_COMPLEX, &A[0][0], chunk*N, MPI_COMPLEX, ROOT, MPI_COMM_WORLD );
	   MPI_Gather( MPI_IN_PLACE, chunk*N, MPI_COMPLEX, &B[0][0], chunk*N, MPI_COMPLEX, ROOT, MPI_COMM_WORLD );
		}
	else{
	   MPI_Gather( &A[chunk*rank][0], chunk*N, MPI_COMPLEX, &A[0][0], chunk*N, MPI_COMPLEX, ROOT, MPI_COMM_WORLD );
	   MPI_Gather( &B[chunk*rank][0], chunk*N, MPI_COMPLEX, &B[0][0], chunk*N, MPI_COMPLEX, ROOT, MPI_COMM_WORLD );
	}

	   if ( rank == ROOT ) t3 = MPI_Wtime();
	
	   if ( rank == ROOT ) {
	      for (i=0;i<N;i++) {
	         for (j=i;j<N;j++) {
	            tmp = A[i][j];
	            A[i][j] = A[j][i];
	            A[j][i] = tmp;
	
	            tmp = B[i][j];
	            B[i][j] = B[j][i];
	            B[j][i] = tmp;
	         }
	      }
	      t4 = MPI_Wtime();
	   }
	   
	   if(rank==0){	
   	   MPI_Scatter( &A[0][0], chunk*N, MPI_COMPLEX, MPI_IN_PLACE, chunk*N, MPI_COMPLEX, ROOT, MPI_COMM_WORLD );
	   MPI_Scatter( &B[0][0], chunk*N, MPI_COMPLEX, MPI_IN_PLACE, chunk*N, MPI_COMPLEX, ROOT, MPI_COMM_WORLD );
		}
	   else{
		MPI_Scatter( &A[0][0], chunk*N, MPI_COMPLEX, &A[chunk*rank][0], chunk*N, MPI_COMPLEX, ROOT, MPI_COMM_WORLD );
		   MPI_Scatter( &B[0][0], chunk*N, MPI_COMPLEX, &B[chunk*rank][0], chunk*N, MPI_COMPLEX, ROOT, MPI_COMM_WORLD );
	}
	   if ( rank == ROOT ) t5 = MPI_Wtime();
	
	
	   for (i= chunk*rank ;i< chunk*(rank+1);i++) {
	         c_fft1d(A[i], N, -1);
	         c_fft1d(B[i], N, -1);
	   }
	
	
	   for (i= chunk*rank ;i< chunk*(rank+1);i++) {
	      for (j=0;j<N;j++) {
	         C[i][j].r = A[i][j].r*B[i][j].r - A[i][j].i*B[i][j].i;
	         C[i][j].i = A[i][j].r*B[i][j].i + A[i][j].i*B[i][j].r;
	      }
	   }
	   for (i= chunk*rank ;i< chunk*(rank+1);i++) {
	      c_fft1d(C[i], N, 1);
	   }
	   if ( rank == ROOT ) t6 = MPI_Wtime();
	
	if(rank==0)
	   MPI_Gather( MPI_IN_PLACE, chunk*N, MPI_COMPLEX, &C[0][0], chunk*N, MPI_COMPLEX, ROOT, MPI_COMM_WORLD );
	else
		MPI_Gather( &C[chunk*rank][0], chunk*N, MPI_COMPLEX, &C[0][0], chunk*N, MPI_COMPLEX, ROOT, MPI_COMM_WORLD );
	
	   if ( rank == ROOT ) t7 = MPI_Wtime();
	
	   if ( rank == ROOT ) {
	      for (i=0;i<N;i++) {
	         for (j=i;j<N;j++) {
	            tmp = C[i][j];
	            C[i][j] = C[j][i];
	            C[j][i] = tmp;
	         }
	      }
	      t8 = MPI_Wtime();
	   }

		IF(RANK==0)	
	   MPI_Scatter( &C[0][0], chunk*N, MPI_COMPLEX, MPI_IN_PLACE, chunk*N, MPI_COMPLEX, ROOT, MPI_COMM_WORLD );
		else
	   MPI_Scatter( &C[0][0], chunk*N, MPI_COMPLEX, &C[chunk*rank][0], chunk*N, MPI_COMPLEX, ROOT, MPI_COMM_WORLD );

	
	   if ( rank == ROOT ) t9 = MPI_Wtime();
	   for (i= chunk*rank ;i< chunk*(rank+1);i++) {
	      c_fft1d(C[i], N, 1);
	   }
	   if ( rank == ROOT ) t10 = MPI_Wtime();
	
	IF(rank==0)
	   MPI_Gather( MPI_IN_PLACE, chunk*N, MPI_COMPLEX, &C[0][0], chunk*N, MPI_COMPLEX, ROOT, MPI_COMM_WORLD );
	else		
	MPI_Gather( &C[chunk*rank][0], chunk*N, MPI_COMPLEX, &C[0][0], chunk*N, MPI_COMPLEX, ROOT, MPI_COMM_WORLD );
	   
	if ( rank == ROOT )
	      time_end = MPI_Wtime();
	
	
	   write_matrix("output_matrix", C);
	
	   if ( rank==0) {
	
	      double tcomputation = (t2-t1) + (t4-t3) + (t6-t5) + (t8-t7) + (t10-t9);
	      double tcommunication = (t1-time_init) + (t3-t2) + (t5-t4) + (t7-t6) + (t9-t8) + (time_end-t10);
	
	      printf("Total time spent: %f ms\n", (time_end-time_init) * 1000 );
	      printf("Time for computation:  %f ms\n", tcomputation * 1000 );
	      printf("Time for communication: %f ms\n", tcommunication * 1000 );
	   }
	
	   MPI_Finalize();
	}
	
	
	
	int write_matrix ( const char* filename, complex matrix[N][N] ) {
	   if ( rank == ROOT ) {
	      int i, j;
	      FILE *fp = fopen(filename,"w");
	
	      for (i=0;i<N;i++) {
	         for (j=0;j<N;j++)
	            fprintf(fp,"   %e",matrix[i][j].r);
	         fprintf(fp,"\n");
	      };
	
	      fclose(fp);
	   }
	}
	int read_input_matrix ( const char* filename, complex matrix[N][N] ) {
	   if ( rank == ROOT ) {
	      int i, j;
	      FILE *fp = fopen(filename,"r");
	
	      if ( !fp ) {
	         printf("This file is not exist: %s\n", filename);
	         exit(1);
	      }
	
	      for (i=0;i<N;i++)
	         for (j=0;j<N;j++) {
	            fscanf(fp,"%g",&matrix[i][j].r);
	            matrix[i][j].i = 0;
	         }
	      fclose(fp);
	   }
	}
	
	
	
	/*
	 ------------------------------------------------------------------------
	 FFT1D            c_fft1d(r,i,-1)
	 Inverse FFT1D    c_fft1d(r,i,+1)
	 ------------------------------------------------------------------------
	*/
	/* ---------- FFT 1D
	   This computes an in-place complex-to-complex FFT
	   r is the real and imaginary arrays of n=2^m points.
	   isign = -1 gives forward transform
	   isign =  1 gives inverse transform
	*/
	
	void c_fft1d(complex *r, int      n, int      isign)
	{
	   int     m,i,i1,j,k,i2,l,l1,l2;
	   float   c1,c2,z;
	   complex t, u;
	
	   if (isign == 0) return;
	
	   /* Do the bit reversal */
	   i2 = n >> 1;
	   j = 0;
	   for (i=0;i<n-1;i++) {
	      if (i < j)
	         C_SWAP(r[i], r[j]);
	      k = i2;
	      while (k <= j) {
	         j -= k;
	         k >>= 1;
	      }
	      j += k;
	   }
	
	   /* m = (int) log2((double)n); */
	   for (i=n,m=0; i>1; m++,i/=2);
	
	   /* Compute the FFT */
	   c1 = -1.0;
	   c2 =  0.0;
	   l2 =  1;
	   for (l=0;l<m;l++) {
	      l1   = l2;
	      l2 <<= 1;
	      u.r = 1.0;
	      u.i = 0.0;
	      for (j=0;j<l1;j++) {
	         for (i=j;i<n;i+=l2) {
	            i1 = i + l1;
	
	            /* t = u * r[i1] */
	            t.r = u.r * r[i1].r - u.i * r[i1].i;
	            t.i = u.r * r[i1].i + u.i * r[i1].r;
	
	            /* r[i1] = r[i] - t */
	            r[i1].r = r[i].r - t.r;
	            r[i1].i = r[i].i - t.i;
	
	            /* r[i] = r[i] + t */
	            r[i].r += t.r;
	            r[i].i += t.i;
	         }
	         z =  u.r * c1 - u.i * c2;
	
	         u.i = u.r * c2 + u.i * c1;
	         u.r = z;
	      }
	      c2 = sqrt((1.0 - c1) / 2.0);
	      if (isign == -1) /* FWD FFT */
	         c2 = -c2;
	      c1 = sqrt((1.0 + c1) / 2.0);
	   }
	
	   /* Scaling for inverse transform */
	   if (isign == 1) {       /* IFFT*/
	      for (i=0;i<n;i++) {
	         r[i].r /= n;
	         r[i].i /= n;
	      }
	   }
	}




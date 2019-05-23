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
	
	   if ( rank==0) printf("\nMPI Data and Task Parallel\n");
	   if ( rank==0) printf("For %d processors\n",p);
	   if ( rank==0) printf("using input files %s and %s\n\n",filename1, filename2);
	
	   
	   int read_input_matrix ( const char* filename, complex matrix[N][N] );
	   int write_matrix ( const char* filename, complex matrix[N][N] );
	   void c_fft1d(complex *r, int n, int isign);
	  	
	
	   int chunk = N / p; 
	   complex A[N][N], B[N][N], C[N][N];
	   int i, j;
	   complex tmp;
	   double time_init, time_end, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14;
	   MPI_Status status;
	
	
	   /* Read files */
	   read_input_matrix (filename1, A);
	   read_input_matrix (filename2, B);
	
	   	
	   /* Initial time */
	   if ( rank == ROOT )
	      time_init = MPI_Wtime();
	
	
	   /* Dividing the processors in 4 groups */ 
	   
	   int sizeOfGroup = p / 4;
	   int rankGroup;
	   int P1_array[sizeOfGroup], P2_array[sizeOfGroup], P3_array[sizeOfGroup], P4_array[sizeOfGroup];
	
	   for(i=0; i<p; i++) {
	      int groupOfProcessor = i / sizeOfGroup;
	      switch(groupOfProcessor){
	      case 0:
	         P1_array[ i%sizeOfGroup ] = i;
	         break;
	      case 1:
	         P2_array[ i%sizeOfGroup ] = i;
	         break;
	      case 2:
	         P3_array[ i%sizeOfGroup ] = i;
	         break;
	      case 3:
	         P4_array[ i%sizeOfGroup ] = i;
	         break;
	      }
	   }
	   
	   MPI_Group world_group, P1, P2, P3, P4; 
	   MPI_Comm P1_comm, P2_comm, P3_comm, P4_comm;
	
	   MPI_Comm_group(MPI_COMM_WORLD, &world_group); 
	
	   /* Create the four groups */
	   int group = rank / sizeOfGroup;
	
	   if ( group == 0 )      { 
	      
	      MPI_Group_incl(world_group, p/4, P1_array, &P1);
	      MPI_Comm_create( MPI_COMM_WORLD, P1, &P1_comm);
	      MPI_Group_rank(P1, &rankGroup);
	   } 
	   else if ( group == 1 ) { 
	
	      MPI_Group_incl(world_group, p/4, P2_array, &P2); 
	      MPI_Comm_create( MPI_COMM_WORLD, P2, &P2_comm);
	      MPI_Group_rank(P2, &rankGroup);
	   } 
	   else if ( group == 2 ) { 
	      MPI_Group_incl(world_group, p/4, P3_array, &P3); 
	      MPI_Comm_create( MPI_COMM_WORLD, P3, &P3_comm);
	      MPI_Group_rank(P3, &rankGroup);
	   } 
	   else if ( group == 3 ) { 
	      MPI_Group_incl(world_group, p/4, P4_array, &P4); 
	      MPI_Comm_create( MPI_COMM_WORLD, P4, &P4_comm);
	      MPI_Group_rank(P4, &rankGroup);
	   } 
	
	   
	   /* Scatter A and B to the other processes. We supose N is divisible by p */
	
	   chunk = N / sizeOfGroup;
	
	   if ( rank == ROOT ){
	
	      for ( i=0; i<sizeOfGroup; i++ ) {
	         if ( P1_array[i]==ROOT ) continue;
	         MPI_Send( &A[chunk*i][0], chunk*N, MPI_COMPLEX, P1_array[i], 0, MPI_COMM_WORLD );
	      }
	      for ( i=0; i<sizeOfGroup; i++ ) {
	         if ( P2_array[i]==ROOT ) continue;
	         MPI_Send( &B[chunk*i][0], chunk*N, MPI_COMPLEX, P2_array[i], 0, MPI_COMM_WORLD );
	      }
	   }
	   else {
	
	      if ( group == 0 )
	         MPI_Recv( &A[chunk*rankGroup][0], chunk*N, MPI_COMPLEX, ROOT, 0, MPI_COMM_WORLD, &status );
	      if ( group == 1 )
	         MPI_Recv( &B[chunk*rankGroup][0], chunk*N, MPI_COMPLEX, ROOT, 0, MPI_COMM_WORLD, &status );
	   }
	
	   MPI_Barrier(MPI_COMM_WORLD); 
	   if ( rank == ROOT ) t1 = MPI_Wtime();
	
	   /* Apply 1D FFT in all rows of A, in group P1 */
	
	   if ( group == 0 )
	      for ( i=chunk*rankGroup; i<chunk*(rankGroup+1); i++ )
	         c_fft1d(A[i], N, -1);
	   /* Apply 1D FFT in all rows of B, in group P2 */
	   
	   if ( group == 1 )
	      for ( i=chunk*rankGroup; i<chunk*(rankGroup+1); i++ )
	         c_fft1d(B[i], N, -1);
	
	   MPI_Barrier(MPI_COMM_WORLD); 
	   if ( rank == ROOT ) t2 = MPI_Wtime();
	
		   /* Gather the row FFTs from A into the first processor of P1 for sequential trasposition */
	
	   if ( group == 0 ) {
	      if ( rankGroup == 0 ) {
	         for ( i=1; i<sizeOfGroup; i++ ) {
	            MPI_Recv( &A[chunk*i][0], chunk*N, MPI_COMPLEX, i, 0, P1_comm, &status );
	         }
	         
	      }
	      else 
	         MPI_Send( &A[chunk*rankGroup][0], chunk*N, MPI_COMPLEX, 0, 0, P1_comm );
	   }
	
	   /* Gather the row FFTs from B into the first processor of P2 for sequential trasposition */
	
	   if ( group == 1 ) {
	      if ( rankGroup == 0 ) {
	         for ( i=1; i<sizeOfGroup; i++ ) {
	            MPI_Recv( &B[chunk*i][0], chunk*N, MPI_COMPLEX, i, 0, P2_comm, &status );
	         }
	         
	      }
	      else 
	         MPI_Send( &B[chunk*rankGroup][0], chunk*N, MPI_COMPLEX, 0, 0, P2_comm );
	   }
	   MPI_Barrier(MPI_COMM_WORLD); 
	   if ( rank == ROOT ) t3 = MPI_Wtime();
	
	   /* Traspose matrix A in P1's main process */
	
	   if ( group == 0 && rankGroup == 0 ) {
	      for (i=0;i<N;i++) {
	         for (j=i;j<N;j++) {
	            tmp = A[i][j];
	            A[i][j] = A[j][i];
	            A[j][i] = tmp;
	         }
	      }
	   }
	
	   /* Traspose matrix B in P2's main process */
	   
	   if ( group == 1 && rankGroup == 0 ) {
	      for (i=0;i<N;i++) {
	         for (j=i;j<N;j++) {
	            tmp = B[i][j];
	            B[i][j] = B[j][i];
	            B[j][i] = tmp;
	         }
	      }
	   }
	   MPI_Barrier(MPI_COMM_WORLD); 
	   if ( rank == ROOT ) t4 = MPI_Wtime();
	
	   /* Scatter the transposed A in the group P1 */
	
	   if ( group == 0 ) {
	      if ( rankGroup == 0 ) {
	         for ( i=1; i<sizeOfGroup; i++ ) {
	            MPI_Send( &A[chunk*i][0], chunk*N, MPI_COMPLEX, i, 0, P1_comm );
	         }
	         
	      }
	      else 
	         MPI_Recv( &A[chunk*rankGroup][0], chunk*N, MPI_COMPLEX, 0, 0, P1_comm, &status );
	   }
	
	   /* Scatter the transposed B in the group P2 */
	
	   if ( group == 1 ) {
	      if ( rankGroup == 0 ) {
	         for ( i=1; i<sizeOfGroup; i++ ) {
	            MPI_Send( &B[chunk*i][0], chunk*N, MPI_COMPLEX, i, 0, P2_comm );
	         }
	         
	      }
	      else 
	         MPI_Recv( &B[chunk*rankGroup][0], chunk*N, MPI_COMPLEX, 0, 0, P2_comm, &status );
	   }
	   MPI_Barrier(MPI_COMM_WORLD); 
	   if ( rank == ROOT ) t5 = MPI_Wtime();
	
	   /* Apply 1D FFT in all rows of A, in group P1. This are actually columns of the original A */
	
	   if ( group == 0 )
	      for ( i=chunk*rankGroup; i<chunk*(rankGroup+1); i++ )
	         c_fft1d(A[i], N, -1);
	
	   /* Apply 1D FFT in all rows of B, in group P2. This are actually columns of the original B */
	
	   if ( group == 1 )
	      for ( i=chunk*rankGroup; i<chunk*(rankGroup+1); i++ )
	         c_fft1d(B[i], N, -1);
	
	
	   MPI_Barrier(MPI_COMM_WORLD); 
	   if ( rank == ROOT ) t6 = MPI_Wtime();
	
	
		   /* Gather A and B into the P3 processors */
	
	   if ( group == 0 )
	      MPI_Send ( &A[chunk*rankGroup][0], chunk*N, MPI_COMPLEX, P3_array[rankGroup], 0, MPI_COMM_WORLD );
	   else if ( group == 1 )
	      MPI_Send ( &B[chunk*rankGroup][0], chunk*N, MPI_COMPLEX, P3_array[rankGroup], 0, MPI_COMM_WORLD );
	
	   else if ( group == 2 ) {
	      MPI_Recv( &A[chunk*rankGroup][0], chunk*N, MPI_COMPLEX, P1_array[rankGroup], 0, MPI_COMM_WORLD, &status );
	      MPI_Recv( &B[chunk*rankGroup][0], chunk*N, MPI_COMPLEX, P2_array[rankGroup], 0, MPI_COMM_WORLD, &status );
	   }
	
	   MPI_Barrier(MPI_COMM_WORLD); 
	   if ( rank == ROOT ) t7 = MPI_Wtime();
	
	
	   /* Point to point multiplication */
	
	   if ( group == 2 ) {
	      for (i= chunk*rankGroup ;i< chunk*(rankGroup+1);i++) {
	         for (j=0;j<N;j++) {
	            C[i][j].r = A[i][j].r*B[i][j].r - A[i][j].i*B[i][j].i;
	            C[i][j].i = A[i][j].r*B[i][j].i + A[i][j].i*B[i][j].r;
	         }
	      }
	   }
	
		   MPI_Barrier(MPI_COMM_WORLD); 
	   if ( rank == ROOT ) t8 = MPI_Wtime();
	
	
	   /* Send the result, which is among the processes of P3, to P4 */
	
	   if ( group == 2 ) {
	      MPI_Send ( &C[chunk*rankGroup][0], chunk*N, MPI_COMPLEX, P4_array[rankGroup], 0, MPI_COMM_WORLD );
	   }
	   else if ( group == 3 ) {
	      MPI_Recv( &C[chunk*rankGroup][0], chunk*N, MPI_COMPLEX, P3_array[rankGroup], 0, MPI_COMM_WORLD, &status );
	   }
		   MPI_Barrier(MPI_COMM_WORLD); // for time meassuring
	   if ( rank == ROOT ) t9 = MPI_Wtime();
	
	   /* Inverse 1D FFT in all rows of C, made by P3. Each processor in P3 will do a part */
	
	   if ( group == 3 )
	      for ( i=chunk*rankGroup; i<chunk*(rankGroup+1); i++ )
	         c_fft1d(C[i], N, 1);
	
	   MPI_Barrier(MPI_COMM_WORLD); // for time meassuring
	   if ( rank == ROOT ) t10 = MPI_Wtime();

	   /* Gather the row FFTs from A into the first processor of P1 for sequential trasposition */
	
	   if ( group == 3 ) {
	      if ( rankGroup == 0 ) {
	         for ( i=1; i<sizeOfGroup; i++ ) {
	            MPI_Recv( &C[chunk*i][0], chunk*N, MPI_COMPLEX, i, 0, P4_comm, &status );
	         }
	         
	      }
	      else 
	         MPI_Send( &C[chunk*rankGroup][0], chunk*N, MPI_COMPLEX, 0, 0, P4_comm );
	   }
	   MPI_Barrier(MPI_COMM_WORLD); 
	   if ( rank == ROOT ) t11 = MPI_Wtime();

	   /* Transpose C sequentially */
	   if ( group == 3 && rankGroup == 0 ) {
	      for (i=0;i<N;i++) {
	         for (j=i;j<N;j++) {
	            tmp = C[i][j];
	            C[i][j] = C[j][i];
	            C[j][i] = tmp;
	         }
	      }
	      
	   }
	   MPI_Barrier(MPI_COMM_WORLD); 
	   if ( rank == ROOT ) t12 = MPI_Wtime();
	   
	   /* Scatter the transposed C in the group P3 */
	
	   if ( group == 3 ) {
	      if ( rankGroup == 0 ) {
	         for ( i=1; i<sizeOfGroup; i++ ) {
	            MPI_Send( &C[chunk*i][0], chunk*N, MPI_COMPLEX, i, 0, P4_comm );
	         }
	         
	      }
	      else 
	         MPI_Recv( &C[chunk*rankGroup][0], chunk*N, MPI_COMPLEX, 0, 0, P4_comm, &status );
	   }
	   MPI_Barrier(MPI_COMM_WORLD); 
	   if ( rank == ROOT ) t13 = MPI_Wtime();

		   /* Inverse 1D FFT in all rows of C, made by P3. Each processor in P3 will do a part */
	
	   if ( group == 3 )
	      for ( i=chunk*rankGroup; i<chunk*(rankGroup+1); i++ )
	         c_fft1d(C[i], N, 1);
	
	
	   MPI_Barrier(MPI_COMM_WORLD); 
	   if ( rank == ROOT ) t14 = MPI_Wtime();
	
	   /* Gather the fragments of C to the ROOT process, which will be in charge of computing time and writing file */
	
	   if ( rank == ROOT ){
	
	      for ( i=0; i<sizeOfGroup; i++ ) {
	         if ( P4_array[i]==ROOT ) continue; /* ROOT process doesn't receive from itself */
	
	         MPI_Recv( &C[chunk*i][0], chunk*N, MPI_COMPLEX, P4_array[i], 0, MPI_COMM_WORLD, &status );
	      }
	   }
	   else if ( group == 3 )
	      MPI_Send( &C[chunk*rankGroup][0], chunk*N, MPI_COMPLEX, ROOT, 0, MPI_COMM_WORLD );
	
	
	   MPI_Barrier(MPI_COMM_WORLD); 
	
	   /* Final time */
	   if ( rank == ROOT )
	      time_end = MPI_Wtime();
	
	
	   /* Write output file */
	   write_matrix("output_matrix", C);
	
	   if ( rank==0) {
	
	      double tcomputation = (t2-t1) + (t4-t3) + (t6-t5) + (t8-t7) + (t10-t9) + (t12-t11) + (t14-t13);
	      double tcommunication = (t1-time_init) + (t3-t2) + (t5-t4) + (t7-t6) + (t9-t8) + (t11-t10) + (t13-t12) + (time_end-t14);
	
	      printf("Total time spent: %f ms\n", (time_end-time_init) * 1000 );
	      printf("Time for computation:  %f ms\n", tcomputation * 1000 );
	      printf("Time for communication:  %f ms\n", tcommunication * 1000 );
	   }
	
	   MPI_Finalize();
	}
	
	
	/* Write the real part of the result matrix */
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
	
	/* Reads the matrix from tha file and inserts the values in the real part */
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


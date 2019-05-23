/*Prefix sum calculation using MPI  */

//Header files
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<mpi.h>
#include<time.h>



// functions for calculation of local sum and buffer size of each processor

long int calculate_local_sum(int* elements, int buffer_start, int buffer_size) {
    long int sum = 0;
    int i;
    for (i = buffer_start; i < (buffer_start+buffer_size); i++) {
        sum += elements[i];
    }
    return sum;
}

int calculate_buffer_size(int rank, int total_processors, int total_elements) {

    int buffer_size;

    buffer_size = total_elements / total_processors;

    if (rank == total_processors -1 && total_elements % total_processors !=0) {
        buffer_size = buffer_size + (total_elements % total_processors);
    }
    return buffer_size;
}

//global variable
int * randomArray;


//main function
int main(int argc, char *argv[])
{
        int rank,total_elements,i,total_processors,buffer_size=0,buffer_start;
        long int total_sum=0,local_buffer_sum=0;
        double startClock,stopClock;

        //receiving command line arguments
        total_processors=atoi(argv[1]);
        total_elements=atoi(argv[2]);

        // array declaration for storing all elements and for local elements
        long * chunk = (long *) malloc(sizeof(long));
        randomArray = (int *) malloc(total_elements * sizeof(int));

	// Generation of random numbers and storing in array
         for (i = 0; i < total_elements; i++)
        {
                randomArray[i] = rand()%1000 +1;
        }

        // Check for processor 0 for displaying input elements
        if(rank == 0)
        {
                printf("\nOriginal Input:\n ");
                for(i = 0; i < total_elements; i++)
                    printf("%d\t",randomArray[i]);
        }

        MPI_Status status;

        // MPI_Init function for initialization
         MPI_Init(&argc,&argv);

        // record start time
        startClock = MPI_Wtime();

        MPI_Comm_size(MPI_COMM_WORLD,&total_processors);
        MPI_Comm_rank(MPI_COMM_WORLD,&rank);

        // barrier function for synchronization
	MPI_Barrier(MPI_COMM_WORLD);

        //calculation of start index of elements assigned to each processor
        buffer_start = ((total_elements / total_processors) * rank);

        // calcualtion of buffer size
        buffer_size = calculate_buffer_size(rank, total_processors, total_elements);

        //calculation of local sum
        local_buffer_sum = calculate_local_sum(randomArray, buffer_start, buffer_size);


        // receiving local sum from other processors to processor 0 if rank is 0 
        if (rank == 0)
         {
                total_sum = local_buffer_sum;
               printf("\nlocal Sum of processor %d: %ld",rank,local_buffer_sum);
                 for(i = 1; i < total_processors; ++i)
                {
                        MPI_Recv(chunk, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
                        total_sum =total_sum +  *chunk;
                }
        }
        //otherwise send local sum to processor 0
        else {
                MPI_Send(&local_buffer_sum, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
                printf("\nlocal Sum of processor %d: %ld",rank,local_buffer_sum);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        // record end time
        stopClock=MPI_Wtime();

        //if rank is 0, display prefi sum and calculate total time
        if (rank == 0) {

                printf("\nPrefix Sum: %ld",total_sum);
                        printf("\nTime Elapsed: %lf milliseconds\n", (double)(stopClock - startClock) * 1000);
        }
        MPI_Finalize();

}




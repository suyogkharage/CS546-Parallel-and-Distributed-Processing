#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define BLOCK_SIZE 16

__global__ void getRowCount(int size, int *matrix)
{
         //calculated the Id the for row
         int row = blockDim.x * blockIdx.x + threadIdx.x + 1;

              if (row <= size)
              {
                    matrix[(size+2)*(size+1)+row] = matrix[(size+2)+row];
                    matrix[row] = matrix[(size+2)*(size) + row];
              }
}

__global__ void getColCount(int size, int *matrix)
{
         // calculated the Id for column 
         int col = blockDim.x * blockIdx.x + threadIdx.x;

              if(col <= size+1)
              {
                    matrix[col*(size+2)+size+1] = matrix[col*(size)+1];
                    matrix[col*(size+2)] = matrix[col*(size+2) + (size)];
              }
}

// device function for calculation of game of life problem
__global__ void GameOfLife(int size, int *matrix, int *newMatrix)
{
    int neighborsCount;
    int col = blockDim.y * blockIdx.y + threadIdx.y + 1;
    int row = blockDim.x * blockIdx.x + threadIdx.x + 1;
    int id = col * (size+2) + row;


    if (col <= size && row <= size)
    {

        neighborsCount = matrix[id+(size+2)] + matrix[id-(size+2)] + matrix[id+1] + matrix[id-1] + matrix[id+(size+3)] + matrix[id-(size+3)]
                         + matrix[id-(size+1)] + matrix[id+(size+1)];

        int element = matrix[id];
        if (element == 1 && neighborsCount < 2)
                newMatrix[id] = 0;

        else if (element == 1 && (neighborsCount == 2 || neighborsCount == 3))
                newMatrix[id] = 1;

        else if (element == 1 && neighborsCount > 3)
                newMatrix[id] = 0;

        else if (element == 0 && neighborsCount == 3)
                newMatrix[id] = 1;

        else
            newMatrix[id] = element;
    }
        __syncthreads();                // this function is used for synchronise threads within a block

}

//display time elapsed
void displayTime(timeval tStart, timeval tEnd, int i)
{
        gettimeofday(&tEnd, NULL);
        double t = ((tEnd.tv_sec - tStart.tv_sec) * 1000.0) + ((tEnd.tv_usec - tStart.tv_usec) / 1000.0);
        printf("Time for %d iterations: %f milliseconds\n",i, t);

}

// display matrix after 10,100 and 1000 iterations
void displayMatrix(int* h_matrix, int colCount)
{
        int i,j;
        for(i=0;i<10;i++)
        {
                for(j=0;j<10;j++)
                {
                        printf("%d ",h_matrix[i*colCount+j]);
                }
                printf("\n");
        }

}

// Here in main function, host process is carried out i.e. assign oiginal matrix and send it to device function and get result back from device.
int main(int argc, char* argv[])
{
    int i,j;
    int* h_matrix;
    int* d_matrix;
    int* d_newmatrix;
    int* d_tempmatrix;

    int size = 15;
    int iterations = 1020;

    struct timeval tStart;
    struct timeval tEnd;
    //size_t unit_size = sizeof(int)*(size+2)*(size+2);
    h_matrix = (int*)malloc(sizeof(int)*(size+2)*(size+2));

    cudaMalloc(&d_matrix, sizeof(int)*(size+2)*(size+2));
    cudaMalloc(&d_newmatrix, sizeof(int)*(size+2)*(size+2));

    for(i = 1; i<=size; i++)
    {
        for(j = 1; j<=size; j++)
        {
            h_matrix[i*(size+2)+j] = rand() % 2;
        }
    }

    printf("Initial Matrix\n");

    for(i=0;i<size;i++)
    {
         for(j=0;j<size;j++)
         {
                printf("%d ",h_matrix[i*size+j]);
         }
         printf("\n");
    }

    cudaMemcpy(d_matrix, h_matrix, sizeof(int)*(size+2)*(size+2), cudaMemcpyHostToDevice);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE,1);
    int  grid = (int)ceil(size/(float)BLOCK_SIZE);
    dim3 gridSize(grid,grid,1);

    dim3 BlockSizeCopy(BLOCK_SIZE,1,1);
    dim3 GridRows((int)ceil(size/(float)BlockSizeCopy.x),1,1);
    dim3 GridCols((int)ceil((size+2)/(float)BlockSizeCopy.x),1,1);


        gettimeofday(&tStart, NULL);
        for (i = 0; i<iterations; i++) {

                getRowCount<<<GridRows, BlockSizeCopy>>>(size, d_matrix);

                getColCount<<<GridCols, BlockSizeCopy>>>(size, d_matrix);

                GameOfLife<<<gridSize, blockSize>>>(size, d_matrix, d_newmatrix);

                d_tempmatrix = d_matrix;
                d_matrix = d_newmatrix;
                d_newmatrix = d_tempmatrix;

                if(i==10)
                {
                        cudaMemcpy(h_matrix, d_matrix, sizeof(int)*(size+2)*(size+2), cudaMemcpyDeviceToHost);
                        printf("\nMatrix at 10 iteration:\n");
                        displayMatrix(h_matrix, size);
                        displayTime(tStart,tEnd,i);
                }
                if(i==100)
                {
                        cudaMemcpy(h_matrix, d_matrix, sizeof(int)*(size+2)*(size+2), cudaMemcpyDeviceToHost);
                        printf("\nMatrix at 100 iteration:\n");
                        displayMatrix(h_matrix, size);
                        displayTime(tStart,tEnd,i);
                }
                if(i==1000)
                {
                        cudaMemcpy(h_matrix, d_matrix, sizeof(int)*(size+2)*(size+2), cudaMemcpyDeviceToHost);
                        printf("\nMatrix at 1000 iteration:\n");
                        displayMatrix(h_matrix, size);
                        displayTime(tStart,tEnd,i);
                }

        }

    cudaMemcpy(h_matrix, d_matrix, sizeof(int)*(size+2)*(size+2), cudaMemcpyDeviceToHost);
    cudaFree(d_matrix);
    cudaFree(d_newmatrix);
    free(h_matrix);
    return 0;
}



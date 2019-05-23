CS546 HW3
Steps to compile and run the program
 
Oper the terminal and do as follow
1- ssh username@login.xsede.org
2- eneter password
3- press 1
4- approve from your smartphone
5- access the comet using 'gsissh comet' command  
6- Check MPI version using command 'module list'
7- if you dont have mvapich2_ib/2.1 run the following command - $ module load mvapich2_ib/2.1

To create and save the file 
1- type 'vim'
2- press i for inserting the code
3- type your c program
4- press escape and then :wq (filename) and press enter

Check the file using command 'll' whether it is successfully created or not

To compile the file
1- $ mpicc -o PrefixSumFinal PrefixSumFinal.c

Creation of job submition file 
1- type 'vim'
2- press i for inserting the code
3- tyep following code-  
#!/bin/bash
#SBATCH --job-name="PrefixSumFinal"
#SBATCH --output="PrefixSumFinal.%j.%N.out"
#SBATCH --par>>on=compute
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --export=ALL
#SBATCH -t 00:10:00

ibrun -np 8 ./PrefixSumFinal

4- press escape and then :wq mpi_job.sh and press enter

If you have to run another program, then change the job-name and output file name in the job submission file.

To submit the job
1- $ sbatch mpi_job.sh

To check the status
1- $ squeue -u (username)

to open the output file 
1- vim (output filename)

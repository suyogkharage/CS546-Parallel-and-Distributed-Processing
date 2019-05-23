CS546 HW5
Steps to compile and run the program
 
Oper the terminal and do as follow
1- ssh username@login.xsede.org
2- eneter password
3- press 1
4- approve from your smartphone
5- access the comet using 'gsissh comet' command  
6- Indtall CUDA on account using 'module load cuda' command
7- check module list using command 'module list'

To create and save the file 
1- type 'vim'
2- press i for inserting the code
3- type your cuda program
4- press escape and then :wq (filename.cu) and press enter

Check the file using command 'll' whether it is successfully created or not

To compile the file
1- $ nvcc -o GameOfLife GameOfLife.cu

Creation of job submition file 
1- type 'vim'
2- press i for inserting the code
3- tyep following code-  

#!/bin/bash
#SBATCH --job-name="GameOfLife"
#SBATCH --output="GameOfLife.%j.%N.out"
#SBATCH -p gpu
#SBATCH --gres=gpu:k80:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --export=ALL
#SBATCH -t 00:10:00

./GameOfLife

4- press escape and then :wq cuda_job.sh and press enter

If you have to run another program, then change the job-name and output file name in the job submission file.

To submit the job
1- $ sbatch cuda_job.sh

To check the status
1- $ squeue -u (username)

to open the output file 
1- vim (output filename)

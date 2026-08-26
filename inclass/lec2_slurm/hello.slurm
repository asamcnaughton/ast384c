#!/bin/bash
#SBATCH -J hello_test          # job name
#SBATCH -o hello_test.%j.out   # stdout -> hello_test.<jobid>.out
#SBATCH -e hello_test.%j.err   # stderr -> hello_test.<jobid>.err
#SBATCH -N 1                   # 1 node
#SBATCH -n 1                   # 1 task
#SBATCH -p development         # short/debug queue: fast turnaround, low walltime limit
#SBATCH -t 00:05:00            # 5 minute walltime

module load python3/3.11
conda activate ast384c

python hello.py

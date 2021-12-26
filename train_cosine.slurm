#!/bin/bash
#SBATCH --job-name="pair-wise"   # Sensible name for the job
#SBATCH --account=ie-idi   # Account for consumed resources
#SBATCH --nodes=8           # Allocate 1 nodes for the job
#SBATCH -c 8                 # Number of cores (can vary)
#SBATCH --gres=gpu:V10032
#SBATCH --time=99-00:00:00    # Upper time limit for the job (DD-HH:MM:SS)
#SBATCH --partition=GPUQ
#SBATCH --mail-user=ying.xu@ntnu.no
#SBATCH --ntasks-per-node=1     # 1 task per node
#SBATCH --nice=2147483645
#SBATCH --priority=TOP
#SBATCH --no-requeue
module purge
cd ${SLURM_SUBMIT_DIR}
WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}

#module load intel/2020b
module load Python/3.8.6-GCCcore-10.2.0
source venv/bin/activate

echo "EfficientnetB7 3+1 DF+F2F+NT train fc first cosine distance"
python -u train.py --epochs 100 --n_classes 3 -b 3 --n_samples 9 --n_classes_total 4 --model_name effb7 --dist_type cosine --train_list data_list/DF_F2F_NT_train_000599_frame30.txt --val_list data_list/DF_F2F_NT_val_600799_frame30.txt --model_output_path model_save/effb7/${SLURM_JOB_ID} --tensorboard_path tensorboard/${SLURM_JOB_ID} | tee output/${SLURM_JOB_ID}.txt

#!/bin/bash
#SBATCH --job-name="pair-wise"   # Sensible name for the job
#SBATCH --account=ie-idi   # Account for consumed resources
#SBATCH --nodes=1           # Allocate 1 nodes for the job
#SBATCH -c 8                 # Number of cores (can vary)
#SBATCH --time=99-00:00:00    # Upper time limit for the job (DD-HH:MM:SS)
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:V10032:2
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

#echo "test 3-2-10 check resnet101 pool_out and distancematrix size"
#python -u train.py --epochs 100 --n_classes 3 -b 2 --n_samples 10 --n_classes_total 5 --train_list data_list/4+0_train.txt --val_list data_list/4+0_val.txt --model_output_path model_save/res101/${SLURM_JOB_ID} --model_name ${SLURM_JOB_ID} --tensorboard_path tensorboard/${SLURM_JOB_ID} | tee output/${SLURM_JOB_ID}.txt

#echo "check pool_out shape resnet101"
#python -u train.py --epochs 100 --n_classes 3 -b 2 --n_samples 10 --n_classes_total 5 --train_list data_list/4+0_train.txt --val_list data_list/4+0_val.txt --model_output_path model_save/res101/${SLURM_JOB_ID} --model_name ${SLURM_JOB_ID} --tensorboard_path tensorboard/${SLURM_JOB_ID} | tee output/${SLURM_JOB_ID}.txt

#echo "fix2500895 mistatke the total number of class to 5, now change to 4, 3+1 DF+F2F+FS train fc first"
#python -u train.py --epochs 100 --n_classes 3 -b 3 --n_samples 3 --n_classes_total 4 --train_list data_list/DF_F2F_FS_train_000599_frame30.txt --val_list data_list/DF_F2F_FS_val_600799_frame30.txt --model_output_path model_save/res101/${SLURM_JOB_ID} --model_name ${SLURM_JOB_ID} --tensorboard_path tensorboard/${SLURM_JOB_ID} | tee output/${SLURM_JOB_ID}.txt
#echo "fix2500896 mistatke the total number of class to 5, now change to 4, 3+1 DF+F2F+NT train fc first"
#python -u train.py --epochs 100 --n_classes 3 -b 3 --n_samples 3 --n_classes_total 4 --train_list data_list/DF_F2F_NT_train_000599_frame30.txt --val_list data_list/DF_F2F_NT_val_600799_frame30.txt --model_output_path model_save/res101/${SLURM_JOB_ID} --model_name ${SLURM_JOB_ID} --tensorboard_path tensorboard/${SLURM_JOB_ID} | tee output/${SLURM_JOB_ID}.txt
#echo "fix2500897 mistatke the total number of class to 5, now change to 4, 3+1 DF+FS+NT train fc first"
#python -u train.py --epochs 100 --n_classes 3 -b 3 --n_samples 3 --n_classes_total 4 --train_list data_list/DF_FS_NT_train_000599_frame30.txt --val_list data_list/DF_FS_NT_val_600799_frame30.txt --model_output_path model_save/res101/${SLURM_JOB_ID} --model_name ${SLURM_JOB_ID} --tensorboard_path tensorboard/${SLURM_JOB_ID} | tee output/${SLURM_JOB_ID}.txt
#echo "fix2500898 mistatke the total number of class to 5, now change to 4, 3+1 F2F+FS+NT train fc first"
#python -u train.py --epochs 100 --n_classes 3 -b 3 --n_samples 3 --n_classes_total 4 --train_list data_list/F2F_FS_NT_train_000599_frame30.txt --val_list data_list/F2F_FS_NT_val_600799_frame30.txt --model_output_path model_save/res101/${SLURM_JOB_ID} --model_name ${SLURM_JOB_ID} --tensorboard_path tensorboard/${SLURM_JOB_ID} | tee output/${SLURM_JOB_ID}.txt

#echo "EfficientnetB7 3+1 DF+F2F+NT train fc first"
#python -u train.py --epochs 100 --n_classes 3 -b 3 --n_samples 9 --n_classes_total 4 --model_name effb7 --train_list data_list/DF_F2F_NT_train_000599_frame30.txt --val_list data_list/DF_F2F_NT_val_600799_frame30.txt --model_output_path model_save/effb7/${SLURM_JOB_ID} --tensorboard_path tensorboard/${SLURM_JOB_ID} | tee output/${SLURM_JOB_ID}.txt
#echo "EfficientnetB0 3+1 DF+F2F+NT train fc first"
#python -u train.py --epochs 100 --n_classes 3 -b 3 --n_samples 9 --n_classes_total 4 --model_name effb0 --train_list data_list/DF_F2F_NT_train_000599_frame30.txt --val_list data_list/DF_F2F_NT_val_600799_frame30.txt --model_output_path model_save/effb0/${SLURM_JOB_ID} --tensorboard_path tensorboard/${SLURM_JOB_ID} | tee output/${SLURM_JOB_ID}.txt
#echo "EfficientnetB1 3+1 DF+F2F+NT train fc first"
#python -u train.py --epochs 100 --n_classes 3 -b 3 --n_samples 9 --n_classes_total 4 --model_name effb1 --train_list data_list/DF_F2F_NT_train_000599_frame30.txt --val_list data_list/DF_F2F_NT_val_600799_frame30.txt --model_output_path model_save/effb1/${SLURM_JOB_ID} --tensorboard_path tensorboard/${SLURM_JOB_ID} | tee output/${SLURM_JOB_ID}.txt
#echo "EfficientnetB2 3+1 DF+F2F+NT train fc first"
#python -u train.py --epochs 100 --n_classes 3 -b 3 --n_samples 9 --n_classes_total 4 --model_name effb2 --train_list data_list/DF_F2F_NT_train_000599_frame30.txt --val_list data_list/DF_F2F_NT_val_600799_frame30.txt --model_output_path model_save/effb2/${SLURM_JOB_ID} --tensorboard_path tensorboard/${SLURM_JOB_ID} | tee output/${SLURM_JOB_ID}.txt
#echo "EfficientnetB3 3+1 DF+F2F+NT train fc first"
#python -u train.py --epochs 100 --n_classes 3 -b 3 --n_samples 9 --n_classes_total 4 --model_name effb3 --train_list data_list/DF_F2F_NT_train_000599_frame30.txt --val_list data_list/DF_F2F_NT_val_600799_frame30.txt --model_output_path model_save/effb3/${SLURM_JOB_ID} --tensorboard_path tensorboard/${SLURM_JOB_ID} | tee output/${SLURM_JOB_ID}.txt
#echo "EfficientnetB4 3+1 DF+F2F+NT train fc first"
#python -u train.py --epochs 100 --n_classes 3 -b 3 --n_samples 9 --n_classes_total 4 --model_name effb4 --train_list data_list/DF_F2F_NT_train_000599_frame30.txt --val_list data_list/DF_F2F_NT_val_600799_frame30.txt --model_output_path model_save/effb4/${SLURM_JOB_ID} --tensorboard_path tensorboard/${SLURM_JOB_ID} | tee output/${SLURM_JOB_ID}.txt
#echo "EfficientnetB5 3+1 DF+F2F+NT train fc first"
#python -u train.py --epochs 100 --n_classes 3 -b 3 --n_samples 9 --n_classes_total 4 --model_name effb5 --train_list data_list/DF_F2F_NT_train_000599_frame30.txt --val_list data_list/DF_F2F_NT_val_600799_frame30.txt --model_output_path model_save/effb5/${SLURM_JOB_ID} --tensorboard_path tensorboard/${SLURM_JOB_ID} | tee output/${SLURM_JOB_ID}.txt


#echo "EfficientnetB0 3+1 DF+F2F+NT train fc first, initial weight=0"
#python -u train.py --weight_init_zero --epochs 100 --n_classes 3 -b 3 --n_samples 9 --n_classes_total 4 --model_name effb0 --train_list data_list/DF_F2F_NT_train_000599_frame30.txt --val_list data_list/DF_F2F_NT_val_600799_frame30.txt --model_output_path model_save/effb0/${SLURM_JOB_ID} --tensorboard_path tensorboard/${SLURM_JOB_ID} | tee output/${SLURM_JOB_ID}.txt
#echo "EfficientnetB1 3+1 DF+F2F+NT train fc first, initial weight=0"
#python -u train.py --weight_init_zero --epochs 100 --n_classes 3 -b 3 --n_samples 9 --n_classes_total 4 --model_name effb1 --train_list data_list/DF_F2F_NT_train_000599_frame30.txt --val_list data_list/DF_F2F_NT_val_600799_frame30.txt --model_output_path model_save/effb1/${SLURM_JOB_ID} --tensorboard_path tensorboard/${SLURM_JOB_ID} | tee output/${SLURM_JOB_ID}.txt
#echo "EfficientnetB2 3+1 DF+F2F+NT train fc first, initial weight=0"
#python -u train.py --weight_init_zero --epochs 100 --n_classes 3 -b 3 --n_samples 9 --n_classes_total 4 --model_name effb2 --train_list data_list/DF_F2F_NT_train_000599_frame30.txt --val_list data_list/DF_F2F_NT_val_600799_frame30.txt --model_output_path model_save/effb2/${SLURM_JOB_ID} --tensorboard_path tensorboard/${SLURM_JOB_ID} | tee output/${SLURM_JOB_ID}.txt
#echo "EfficientnetB3 3+1 DF+F2F+NT train fc first, initial weight=0"
#python -u train.py --weight_init_zero --epochs 100 --n_classes 3 -b 3 --n_samples 9 --n_classes_total 4 --model_name effb3 --train_list data_list/DF_F2F_NT_train_000599_frame30.txt --val_list data_list/DF_F2F_NT_val_600799_frame30.txt --model_output_path model_save/effb3/${SLURM_JOB_ID} --tensorboard_path tensorboard/${SLURM_JOB_ID} | tee output/${SLURM_JOB_ID}.txt
#echo "EfficientnetB4 3+1 DF+F2F+NT train fc first, initial weight=0"
#python -u train.py --weight_init_zero --epochs 100 --n_classes 3 -b 3 --n_samples 9 --n_classes_total 4 --model_name effb4 --train_list data_list/DF_F2F_NT_train_000599_frame30.txt --val_list data_list/DF_F2F_NT_val_600799_frame30.txt --model_output_path model_save/effb4/${SLURM_JOB_ID} --tensorboard_path tensorboard/${SLURM_JOB_ID} | tee output/${SLURM_JOB_ID}.txt
#echo "EfficientnetB5 3+1 DF+F2F+NT train fc first, initial weight=0"
#python -u train.py --weight_init_zero --epochs 100 --n_classes 3 -b 3 --n_samples 9 --n_classes_total 4 --model_name effb5 --train_list data_list/DF_F2F_NT_train_000599_frame30.txt --val_list data_list/DF_F2F_NT_val_600799_frame30.txt --model_output_path model_save/effb5/${SLURM_JOB_ID} --tensorboard_path tensorboard/${SLURM_JOB_ID} | tee output/${SLURM_JOB_ID}.txt
#echo "EfficientnetB6 3+1 DF+F2F+NT train fc first, initial weight=0"
#python -u train.py --weight_init_zero --epochs 100 --n_classes 3 -b 3 --n_samples 9 --n_classes_total 4 --model_name effb6 --train_list data_list/DF_F2F_NT_train_000599_frame30.txt --val_list data_list/DF_F2F_NT_val_600799_frame30.txt --model_output_path model_save/effb6/${SLURM_JOB_ID} --tensorboard_path tensorboard/${SLURM_JOB_ID} | tee output/${SLURM_JOB_ID}.txt
#echo "EfficientnetB7 3+1 DF+F2F+NT train fc first, initial weight=0"
#python -u train.py --weight_init_zero --epochs 100 --n_classes 3 -b 3 --n_samples 9 --n_classes_total 4 --model_name effb7 --train_list data_list/DF_F2F_NT_train_000599_frame30.txt --val_list data_list/DF_F2F_NT_val_600799_frame30.txt --model_output_path model_save/effb7/${SLURM_JOB_ID} --tensorboard_path tensorboard/${SLURM_JOB_ID} | tee output/${SLURM_JOB_ID}.txt
echo "Resnet101 3+1 DF+F2F+NT train fc first, initial weight=0"
python -u train.py --weight_init_zero --epochs 100 --n_classes 3 -b 3 --n_samples 9 --n_classes_total 4 --model_name res101 --train_list data_list/DF_F2F_NT_train_000599_frame30.txt --val_list data_list/DF_F2F_NT_val_600799_frame30.txt --model_output_path model_save/res101/${SLURM_JOB_ID} --tensorboard_path tensorboard/${SLURM_JOB_ID} | tee output/${SLURM_JOB_ID}.txt


#echo "EfficientnetB0 3+1 DF+F2F+NT train fc first, cosine distance"
#python -u train.py --dist_type cosine --epochs 100 --n_classes 3 -b 3 --n_samples 9 --n_classes_total 4 --model_name effb0 --train_list data_list/DF_F2F_NT_train_000599_frame30.txt --val_list data_list/DF_F2F_NT_val_600799_frame30.txt --model_output_path model_save/effb0/${SLURM_JOB_ID} --tensorboard_path tensorboard/${SLURM_JOB_ID} | tee output/${SLURM_JOB_ID}.txt
#echo "EfficientnetB1 3+1 DF+F2F+NT train fc first, cosine distance"
#python -u train.py --dist_type cosine --epochs 100 --n_classes 3 -b 3 --n_samples 9 --n_classes_total 4 --model_name effb1 --train_list data_list/DF_F2F_NT_train_000599_frame30.txt --val_list data_list/DF_F2F_NT_val_600799_frame30.txt --model_output_path model_save/effb1/${SLURM_JOB_ID} --tensorboard_path tensorboard/${SLURM_JOB_ID} | tee output/${SLURM_JOB_ID}.txt
#echo "EfficientnetB2 3+1 DF+F2F+NT train fc first, cosine distance"
#python -u train.py --dist_type cosine --epochs 100 --n_classes 3 -b 3 --n_samples 9 --n_classes_total 4 --model_name effb2 --train_list data_list/DF_F2F_NT_train_000599_frame30.txt --val_list data_list/DF_F2F_NT_val_600799_frame30.txt --model_output_path model_save/effb2/${SLURM_JOB_ID} --tensorboard_path tensorboard/${SLURM_JOB_ID} | tee output/${SLURM_JOB_ID}.txt
#echo "EfficientnetB3 3+1 DF+F2F+NT train fc first, cosine distance"
#python -u train.py --dist_type cosine --epochs 100 --n_classes 3 -b 3 --n_samples 9 --n_classes_total 4 --model_name effb3 --train_list data_list/DF_F2F_NT_train_000599_frame30.txt --val_list data_list/DF_F2F_NT_val_600799_frame30.txt --model_output_path model_save/effb3/${SLURM_JOB_ID} --tensorboard_path tensorboard/${SLURM_JOB_ID} | tee output/${SLURM_JOB_ID}.txt
#echo "EfficientnetB4 3+1 DF+F2F+NT train fc first, cosine distance"
#python -u train.py --dist_type cosine --epochs 100 --n_classes 3 -b 3 --n_samples 9 --n_classes_total 4 --model_name effb4 --train_list data_list/DF_F2F_NT_train_000599_frame30.txt --val_list data_list/DF_F2F_NT_val_600799_frame30.txt --model_output_path model_save/effb4/${SLURM_JOB_ID} --tensorboard_path tensorboard/${SLURM_JOB_ID} | tee output/${SLURM_JOB_ID}.txt
#echo "EfficientnetB5 3+1 DF+F2F+NT train fc first, cosine distance"
#python -u train.py --dist_type cosine --epochs 100 --n_classes 3 -b 3 --n_samples 9 --n_classes_total 4 --model_name effb5 --train_list data_list/DF_F2F_NT_train_000599_frame30.txt --val_list data_list/DF_F2F_NT_val_600799_frame30.txt --model_output_path model_save/effb5/${SLURM_JOB_ID} --tensorboard_path tensorboard/${SLURM_JOB_ID} | tee output/${SLURM_JOB_ID}.txt
#echo "EfficientnetB6 3+1 DF+F2F+NT train fc first, cosine distance"
#python -u train.py --dist_type cosine --epochs 100 --n_classes 3 -b 3 --n_samples 9 --n_classes_total 4 --model_name effb6 --train_list data_list/DF_F2F_NT_train_000599_frame30.txt --val_list data_list/DF_F2F_NT_val_600799_frame30.txt --model_output_path model_save/effb6/${SLURM_JOB_ID} --tensorboard_path tensorboard/${SLURM_JOB_ID} | tee output/${SLURM_JOB_ID}.txt
#echo "EfficientnetB7 3+1 DF+F2F+NT train fc first, cosine distance"
#python -u train.py --dist_type cosine --epochs 100 --n_classes 3 -b 3 --n_samples 9 --n_classes_total 4 --model_name effb7 --train_list data_list/DF_F2F_NT_train_000599_frame30.txt --val_list data_list/DF_F2F_NT_val_600799_frame30.txt --model_output_path model_save/effb7/${SLURM_JOB_ID} --tensorboard_path tensorboard/${SLURM_JOB_ID} | tee output/${SLURM_JOB_ID}.txt
#echo "Resnet101 3+1 DF+F2F+NT train fc first, cosine distance"
#python -u train.py --dist_type cosine --epochs 100 --n_classes 3 -b 3 --n_samples 9 --n_classes_total 4 --model_name res101 --train_list data_list/DF_F2F_NT_train_000599_frame30.txt --val_list data_list/DF_F2F_NT_val_600799_frame30.txt --model_output_path model_save/res101/${SLURM_JOB_ID} --tensorboard_path tensorboard/${SLURM_JOB_ID} | tee output/${SLURM_JOB_ID}.txt

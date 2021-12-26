#!/bin/bash
#SBATCH --job-name="API-test"   # Sensible name for the job
#SBATCH --account=ie-idi   # Account for consumed resources
#SBATCH --nodes=1           # Allocate 1 nodes for the job
#SBATCH -c 8                 # Number of cores (can vary)
#SBATCH --gres=gpu:V100
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

#echo "test 2475004 model"
#python -u test.py -b 1 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/res101/2475004/model_best.pth.tar --output_path test_results/df_test_detail_${SLURM_JOB_ID}_model_2475004.txt| tee test_results/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --test_list data_list/face2face_test_800999_frame30_label2.txt --resume model_save/res101/2475004/model_best.pth.tar --output_path test_results/f2f_test_detail_${SLURM_JOB_ID}_model_2475004.txt| tee test_results/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --test_list data_list/faceswap_test_800999_frame30_label3.txt --resume model_save/res101/2475004/model_best.pth.tar --output_path test_results/fs_test_detail_${SLURM_JOB_ID}_model_2475004.txt| tee test_results/fs_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --test_list data_list/neuraltextures_test_800999_frame30.txt --resume model_save/res101/2475004/model_best.pth.tar --output_path test_results/nt_test_detail_${SLURM_JOB_ID}_model_2475004.txt| tee test_results/nt_test_${SLURM_JOB_ID}.txt

#echo "test 2475034 model"
#python -u test.py -b 1 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/res101/2475034/model_best.pth.tar --output_path test_results/2475034/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2475034/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --test_list data_list/face2face_test_800999_frame30_label2.txt --resume model_save/res101/2475034/model_best.pth.tar --output_path test_results/2475034/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2475034/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --test_list data_list/faceswap_test_800999_frame30_label3.txt --resume model_save/res101/2475034/model_best.pth.tar --output_path test_results/2475034/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2475034/fs_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --test_list data_list/neuraltextures_test_800999_frame30.txt --resume model_save/res101/2475034/model_best.pth.tar --output_path test_results/2475034/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2475034/nt_test_${SLURM_JOB_ID}.txt


#echo "test 2506357 model closed set"
#python -u test.py -b 1 --n_classes_total 4 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/res101/2506357/model_best.pth.tar --output_path test_results/2506357/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2506357/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --n_classes_total 4 --test_list data_list/face2face_test_800999_frame30_label2.txt --resume model_save/res101/2506357/model_best.pth.tar --output_path test_results/2506357/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2506357/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --n_classes_total 4 --test_list data_list/faceswap_test_800999_frame30_label3.txt --resume model_save/res101/2506357/model_best.pth.tar --output_path test_results/2506357/close/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2506357/close/fs_test_${SLURM_JOB_ID}.txt

#echo "test 2506357 model open set"
#python -u test_binary.py -b 1 --n_classes_total 4 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/res101/2506357/model_best.pth.tar --output_path test_results/2506357/open/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2506357/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_binary.py -b 1 --n_classes_total 4 --test_list data_list/face2face_test_800999_frame30_label2.txt --resume model_save/res101/2506357/model_best.pth.tar --output_path test_results/2506357/open/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2506357/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_binary.py -b 1 --n_classes_total 4 --test_list data_list/faceswap_test_800999_frame30_label3.txt --resume model_save/res101/2506357/model_best.pth.tar --output_path test_results/2506357/open/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2506357/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_binary.py -b 1 --n_classes_total 4 --test_list data_list/neuraltextures_test_800999_frame30_label4.txt --resume model_save/res101/2506357/model_best.pth.tar --output_path test_results/2506357/open/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2506357/open/nt_test_${SLURM_JOB_ID}.txt


#echo "test 2506358 model closed set"
#python -u test.py -b 1 --n_classes_total 4 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/res101/2506358/model_best.pth.tar --output_path test_results/2506358/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2506358/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --n_classes_total 4 --test_list data_list/face2face_test_800999_frame30_label2.txt --resume model_save/res101/2506358/model_best.pth.tar --output_path test_results/2506358/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2506358/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --n_classes_total 4 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/res101/2506358/model_best.pth.tar --output_path test_results/2506358/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2506358/close/nt_test_${SLURM_JOB_ID}.txt

#echo "test 2506358 model open set"
#python -u test_binary.py -b 1 --n_classes_total 4 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/res101/2506358/model_best.pth.tar --output_path test_results/2506358/open/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2506358/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_binary.py -b 1 --n_classes_total 4 --test_list data_list/face2face_test_800999_frame30_label4.txt --resume model_save/res101/2506358/model_best.pth.tar --output_path test_results/2506358/open/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2506358/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_binary.py -b 1 --n_classes_total 4 --test_list data_list/faceswap_test_800999_frame30_label2.txt --resume model_save/res101/2506358/model_best.pth.tar --output_path test_results/2506358/open/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2506358/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_binary.py -b 1 --n_classes_total 4 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/res101/2506358/model_best.pth.tar --output_path test_results/2506358/open/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2506358/open/nt_test_${SLURM_JOB_ID}.txt


#echo "test 2506359 model closed set"
#python -u test.py -b 1 --n_classes_total 4 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/res101/2506359/model_best.pth.tar --output_path test_results/2506359/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2506359/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --n_classes_total 4 --test_list data_list/faceswap_test_800999_frame30_label2.txt --resume model_save/res101/2506359/model_best.pth.tar --output_path test_results/2506359/close/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2506359/close/fs_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --n_classes_total 4 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/res101/2506359/model_best.pth.tar --output_path test_results/2506359/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2506359/close/nt_test_${SLURM_JOB_ID}.txt

#echo "test 2506359 model open set"
#python -u test_binary.py -b 1 --n_classes_total 4 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/res101/2506359/model_best.pth.tar --output_path test_results/2506359/open/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2506359/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_binary.py -b 1 --n_classes_total 4 --test_list data_list/face2face_test_800999_frame30_label2.txt --resume model_save/res101/2506359/model_best.pth.tar --output_path test_results/2506359/open/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2506359/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_binary.py -b 1 --n_classes_total 4 --test_list data_list/faceswap_test_800999_frame30_label4.txt --resume model_save/res101/2506359/model_best.pth.tar --output_path test_results/2506359/open/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2506359/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_binary.py -b 1 --n_classes_total 4 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/res101/2506359/model_best.pth.tar --output_path test_results/2506359/open/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2506359/open/nt_test_${SLURM_JOB_ID}.txt


#echo "test 2506360 model closed set"
#python -u test.py -b 1 --n_classes_total 4 --test_list data_list/face2face_test_800999_frame30_label1.txt --resume model_save/res101/2506360/model_best.pth.tar --output_path test_results/2506360/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2506360/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --n_classes_total 4 --test_list data_list/faceswap_test_800999_frame30_label2.txt --resume model_save/res101/2506360/model_best.pth.tar --output_path test_results/2506360/close/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2506360/close/fs_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --n_classes_total 4 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/res101/2506360/model_best.pth.tar --output_path test_results/2506360/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2506360/close/nt_test_${SLURM_JOB_ID}.txt

#echo "test 2506360 model open set"
#python -u test_binary.py -b 1 --n_classes_total 4 --test_list data_list/deepfakes_test_800999_frame30_label4.txt --resume model_save/res101/2506360/model_best.pth.tar --output_path test_results/2506360/open/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2506360/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_binary.py -b 1 --n_classes_total 4 --test_list data_list/face2face_test_800999_frame30_label1.txt --resume model_save/res101/2506360/model_best.pth.tar --output_path test_results/2506360/open/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2506360/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_binary.py -b 1 --n_classes_total 4 --test_list data_list/faceswap_test_800999_frame30_label2.txt --resume model_save/res101/2506360/model_best.pth.tar --output_path test_results/2506360/open/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2506360/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_binary.py -b 1 --n_classes_total 4 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/res101/2506360/model_best.pth.tar --output_path test_results/2506360/open/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2506360/open/nt_test_${SLURM_JOB_ID}.txt


#echo "test 2521041 model closed set"
#python -u test.py -b 1 --n_classes_total 4 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/efficientnet/2521041/model_best.pth.tar --output_path test_results/2521041/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2521041/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --n_classes_total 4 --test_list data_list/face2face_test_800999_frame30_label2.txt --resume model_save/efficientnet/2521041/model_best.pth.tar --output_path test_results/2521041/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2521041/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --n_classes_total 4 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/efficientnet/2521041/model_best.pth.tar --output_path test_results/2521041/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2521041/close/nt_test_${SLURM_JOB_ID}.txt

#echo "test 2521041 model open set"
#python -u test_binary.py -b 1 --n_classes_total 4 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/efficientnet/2521041/model_best.pth.tar --output_path test_results/2521041/open/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2521041/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_binary.py -b 1 --n_classes_total 4 --test_list data_list/face2face_test_800999_frame30_label4.txt --resume model_save/efficientnet/2521041/model_best.pth.tar --output_path test_results/2521041/open/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2521041/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_binary.py -b 1 --n_classes_total 4 --test_list data_list/faceswap_test_800999_frame30_label2.txt --resume model_save/efficientnet/2521041/model_best.pth.tar --output_path test_results/2521041/open/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2521041/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_binary.py -b 1 --n_classes_total 4 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/efficientnet/2521041/model_best.pth.tar --output_path test_results/2521041/open/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2521041/open/nt_test_${SLURM_JOB_ID}.txt


#echo "test 2521040 model closed set"
#python -u test.py -b 1 --n_classes_total 4 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/efficientnet/2521040/model_best.pth.tar --output_path test_results/2521040/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2521040/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --n_classes_total 4 --test_list data_list/face2face_test_800999_frame30_label2.txt --resume model_save/efficientnet/2521040/model_best.pth.tar --output_path test_results/2521040/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2521040/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --n_classes_total 4 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/efficientnet/2521040/model_best.pth.tar --output_path test_results/2521040/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2521040/close/nt_test_${SLURM_JOB_ID}.txt

#echo "test 2521040 model open set"
#python -u test_binary.py -b 1 --n_classes_total 4 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/efficientnet/2521040/model_best.pth.tar --output_path test_results/2521040/open/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2521040/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_binary.py -b 1 --n_classes_total 4 --test_list data_list/face2face_test_800999_frame30_label4.txt --resume model_save/efficientnet/2521040/model_best.pth.tar --output_path test_results/2521040/open/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2521040/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_binary.py -b 1 --n_classes_total 4 --test_list data_list/faceswap_test_800999_frame30_label2.txt --resume model_save/efficientnet/2521040/model_best.pth.tar --output_path test_results/2521040/open/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2521040/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_binary.py -b 1 --n_classes_total 4 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/efficientnet/2521040/model_best.pth.tar --output_path test_results/2521040/open/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2521040/open/nt_test_${SLURM_JOB_ID}.txt


#echo "test 2532498 model closed set"
#python -u test.py -b 1 --n_classes_total 4 --model_name effb6 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/effb6/2532498/model_best.pth.tar --output_path test_results/2532498/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2532498/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --n_classes_total 4 --model_name effb6 --test_list data_list/face2face_test_800999_frame30_label2.txt --resume model_save/effb6/2532498/model_best.pth.tar --output_path test_results/2532498/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2532498/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --n_classes_total 4 --model_name effb6 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/effb6/2532498/model_best.pth.tar --output_path test_results/2532498/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2532498/close/nt_test_${SLURM_JOB_ID}.txt

#echo "test 2532498 model open set"
#python -u test_binary.py -b 1 --n_classes_total 4 --model_name effb6 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/effb6/2532498/model_best.pth.tar --output_path test_results/2532498/open/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2532498/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_binary.py -b 1 --n_classes_total 4 --model_name effb6 --test_list data_list/face2face_test_800999_frame30_label4.txt --resume model_save/effb6/2532498/model_best.pth.tar --output_path test_results/2532498/open/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2532498/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_binary.py -b 1 --n_classes_total 4 --model_name effb6 --test_list data_list/faceswap_test_800999_frame30_label2.txt --resume model_save/effb6/2532498/model_best.pth.tar --output_path test_results/2532498/open/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2532498/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_binary.py -b 1 --n_classes_total 4 --model_name effb6 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/effb6/2532498/model_best.pth.tar --output_path test_results/2532498/open/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2532498/open/nt_test_${SLURM_JOB_ID}.txt


#echo "test 2543837 model closed set"
#python -u test.py -b 1 --n_classes_total 4 --model_name effb7 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/effb7/2543837/model_best.pth.tar --output_path test_results/2543837/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2543837/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --n_classes_total 4 --model_name effb7 --test_list data_list/face2face_test_800999_frame30_label2.txt --resume model_save/effb7/2543837/model_best.pth.tar --output_path test_results/2543837/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2543837/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --n_classes_total 4 --model_name effb7 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/effb7/2543837/model_best.pth.tar --output_path test_results/2543837/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2543837/close/nt_test_${SLURM_JOB_ID}.txt

#echo "test 2543837 model open set"
#python -u test_binary.py -b 1 --n_classes_total 4 --model_name effb7 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/effb7/2543837/model_best.pth.tar --output_path test_results/2543837/open/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2543837/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_binary.py -b 1 --n_classes_total 4 --model_name effb7 --test_list data_list/face2face_test_800999_frame30_label4.txt --resume model_save/effb7/2543837/model_best.pth.tar --output_path test_results/2543837/open/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2543837/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_binary.py -b 1 --n_classes_total 4 --model_name effb7 --test_list data_list/faceswap_test_800999_frame30_label2.txt --resume model_save/effb7/2543837/model_best.pth.tar --output_path test_results/2543837/open/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2543837/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_binary.py -b 1 --n_classes_total 4 --model_name effb7 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/effb7/2543837/model_best.pth.tar --output_path test_results/2543837/open/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2543837/open/nt_test_${SLURM_JOB_ID}.txt


#echo "test 2546248 model closed set"
#python -u test.py -b 1 --n_classes_total 4 --model_name effb0 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/effb0/2546248/model_best.pth.tar --output_path test_results/2546248/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546248/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --n_classes_total 4 --model_name effb0 --test_list data_list/face2face_test_800999_frame30_label2.txt --resume model_save/effb0/2546248/model_best.pth.tar --output_path test_results/2546248/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546248/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --n_classes_total 4 --model_name effb0 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/effb0/2546248/model_best.pth.tar --output_path test_results/2546248/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546248/close/nt_test_${SLURM_JOB_ID}.txt

#echo "test 2546248 model open set"
#python -u test_binary.py -b 1 --n_classes_total 4 --model_name effb0 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/effb0/2546248/model_best.pth.tar --output_path test_results/2546248/open/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546248/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_binary.py -b 1 --n_classes_total 4 --model_name effb0 --test_list data_list/face2face_test_800999_frame30_label4.txt --resume model_save/effb0/2546248/model_best.pth.tar --output_path test_results/2546248/open/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546248/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_binary.py -b 1 --n_classes_total 4 --model_name effb0 --test_list data_list/faceswap_test_800999_frame30_label2.txt --resume model_save/effb0/2546248/model_best.pth.tar --output_path test_results/2546248/open/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546248/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_binary.py -b 1 --n_classes_total 4 --model_name effb0 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/effb0/2546248/model_best.pth.tar --output_path test_results/2546248/open/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546248/open/nt_test_${SLURM_JOB_ID}.txt


#echo "test 2546254 model closed set"
#python -u test.py -b 1 --n_classes_total 4 --model_name effb1 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/effb1/2546254/model_best.pth.tar --output_path test_results/2546254/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546254/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --n_classes_total 4 --model_name effb1 --test_list data_list/face2face_test_800999_frame30_label2.txt --resume model_save/effb1/2546254/model_best.pth.tar --output_path test_results/2546254/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546254/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --n_classes_total 4 --model_name effb1 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/effb1/2546254/model_best.pth.tar --output_path test_results/2546254/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546254/close/nt_test_${SLURM_JOB_ID}.txt

#echo "test 2546254 model open set"
#python -u test_binary.py -b 1 --n_classes_total 4 --model_name effb1 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/effb1/2546254/model_best.pth.tar --output_path test_results/2546254/open/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546254/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_binary.py -b 1 --n_classes_total 4 --model_name effb1 --test_list data_list/face2face_test_800999_frame30_label4.txt --resume model_save/effb1/2546254/model_best.pth.tar --output_path test_results/2546254/open/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546254/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_binary.py -b 1 --n_classes_total 4 --model_name effb1 --test_list data_list/faceswap_test_800999_frame30_label2.txt --resume model_save/effb1/2546254/model_best.pth.tar --output_path test_results/2546254/open/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546254/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_binary.py -b 1 --n_classes_total 4 --model_name effb1 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/effb1/2546254/model_best.pth.tar --output_path test_results/2546254/open/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546254/open/nt_test_${SLURM_JOB_ID}.txt


#echo "test 2546255 model closed set"
#python -u test.py -b 1 --n_classes_total 4 --model_name effb2 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/effb2/2546255/model_best.pth.tar --output_path test_results/2546255/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546255/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --n_classes_total 4 --model_name effb2 --test_list data_list/face2face_test_800999_frame30_label2.txt --resume model_save/effb2/2546255/model_best.pth.tar --output_path test_results/2546255/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546255/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --n_classes_total 4 --model_name effb2 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/effb2/2546255/model_best.pth.tar --output_path test_results/2546255/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546255/close/nt_test_${SLURM_JOB_ID}.txt

#echo "test 2546255 model open set"
#python -u test_binary.py -b 1 --n_classes_total 4 --model_name effb2 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/effb2/2546255/model_best.pth.tar --output_path test_results/2546255/open/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546255/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_binary.py -b 1 --n_classes_total 4 --model_name effb2 --test_list data_list/face2face_test_800999_frame30_label4.txt --resume model_save/effb2/2546255/model_best.pth.tar --output_path test_results/2546255/open/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546255/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_binary.py -b 1 --n_classes_total 4 --model_name effb2 --test_list data_list/faceswap_test_800999_frame30_label2.txt --resume model_save/effb2/2546255/model_best.pth.tar --output_path test_results/2546255/open/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546255/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_binary.py -b 1 --n_classes_total 4 --model_name effb2 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/effb2/2546255/model_best.pth.tar --output_path test_results/2546255/open/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546255/open/nt_test_${SLURM_JOB_ID}.txt


#echo "test 2546256 model closed set"
#python -u test.py -b 1 --n_classes_total 4 --model_name effb3 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/effb3/2546256/model_best.pth.tar --output_path test_results/2546256/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546256/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --n_classes_total 4 --model_name effb3 --test_list data_list/face2face_test_800999_frame30_label2.txt --resume model_save/effb3/2546256/model_best.pth.tar --output_path test_results/2546256/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546256/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --n_classes_total 4 --model_name effb3 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/effb3/2546256/model_best.pth.tar --output_path test_results/2546256/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546256/close/nt_test_${SLURM_JOB_ID}.txt

#echo "test 2546256 model open set"
#python -u test_binary.py -b 1 --n_classes_total 4 --model_name effb3 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/effb3/2546256/model_best.pth.tar --output_path test_results/2546256/open/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546256/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_binary.py -b 1 --n_classes_total 4 --model_name effb3 --test_list data_list/face2face_test_800999_frame30_label4.txt --resume model_save/effb3/2546256/model_best.pth.tar --output_path test_results/2546256/open/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546256/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_binary.py -b 1 --n_classes_total 4 --model_name effb3 --test_list data_list/faceswap_test_800999_frame30_label2.txt --resume model_save/effb3/2546256/model_best.pth.tar --output_path test_results/2546256/open/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546256/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_binary.py -b 1 --n_classes_total 4 --model_name effb3 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/effb3/2546256/model_best.pth.tar --output_path test_results/2546256/open/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546256/open/nt_test_${SLURM_JOB_ID}.txt


#echo "test 2546261 model closed set"
#python -u test.py -b 1 --n_classes_total 4 --model_name effb4 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/effb4/2546261/model_best.pth.tar --output_path test_results/2546261/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546261/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --n_classes_total 4 --model_name effb4 --test_list data_list/face2face_test_800999_frame30_label2.txt --resume model_save/effb4/2546261/model_best.pth.tar --output_path test_results/2546261/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546261/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --n_classes_total 4 --model_name effb4 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/effb4/2546261/model_best.pth.tar --output_path test_results/2546261/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546261/close/nt_test_${SLURM_JOB_ID}.txt

#echo "test 2546261 model open set"
#python -u test_binary.py -b 1 --n_classes_total 4 --model_name effb4 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/effb4/2546261/model_best.pth.tar --output_path test_results/2546261/open/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546261/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_binary.py -b 1 --n_classes_total 4 --model_name effb4 --test_list data_list/face2face_test_800999_frame30_label4.txt --resume model_save/effb4/2546261/model_best.pth.tar --output_path test_results/2546261/open/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546261/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_binary.py -b 1 --n_classes_total 4 --model_name effb4 --test_list data_list/faceswap_test_800999_frame30_label2.txt --resume model_save/effb4/2546261/model_best.pth.tar --output_path test_results/2546261/open/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546261/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_binary.py -b 1 --n_classes_total 4 --model_name effb4 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/effb4/2546261/model_best.pth.tar --output_path test_results/2546261/open/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546261/open/nt_test_${SLURM_JOB_ID}.txt


echo "test 2546262 model closed set"
python -u test.py -b 1 --n_classes_total 4 --model_name effb5 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/effb5/2546262/model_best.pth.tar --output_path test_results/2546262/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546262/close/df_test_${SLURM_JOB_ID}.txt
python -u test.py -b 1 --n_classes_total 4 --model_name effb5 --test_list data_list/face2face_test_800999_frame30_label2.txt --resume model_save/effb5/2546262/model_best.pth.tar --output_path test_results/2546262/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546262/close/f2f_test_${SLURM_JOB_ID}.txt
python -u test.py -b 1 --n_classes_total 4 --model_name effb5 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/effb5/2546262/model_best.pth.tar --output_path test_results/2546262/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546262/close/nt_test_${SLURM_JOB_ID}.txt

echo "test 2546262 model open set"
python -u test_binary.py -b 1 --n_classes_total 4 --model_name effb5 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/effb5/2546262/model_best.pth.tar --output_path test_results/2546262/open/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546262/open/df_test_${SLURM_JOB_ID}.txt
python -u test_binary.py -b 1 --n_classes_total 4 --model_name effb5 --test_list data_list/face2face_test_800999_frame30_label4.txt --resume model_save/effb5/2546262/model_best.pth.tar --output_path test_results/2546262/open/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546262/open/f2f_test_${SLURM_JOB_ID}.txt
python -u test_binary.py -b 1 --n_classes_total 4 --model_name effb5 --test_list data_list/faceswap_test_800999_frame30_label2.txt --resume model_save/effb5/2546262/model_best.pth.tar --output_path test_results/2546262/open/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546262/open/fs_test_${SLURM_JOB_ID}.txt
python -u test_binary.py -b 1 --n_classes_total 4 --model_name effb5 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/effb5/2546262/model_best.pth.tar --output_path test_results/2546262/open/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546262/open/nt_test_${SLURM_JOB_ID}.txt

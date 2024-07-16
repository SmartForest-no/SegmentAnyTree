#!/bin/bash

DATA_PATH=/home/nibio/data/test_data_agnostic_instanceSeg_treeins_agnostic_sparse_1000_500_100_10/results_/for_instance_no_outer_out/final_results
TMP_DIR=/home/nibio/mutable-outside-world/tmp_per_class_metric

# remove tmp dir if exists
rm -rf ${TMP_DIR}

# create tmp dir
mkdir ${TMP_DIR}

# copy all files to tmp dir
cp ${DATA_PATH}/*.laz ${TMP_DIR}

# create new folders with the following names NIBIO, SCION, RMIT, CULS, SCION, TUWIEN in tmp dir
mkdir ${TMP_DIR}/NIBIO
mkdir ${TMP_DIR}/SCION
mkdir ${TMP_DIR}/RMIT
mkdir ${TMP_DIR}/CULS
mkdir ${TMP_DIR}/TUWIEN

# move files to the corresponding folders
mv ${TMP_DIR}/*NIBIO_*.laz ${TMP_DIR}/NIBIO
mv ${TMP_DIR}/*SCION_*.laz ${TMP_DIR}/SCION
mv ${TMP_DIR}/*RMIT_*.laz ${TMP_DIR}/RMIT
mv ${TMP_DIR}/*CULS_*.laz ${TMP_DIR}/CULS
mv ${TMP_DIR}/*TUWIEN_*.laz ${TMP_DIR}/TUWIEN

# run metrics for each folder
python3 metrics/instance_segmentation_metrics_in_folder.py \
    --gt_las_folder_path ${TMP_DIR}/NIBIO/ \
    --target_las_folder_path ${TMP_DIR}/NIBIO/ \
    --output_folder_path ${TMP_DIR}/NIBIO/metrics_out \
    --remove_ground --verbose

python3 metrics/instance_segmentation_metrics_in_folder.py \
    --gt_las_folder_path ${TMP_DIR}/SCION/ \
    --target_las_folder_path ${TMP_DIR}/SCION/ \
    --output_folder_path ${TMP_DIR}/SCION/metrics_out \
    --remove_ground --verbose

python3 metrics/instance_segmentation_metrics_in_folder.py \
    --gt_las_folder_path ${TMP_DIR}/RMIT/ \
    --target_las_folder_path ${TMP_DIR}/RMIT/ \
    --output_folder_path ${TMP_DIR}/RMIT/metrics_out \
    --remove_ground --verbose

python3 metrics/instance_segmentation_metrics_in_folder.py \
    --gt_las_folder_path ${TMP_DIR}/CULS/ \
    --target_las_folder_path ${TMP_DIR}/CULS/ \
    --output_folder_path ${TMP_DIR}/CULS/metrics_out \
    --remove_ground --verbose

python3 metrics/instance_segmentation_metrics_in_folder.py \
    --gt_las_folder_path ${TMP_DIR}/TUWIEN/ \
    --target_las_folder_path ${TMP_DIR}/TUWIEN/ \
    --output_folder_path ${TMP_DIR}/TUWIEN/metrics_out \
    --remove_ground --verbose


# go to each folder and run metrics (metrics_out) and get summary_metrics_all_plots.csv
# take the following columns that contains: detection rate, omissions, commission, rmse_hight, f1_score


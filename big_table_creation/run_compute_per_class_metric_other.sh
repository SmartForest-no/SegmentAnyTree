#!/bin/bash
TMP_DIR=/home/nibio/mutable-outside-world/tmp_per_class_metric

# remove tmp dir if exists
rm -rf ${TMP_DIR}

# create tmp dir
mkdir ${TMP_DIR}

mkdir ${TMP_DIR}/austrian_plot_out
mkdir ${TMP_DIR}/english_plot_out
mkdir ${TMP_DIR}/german_plot_out
mkdir ${TMP_DIR}/mls_out

cp /home/nibio/data/test_data_agnostic_instanceSeg_treeins_agnostic_sparse_1000_500_100_10/results_/austrian_plot_out/final_results/*.laz ${TMP_DIR}/austrian_plot_out
cp /home/nibio/data/test_data_agnostic_instanceSeg_treeins_agnostic_sparse_1000_500_100_10/results_/english_plot_out/final_results/*.laz ${TMP_DIR}/english_plot_out
cp /home/nibio/data/test_data_agnostic_instanceSeg_treeins_agnostic_sparse_1000_500_100_10/results_/german_plot_out/final_results/*.laz ${TMP_DIR}/german_plot_out
cp /home/nibio/data/test_data_agnostic_instanceSeg_treeins_agnostic_sparse_1000_500_100_10/results_/mls_out/final_results/*.laz ${TMP_DIR}/mls_out

# copy metics folder for each
cp -r /home/nibio/data/test_data_agnostic_instanceSeg_treeins_agnostic_sparse_1000_500_100_10/results_/austrian_plot_out/metrics_out ${TMP_DIR}/austrian_plot_out
cp -r /home/nibio/data/test_data_agnostic_instanceSeg_treeins_agnostic_sparse_1000_500_100_10/results_/english_plot_out/metrics_out ${TMP_DIR}/english_plot_out
cp -r /home/nibio/data/test_data_agnostic_instanceSeg_treeins_agnostic_sparse_1000_500_100_10/results_/german_plot_out/metrics_out ${TMP_DIR}/german_plot_out
cp -r /home/nibio/data/test_data_agnostic_instanceSeg_treeins_agnostic_sparse_1000_500_100_10/results_/mls_out/metrics_out ${TMP_DIR}/mls_out

echo "Done copying files"
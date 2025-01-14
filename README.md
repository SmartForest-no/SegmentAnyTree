 
![SegmentAnyTree_logo](https://github.com/user-attachments/assets/8849a4b2-3bb3-4c6d-b1f1-13f91efc0936)

## Description
This repo includes the code for training and inference for the method developed by [Wielgosz et al. (2024)SegmentAnyTree: A sensor and platform agnostic deep learning model for tree segmentation using laser scanning data. Remote Sensing of Environment](https://www.sciencedirect.com/science/article/pii/S0034425724003936). 

Under the hood, SegmentAnyTree relies on the [torch-points3d framework](https://github.com/torch-points3d/torch-points3d) as the code base. So please take a look there for more information regarding the training and parametrization of the code.

## Usage
The code has been tested on a Linux machine and it relies on a docker image. The method has not been tested in a Windows environment and parts of the code (e.g. Minkowski Engine) might not be available for Windows.

### Using the docker image

A pre-built docker image can be pulled from [here](https://hub.docker.com/repository/docker/donaldmaen/segment-any-tree/general)

In order to run code using docker container you should edit the content of `run_docker_locally.sh` file.  You should change the following lines:
```
docker run -it --gpus all \
    --name $CONTAINER_NAME \
    --mount type=bind,source=/home/nibio/mutable-outside-world/code/PanopticSegForLargeScalePointCloud_maciej/bucket_in_folder,target=/home/nibio/mutable-outside-world/bucket_in_folder \
    --mount type=bind,source=/home/nibio/mutable-outside-world/code/PanopticSegForLargeScalePointCloud_maciej/bucket_out_folder,target=/home/nibio/mutable-outside-world/bucket_out_folder \
    $IMAGE_NAME
```

You should change the mounting folders : 
```
 --mount type=bind,source=/home/nibio/mutable-outside-world/code/PanopticSegForLargeScalePointCloud_maciej/bucket_in_folder
```
and 
```
 --mount type=bind,source=/home/nibio/mutable-outside-world/code/PanopticSegForLargeScalePointCloud_maciej/bucket_out_folder
```
to match your folders where you keep your local point clound files (las, ply, laz or zip ) to be processed. 

Once you introduce changes to `run_docker_locally.sh` file, you should run it: `bash run_docker_locally.sh` and expect the results in your output folder e.g. `/home/nibio/mutable-outside-world/code/PanopticSegForLargeScalePointCloud_maciej/bucket_out_folder`.


## Inference
This section explains how to use the inference script (`run_inference.sh`) to process data and manage the output. This is to be used if you do not run using docker container.
Follow the steps below for successful execution. 

### Steps to Use the Script

1. **Set Up Environment**:
   - The script defines the working directory (`WORK_DIR`) and ensures all necessary modules are accessible by updating the `PYTHONPATH`. No changes are needed unless the working directory path must be modified.
   ```bash
   WORK_DIR='/home/nibio/mutable-outside-world'
   export PYTHONPATH=$WORK_DIR:$PYTHONPATH
   ```

2. **Provide Input Parameters**:
   - The script requires three parameters to run:
     1. **SOURCE_DIR**: The input directory with files to process.
     2. **DEST_DIR**: The output directory where results will be saved.
     3. **CLEAN_OUTPUT_DIR**: Set to `true` or `false` to decide if the output directory should be cleaned before execution.

   - If no parameters are provided, default values are used:
   ```bash
   SOURCE_DIR="$WORK_DIR/data_for_test"
   DEST_DIR="$WORK_DIR/data_for_test_results"
   CLEAN_OUTPUT_DIR=true
   ```

3. **Run the Script**:
   - To execute the script, use the following command:
   ```bash
   bash run_inference.sh <path_to_input_dir> <path_to_output_dir> <clean_output_dir>
   ```

4. **Cleaning the Output Directory**:
   - If `CLEAN_OUTPUT_DIR` is set to `true`, the script will remove existing contents from the output directory before processing.

5. **File Preparation**:
   - The script copies input files to an `input_data` folder in the output directory to avoid modifying the original files:
   ```bash
   cp -r "$SOURCE_DIR/"* "$DEST_DIR/input_data/"
   ```

6. **Run Python Scripts**:
   - The script runs multiple Python scripts for tasks like updating the `eval.yaml` file, renaming files, and performing UTM normalization. These scripts are called sequentially to ensure correct data preparation.

7. **Inference Execution**:
   - After preparing the files, the script runs the inference pipeline with this command:
   ```bash
   bash large_PC_predict.sh "$DEST_DIR"
   ```

8. **Post-Processing and Results**:
   - After inference, the script processes and renames output files, stores them in a `final_results` folder, and counts the number of result files generated:
 


## Training
Please follow the command to train the model.
`python train.py task=panoptic data=panoptic/treeins models=panoptic/area4_ablation_3heads model_name=PointGroup-PAPER training=treeins job_name=treeins_my_first_run`

In order to train the model you have to prepare the data. You can take a look at file : `sample_data_conversion.py` to check how it may be done. 

## Issues
If you encounter any issues with the code please provide your feedback by raising an issue in this repo rather than contacting the paper authors!

## Citation
If you use the code or data in this repository for your research or project, please make sure to cite the associated article:

```
@article{WIELGOSZ2024114367,
title = {SegmentAnyTree: A sensor and platform agnostic deep learning model for tree segmentation using laser scanning data},
journal = {Remote Sensing of Environment},
volume = {313},
pages = {114367},
year = {2024},
issn = {0034-4257},
doi = {https://doi.org/10.1016/j.rse.2024.114367},
url = {https://www.sciencedirect.com/science/article/pii/S0034425724003936},
author = {Maciej Wielgosz and Stefano Puliti and Binbin Xiang and Konrad Schindler and Rasmus Astrup},
keywords = {3D deep learning, Instance segmentation, ITC, ALS, TLS, Drones}
}
```

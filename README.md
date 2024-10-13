 

## Inference
For inference run `run_inference.sh input_folder output_folder`. `input_folder` folder should contains las files for processing. `output_folder` contains results in `final_results` subfolder.

## Training
Please follow the command to train the model.
`python train.py task=panoptic data=panoptic/treeins models=panoptic/area4_ablation_3heads model_name=PointGroup-PAPER training=treeins job_name=treeins_my_first_run`

In order to train the model you have to prepare the data. You can take a look at file : `sample_data_conversion.py` to check how it may be done. 

## Framework used as a code base
This framework was used https://github.com/torch-points3d/torch-points3d as the code base. So please take a look there for more information regarding the training and parametrization of the code.


## Issues
Please provide problems with the code as issuses in this repo.


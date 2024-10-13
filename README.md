 
![SegmentAnyTree_logo](https://github.com/user-attachments/assets/8849a4b2-3bb3-4c6d-b1f1-13f91efc0936)

## Description
This repo includes the code for training and inference for the method developed by [Wielgosz et al. (2024)SegmentAnyTree: A sensor and platform agnostic deep learning model for tree segmentation using laser scanning data. Remote Sensing of Environment](https://www.sciencedirect.com/science/article/pii/S0034425724003936). 

Under the hood, SegmentAnyTree relies on the [torch-points3d framework](https://github.com/torch-points3d/torch-points3d) as the code base. So please take a look there for more information regarding the training and parametrization of the code.

## Usage
The code has been tested on a Linux machine, ... and it relies on a docker image. The method has not been tested in a Windows environment and we do not plan on 

### Using the docker image
...

## Inference
For inference run `run_inference.sh input_folder output_folder`. `input_folder` folder should contains las files for processing. `output_folder` contains results in `final_results` subfolder.

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

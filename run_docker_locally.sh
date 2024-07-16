#!/bin/bash

CONTAINER_NAME="test_e2e_instance"
IMAGE_NAME="nibio/e2e-instance"

# Check if the container exists
if [ $(docker container ls -a -q -f name=$CONTAINER_NAME) ]; then
    echo "Removing existing container $CONTAINER_NAME"
    docker container rm $CONTAINER_NAME
else
    echo "Container $CONTAINER_NAME does not exist."
fi

# Check if the image exists
# if [ $(docker image ls -q -f reference=$IMAGE_NAME) ]; then
#     echo "Removing existing image $IMAGE_NAME"
#     docker image rm $IMAGE_NAME
# else
#     echo "Image $IMAGE_NAME does not exist."
# fi

# ./build.sh
docker build -t $IMAGE_NAME .

echo "Running the container"
# docker run -it --gpus all --name $CONTAINER_NAME $IMAGE_NAME > e2e-instance.log 2>&1

docker run -it --gpus all \
    --name $CONTAINER_NAME \
    --mount type=bind,source=/home/nibio/mutable-outside-world/code/PanopticSegForLargeScalePointCloud_maciej/bucket_in_folder,target=/home/nibio/mutable-outside-world/bucket_in_folder \
    --mount type=bind,source=/home/nibio/mutable-outside-world/code/PanopticSegForLargeScalePointCloud_maciej/bucket_out_folder,target=/home/nibio/mutable-outside-world/bucket_out_folder \
    $IMAGE_NAME 





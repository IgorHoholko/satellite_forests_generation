#!/bin/bash

if [ "$USE_GPU" = "1" ]; then
    docker build --rm --tag fr_gpu --build-arg GPU=ON --build-arg IMAGE=nvidia/cuda:10.2-cudnn8-runtime-ubuntu18.04 .
else
    docker build --rm --tag fr_cpu --build-arg GPU=OFF .
fi


xhost +

if [ "$USE_GPU" = "1" ]; then
    docker run --rm -ti --net=host --ipc=host   -v $(pwd):/root   -e DISPLAY=$DISPLAY  -v /tmp/.X11-unix:/tmp/.X11-unix --name fr_gpu --runtime=nvidia -d --gpus all fr_gpu
else
    docker run --rm -ti --net=host --ipc=host   -v $(pwd):/root   -e DISPLAY=$DISPLAY    -v /tmp/.X11-unix:/tmp/.X11-unix     --name fr_cpu -d fr_cpu
fi


echo "Server is set up. Run: "
echo "> docker exec -it fr_cpu bash"
echo "or fr_gpu if you set up with USE_GPU"
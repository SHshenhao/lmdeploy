
# export NCCL_LIBRARIES=
docker run --gpus device=7 --rm --shm-size 16g --name lmbuild -v /home/shenhao/work/lmdeploy:/home/shenhao/work/lmdeploy -it openmmlab/lmdeploy-builder:cuda11.8
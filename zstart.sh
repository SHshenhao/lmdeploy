#python profile_generation.py --model-path /nvme/share/share/shenhao/workspace --concurrency 1 --prompt-tokens 64 --completion-tokens 512
# export NCCL_LIBRARIES=
docker run --gpus device=7 --rm --shm-size 16g --name lmbuild -v /home/shenhao/work/lmdeploy:/home/shenhao/work/lmdeploy -it openmmlab/lmdeploy-builder:cuda11.8
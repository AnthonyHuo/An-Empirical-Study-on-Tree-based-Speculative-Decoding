CUDA_VISIBLE_DEVICES=0 python benchmark_inference.py --P 128 --D 1 --offloading

CUDA_VISIBLE_DEVICES=0 python benchmark_inference.py --P 128 --D 2 --offloading

CUDA_VISIBLE_DEVICES=0 python benchmark_inference.py --P 128 --D 4 --offloading

CUDA_VISIBLE_DEVICES=0 python benchmark_inference.py --P 128 --D 8 --offloading

CUDA_VISIBLE_DEVICES=0 python benchmark_inference.py --P 128 --D 16 --offloading

CUDA_VISIBLE_DEVICES=0 python benchmark_inference.py --P 128 --D 32 --offloading

CUDA_VISIBLE_DEVICES=0 python benchmark_inference.py --P 128 --D 64 --offloading

CUDA_VISIBLE_DEVICES=0 python benchmark_inference.py --P 128 --D 128 --offloading

CUDA_VISIBLE_DEVICES=0 python benchmark_inference.py --P 128 --D 256 --offloading

CUDA_VISIBLE_DEVICES=0 python benchmark_inference.py --P 128 --D 512 --offloading

CUDA_VISIBLE_DEVICES=0 python benchmark_inference.py --P 128 --D 1024 --offloading

CUDA_VISIBLE_DEVICES=0 python benchmark_inference.py --P 128 --D 1536 --M 2048 --offloading

CUDA_VISIBLE_DEVICES=0 python benchmark_inference.py --P 128 --D 2048 --M 2560 --offloading

CUDA_VISIBLE_DEVICES=0 python benchmark_inference.py --P 128 --D 2560 --M 4096 --offloading

CUDA_VISIBLE_DEVICES=0 python benchmark_inference.py --P 128 --D 3120 --M 4096 --offloading


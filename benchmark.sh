CUDA_VISIBLE_DEVICES=0 python benchmark_inference.py --P 128 --D 1 --offloading

CUDA_VISIBLE_DEVICES=0 python benchmark_inference.py --P 128 --D 2 --offloading

CUDA_VISIBLE_DEVICES=0 python benchmark_inference.py --P 128 --D 4 --offloading

CUDA_VISIBLE_DEVICES=0 python benchmark_inference.py --P 128 --D 8 --offloading

CUDA_VISIBLE_DEVICES=0 python benchmark_inference.py --P 128 --D 16 --offloading

CUDA_VISIBLE_DEVICES=0 python benchmark_inference.py --P 128 --D 32 --offloading

CUDA_VISIBLE_DEVICES=0 python benchmark_inference.py --P 128 --D 64 --offloading

CUDA_VISIBLE_DEVICES=0 python benchmark_inference.py --P 128 --D 128 --offloading
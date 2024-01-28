bash benchmark.sh >> benchmark_13b_model.log

CUDA_VISIBLE_DEVICES=0 python testbed.py --model  JackFram/llama-68m  --target meta-llama/Llama-2-13b-hf  --T 0.01 --P 1.0 --B 128  --DP 0.99 --W 32 --start 0 --end 10 --Mode baseline --M  384 --dataset ./dataset/c4_small.json >> baseline.log

CUDA_VISIBLE_DEVICES=0 python testbed.py --model  JackFram/llama-68m  --target meta-llama/Llama-2-13b-hf  --T 0.01 --P 1.0 --B 128  --DP 0.99 --W 32 --start 0 --end 10 --Mode greedy --M  384 --dataset ./dataset/c4_small.json >> results.log

CUDA_VISIBLE_DEVICES=0 python testbed.py --model  JackFram/llama-68m  --target meta-llama/Llama-2-13b-hf  --T 0.01 --P 1.0 --B 128  --DP 0.99 --W 32 --start 0 --end 10 --Mode benchmark --M  384 --dataset ./dataset/c4_small.json >> benchmark.log

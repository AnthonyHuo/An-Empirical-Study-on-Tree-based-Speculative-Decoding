bash benchmark.sh >> benchmark.log

CUDA_VISIBLE_DEVICES=0 python testbed.py --model  JackFram/llama-68m  --target meta-llama/Llama-2-13b-hf  --T 0.01 --P 1.0 --B 128  --DP 0.99 --W 32 --start 0 --end 10 --Mode baseline --M  384 --dataset ./dataset/c4_small.json >> benchmark.log

CUDA_VISIBLE_DEVICES=0 python testbed.py --model  JackFram/llama-68m  --target meta-llama/Llama-2-13b-hf  --T 0.01 --P 1.0 --B 128  --DP 0.99 --W 32 --start 0 --end 10 --Mode greedy --M  384 --dataset ./dataset/c4_small.json >> benchmark.log

CUDA_VISIBLE_DEVICES=0 python testbed.py --model  JackFram/llama-68m  --target meta-llama/Llama-2-13b-hf  --T 0.01 --P 1.0 --B 128  --DP 0.99 --W 32 --start 0 --end 10 --Mode benchmark --M  384 --dataset ./dataset/c4_small.json >> benchmark.log

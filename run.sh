CUDA_VISIBLE_DEVICES=0 python testbed.py --model  JackFram/llama-68m  --target meta-llama/Llama-2-7b-hf  --T 0.6 --P 0.9 --B 128  --DP 0.99 --W 32 --start 0 --end 10 --Mode benchmark --static >> overhead.log

python benchmark_inference.py --D 1 >> evaluation.log
python benchmark_inference.py --D 8 >> evaluation.log
python benchmark_inference.py --D 16 >> evaluation.log
python benchmark_inference.py --D 32 >> evaluation.log
python benchmark_inference.py --D 64 >> evaluation.log
python benchmark_inference.py --D 128 >> evaluation.log
python benchmark_inference.py --D 192 >> evaluation.log
python benchmark_inference.py --D 256 >> evaluation.log
python benchmark_inference.py --D 384 >> evaluation.log
python benchmark_inference.py --D 512 >> evaluation.log
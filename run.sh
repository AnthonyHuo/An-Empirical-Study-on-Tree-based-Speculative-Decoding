CUDA_VISIBLE_DEVICES=5 python deepspeed_verify.py --P 32 >> 7b.log
CUDA_VISIBLE_DEVICES=5 python deepspeed_verify.py --P 64 >> 7b.log
CUDA_VISIBLE_DEVICES=5 python deepspeed_verify.py --P 128 >> 7b.log


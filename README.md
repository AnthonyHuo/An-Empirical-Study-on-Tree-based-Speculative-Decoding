# Project Name

## Environment Setup

To set up the environment for this project, please follow the steps below. The commands will install the required dependencies, including specific versions of PyTorch, Transformers, and other libraries.

### Prerequisites

- Python 3.8 or higher
- CUDA 12.1 for GPU acceleration (recommended)

### Installation Steps

1. **Install PyTorch and related libraries**

   First, install PyTorch, Torchvision, and Torchaudio with CUDA support using the following command:

   ```bash
   pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

CUDA_VISIBLE_DEVICES=7 python test_greedyS_dy.py --model JackFram/llama-68m --target meta-llama/Llama-2-13b-hf --T 0.6 --P 0.9 --start 0 --end 200 --Mode greedy --M 1024 --growmap /home/zhuominc/Sequoia_mingxiaohuo/growmaps/68m_7b-64.pt --dataset cnn

CUDA_VISIBLE_DEVICES=2 python test_greedyS.py \
    --model JackFram/llama-68m \
    --target meta-llama/Llama-2-7b-hf \
    --T 0.6 \
    --P 1.0 \
    --start 0 \
    --end 3500 \
    --Mode greedy \
    --M 1024 \
    --growmap /home/zhuominc/Sequoia_mingxiaohuo/growmaps/A100-OpenWebText-68m-7b-stochastic.pt \
    --dataset openwebtext

CUDA_VISIBLE_DEVICES=0 python test_greedyS.py \
    --model /home/zhuominc/Sequoia_mingxiaohuo/outputs2/checkpoint-50/ \
    --target meta-llama/Llama-2-7b-hf \
    --T 0.6 \
    --P 1.0 \
    --start 1800 \
    --end 2000 \
    --Mode greedy \
    --M 1024 \
    --growmap /home/zhuominc/Sequoia_mingxiaohuo/growmaps/A100-OpenWebText-68m-7b-stochastic.pt \
    --dataset cnn

CUDA_VISIBLE_DEVICES=6 python test_greedyS_logit.py \
    --model  /home/zhuominc/Sequoia_mingxiaohuo/outputs/checkpoint-1000/ \
    --target meta-llama/Llama-2-7b-hf \
    --T 0.6 \
    --P 1.0 \
    --start 0 \
    --end 3500 \
    --Mode baseline \
    --M 1024 \
    --growmap /home/zhuominc/Sequoia_mingxiaohuo/growmaps/A100-OpenWebText-68m-7b-stochastic.pt \
    --dataset openwebtext

CUDA_VISIBLE_DEVICES=9 python test_greedyS_logit.py     --model JackFram/llama-160m     --target meta-llama/Llama-2-13b-hf     --T 0.6     --P 1.0     --start 0     --end 2000     --Mode baseline     --M 1024     --growmap /home/zhuominc/Sequoia_mingxiaohuo/growmaps/A100-OpenWebText-68m-7b-stochastic.pt     --dataset cnn

CUDA_VISIBLE_DEVICES=7 python test_greedyS_logit.py     --model  /home/zhuominc/Sequoia_mingxiaohuo/outputs/checkpoint-1000/     --target meta-llama/Llama-2-13b-hf     --T 0.6     --P 1.0     --start 0     --end 3500     --Mode baseline     --M 1024     --growmap /home/zhuominc/Sequoia_mingxiaohuo/growmaps/A100-OpenWebText-68m-7b-stochastic.pt     --dataset openwebtext

CUDA_VISIBLE_DEVICES=5 python test_greedyS.py     --model JackFram/llama-68m  --target meta-llama/Llama-2-7b-hf     --T 0.8     --P 1.0     --start 3500    --end 3700     --Mode greedy     --M 1024     --growmap /home/zhuominc/Sequoia_mingxiaohuo/growmaps/A100-OpenWebText-68m-7b-stochastic.pt     --dataset openwebtext

CUDA_VISIBLE_DEVICES=5 python test_greedyS_dy.py \
    --model JackFram/llama-68m \
    --target meta-llama/Llama-2-7b-hf \
    --T 0.6 \
    --P 0.9 \
    --start 0 \
    --end 200 \
    --Mode greedy \
    --M 1024 \
    --growmap /home/zhuominc/Sequoia_mingxiaohuo/growmaps/A100-OpenWebText-68m-7b-stochastic.pt \
    --dataset openwebtext

CUDA_VISIBLE_DEVICES=8 python test_llava.py \
    --model anthonycmu/llava-1.5-160m \
    --target llava-hf/llava-1.5-7b-hf \
    --T 0.01 \
    --P 1.0 \
    --Mode greedy \
    --M 1024 \
    --growmap /home/zhuominc/Sequoia_mingxiaohuo/growmaps/A100-OpenWebText-68m-7b-stochastic.pt

CUDA_VISIBLE_DEVICES=9 python test_llava.py \
    --model anthonycmu/llava-1.5-160m \
    --target llava-hf/llava-1.5-7b-hf \
    --T 0.01 \
    --P 1.0 \
    --Mode baseline \
    --M 1024 \
    --growmap /home/zhuominc/Sequoia_mingxiaohuo/growmaps/A100-OpenWebText-68m-7b-stochastic.pt
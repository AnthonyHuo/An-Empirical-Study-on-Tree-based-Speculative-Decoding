# Empirical Study on Tree-based Speculative Decoding

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
2. **Install other dependencies**

Install the required libraries for Transformers, Accelerate, Datasets, and others:


pip install transformers==4.36.2
pip install accelerate==0.26.1
pip install datasets==2.16.1
pip install einops
pip install protobuf
pip install sentencepiece
pip install typing-extensions
## Running Experiments
Use the following commands to run various experiments with different models and datasets. Make sure to adjust the CUDA_VISIBLE_DEVICES according to your GPU availability.

Command Examples
Running Greedy Speculative Decoding with JackFram/llama-68m on CNN Dataset

bash

CUDA_VISIBLE_DEVICES=7 python test_greedyS_dy.py \
    --model JackFram/llama-68m \
    --target meta-llama/Llama-2-13b-hf \
    --T 0.6 \
    --P 0.9 \
    --start 0 \
    --end 200 \
    --Mode greedy \
    --M 1024 \
    --growmap /home/zhuominc/Sequoia_mingxiaohuo/growmaps/68m_7b-64.pt \
    --dataset cnn
Running Greedy Speculative Decoding with OpenWebText Dataset

bash
复制代码
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
Running Baseline Speculative Decoding

bash
复制代码
CUDA_VISIBLE_DEVICES=6 python test_greedyS_logit.py \
    --model /home/zhuominc/Sequoia_mingxiaohuo/outputs/checkpoint-1000/ \
    --target meta-llama/Llama-2-7b-hf \
    --T 0.6 \
    --P 1.0 \
    --start 0 \
    --end 3500 \
    --Mode baseline \
    --M 1024 \
    --growmap /home/zhuominc/Sequoia_mingxiaohuo/growmaps/A100-OpenWebText-68m-7b-stochastic.pt \
    --dataset openwebtext
Running Speculative Decoding with JackFram/llama-160m on CNN Dataset

bash
复制代码
CUDA_VISIBLE_DEVICES=9 python test_greedyS_logit.py \
    --model JackFram/llama-160m \
    --target meta-llama/Llama-2-13b-hf \
    --T 0.6 \
    --P 1.0 \
    --start 0 \
    --end 2000 \
    --Mode baseline \
    --M 1024 \
    --growmap /home/zhuominc/Sequoia_mingxiaohuo/growmaps/A100-OpenWebText-68m-7b-stochastic.pt \
    --dataset cnn
Running LLAVA Model Test

bash
复制代码
CUDA_VISIBLE_DEVICES=8 python test_llava.py \
    --model anthonycmu/llava-1.5-160m \
    --target llava-hf/llava-1.5-7b-hf \
    --T 0.01 \
    --P 1.0 \
    --Mode greedy \
    --M 1024 \
    --growmap /home/zhuominc/Sequoia_mingxiaohuo/growmaps/A100-OpenWebText-68m-7b-stochastic.pt
Running Baseline LLAVA Model Test

bash

CUDA_VISIBLE_DEVICES=9 python test_llava.py \
    --model anthonycmu/llava-1.5-160m \
    --target llava-hf/llava-1.5-7b-hf \
    --T 0.01 \
    --P 1.0 \
    --Mode baseline \
    --M 1024 \
    --growmap /home/zhuominc/Sequoia_mingxiaohuo/growmaps/A100-OpenWebText-68m-7b-stochastic.pt
These commands allow you to perform various speculative decoding experiments using different models and configurations. Adjust parameters as needed for your specific use case.



This README guide includes installation instructions, setup details, and example commands 
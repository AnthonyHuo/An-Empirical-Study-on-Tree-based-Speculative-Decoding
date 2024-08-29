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

    ```bash
    pip install transformers==4.36.2
    pip install accelerate==0.26.1
    pip install datasets==2.16.1
    pip install einops
    pip install protobuf
    pip install sentencepiece
    pip install typing-extensions

## Running Experiments

    Use the following commands to run various experiments with different models and datasets. Make sure to adjust the CUDA_VISIBLE_DEVICES according to your GPU availability.
1. **Static Tree Method**
    Command Examples
    Running Greedy Speculative Decoding with JackFram/llama-68m on CNN Dataset

    ```bash

    CUDA_VISIBLE_DEVICES=7 python test_greedyS.py \
        --model JackFram/llama-68m \
        --target meta-llama/Llama-2-13b-hf \
        --T 0.6 \
        --P 0.9 \
        --start 0 \
        --end 200 \
        --Mode greedy \
        --M 1024 \
        --growmap ./growmaps/68m_7b-64.pt \
        --dataset cnn
    Running Greedy Speculative Decoding with OpenWebText Dataset

2. **Common-Sense Token Verfication**
    Command Examples
    Running Greedy Speculative Decoding with Common-Sense Token Verfication on CNN Dataset, it is cool let's try it! It can improve the average acceptance length from about 0.3 to 0.5

    ```bash

    CUDA_VISIBLE_DEVICES=7 python test_greedyS_commonsense.py \
        --model JackFram/llama-68m \
        --target meta-llama/Llama-2-13b-hf \
        --T 0.6 \
        --P 0.9 \
        --start 0 \
        --end 200 \
        --Mode greedy \
        --M 1024 \
        --growmap ./growmaps/68m_7b-64.pt \
        --dataset cnn

3. **Dynamic-Tree Algorithm**
    Command Examples
    It is faster than static tree, especially when temperature is high.
    
    ```bash

    CUDA_VISIBLE_DEVICES=7 python test_greedyS_dy.py \
        --model JackFram/llama-68m \
        --target meta-llama/Llama-2-13b-hf \
        --T 0.6 \
        --P 0.9 \
        --start 0 \
        --end 200 \
        --Mode greedy \
        --M 1024 \
        --growmap ./growmaps/68m_7b-64.pt \
        --dataset cnn

4. **Run kl Divergency test in advance**
    Command Examples
    We perform a method to test draft model potential in advance using KL Divergency.
    
    ```bash

    CUDA_VISIBLE_DEVICES=7 python test_greedyS_logit.py \
        --model JackFram/llama-68m \
        --target meta-llama/Llama-2-13b-hf \
        --T 0.6 \
        --P 0.9 \
        --start 0 \
        --end 3500 \
        --Mode baseline \
        --M 1024 \
        --growmap ./growmaps/68m_7b-64.pt \
        --dataset openwebtext

    After running it, you can get an potential score, for the same dataset small score means a strong draft model.

5. **Spec-LLava**
    Command Examples
    We perform a method to acclerate llava
     
    First, you can try our draft llava model, which very similar to llava original model, llava-160m and llava-68m, their peformance is good, can be used in some small model scenes, like robotics, driving!

    ```bash

    processor = AutoProcessor.from_pretrained("anthonycmu/llava-1.5-160m")
    processor = AutoProcessor.from_pretrained("anthonycmu/llava-1.5-68m")

    

    CUDA_VISIBLE_DEVICES=7 python test_greedyS_logit.py \
        --model JackFram/llama-68m \
        --target meta-llama/Llama-2-13b-hf \
        --T 0.6 \
        --P 0.9 \
        --start 0 \
        --end 3500 \
        --Mode baseline \
        --M 1024 \
        --growmap ./growmaps/68m_7b-64.pt \
        --dataset openwebtext


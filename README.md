## run the code ## 
CUDA_VISIBLE_DEVICES=6 python test_greedyS_dy.py --model JackFram/llama-68m --target meta-llama/Llama-2-13b-hf --T 0.01 --P 0.9 --start 0 --end 200 --Mode greedy --M 1024 --growmap /home/zhuominc/Sequoia_mingxiaohuo/growmaps/68m_13b-greedy.pt  --dataset cnn

you will get an acceptance rate which is about 3.29
if you want to change the draft_step change the self.draft step
## ! mainly check the code in GreedySTree_dy.py, about the if the attn_mask and position id and storage id is right. ##


#### 1、环境  
* fp16占用22G显存  
* INT8占用   

#### 2、运行命令  

* 需要更改里面的chatglm2对应的model目录和文件目录  
> CUDA_VISIBLE_DEVICES=0 python3 train.py  

#### 3、训练日志  
* Epoch 1 / 50  

* 100%|███████████████████████████████| 1392/1392 [08:51<00:00,  2.62it/s, lr=0.0005, train_loss=3.07]  
* 100%|████████████████████████████████████████████████| 15/15 [00:01<00:00, 10.03it/s, val_loss=2.89]  

#### 4、分布式  

* deepspeed分布式运行脚本
> CUDA_VISIBLE_DEVICES=3 deepspeed --master_port 12345 --num_gpus=1 train2.py \
    ----deepspeed conf/mydeepspeed.json


#### 5、推理  

* SFT微调之后，推理脚本
> CUDA_VISIBLE_DEVICES=0 python3 inference.py  

* 99%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌ | 99/100 [01:29<00:00,  1.23it/s]  
 
> query： 江北核心区，燕子矶，迈皋桥，怎么选择
> 
> response： 你好！个人建议是优先考虑核心区和城南中心。如果预算有限的话可以先摇燕然居  

------------  
* 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [01:29<00:00,  1.12it/s]

####  6、数据集  
* 医疗数据集可以参考：https://huggingface.co/datasets/shibing624/medical  

###  7、base合并Lora权重  
* 合并脚本  
> CUDA_VISIBLE_DEVICES=0 python3 merge_lora2base.py \
  --base_model_name_or_path /home/xxm/model/new/chatglm-6b \
  --peft_model_path /home/xxm/下载/ChatGLM-Efficient-Tuning/output/lora_estate_qa5 \
  --output_dir /home/xxm/fsdownload/chatglm2_lora/output/merged_chatglm_lora \
  --model_type chatglm

### 8、reward  

* 运行脚本
> CUDA_VISIBLE_DEVICES=0 python3 reward.py \
    --lr_scheduler_type cosine \
    --learning_rate 5e-4 \
    --do_train \
    --do_eval \
    --output_dir /home/xxm/fsdownload/chatglm2_lora/output/reward \
    --use_v2 \
    --model_path /home/xxm/model/chatglm2-6b \
    --checkpoint_dir /home/xxm/fsdownload/chatglm2_lora/output/estate_qa0 \
    --num_train_epochs 5.0 \
    --save_steps 500 \
    --reward_filename /home/xxm/fsdownload/chatglm2_lora/data/estate_reward.json \
    --per_device_train_batch_size 8 \
    --fp16

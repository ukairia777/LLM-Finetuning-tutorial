# LLM-Finetuning-tutorial

- If you're fine-tuning LLaMa-2 7B, please add bf16=True and change fp16=False in the HF trainer. LLaMa-1 7B works as is. This only applies to LLaMa-2 7B. Additionally, if you are using 1 GPU, please change ddp_find_unused_paramters=False in the HF trainer. We will be updating the fine-tuning script to handle these changes automatically.
- Example for how to calcualte gradient accumulation steps using 2 GPUs: = global_batch_size / micro_batch_size / num_gpus = 16 / 1 / 2 = 8.

### SFT-LoRA
```
!git clone https://github.com/ukairia777/LLM-Finetuning-tutorial.git
%cd LLM-Finetuning-tutorial
!pip install -r requirements.txt

!torchrun finetune.py \
    --base_model beomi/llama-2-ko-7b \
    --data-path '학습할 데이터셋' \
    --output_dir ./fine-tuned-llama-2-ko-7b \
    --batch_size 64 \
    --micro_batch_size 1 \
    --num_epochs 15 \
    --learning_rate 1e-5 \
    --cutoff_len 2048 \
    --val_set_size 0 \
    --lora_r 16 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[gate_proj, down_proj, up_proj]' \
    --train_on_inputs False \
    --add_eos_token True \
    --group_by_length False \
    --prompt_template_name alpaca \
    --lr_scheduler 'linear' \
    --warmup_steps 0
```

### DPO-LoRA
```
!wget https://raw.githubusercontent.com/ukairia777/LLM-Finetuning-tutorial/main/DPO.py
!pip install trl==0.7.9 peft==0.7.1 accelerate==0.26.1 datasets==2.16.1 bitsandbytes==0.42.0 scipy==1.11.4 sentencepiece==0.1.99 fire==0.5.0
!pip install transformers==4.37.2

!python DPO.py \
    --base_model Qwen/Qwen1.5-72B \
    --data-path  Intel/orca_dpo_pairs \
    --output_dir ./lora \
    --num_epochs 3 \
    --batch_size 16 \
    --micro_batch_size 2 \
    --learning_rate 1e-5 \
    --lora_r 16 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    # --lora_target_modules ["embed_tokens", "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"] \
    --lora_target_modules ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"] \
    --lr_scheduler 'linear' \
    --warmup_ratio 0.1 \
    --cutoff_len 4096 \
```

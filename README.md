# LLM-Finetuning-tutorial
LLM을 튜닝하기 위한 튜토리얼 자료입니다.

- If you're fine-tuning LLaMa-2 7B, please add bf16=True and change fp16=False in the HF trainer. LLaMa-1 7B works as is. This only applies to LLaMa-2 7B. Additionally, if you are using 1 GPU, please change ddp_find_unused_paramters=False in the HF trainer. We will be updating the fine-tuning script to handle these changes automatically.

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

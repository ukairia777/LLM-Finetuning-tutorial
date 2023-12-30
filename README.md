# LLM-Finetuning-tutorial
LLM을 튜닝하기 위한 튜토리얼 자료입니다.

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

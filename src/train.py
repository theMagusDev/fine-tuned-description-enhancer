import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from dotenv import load_dotenv
from peft import LoraConfig
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer
try:
    from .data_utils import load_avito_dataset
except ImportError:  # Support ``python src/train.py``.
    from data_utils import load_avito_dataset

load_dotenv()

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5-7B with QLoRA")
    parser.add_argument(
        "--data-path", "--data_path", dest="data_path", required=True, help="Path to .jsonl dataset"
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        default="./qwen-avito-finetuned",
        help="Trainer checkpoints directory",
    )
    parser.add_argument(
        "--adapter-output-dir",
        default="./qwen-avito-adapter",
        help="Final LoRA adapter and tokenizer directory",
    )
    return parser.parse_args()

def train():
    args = parse_args()
    model_id = "Qwen/Qwen2.5-7B-Instruct"

    train_ds, test_ds = load_avito_dataset(args.data_path)

    # QLoRA конфигурация
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Загрузка модели и токенизатора
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # LoRA параметры
    peft_config = LoraConfig(
        lora_alpha=128,
        lora_dropout=0.05,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # Настройки SFT
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        max_length=1024,
        dataset_text_field="text",
        completion_only_loss=True,
        num_train_epochs=3,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_ratio=0.1,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,
        logging_steps=10,
        optim="paged_adamw_32bit",
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        args=sft_config,
        peft_config=peft_config,
    )

    print("--- Начинаем обучение ---")
    trainer.train()
    
    trainer.model.save_pretrained(args.adapter_output_dir)
    tokenizer.save_pretrained(args.adapter_output_dir)
    print(f"--- Обучение завершено. Адаптер сохранен в {args.adapter_output_dir}. ---")

if __name__ == "__main__":
    train()

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from data_utils import load_avito_dataset

def train():
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    dataset_path = "/kaggle/input/avito-descriptions-enhanced/descriptions_enhancement_avito.jsonl"
    
    # Загрузка данных
    train_ds, test_ds = load_avito_dataset(dataset_path)

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
        output_dir="./qwen-avito-finetuned",
        max_length=1024,
        dataset_text_field="text",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        load_best_model_at_end=True,
        bf16=True,
        report_to="none"
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
    
    trainer.model.save_pretrained("qwen-avito-adapter")
    tokenizer.save_pretrained("qwen-avito-adapter")
    print("--- Обучение завершено. Адаптер сохранен. ---")

if __name__ == "__main__":
    train()
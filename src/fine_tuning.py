import os
import torch
import json
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments, TextStreamer
from unsloth import FastLanguageModel

# 1. Configuration
class FineTuningConfig:
    MODEL_NAME = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
    MAX_SEQ_LENGTH = 2048
    DTYPE = None
    LOAD_IN_4BIT = True
    DATASET_PATH = "final_avito_dataset_1500.jsonl"
    OUTPUT_DIR = "outputs"

# 2. Model & Tokenizer Setup
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = FineTuningConfig.MODEL_NAME,
    max_seq_length = FineTuningConfig.MAX_SEQ_LENGTH,
    dtype = FineTuningConfig.DTYPE,
    load_in_4bit = FineTuningConfig.LOAD_IN_4BIT,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# 3. Data Preparation
prompt_style = """Отредактируй описание товара для Авито. 
Будь грамотным, добавь в текст структуру и привлекательность, а также строго придерживайся фактов из исходного текста.

### Категория:
{}

### Исходное описание:
{}

### Улучшенное описание:
{}"""

EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    categories = [f"{c} / {sc}" for c, sc in zip(examples["base_category_name"], examples["base_subcategory_name"])]
    originals = examples["base_description"]
    generateds = examples["generated_description"]
    texts = [prompt_style.format(c, o, g) + EOS_TOKEN for c, o, g in zip(categories, originals, generateds)]
    return {"text": texts}

# Load and split dataset
if os.path.exists(FineTuningConfig.DATASET_PATH):
    with open(FineTuningConfig.DATASET_PATH, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    
    full_dataset = Dataset.from_list(data)
    dataset_split = full_dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = dataset_split["train"].map(formatting_prompts_func, batched=True)
    test_dataset = dataset_split["test"]
    print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
else:
    raise FileNotFoundError(f"Dataset not found at {FineTuningConfig.DATASET_PATH}")

# 4. Training
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    dataset_text_field = "text",
    max_seq_length = FineTuningConfig.MAX_SEQ_LENGTH,
    dataset_num_proc = 2,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = FineTuningConfig.OUTPUT_DIR,
    ),
)

trainer.train()

# 5. Inference Example
FastLanguageModel.for_inference(model)
sample = test_dataset[0]
inputs = tokenizer(
    [prompt_style.format(sample["base_category_name"], sample["base_description"], "")], 
    return_tensors = "pt"
).to("cuda")

print("\n--- Model Output Example ---")
_ = model.generate(**inputs, streamer = TextStreamer(tokenizer), max_new_tokens = 512)

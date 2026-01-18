import os
import json
import time
import logging
import pandas as pd
import numpy as np
from typing import Optional, Dict, List
from tqdm.auto import tqdm
from openai import OpenAI, APIError, RateLimitError, APITimeoutError
from kaggle_secrets import UserSecretsClient

class Config:
    # Paths
    INPUT_PARQUET = "/kaggle/input/avito-data/data/test_part_0001.snappy.parquet"
    OUTPUT_DATASET = "avito_train_dataset.jsonl"
    LOG_FILE = "generation_process.log"
    PREVIOUS_PROGRESS = "/kaggle/input/avito-700-samples/avito_train_dataset-united.jsonl"

    # Sampling parameters
    SAMPLE_SIZE = 1500
    MIN_LEN = 30
    MAX_LEN = 150

    # API Configuration (OpenRouter / DeepSeek)
    MODEL_NAME = "deepseek/deepseek-v3.2-speciale"
    API_BASE = "https://openrouter.ai/api/v1"

    # Generation parameters
    TEMPERATURE = 0.9
    TOP_P = 0.9
    FREQ_PENALTY = 0.1
    PRESENCE_PENALTY = 0.1
    REP_PENALTY = 1.05
    REASONING_EFFORT = "low"

    # Retry logic
    MAX_RETRIES = 5
    RETRY_DELAY = 2 

def setup_logging():
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(Config.LOG_FILE, mode="a", encoding="utf-8")
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(logging.INFO)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

logger = setup_logging()

def init_api_client():
    try:
        user_secrets = UserSecretsClient()
        api_key = user_secrets.get_secret("OPENROUTER_API_KEY")
        client = OpenAI(base_url=Config.API_BASE, api_key=api_key)
        logger.info("OpenAI client initialized successfully.")
        return client
    except Exception as e:
        logger.error(f"Failed to retrieve API key: {e}")
        raise e

client = init_api_client()

def load_and_filter_data():
    logger.info("Loading data from Parquet...")
    cols = ["base_item_id", "base_title", "base_description",
            "base_category_name", "base_subcategory_name"]

    df = pd.read_parquet(Config.INPUT_PARQUET, columns=cols)
    df["desc_len"] = df["base_description"].str.len().fillna(0)
    
    filtered_df = df[
        (df["desc_len"] >= Config.MIN_LEN) &
        (df["desc_len"] <= Config.MAX_LEN)
    ].copy()

    processed_ids = set()

    # Load previously processed IDs to avoid duplicates
    for path in [Config.PREVIOUS_PROGRESS, Config.OUTPUT_DATASET]:
        if os.path.exists(path):
            logger.info(f"Reading processed IDs from: {path}")
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        processed_ids.add(json.loads(line).get("id"))
                    except: continue

    logger.info(f"Total already processed: {len(processed_ids)} records.")
    filtered_df = filtered_df[~filtered_df["base_item_id"].isin(processed_ids)]

    remaining_needed = Config.SAMPLE_SIZE - len(processed_ids)
    if remaining_needed <= 0:
        logger.info("Target sample size reached.")
        return pd.DataFrame()

    if len(filtered_df) > remaining_needed:
        filtered_df = filtered_df.sample(remaining_needed, random_state=42)

    logger.info(f"Remaining to process: {len(filtered_df)} records.")
    return filtered_df

class DescriptionGenerator:
    def __init__(self, client):
        self.client = client
        self.system_prompt = (
            "Ты — ведущий AI-копирайтер платформы Авито. Твоя задача: профессионально отредактировать описание товара, "
            "сделав его привлекательным, структурированным и грамотным.\n\n"
            "СЛЕДУЙ ПРАВИЛАМ СТРОГО:\n"
            "1. ИСПРАВЛЕНИЕ: Исправь все опечатки, пунктуационные и грамматические ошибки исходного текста.\n"
            "2. ФАКТЫ: Оставь все детали из исходного описания. Категорически запрещено выдумывать состояние товара "
            "или детали, которых нет в тексте.\n"
            "3. ЭКСПЕРТНОСТЬ: Добавь 1-2 предложения с общеизвестными преимуществами данной модели товара.\n"
            "4. СТРУКТУРА:\n"
            "   - Разбей текст на логические абзацы.\n"
            "   - Обязательно добавь один маркированный список (через точку '•').\n"
            "   - Лимит эмодзи: строго не более 2 штук.\n"
            "5. СТИЛЬ: Профессиональный, человечный, лаконичный.\n"
            "6. ОГРАНИЧЕНИЕ ДЛИНЫ: Ориентируйся на объем 40–80 слов. "
            "ВАЖНО: Если исходное описание очень короткое, не пытайся растягивать текст «водой» или выдумками. "
            "В таком случае лучше оставить описание коротким (20-40 слов), но честным и полезным.\n\n"
            "Выдавай ТОЛЬКО финальный текст объявления без вводных фраз."
        )

    def generate(self, row) -> Optional[str]:
        prompt = (f"Улучши следующее объявление:\n"
            f"Категория: {row['base_category_name']} / {row['base_subcategory_name']}\n"
            f"Заголовок: {row['base_title']}\n"
            f"Исходное описание: {row['base_description']}\n\n"
            "Результат:")

        for attempt in range(Config.MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=Config.MODEL_NAME,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=Config.TEMPERATURE,
                    top_p=Config.TOP_P,
                    extra_body={"reasoning_effort": Config.REASONING_EFFORT}
                )
                return response.choices[0].message.content.strip()
            except (APITimeoutError, RateLimitError):
                time.sleep(Config.RETRY_DELAY * (attempt + 1))
            except Exception as e:
                logger.error(f"Error at ID {row['base_item_id']}: {e}")
                return None
        return None

if __name__ == "__main__":
    work_df = load_and_filter_data()
    if not work_df.empty:
        generator = DescriptionGenerator(client)
        with open(Config.OUTPUT_DATASET, "a", encoding="utf-8", buffering=1) as f_out:
            for _, row in tqdm(work_df.iterrows(), total=len(work_df)):
                text = generator.generate(row)
                if text:
                    record = {
                        "id": row["base_item_id"],
                        "base_category_name": row["base_category_name"],
                        "base_subcategory_name": row["base_subcategory_name"],
                        "base_description": row["base_description"],
                        "generated_description": text
                    }
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

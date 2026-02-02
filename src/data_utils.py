import json
import pandas as pd
from datasets import Dataset

def format_instruction(sample):
    """Форматирует входные данные в текстовый промпт для модели."""
    return f"""### Instruction:
{sample['instruction']}

### Context:
Категория: {sample['category_context']}
Товар: {sample['title']}

### Original Description:
{sample['original_description']}

### Improved Description:
{sample['generated_description']}"""

def load_avito_dataset(file_path, test_size=50):
    """Загружает .jsonl и возвращает HuggingFace Datasets."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip().startswith('{'):
                data.append(json.loads(line))

    df = pd.DataFrame(data)
    df['text'] = df.apply(format_instruction, axis=1)

    train_df = df.iloc[:-test_size]
    test_df = df.iloc[-test_size:]

    return Dataset.from_pandas(train_df), Dataset.from_pandas(test_df)
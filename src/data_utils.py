import json
import pandas as pd
from datasets import Dataset


REQUIRED_FIELDS = (
    "instruction",
    "category_context",
    "title",
    "original_description",
    "generated_description",
)

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
    """Load training rows and keep the final rows as the validation split.

    The returned second dataset is the same held-out validation split used as
    ``eval_dataset`` during training; it is not an independent test set.
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip().startswith('{'):
                data.append(json.loads(line))

    df = pd.DataFrame(data)
    missing_fields = [field for field in REQUIRED_FIELDS if field not in df.columns]
    if missing_fields:
        raise ValueError(f"Dataset is missing required fields: {missing_fields}")
    if test_size <= 0 or len(df) <= test_size:
        raise ValueError("test_size must leave at least one row in the training split")

    df['text'] = df.apply(format_instruction, axis=1)

    train_df = df.iloc[:-test_size]
    test_df = df.iloc[-test_size:]

    return Dataset.from_pandas(train_df), Dataset.from_pandas(test_df)

import os
import re
import json
import torch
import numpy as np
from tqdm.auto import tqdm
from datasets import Dataset
from unsloth import FastLanguageModel

# Metrics libraries
from bert_score import score as bert_score_func
from rouge_score import rouge_scorer

class EvalConfig:
    MODEL_NAME = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
    ADAPTER_PATH = "outputs" # Path to your LoRA weights
    TEST_DATA_PATH = "test_samples_holdout.jsonl"
    MAX_SAMPLES = 50 # To save time during evaluation
    MAX_NEW_TOKENS = 128

class MetricsEvaluator:
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    def check_structure(self, text: str) -> bool:
        """Checks if the text contains bullet points and at least one emoji."""
        has_bullets = bool(re.search(r"•|- ", text))
        # Basic emoji/special symbol detection
        has_emoji = bool(re.search(r"[^\w\s,.\(\)\-:]", text))
        return has_bullets and has_emoji

    def calculate_metrics(self, predictions: list, references: list):
        # 1. BERTScore (Semantic similarity)
        P, R, F1 = bert_score_func(predictions, references, lang="ru", verbose=False)
        avg_bert = F1.mean().item()

        # 2. ROUGE-L (Structural similarity)
        rouge_scores = [self.rouge_scorer.score(ref, pred)["rougeL"].fmeasure 
                        for ref, pred in zip(references, predictions)]
        avg_rouge = np.mean(rouge_scores)

        # 3. Structure Compliance
        compliance = [self.check_structure(pred) for pred in predictions]
        avg_compliance = np.mean(compliance)

        return {
            "bert": avg_bert,
            "rouge": avg_rouge,
            "structure": avg_compliance
        }

def run_evaluation(model, tokenizer, dataset, prompt_style):
    FastLanguageModel.for_inference(model)
    predictions = []
    references = []

    print(f"Running inference on {len(dataset)} samples...")
    for item in tqdm(dataset):
        prompt = prompt_style.format(item["base_category_name"], item["base_description"], "")
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
        
        outputs = model.generate(**inputs, max_new_tokens=EvalConfig.MAX_NEW_TOKENS, use_cache=True)
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract model response after the prompt
        marker = "### Improved Description:"
        if marker in full_text:
            generated_part = full_text.split(marker)[1].strip()
        else:
            generated_part = full_text.strip()
            
        predictions.append(generated_part)
        references.append(item["generated_description"])

    return predictions, references

if __name__ == "__main__":
    # 1. Load Data
    if not os.path.exists(EvalConfig.TEST_DATA_PATH):
        raise FileNotFoundError(f"Test data not found at {EvalConfig.TEST_DATA_PATH}")
    
    with open(EvalConfig.TEST_DATA_PATH, "r", encoding="utf-8") as f:
        test_data = [json.loads(line) for line in f][:EvalConfig.MAX_SAMPLES]
    
    prompt_style = """Edit this product description for Avito. 
Be professional, add structure, and strictly follow the facts from the source.

### Category:
{}

### Source Description:
{}

### Improved Description:
{}"""

    evaluator = MetricsEvaluator()

    # 2. EVALUATE BASE MODEL
    print("\n--- Phase 1: Evaluating Base Model ---")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = EvalConfig.MODEL_NAME,
        load_in_4bit = True,
    )
    
    base_preds, base_refs = run_evaluation(model, tokenizer, test_data, prompt_style)
    base_results = evaluator.calculate_metrics(base_preds, base_refs)

    # 3. EVALUATE FINE-TUNED MODEL
    print("\n--- Phase 2: Evaluating Fine-tuned Model ---")
    if os.path.exists(EvalConfig.ADAPTER_PATH):
        model.load_adapter(EvalConfig.ADAPTER_PATH)
        ft_preds, ft_refs = run_evaluation(model, tokenizer, test_data, prompt_style)
        ft_results = evaluator.calculate_metrics(ft_preds, ft_refs)
    else:
        print("Warning: Adapter path not found. Skipping Fine-tuned evaluation.")
        ft_results = None

    # 4. FINAL COMPARISON
    print("\n" + "="*50)
    print(f"{'Metric':<25} | {'Base Model':<12} | {'Fine-tuned':<12}")
    print("-" * 50)
    
    metrics_to_show = [
        ("BERTScore (Semantics)", "bert"),
        ("ROUGE-L (Structure)", "rouge"),
        ("Format Compliance (%)", "structure")
    ]

    for label, key in metrics_to_show:
        base_val = base_results[key]
        ft_val = ft_results[key] if ft_results else 0.0
        
        # Format as percentage for structure
        if key == "structure":
            print(f"{label:<25} | {base_val*100:>11.1f}% | {ft_val*100:>11.1f}%")
        else:
            print(f"{label:<25} | {base_val:>12.4f} | {ft_val:>12.4f}")
    
    print("="*50)
    
import torch
import evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from data_utils import load_avito_dataset

def run_evaluation(model_path="qwen-avito-adapter", base_model_id="Qwen/Qwen2.5-7B-Instruct", num_samples=10):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, test_dataset = load_avito_dataset("/kaggle/input/avito-descriptions-enhanced/descriptions_enhancement_avito.jsonl")

    # Загрузка базовой модели и адаптера
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, model_path).to(device)
    model.eval()

    rouge = evaluate.load("rouge")
    bertscore = evaluate.load("bertscore")

    predictions, references = [], []

    for i in range(min(num_samples, len(test_dataset))):
        sample = test_dataset[i]
        split_point = "### Improved Description:\n"
        input_prompt = sample['text'].split(split_point)[0] + split_point
        
        inputs = tokenizer(input_prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.4, do_sample=True)
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(input_prompt, "").strip()
        predictions.append(response)
        references.append(sample['generated_description'])

    # Считаем метрики
    rouge_results = rouge.compute(predictions=predictions, references=references)
    bert_results = bertscore.compute(predictions=predictions, references=references, lang="ru")

    print("\n--- Результаты валидации ---")
    print(f"ROUGE-1: {rouge_results['rouge1']:.4f}")
    print(f"BERTScore F1: {sum(bert_results['f1']) / len(bert_results['f1']):.4f}")

if __name__ == "__main__":
    run_evaluation()
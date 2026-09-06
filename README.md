# Avito Description Enhancer

Fine-tuning `Qwen/Qwen2.5-7B-Instruct` to improve Russian e-commerce and Avito product descriptions under limited GPU resources.

## Project goal

The model rewrites a title, category context, and original user description into clearer, more structured sales copy while following the facts supplied in the input. The project focuses on a reproducible QLoRA workflow that can be trained on modest cloud GPUs.

## 🔗 Live Demo

**Try it now on Google Colab:** [Avito Description Enhancer client](https://colab.research.google.com/drive/1E5yIj2imosq5qyugyMBix3pE0oRjWOs7?usp=sharing)

*(No local setup required)*

## 📌 Project Overview

User-generated content often suffers from poor formatting, grammatical errors, and lack of structure. This project addresses these issues by fine-tuning a **Qwen 2.5 7B** model to act as a professional e-commerce copywriter.
>>>>>>> 93ad098ac323a8292b4265bbc47228960121d17f

## Dataset

The training dataset contains approximately 1.5k examples derived from open Avito data. Target descriptions were generated through knowledge distillation with DeepSeek V3.2 Speciale as the teacher model. The prepared dataset is published on [Kaggle](https://www.kaggle.com/datasets/yuriymagus/avito-descriptions-enhanced); a local copy can be placed at `data/descriptions_enhancement_avito.jsonl`.

Each training row contains:

- `instruction`
- `category_context`
- `title`
- `original_description`
- `generated_description`

The final 50 rows are held out from gradient updates and used as the validation split.

## Fine-tuning

- Base model: [`Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- Method: QLoRA with 4-bit NF4 quantization
- Stack: Transformers, TRL, PEFT, bitsandbytes, and Datasets
- Training hardware: 2× NVIDIA Tesla T4
- Published adapter: [`yuriy-magus/Qwen2.5-7B-Ecom-Refiner`](https://huggingface.co/yuriy-magus/Qwen2.5-7B-Ecom-Refiner)

The LoRA configuration uses rank 64, alpha 128, dropout 0.05, and targets the attention and MLP projection layers. Training runs for three epochs with evaluation every 50 steps and selects the best checkpoint by `eval_loss`.

## Evaluation

The deterministic evaluation compares the original base model with the same model plus the published LoRA adapter on the same 50 held-out validation examples. Both paths use the tokenizer stored in the adapter repository, identical leak-free prompts, identical generation settings, and `do_sample=False`. Baseline inference uses PEFT's `disable_adapter()` context, so only one 7B model needs to be loaded.

| Metric | Base | Base + LoRA | Relative uplift |
|---|---:|---:|---:|
| BERTScore F1 | 0.684 | 0.761 | +11.2% |
| ROUGE-1 | 0.435 | 0.616 | +41.6% |
| ROUGE-L | 0.432 | 0.595 | +37.7% |

The evaluation implementation also calculates BERTScore Precision and Recall and records absolute deltas. The committed aggregate values are stored in [`results/eval_metrics.json`](results/eval_metrics.json) and [`results/eval_summary.csv`](results/eval_summary.csv), rather than existing only in this README.

Important limitation: these 50 examples were used as `eval_dataset` during fine-tuning and for best-checkpoint selection. They are therefore a held-out validation set, not an independent test set. No claim is made about factuality, hallucination reduction, format-compliance percentage, or statistical significance.

## Reproduction

Install the direct dependencies:

```bash
python -m pip install -r requirements.txt
```

If a Hugging Face token is required for model or dataset access, copy `.env.example` to `.env` and set `HF_TOKEN` there. The real `.env` file is excluded by `.gitignore`.

Run training only when a new adapter is required:

```bash
python src/train.py \
  --data-path data/descriptions_enhancement_avito.jsonl \
  --output-dir qwen-avito-finetuned \
  --adapter-output-dir qwen-avito-adapter
```

Run the deterministic base-vs-adapter evaluation without repeating fine-tuning:

```bash
python src/evaluation.py \
  --data-path data/descriptions_enhancement_avito.jsonl \
  --adapter yuriy-magus/Qwen2.5-7B-Ecom-Refiner \
  --output-dir results
```

The adapter and its tokenizer are downloaded from Hugging Face. Evaluation checkpoints all paired base/fine-tuned predictions in `results/eval_predictions.jsonl` and writes the aggregate files in `results/`. A CUDA GPU is required for the 4-bit 7B evaluation path.

For a sampling-based interactive example (separate from deterministic evaluation):

```bash
python src/inference.py \
  --category-context "Личные вещи / Одежда" \
  --title "Зимняя куртка" \
  --input "Продам теплую куртку, носил один сезон"
```

## Repository layout

- `notebooks/avito-desc-model-fine-tuning.ipynb` — historical training workflow plus the current deterministic evaluation section
- `notebooks/avito-fine-tune-new.ipynb` — source notebook for the corrected evaluation methodology
- `src/train.py` — CLI training entry point
- `src/evaluation.py` — deterministic paired evaluation and artifact generation
- `src/inference.py` — sampling-based demo inference
- `src/data_utils.py` — dataset loading, formatting, and validation split
- `src/dataset_generation.py` — original knowledge-distillation data generation workflow
- `results/` — committed aggregate metrics and generated paired predictions after an evaluation run

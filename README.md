# Avito Description Enhancer: Fine-Tuning Qwen 2.5 7B via Knowledge Distillation

A specialized LLM service designed to transform low-quality user-generated product descriptions into professional, structured, and selling ad copy for the **Avito** marketplace.

## 🔗 Live Demo

**Try it now on Google Colab:** [Avito Description Enhancer client](https://colab.research.google.com/drive/1E5yIj2imosq5qyugyMBix3pE0oRjWOs7?usp=sharing)

*(No local setup required)*

## 📌 Project Overview

User-generated content often suffers from poor formatting, grammatical errors, and lack of structure. This project addresses these issues by fine-tuning a **Qwen 2.5 7B** model to act as a professional e-commerce copywriter.

The core approach relies on **Knowledge Distillation**: using a powerful "Teacher" model (**DeepSeek V3.2 Speciale**) to generate high-quality training data for a smaller, more efficient "Student" model.

## 🚀 Key Features

* **Grammar & Style Correction:** Automatic fixing of typos and syntax errors.
* **Structured Output:** Generates logical paragraphs and bulleted lists.
* **E-commerce Optimization:** Adds 1-2 expert sentences about product benefits while maintaining a concise length (40-80 words).
* **High Efficiency:** Optimized for training and inference on consumer-grade hardware (e.g., NVIDIA T4).

## 🛠 Tech Stack

* **Base Model:** [Qwen 2.5 7B Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
* **Training:** QLoRA, 4-bit quantization
* **Teacher Model:** DeepSeek V3.2 Speciale (via OpenRouter API)
* **Libraries:** PyTorch, Transformers, PEFT, Bitsandbytes, TRL, Datasets
* **Metrics:** BERTScore, ROUGE-L, Structure Compliance Rate

## 📊 Pipeline Architecture

### 1. Data Generation (Distillation)

* Extracted 1,500 raw samples from the Avito Open Dataset.
* Processed via DeepSeek V3 to create "Golden Targets" following strict editorial guidelines.
* *Script:* `avito_dataset_gen.py`

### 2. Fine-Tuning

* Applied QLoRA
* Trained on two **Tesla T4 GPUs** (Kaggle environment)

### 3. Evaluation

* Comparative analysis between the Base Model and the Fine-Tuned version.
* *Script:* `evaluation.py`

## 📂 Repository Structure

* `avito_dataset_gen.py`: Script for synthetic data generation via API.
* `train.py`: Main training script.
* `evaluation.py`: Quantitative assessment of model performance.
* `data_utils.py`: Helpful functions for data processing. 
* `requirements.txt`: List of necessary dependencies.

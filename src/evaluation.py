"""Deterministic base-vs-LoRA evaluation on the 50-row validation split.

The split was used as ``eval_dataset`` during fine-tuning and for best-checkpoint
selection. Results produced here are validation results, not an independent test
estimate.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as importlib_metadata
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
import torch
from dotenv import load_dotenv
from huggingface_hub import model_info
from peft import PeftConfig, PeftModel
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    from .data_utils import load_avito_dataset
except ImportError:  # Support ``python src/evaluation.py``.
    from data_utils import load_avito_dataset


BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_ID = "yuriy-magus/Qwen2.5-7B-Ecom-Refiner"
EXPECTED_VALIDATION_SIZE = 50
MAX_NEW_TOKENS = 256
BERTSCORE_MODEL_TYPE = "bert-base-multilingual-cased"
SOURCE_FIELDS = (
    "instruction",
    "category_context",
    "title",
    "original_description",
)
PREDICTION_SOURCE_FIELDS = (
    "index",
    "category_context",
    "title",
    "original_description",
    "reference",
)

load_dotenv()


def build_inference_prompt(
    instruction: str,
    category_context: str,
    title: str,
    original_description: str,
) -> str:
    """Build the exact training prefix without reading the target completion."""
    return f"""### Instruction:
{instruction}

### Context:
Категория: {category_context}
Товар: {title}

### Original Description:
{original_description}

### Improved Description:
"""


def generation_kwargs(tokenizer: Any, max_new_tokens: int = MAX_NEW_TOKENS) -> dict[str, Any]:
    """Return the generation settings shared by both evaluated systems."""
    return {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }


def generate_deterministic(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """Generate one completion and decode only newly generated token IDs."""
    inputs = tokenizer(prompt, return_tensors="pt")
    input_length = inputs["input_ids"].shape[1]
    input_device = model.get_input_embeddings().weight.device
    inputs = {name: tensor.to(input_device) for name, tensor in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            **generation_kwargs(tokenizer, max_new_tokens=max_new_tokens),
        )

    generated_ids = outputs[:, input_length:]
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()


def _stable_index(sample: Mapping[str, Any], position: int) -> int | str:
    value = sample.get("__index_level_0__", position)
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def build_prediction_records(
    validation_dataset: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Create paired-record shells and matching leak-free prompts."""
    records: list[dict[str, Any]] = []
    prompts: list[str] = []
    for position, sample in enumerate(validation_dataset):
        missing = [field for field in (*SOURCE_FIELDS, "generated_description") if field not in sample]
        if missing:
            raise ValueError(f"Validation row {position} is missing fields: {missing}")

        prompts.append(build_inference_prompt(*(str(sample[field]) for field in SOURCE_FIELDS)))
        records.append(
            {
                "index": _stable_index(sample, position),
                "category_context": sample["category_context"],
                "title": sample["title"],
                "original_description": sample["original_description"],
                "reference": sample["generated_description"],
                "base_prediction": None,
                "finetuned_prediction": None,
            }
        )

    if len(records) != EXPECTED_VALIDATION_SIZE:
        raise ValueError(
            f"Expected the unchanged {EXPECTED_VALIDATION_SIZE}-row validation split, "
            f"got {len(records)} rows."
        )
    if not all(prompt.endswith("### Improved Description:\n") for prompt in prompts):
        raise AssertionError("An inference prompt does not end at the completion marker.")
    return records, prompts


def validation_fingerprint(records: Sequence[Mapping[str, Any]], prompts: Sequence[str]) -> str:
    components = []
    for record, prompt in zip(records, prompts):
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        reference_hash = hashlib.sha256(str(record["reference"]).encode("utf-8")).hexdigest()
        components.append(f"{prompt_hash}:{reference_hash}")
    return hashlib.sha256("\n".join(components).encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def save_paired_predictions(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    """Atomically checkpoint all paired records in the single final JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(temporary_path, path)


def resume_compatible_predictions(
    path: Path,
    fresh_records: list[dict[str, Any]],
    *,
    reset: bool = False,
) -> list[dict[str, Any]]:
    """Resume predictions only when they belong to the exact same validation rows."""
    if reset or not path.exists():
        save_paired_predictions(path, fresh_records)
        return fresh_records

    previous_records = read_jsonl(path)
    if len(previous_records) != EXPECTED_VALIDATION_SIZE:
        raise ValueError(
            f"{path} contains {len(previous_records)} rows; pass --reset-predictions "
            "to replace an incompatible file."
        )

    for fresh, previous in zip(fresh_records, previous_records):
        if any(fresh[field] != previous.get(field) for field in PREDICTION_SOURCE_FIELDS):
            raise ValueError(
                "Existing predictions do not match this validation split; "
                "pass --reset-predictions to start a fresh run."
            )
        fresh["base_prediction"] = previous.get("base_prediction")
        fresh["finetuned_prediction"] = previous.get("finetuned_prediction")
        for metric_field in ("base_bertscore_f1", "finetuned_bertscore_f1"):
            if metric_field in previous:
                fresh[metric_field] = previous[metric_field]
    return fresh_records


def load_evaluation_model(
    base_model_id: str = BASE_MODEL_ID,
    adapter_id: str = ADAPTER_ID,
    *,
    base_model_revision: str | None = None,
    adapter_revision: str | None = None,
) -> tuple[Any, Any]:
    """Load one 4-bit base model and attach the published LoRA adapter."""
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for the 4-bit 7B evaluation path.")

    adapter_config = PeftConfig.from_pretrained(adapter_id, revision=adapter_revision)
    if adapter_config.base_model_name_or_path != base_model_id:
        raise ValueError(
            f"Adapter expects {adapter_config.base_model_name_or_path}, not {base_model_id}."
        )
    if not str(adapter_config.task_type).endswith("CAUSAL_LM"):
        raise ValueError(f"Unexpected adapter task type: {adapter_config.task_type}")
    if getattr(adapter_config, "bias", None) != "none":
        raise ValueError("disable_adapter() requires adapter bias='none' for this comparison.")

    tokenizer = AutoTokenizer.from_pretrained(
        adapter_id,
        revision=adapter_revision,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        revision=base_model_revision,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(
        base_model,
        adapter_id,
        revision=adapter_revision,
        is_trainable=False,
    )
    model.eval()
    model.config.use_cache = True
    if not hasattr(model, "disable_adapter"):
        raise RuntimeError("Installed PEFT lacks the context-managed disable_adapter() API.")
    return model, tokenizer


def run_paired_inference(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    prompts: Sequence[str],
    predictions_path: Path,
    *,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> None:
    """Checkpoint base and adapter completions after every generated sample."""
    with model.disable_adapter():
        for position, record in enumerate(tqdm(records, desc="Base model")):
            if record.get("base_prediction") is not None:
                continue
            record["base_prediction"] = generate_deterministic(
                model,
                tokenizer,
                prompts[position],
                max_new_tokens=max_new_tokens,
            )
            save_paired_predictions(predictions_path, records)

    model.set_adapter("default")
    model.eval()
    for position, record in enumerate(tqdm(records, desc="Base + adapter")):
        if record.get("finetuned_prediction") is not None:
            continue
        record["finetuned_prediction"] = generate_deterministic(
            model,
            tokenizer,
            prompts[position],
            max_new_tokens=max_new_tokens,
        )
        save_paired_predictions(predictions_path, records)


def _summary_rows(metric_values: Mapping[str, tuple[float, float]]) -> list[dict[str, Any]]:
    rows = []
    for metric_name, (base_value, finetuned_value) in metric_values.items():
        base_value = float(base_value)
        finetuned_value = float(finetuned_value)
        delta = finetuned_value - base_value
        relative_uplift = delta / base_value * 100.0 if base_value else None
        rows.append(
            {
                "Metric": metric_name,
                "Base": base_value,
                "Fine-tuned": finetuned_value,
                "Absolute delta": delta,
                "Relative uplift": relative_uplift,
            }
        )
    return rows


def calculate_metrics(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Calculate only BERTScore P/R/F1 and ROUGE-1/ROUGE-L."""
    import evaluate

    if len(records) != EXPECTED_VALIDATION_SIZE:
        raise ValueError(f"Expected {EXPECTED_VALIDATION_SIZE} paired records.")
    if any(
        row.get("base_prediction") is None or row.get("finetuned_prediction") is None
        for row in records
    ):
        raise ValueError("Both predictions must be present for every validation row.")

    references = [row["reference"] for row in records]
    base_predictions = [row["base_prediction"] for row in records]
    finetuned_predictions = [row["finetuned_prediction"] for row in records]

    rouge = evaluate.load("rouge")
    base_rouge = rouge.compute(predictions=base_predictions, references=references)
    finetuned_rouge = rouge.compute(predictions=finetuned_predictions, references=references)

    bertscore = evaluate.load("bertscore")
    bert_all = bertscore.compute(
        predictions=base_predictions + finetuned_predictions,
        references=references + references,
        lang="ru",
        model_type=BERTSCORE_MODEL_TYPE,
        rescale_with_baseline=False,
        batch_size=16,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    n = len(records)
    base_bert = {key: bert_all[key][:n] for key in ("precision", "recall", "f1")}
    finetuned_bert = {key: bert_all[key][n:] for key in ("precision", "recall", "f1")}

    for position, record in enumerate(records):
        record["base_bertscore_f1"] = float(base_bert["f1"][position])
        record["finetuned_bertscore_f1"] = float(finetuned_bert["f1"][position])

    metric_values = {
        "BERTScore Precision": (
            sum(base_bert["precision"]) / n,
            sum(finetuned_bert["precision"]) / n,
        ),
        "BERTScore Recall": (
            sum(base_bert["recall"]) / n,
            sum(finetuned_bert["recall"]) / n,
        ),
        "BERTScore F1": (sum(base_bert["f1"]) / n, sum(finetuned_bert["f1"]) / n),
        "ROUGE-1": (base_rouge["rouge1"], finetuned_rouge["rouge1"]),
        "ROUGE-L": (base_rouge["rougeL"], finetuned_rouge["rougeL"]),
    }
    return _summary_rows(metric_values)


def save_metric_artifacts(
    output_dir: Path,
    records: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    *,
    base_model_id: str,
    adapter_id: str,
    base_model_revision: str,
    adapter_revision: str,
    fingerprint: str,
    generation: Mapping[str, Any],
    library_versions: Mapping[str, str],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "eval_predictions.jsonl"
    save_paired_predictions(predictions_path, records)
    pd.DataFrame(summary_rows).to_csv(output_dir / "eval_summary.csv", index=False)

    payload = {
        "base_model": base_model_id,
        "adapter": adapter_id,
        "base_model_revision": base_model_revision,
        "adapter_revision": adapter_revision,
        "validation_size": EXPECTED_VALIDATION_SIZE,
        "evaluation_split": "held-out validation split: df.iloc[-50:]",
        "validation_set_fingerprint_sha256": fingerprint,
        "generation": dict(generation),
        "bertscore_model_type": BERTSCORE_MODEL_TYPE,
        "library_versions": dict(library_versions),
        "format_compliance": None,
        "format_compliance_note": (
            "No objective binary format is defined by the training instruction; "
            "the previously reported 91% figure has no reproducible procedure "
            "in the project artifacts and is excluded."
        ),
        "statistical_significance_tested": False,
        "metrics": summary_rows,
    }
    with (output_dir / "eval_metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    return payload


def run_evaluation(
    data_path: str,
    *,
    adapter_id: str = ADAPTER_ID,
    base_model_id: str = BASE_MODEL_ID,
    output_dir: str = "results",
    max_new_tokens: int = MAX_NEW_TOKENS,
    reset_predictions: bool = False,
) -> dict[str, Any]:
    """Run the full deterministic paired validation workflow."""
    _, validation_dataset = load_avito_dataset(data_path, test_size=EXPECTED_VALIDATION_SIZE)
    fresh_records, prompts = build_prediction_records(validation_dataset)
    fingerprint = validation_fingerprint(fresh_records, prompts)

    output_path = Path(output_dir)
    predictions_path = output_path / "eval_predictions.jsonl"
    records = resume_compatible_predictions(
        predictions_path,
        fresh_records,
        reset=reset_predictions,
    )

    base_model_revision = model_info(base_model_id).sha
    adapter_revision = model_info(adapter_id).sha
    library_versions = {
        name: importlib_metadata.version(name)
        for name in ("transformers", "peft", "bitsandbytes", "evaluate", "bert-score")
    }
    model, tokenizer = load_evaluation_model(
        base_model_id,
        adapter_id,
        base_model_revision=base_model_revision,
        adapter_revision=adapter_revision,
    )
    resolved_generation = generation_kwargs(tokenizer, max_new_tokens=max_new_tokens)
    run_paired_inference(
        model,
        tokenizer,
        records,
        prompts,
        predictions_path,
        max_new_tokens=max_new_tokens,
    )
    del model, tokenizer
    torch.cuda.empty_cache()

    summary_rows = calculate_metrics(records)
    return save_metric_artifacts(
        output_path,
        records,
        summary_rows,
        base_model_id=base_model_id,
        adapter_id=adapter_id,
        base_model_revision=base_model_revision,
        adapter_revision=adapter_revision,
        fingerprint=fingerprint,
        generation=resolved_generation,
        library_versions=library_versions,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", "--data_path", dest="data_path", required=True)
    parser.add_argument(
        "--adapter",
        "--model_path",
        dest="adapter_id",
        default=ADAPTER_ID,
        help="Published adapter ID or a compatible local adapter path.",
    )
    parser.add_argument(
        "--base-model",
        "--base_model_id",
        dest="base_model_id",
        default=BASE_MODEL_ID,
    )
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--reset-predictions", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_evaluation(
        args.data_path,
        adapter_id=args.adapter_id,
        base_model_id=args.base_model_id,
        output_dir=args.output_dir,
        max_new_tokens=args.max_new_tokens,
        reset_predictions=args.reset_predictions,
    )
    print(pd.DataFrame(payload["metrics"]).to_string(index=False))


if __name__ == "__main__":
    main()

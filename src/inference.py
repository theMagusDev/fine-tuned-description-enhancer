"""Interactive/demo inference with the published Avito LoRA adapter."""

import argparse

import torch
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .evaluation import ADAPTER_ID, BASE_MODEL_ID, build_inference_prompt
except ImportError:  # Support ``python src/inference.py``.
    from evaluation import ADAPTER_ID, BASE_MODEL_ID, build_inference_prompt


DEFAULT_INSTRUCTION = (
    "Отредактируй описание товара для Авито. Будь грамотным, добавь в текст "
    "структуру и привлекательность, а также строго придерживайся фактов из "
    "исходного текста."
)

load_dotenv()


def load_model(base_model_id: str = BASE_MODEL_ID, adapter_id: str = ADAPTER_ID):
    """Load the base model, adapter, and tokenizer saved with the adapter."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(adapter_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, adapter_id, is_trainable=False)
    if device == "cpu":
        model.to(device)
    model.eval()
    return model, tokenizer


def generate_description(
    model,
    tokenizer,
    *,
    instruction: str,
    category_context: str,
    title: str,
    original_description: str,
) -> str:
    """Generate a demo response using sampling (evaluation remains deterministic)."""
    prompt = build_inference_prompt(
        instruction,
        category_context,
        title,
        original_description,
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    input_length = inputs["input_ids"].shape[1]
    input_device = model.get_input_embeddings().weight.device
    inputs = {name: tensor.to(input_device) for name, tensor in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[:, input_length:]
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an improved Avito description")
    parser.add_argument("--base-model", default=BASE_MODEL_ID)
    parser.add_argument("--adapter", default=ADAPTER_ID)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--category-context", default="Не указана")
    parser.add_argument("--title", default="Не указан")
    parser.add_argument(
        "--input",
        dest="original_description",
        default="Продам велосипед, красный, едет нормально.",
        help="Original product description",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, tokenizer = load_model(args.base_model, args.adapter)
    result = generate_description(
        model,
        tokenizer,
        instruction=args.instruction,
        category_context=args.category_context,
        title=args.title,
        original_description=args.original_description,
    )
    print(result)


if __name__ == "__main__":
    main()

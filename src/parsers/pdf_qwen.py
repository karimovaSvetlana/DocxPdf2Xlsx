"""
PDF table extraction using Qwen3-VL model with parallel processing.

This parser uses the Qwen3-VL vision-language model to extract structured
table data from PDF pages with support for resumable processing.
"""

import os
import torch
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from pdf2image import convert_from_path
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from torch.utils.data import DataLoader, Dataset

import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.json_utils import clean_extracted_data, extract_json_from_text


BATCH_SIZE = 1
MODEL_PATH = "Qwen/Qwen3-VL-8B-Instruct"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


print(f"Loading model {MODEL_PATH}...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    # attn_implementation="flash_attention_2",  # Critical for speed and memory
    device_map="auto",
    trust_remote_code=True
).eval()

processor = AutoProcessor.from_pretrained(MODEL_PATH)


class PDFPagesDataset(Dataset):
    """Dataset for PDF pages with prompt preparation."""

    def __init__(self, pages, prompt):
        self.pages = pages
        self.prompt = prompt

    def __len__(self):
        return len(self.pages)

    def __getitem__(self, idx):
        page_image = self.pages[idx].convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": page_image,
                    },
                    {"type": "text", "text": self.prompt}
                ],
            }
        ]

        # Qwen3-VL requires special preparation via chat_template
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Prepare visual data + tokenization
        inputs = processor(
            text=[text],
            images=[page_image],
            padding=True,
            return_tensors="pt"
        )

        return {k: v.squeeze(0) for k, v in inputs.items()}


def load_processed_indices(log_path):
    """
    Load already processed page indices from log file.

    Returns:
        Dict mapping page index to output text
    """
    processed_indices = {}
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    # Store both index and text for final assembly
                    processed_indices[data["n"]] = data["output_text"]
                except:
                    continue
    return processed_indices


def process_pdf_parallel(pdf_path, output_dir, prompt_path):
    """
    Process PDF with parallel page processing and resume support.

    Args:
        pdf_path: Path to input PDF file
        output_dir: Directory for output files
        prompt_path: Path to prompt template file
    """
    file_stem = Path(pdf_path).stem
    target_dir = Path(output_dir) / file_stem
    target_dir.mkdir(parents=True, exist_ok=True)

    output_xlsx = target_dir / f"{file_stem}.xlsx"
    log_file = target_dir / "raw_outputs.jsonl"

    # STEP 1: Load progress
    processed_data = load_processed_indices(log_file)
    print(f"Found {len(processed_data)} already processed pages")

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading PDF...")
    pages = convert_from_path(pdf_path)

    # Prepare prompt
    json_schema = {
        "rows": [{
            "наименование": "", "рз": "", "пр": "", "цср": "", "вр": "",
            "2024_год": "", "2025_год": "", "2026_год": ""
        }]
    }
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()
    prompt_text = prompt_template.replace("{INPUTT}", json.dumps(json_schema, ensure_ascii=False))

    dataset = PDFPagesDataset(pages, prompt_text)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda batch: processor.tokenizer.pad(batch, return_tensors="pt"),
        num_workers=4,
    )

    # Fill results with already processed pages
    all_results = [None] * len(pages)
    for idx, text in processed_data.items():
        if idx < len(all_results):
            all_results[idx] = text

    print(f"Starting processing of {len(pages)} pages...")

    # STEP 2: Process with skipping already done pages
    for current_idx, batch in enumerate(tqdm(dataloader)):
        # Skip if this page is already processed
        if current_idx in processed_data:
            continue

        batch_cuda = {k: v.to(model.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

        with torch.no_grad():
            generated_ids = model.generate(
                **batch_cuda,
                max_new_tokens=4096,
                do_sample=False
            )

        input_len = batch_cuda["input_ids"].shape[1]

        with open(log_file, "a", encoding="utf-8") as f:
            for i in range(len(generated_ids)):
                # Calculate real page index
                real_page_idx = current_idx + i

                output_text = processor.decode(generated_ids[i][input_len:], skip_special_tokens=True)
                all_results[real_page_idx] = output_text

                log_entry = {
                    "n": real_page_idx,
                    "timestamp": datetime.now().strftime('%H:%M:%S'),
                    "output_text": output_text,
                    "parsed_text": extract_json_from_text(output_text),
                }
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    # STEP 3: Final assembly
    final_rows = []
    for i, text in enumerate(all_results):
        if text is None:
            continue

        parsed_json = extract_json_from_text(text)
        if parsed_json and "rows" in parsed_json:
            cleaned_rows = clean_extracted_data(parsed_json["rows"])
            final_rows.extend(cleaned_rows)

    if final_rows:
        pd.DataFrame(final_rows).to_excel(output_xlsx, index=False)
        print(f"Done. Saved to {output_xlsx}")


def main():
    """CLI entry point."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pdf_qwen.py <input.pdf> <output_dir> <prompt.txt>")
        print("Example: python pdf_qwen.py data/input/file.pdf data/output data/prompts/prompt.txt")
        sys.exit(1)

    pdf_path = sys.argv[1] if len(sys.argv) >= 2 else "data/input/2024_Прил15_закон.pdf"
    output_dir = sys.argv[2] if len(sys.argv) >= 3 else "data/output"
    prompt_path = sys.argv[3] if len(sys.argv) >= 4 else "./data/prompts/prompt_v1.txt"

    process_pdf_parallel(pdf_path, output_dir, prompt_path)


if __name__ == "__main__":
    main()

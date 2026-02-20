"""
PDF table extraction using GLM-OCR model.

This parser uses the GLM-OCR vision-language model to extract structured
table data from PDF pages and convert them to Excel format.
"""

import torch
import json
import re
import pandas as pd
from datetime import datetime
from pathlib import Path
from pdf2image import convert_from_path
from transformers import AutoProcessor, AutoModelForImageTextToText


MODEL_PATH = "zai-org/GLM-OCR"

# Detect device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

print("Loading model (this may take a while)...")
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    device_map="auto",
    trust_remote_code=True
)
model.eval()


def clean_extracted_data(rows):
    """
    Merge multi-line entries in extracted table data.

    Args:
        rows: List of row dictionaries

    Returns:
        List of cleaned row dictionaries
    """
    if not rows:
        return []

    cleaned = []
    current_row = None

    for row in rows:
        # Check if row contains financial data (year fields)
        has_data = any(str(row.get(k, "")).strip() for k in ["2024_год", "2025_год", "2026_год"])

        if not has_data and current_row is not None:
            # No numbers - append text to previous row's description
            current_row["наименование"] += " " + row.get("наименование", "")
        else:
            # Has numbers or first row - create new entry
            if current_row:
                cleaned.append(current_row)
            current_row = row

    if current_row:
        cleaned.append(current_row)

    return cleaned


def process_pdf_to_excel(pdf_path, output_xlsx):
    """
    Extract tables from PDF and save to Excel.

    Args:
        pdf_path: Path to input PDF file
        output_xlsx: Path to output Excel file
    """
    pages = convert_from_path(pdf_path)
    all_extracted_data = []

    json_schema = {
        "rows": [{
            "наименование": "", "рз": "", "пр": "", "цср": "", "вр": "",
            "2024_год": "", "2025_год": "", "2026_год": ""
        }]
    }

    prompt_text = f"""Please extract the information from the image in the following JSON format.

Rules:
1. "наименование" (Description) often spans multiple lines. You MUST merge these lines into a single string.
2. If a line has text but NO numbers, it is a continuation of the previous "наименование".
3. The numbers are usually at the bottom of the text block - use this to identify the end of a record.
4. Strict JSON output only.

Schema:
{json.dumps(json_schema, ensure_ascii=False)}
"""

    for i, page_image in enumerate(pages):
        start_time = datetime.now()
        print(f"[{start_time.strftime('%H:%M:%S')}] Processing page {i+1}...")

        # Prepare messages according to model format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": page_image.convert("RGB")},
                    {"type": "text", "text": prompt_text}
                ],
            }
        ]

        # Apply chat template
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device)

        # Remove extra keys (vLLM/Transformers compatibility)
        inputs.pop("token_type_ids", None)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=4096,
                do_sample=False  # Disable sampling for OCR (more deterministic)
            )

        # Decode only the generated tokens (skip input)
        input_len = inputs["input_ids"].shape[1]
        output_text = processor.decode(generated_ids[0][input_len:], skip_special_tokens=True)

        end_time = datetime.now()
        duration = str(end_time - start_time).split(".")[0]
        print(f"[{end_time.strftime('%H:%M:%S')}] Response from page {i+1}:\n{output_text[:100]}...\nDuration: {duration}")

        with open("output_texts.jsonl", "a") as f:
            f.write(json.dumps({"n": i, "output_text": output_text}, ensure_ascii=False) + '\n')

        try:
            json_str = re.search(r'\{.*\}', output_text, re.DOTALL).group()
            page_data = json.loads(json_str)
            if "rows" in page_data:
                all_extracted_data.extend(page_data["rows"])
            else:
                print(f"Unable to parse: {page_data}")
        except Exception:
            print(f"JSON parsing error on page {i+1}")

    if all_extracted_data:
        pd.DataFrame(all_extracted_data).to_excel(output_xlsx, index=False)
        print(f"Saved to {output_xlsx}")


def main():
    """CLI entry point."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pdf_glm_ocr.py <input.pdf> [output.xlsx]")
        sys.exit(1)

    pdf_path = sys.argv[1]
    output_xlsx = sys.argv[2] if len(sys.argv) >= 3 else "output.xlsx"

    process_pdf_to_excel(pdf_path, output_xlsx)


if __name__ == "__main__":
    main()
